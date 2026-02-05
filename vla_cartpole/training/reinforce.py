"""Actor-Critic (A2C) training implementation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from vla_cartpole.utils.text import make_bow_instruction


def compute_returns(rewards, gamma=0.99):
    """Compute discounted returns."""
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)
    return returns


def compute_gae(rewards, values, next_value, dones, gamma=0.99, lam=0.95):
    """Compute GAE advantages and bootstrapped returns."""
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=values[0].device)
    values_t = torch.stack(values).squeeze()
    dones_t = torch.tensor(dones, dtype=torch.float32, device=values_t.device)

    advantages = torch.zeros_like(rewards_t)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_val = next_value if t == len(rewards) - 1 else values_t[t + 1]
        next_nonterminal = 1.0 - dones_t[t]
        delta = rewards_t[t] + gamma * next_val * next_nonterminal - values_t[t]
        gae = delta + gamma * lam * next_nonterminal * gae
        advantages[t] = gae

    returns_t = advantages + values_t
    return advantages, returns_t


def compute_gae_batch(rewards_t, values_t, next_value_t, dones_t, gamma=0.99, lam=0.95):
    """Compute GAE for batched rollouts.

    Args:
        rewards_t: Tensor of shape [T, N]
        values_t: Tensor of shape [T, N]
        next_value_t: Tensor of shape [N]
        dones_t: Tensor of shape [T, N] with 1.0 for terminal transitions
    """
    T = rewards_t.shape[0]
    advantages = torch.zeros_like(rewards_t)
    gae = torch.zeros_like(next_value_t)

    for t in reversed(range(T)):
        next_val = next_value_t if t == T - 1 else values_t[t + 1]
        next_nonterminal = 1.0 - dones_t[t]
        delta = rewards_t[t] + gamma * next_val * next_nonterminal - values_t[t]
        gae = delta + gamma * lam * next_nonterminal * gae
        advantages[t] = gae

    returns_t = advantages + values_t
    return advantages, returns_t


def train_vla(env, model, instruction, num_episodes=1000, lr=3e-4, 
              gamma=0.99, max_steps=200, device='cpu', print_every=50,
              entropy_coef=0.01, value_coef=0.5, actor_lr=None, critic_lr=None,
              gae_lambda=0.95,
              num_envs: int = 1,
              rollout_steps: int = 32,
              checkpoint_every_steps: int | None = 1000,
              checkpoint_dir: str | Path = "checkpoints",
              checkpoint_latest_every_steps: int | None = None,
              checkpoint_latest_path: str | Path | None = "model.pth",
              eval_every_steps: int | None = 1000,
              eval_num_episodes: int = 5,
              eval_max_steps: int = 200,
              eval_env_factory=None):
    """Train the VLA model using Advantage Actor-Critic (A2C).
    
    Args:
        env: Gymnasium environment
        model: VLA model with actor and critic heads
        instruction: Text instruction string
        num_episodes: Number of training episodes
        lr: Learning rate (used if actor_lr/critic_lr not specified)
        gamma: Discount factor
        max_steps: Maximum steps per episode
        device: Device to run on
        print_every: Print progress every N episodes
        entropy_coef: Coefficient for entropy bonus
        value_coef: Coefficient for value loss
        actor_lr: Separate learning rate for actor (policy)
        critic_lr: Separate learning rate for critic (value)
        gae_lambda: Lambda for Generalized Advantage Estimation
        num_envs: Number of parallel environments to run (uses batched model forward)
        rollout_steps: Number of environment steps per optimization update
        checkpoint_every_steps: Save a checkpoint every N environment steps (None to disable)
        checkpoint_dir: Directory to write numbered checkpoints into
        checkpoint_latest_every_steps: Save latest weights every N environment steps (defaults to checkpoint_every_steps)
        checkpoint_latest_path: Optional path to also write the latest weights to (None to disable)
        eval_every_steps: Run evaluation every N environment steps (None to disable)
        eval_num_episodes: Number of evaluation episodes per eval run
        eval_max_steps: Max steps per evaluation episode
        eval_env_factory: Optional callable that returns a fresh eval environment
        
    Returns:
        Tuple of (episode_rewards, episode_lengths) lists
    """
    # Use separate optimizers for actor and critic for stability
    if actor_lr is None:
        actor_lr = lr
    if critic_lr is None:
        critic_lr = lr
    
    # Separate parameter groups
    actor_params = list(model.actor.parameters())
    critic_params = list(model.critic.parameters())
    shared_params = list(model.vision.parameters()) + list(model.text_embed.parameters())
    
    optimizer = optim.Adam([
        {'params': shared_params, 'lr': lr},
        {'params': actor_params, 'lr': actor_lr},
        {'params': critic_params, 'lr': critic_lr},
    ])
    
    model.train()
    
    episode_rewards = []
    episode_lengths = []
    
    bow = make_bow_instruction(instruction).unsqueeze(0).to(device)

    global_step = 0
    checkpoint_due_steps: list[int] = []
    checkpoint_latest_due_steps: list[int] = []
    eval_due_steps: list[int] = []

    checkpoint_dir_path = Path(checkpoint_dir) if checkpoint_every_steps else None
    latest_path = Path(checkpoint_latest_path) if checkpoint_latest_path else None
    if checkpoint_latest_every_steps is None:
        checkpoint_latest_every_steps = checkpoint_every_steps

    next_checkpoint_step = checkpoint_every_steps if checkpoint_every_steps else None
    next_checkpoint_latest_step = (
        checkpoint_latest_every_steps
        if (latest_path is not None and checkpoint_latest_every_steps)
        else None
    )
    next_eval_step = eval_every_steps if eval_every_steps else None

    eval_env = None

    if num_envs < 1:
        raise ValueError("num_envs must be >= 1")
    if rollout_steps < 1:
        raise ValueError("rollout_steps must be >= 1")

    envs = [env]
    if num_envs > 1:
        envs.extend(env.__class__() for _ in range(num_envs - 1))

    obs_list = []
    for e in envs:
        o, _ = e.reset()
        obs_list.append(o)

    running_rewards = [0.0 for _ in range(num_envs)]
    running_lengths = [0 for _ in range(num_envs)]
    episodes_completed = 0

    while episodes_completed < num_episodes:
        log_probs_steps = []
        values_steps = []
        rewards_steps = []
        entropies_steps = []
        dones_steps = []

        for _ in range(rollout_steps):
            obs_batch = np.stack(obs_list, axis=0)
            img_t = torch.tensor(obs_batch, dtype=torch.float32, device=device)
            bow_batch = bow.expand(num_envs, -1)

            logits, value = model.get_action_and_value(img_t, bow_batch)
            action_dist = torch.distributions.Categorical(logits=logits)
            action = action_dist.sample()

            log_probs_steps.append(action_dist.log_prob(action))
            values_steps.append(value)
            entropies_steps.append(action_dist.entropy())

            actions_cpu = action.detach().cpu().numpy().tolist()

            next_obs_list = [None for _ in range(num_envs)]
            reward_list = [0.0 for _ in range(num_envs)]
            done_list = [False for _ in range(num_envs)]

            for i, e in enumerate(envs):
                next_obs, reward, terminated, truncated, _ = e.step(actions_cpu[i])
                running_rewards[i] += float(reward)
                running_lengths[i] += 1

                done = bool(terminated or truncated or (running_lengths[i] >= max_steps))
                reward_list[i] = float(reward)
                done_list[i] = done

                if done:
                    episode_rewards.append(running_rewards[i])
                    episode_lengths.append(running_lengths[i])
                    episodes_completed += 1

                    running_rewards[i] = 0.0
                    running_lengths[i] = 0

                    next_obs, _ = e.reset()

                next_obs_list[i] = next_obs

            obs_list = next_obs_list

            rewards_steps.append(torch.tensor(reward_list, dtype=torch.float32, device=device))
            dones_steps.append(torch.tensor(done_list, dtype=torch.float32, device=device))

            global_step += num_envs
            if next_checkpoint_step is not None and checkpoint_every_steps:
                while global_step >= next_checkpoint_step:
                    checkpoint_due_steps.append(next_checkpoint_step)
                    next_checkpoint_step += checkpoint_every_steps
            if next_checkpoint_latest_step is not None and checkpoint_latest_every_steps:
                while global_step >= next_checkpoint_latest_step:
                    checkpoint_latest_due_steps.append(next_checkpoint_latest_step)
                    next_checkpoint_latest_step += checkpoint_latest_every_steps
            if next_eval_step is not None and eval_every_steps:
                while global_step >= next_eval_step:
                    eval_due_steps.append(next_eval_step)
                    next_eval_step += eval_every_steps

            if episodes_completed >= num_episodes:
                break

        # Bootstrap value at the end of rollout
        with torch.no_grad():
            obs_batch = np.stack(obs_list, axis=0)
            next_img_t = torch.tensor(obs_batch, dtype=torch.float32, device=device)
            bow_batch = bow.expand(num_envs, -1)
            next_value = model.get_value(next_img_t, bow_batch)

        log_probs_t = torch.stack(log_probs_steps)  # [T, N]
        values_t = torch.stack(values_steps)         # [T, N]
        entropies_t = torch.stack(entropies_steps)   # [T, N]
        rewards_t = torch.stack(rewards_steps)       # [T, N]
        dones_t = torch.stack(dones_steps)           # [T, N]

        advantages_t, returns_t = compute_gae_batch(
            rewards_t=rewards_t,
            values_t=values_t,
            next_value_t=next_value,
            dones_t=dones_t,
            gamma=gamma,
            lam=gae_lambda,
        )

        # Normalize advantages for stability (important!)
        if advantages_t.numel() > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        actor_loss = -(log_probs_t * advantages_t).mean()
        critic_loss = F.smooth_l1_loss(values_t, returns_t)
        entropy_loss = -entropies_t.mean()
        loss = actor_loss + value_coef * critic_loss + entropy_coef * entropy_loss

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.actor.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(model.critic.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(model.vision.parameters(), max_norm=1.0)

        optimizer.step()

        if episodes_completed > 0 and (episodes_completed % print_every == 0):
            avg_reward = np.mean(episode_rewards[-print_every:])
            avg_length = np.mean(episode_lengths[-print_every:])
            avg_entropy = entropies_t.mean().item()
            avg_value = values_t.mean().item()
            print(
                f"Episodes {episodes_completed}/{num_episodes} | "
                f"Reward: {avg_reward:.1f} | "
                f"Length: {avg_length:.1f} | "
                f"Entropy: {avg_entropy:.3f} | "
                f"Value: {avg_value:.1f}"
            )
        
        if checkpoint_due_steps and checkpoint_dir_path is not None:
            checkpoint_dir_path.mkdir(parents=True, exist_ok=True)
            for due_step in checkpoint_due_steps:
                checkpoint_path = checkpoint_dir_path / f"model_step_{due_step}.pth"
                torch.save(model.state_dict(), checkpoint_path)
                print(f"[checkpoint step {due_step}] saved to '{checkpoint_path}'")
            checkpoint_due_steps.clear()

        if checkpoint_latest_due_steps and latest_path is not None:
            # Only need the most recent step because this file is overwritten.
            torch.save(model.state_dict(), latest_path)
            checkpoint_latest_due_steps.clear()

        if eval_due_steps:
            from vla_cartpole.evaluation import evaluate_model

            if eval_env is None:
                if eval_env_factory is not None:
                    eval_env = eval_env_factory()
                else:
                    eval_env = env.__class__()

            was_training = model.training
            for due_step in eval_due_steps:
                results = evaluate_model(
                    env=eval_env,
                    model=model,
                    instruction=instruction,
                    num_episodes=eval_num_episodes,
                    max_steps=eval_max_steps,
                    device=device,
                    verbose=False,
                )
                print(
                    f"[eval step {due_step}] "
                    f"avg_reward={results['avg_reward']:.2f}±{results['std_reward']:.2f} | "
                    f"avg_length={results['avg_length']:.1f}±{results['std_length']:.1f}"
                )
            if was_training:
                model.train()
            eval_due_steps.clear()
    
    model.eval()
    return episode_rewards, episode_lengths


# Keep old function for backward compatibility
def collect_episode(env, model, instruction, max_steps=100, device='cpu'):
    """Collect one episode of experience."""
    obs, _ = env.reset()
    bow = make_bow_instruction(instruction).unsqueeze(0).to(device)
    
    observations = []
    actions = []
    rewards = []
    log_probs = []
    
    for step in range(max_steps):
        img_t = torch.tensor(obs).float().unsqueeze(0).to(device)
        logits = model(img_t, bow, return_logits=True)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        obs_next, reward, terminated, truncated, _ = env.step(action.item())
        
        observations.append(obs)
        actions.append(action.item())
        rewards.append(reward)
        log_probs.append(log_prob)
        
        obs = obs_next
        
        if terminated or truncated:
            break
    
    return {
        'observations': observations,
        'actions': actions,
        'rewards': rewards,
        'log_probs': log_probs,
        'episode_length': len(rewards)
    }
