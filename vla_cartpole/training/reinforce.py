"""Actor-Critic (A2C) training implementation."""

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


def train_vla(env, model, instruction, num_episodes=1000, lr=3e-4, 
              gamma=0.99, max_steps=200, device='cpu', print_every=50,
              entropy_coef=0.01, value_coef=0.5, actor_lr=None, critic_lr=None,
              gae_lambda=0.95):
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
    
    for episode in range(num_episodes):
        # Collect episode data
        obs, _ = env.reset()
        
        log_probs = []
        values = []
        rewards = []
        entropies = []
        dones = []
        
        last_obs = obs
        last_terminated = False
        last_truncated = False

        for step in range(max_steps):
            img_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Get action and value from model
            logits, value = model.get_action_and_value(img_t, bow)
            
            # Sample action
            action_dist = torch.distributions.Categorical(logits=logits)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            entropy = action_dist.entropy()
            
            # Take action in environment
            obs, reward, terminated, truncated, _ = env.step(action.item())
            last_obs = obs
            last_terminated = terminated
            last_truncated = truncated
            
            # Store data
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            entropies.append(entropy)
            dones.append(terminated)
            
            if terminated or truncated:
                break
        
        # Skip empty episodes
        if len(rewards) == 0:
            continue
        
        # Bootstrap value for truncated episodes, zero for true terminal
        if last_terminated:
            next_value = torch.tensor(0.0, device=device)
        else:
            with torch.no_grad():
                next_img_t = torch.tensor(last_obs, dtype=torch.float32).unsqueeze(0).to(device)
                next_value = model.get_value(next_img_t, bow).squeeze()

        # Compute GAE advantages and returns
        advantages_t, returns_t = compute_gae(
            rewards=rewards,
            values=values,
            next_value=next_value,
            dones=dones,
            gamma=gamma,
            lam=gae_lambda,
        )
        
        log_probs_t = torch.stack(log_probs).squeeze()
        values_t = torch.stack(values).squeeze()
        entropies_t = torch.stack(entropies).squeeze()
        
        # Normalize advantages for stability (important!)
        if len(advantages_t) > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        # Actor loss: policy gradient with advantage
        actor_loss = -(log_probs_t * advantages_t).mean()
        
        # Critic loss: MSE between predicted values and returns (use Huber for stability)
        critic_loss = F.smooth_l1_loss(values_t, returns_t)
        
        # Entropy bonus (maximize entropy = minimize negative entropy)
        entropy_loss = -entropies_t.mean()
        
        # Total loss
        loss = actor_loss + value_coef * critic_loss + entropy_coef * entropy_loss
        
        # Update model
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients (separately for actor to prevent catastrophic updates)
        torch.nn.utils.clip_grad_norm_(model.actor.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(model.critic.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(model.vision.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track metrics
        total_reward = sum(rewards)
        episode_rewards.append(total_reward)
        episode_lengths.append(len(rewards))
        
        if (episode + 1) % print_every == 0:
            avg_reward = np.mean(episode_rewards[-print_every:])
            avg_length = np.mean(episode_lengths[-print_every:])
            avg_entropy = entropies_t.mean().item()
            avg_value = values_t.mean().item()
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Reward: {avg_reward:.1f} | "
                  f"Length: {avg_length:.1f} | "
                  f"Entropy: {avg_entropy:.3f} | "
                  f"Value: {avg_value:.1f}")
    
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
