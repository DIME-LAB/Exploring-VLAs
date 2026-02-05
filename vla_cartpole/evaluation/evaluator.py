"""Model evaluation utilities."""

import numpy as np
import torch

from vla_cartpole.utils.text import make_bow_instruction


def evaluate_model(env, model, instruction, num_episodes=5, max_steps=100, device='cpu', verbose=True):
    """Evaluate a trained model on multiple episodes.
    
    Args:
        env: Gymnasium environment
        model: Trained VLA model (should be in eval mode)
        instruction: Text instruction string
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        device: Device to run on ('cpu' or 'cuda')
        verbose: If True, prints per-episode results
        
    Returns:
        Dictionary with keys: rewards, lengths, avg_reward, std_reward, avg_length, std_length
    """
    model.eval()
    bow = make_bow_instruction(instruction).unsqueeze(0).to(device)
    
    test_rewards = []
    test_lengths = []
    
    for test_ep in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            img_t = torch.tensor(obs).float().unsqueeze(0).to(device)
            
            with torch.no_grad():
                action_prob = model(img_t, bow)
                action = torch.argmax(action_prob, dim=-1).item()
            
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        test_rewards.append(total_reward)
        test_lengths.append(steps)
        if verbose:
            print(f"Episode {test_ep+1}: Reward={total_reward:.3f}, Length={steps}")
    
    return {
        'rewards': test_rewards,
        'lengths': test_lengths,
        'avg_reward': np.mean(test_rewards),
        'std_reward': np.std(test_rewards),
        'avg_length': np.mean(test_lengths),
        'std_length': np.std(test_lengths),
    }
