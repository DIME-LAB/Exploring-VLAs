#!/usr/bin/env python3
"""Main training script for VLA CartPole."""

import os
import sys

# Ensure deterministic Python hashing (must be set at interpreter startup).
if __name__ == "__main__" and os.environ.get("PYTHONHASHSEED") is None:
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import torch

from vla_cartpole.env import MiniCartPoleVisionEnv
from vla_cartpole.models import MiniVLA
from vla_cartpole.training import train_vla
from vla_cartpole.utils.visualization import plot_training_progress


def main(num_episodes: int = 6000):
    """Train the VLA model."""
    # Configuration
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    instruction = "keep the pole upright"
    print(f"Training with instruction: '{instruction}'")
    print("-" * 60)
    
    # Create environment and model
    env = MiniCartPoleVisionEnv()
    model = MiniVLA().to(device)
    
    # Print model size
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print("-" * 60)
    
    # Train the model using Actor-Critic
    rewards, lengths = train_vla(
        env=env,
        model=model,
        instruction=instruction,
        num_episodes=num_episodes,
        lr=5e-4,  # Shared feature learning rate
        actor_lr=5e-4,  # Policy learning rate
        critic_lr=1e-3,  # Higher for faster value learning
        gamma=0.99,
        max_steps=200,
        device=device,
        print_every=50,
        entropy_coef=0.002,  # Small entropy bonus
        value_coef=0.5,  # Balanced critic weight
        gae_lambda=0.95,
        num_envs=50,  # More parallel environments for stable updates
        rollout_steps=32,
        checkpoint_every_steps=100_000,
        checkpoint_dir="checkpoints",
        checkpoint_latest_every_steps=1000,
        checkpoint_latest_path="model.pth",
        eval_every_steps=100_000,
        eval_num_episodes=5,
        eval_max_steps=200,
    )
    
    # Plot training progress
    plot_training_progress(rewards, lengths)
    
    # Print final statistics
    print(f"\nFinal performance:")
    print(f"  Last 20 episodes - Avg Reward: {sum(rewards[-20:])/len(rewards[-20:]):.3f}, "
          f"Avg Length: {sum(lengths[-20:])/len(lengths[-20:]):.1f}")
    print(f"  Best episode - Reward: {max(rewards):.3f}, Length: {lengths[rewards.index(max(rewards))]}")
    
    # Save model
    torch.save(model.state_dict(), 'model.pth')
    print(f"\nModel saved to 'model.pth'")


if __name__ == "__main__":
    main()
