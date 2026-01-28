"""Visualization utilities."""

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_observation_and_probs(obs, probs, action, reward, step_num=None):
    """Plot observation and action probabilities side by side.
    
    Args:
        obs: Observation image (numpy array)
        probs: Action probabilities (numpy array)
        action: Action taken (int)
        reward: Reward received (float)
        step_num: Optional step number for title
    """
    plt.figure(figsize=(8, 3))
    
    # Observation
    plt.subplot(1, 2, 1)
    plt.imshow(obs)
    plt.axis("off")
    title = f"Step {step_num+1} | Reward: {reward:.3f}" if step_num is not None else f"Reward: {reward:.3f}"
    plt.title(title)
    
    # Action probabilities
    plt.subplot(1, 2, 2)
    plt.bar(["Left(0)", "Right(1)"], probs, color=['blue', 'green'])
    plt.ylim(0, 1)
    plt.ylabel("Probability")
    action_name = 'Right' if action == 1 else 'Left'
    plt.title(f"Action Probabilities\nAction: {action} ({action_name})")
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


def plot_training_progress(rewards, lengths, window=20):
    """Plot training progress with rewards and episode lengths.
    
    Args:
        rewards: List of episode rewards
        lengths: List of episode lengths
        window: Window size for moving average
    """
    plt.figure(figsize=(12, 4))

    # Rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training: Episode Rewards')
    plt.grid(True, alpha=0.3)

    # Moving average
    if len(rewards) > window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg, 'r-', linewidth=2, 
                label=f'{window}-episode avg')
        plt.legend()

    # Episode lengths
    plt.subplot(1, 2, 2)
    plt.plot(lengths)
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')
    plt.title('Training: Episode Lengths')
    plt.grid(True, alpha=0.3)

    if len(lengths) > window:
        moving_avg = np.convolve(lengths, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(lengths)), moving_avg, 'r-', linewidth=2, 
                label=f'{window}-episode avg')
        plt.legend()

    plt.tight_layout()
    plt.show()
