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


def plot_episode_summary(
    *,
    frames,
    rewards,
    actions,
    probs_left,
    probs_right,
    values,
    xs,
    thetas,
    title="Episode summary",
    num_frame_samples: int = 8,
):
    """Plot a single episode rollout summary.

    Shows a small frame strip plus time series for reward/value, policy probs/actions,
    and environment state (x/theta).
    """
    T = len(rewards)
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.2, 1.2])

    ax_frames = fig.add_subplot(gs[0, :])
    ax_reward = fig.add_subplot(gs[1, 0])
    ax_policy = fig.add_subplot(gs[1, 1])
    ax_state = fig.add_subplot(gs[2, :])

    ax_frames.set_title(title)
    ax_frames.axis("off")

    if frames:
        sample_count = min(num_frame_samples, len(frames))
        idxs = np.linspace(0, len(frames) - 1, sample_count).astype(int)
        strip = np.concatenate([frames[i] for i in idxs], axis=1)
        ax_frames.imshow(strip)

    steps = np.arange(T)

    ax_reward.plot(steps, rewards, label="reward", linewidth=1)
    ax_reward.plot(steps, values, label="value", linewidth=1)
    ax_reward.set_xlabel("step")
    ax_reward.set_title("Reward / Value")
    ax_reward.grid(True, alpha=0.3)
    ax_reward.legend()

    ax_policy.plot(steps, probs_left, label="P(left)", linewidth=1)
    ax_policy.plot(steps, probs_right, label="P(right)", linewidth=1)
    ax_policy.scatter(steps, actions, s=10, label="action (0/1)", alpha=0.6)
    ax_policy.set_ylim(-0.05, 1.05)
    ax_policy.set_xlabel("step")
    ax_policy.set_title("Policy")
    ax_policy.grid(True, alpha=0.3)
    ax_policy.legend()

    ax_state.plot(steps, xs, label="x", linewidth=1)
    ax_state.plot(steps, thetas, label="theta", linewidth=1)
    ax_state.set_xlabel("step")
    ax_state.set_title("State")
    ax_state.grid(True, alpha=0.3)
    ax_state.legend()

    fig.tight_layout()
    return fig
