#!/usr/bin/env python3
"""Evaluation script for trained VLA model."""

import os
import sys

# Ensure deterministic Python hashing (must be set at interpreter startup).
if __name__ == "__main__" and os.environ.get("PYTHONHASHSEED") is None:
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import torch

from vla_cartpole.env import MiniCartPoleVisionEnv
from vla_cartpole.evaluation import evaluate_model
from vla_cartpole.models import MiniVLA
from vla_cartpole.utils.runtime import pick_device


def main():
    """Evaluate a trained model."""
    # Configuration
    device = pick_device()
    instruction = "keep the pole upright"
    
    # Create environment and model
    env = MiniCartPoleVisionEnv()
    model = MiniVLA().to(device)
    
    # Load trained weights (if available)
    try:
        model.load_state_dict(torch.load('model.pth', map_location=device))
        print("Loaded trained model from 'model.pth'")
    except FileNotFoundError:
        print("Warning: No trained model found. Using random weights.")
    
    model.eval()
    
    # Evaluate
    print("Testing trained model:")
    print("-" * 60)
    
    results = evaluate_model(
        env=env,
        model=model,
        instruction=instruction,
        num_episodes=5,
        max_steps=100,
        device=device
    )
    
    print("-" * 60)
    print(f"Average Reward: {results['avg_reward']:.3f} ± {results['std_reward']:.3f}")
    print(f"Average Length: {results['avg_length']:.1f} ± {results['std_length']:.1f}")


if __name__ == "__main__":
    main()
