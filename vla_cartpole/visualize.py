#!/usr/bin/env python3
"""Visualization script for trained VLA model behavior."""

import torch

from vla_cartpole.env import MiniCartPoleVisionEnv
from vla_cartpole.models import MiniVLA
from vla_cartpole.utils.text import make_bow_instruction
from vla_cartpole.utils.visualization import plot_observation_and_probs


def main():
    """Visualize model behavior step by step."""
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    bow = make_bow_instruction(instruction).unsqueeze(0).to(device)
    
    # Run episode and visualize
    obs, _ = env.reset()
    
    for step in range(10):
        img_t = torch.tensor(obs).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            probs = model(img_t, bow).squeeze().cpu().numpy()
            action = int(probs.argmax())
        
        obs, reward, done, _, _ = env.step(action)
        
        plot_observation_and_probs(obs, probs, action, reward, step)
        
        if done:
            print(f"Episode finished at step {step+1}")
            break


if __name__ == "__main__":
    main()
