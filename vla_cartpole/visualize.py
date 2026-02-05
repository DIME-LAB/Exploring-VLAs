#!/usr/bin/env python3
"""Visualization script for trained VLA model behavior."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure deterministic Python hashing (must be set at interpreter startup).
if __name__ == "__main__" and os.environ.get("PYTHONHASHSEED") is None:
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

# Allow running as a script (`python vla_cartpole/visualize.py`) as well as a module
# (`python -m vla_cartpole.visualize`) by ensuring the repo root is on sys.path.
if __name__ == "__main__":
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

import numpy as np
import torch
from PIL import Image

from vla_cartpole.evaluation import evaluate_model

from vla_cartpole.env import MiniCartPoleVisionEnv
from vla_cartpole.models import MiniVLA
from vla_cartpole.utils.text import make_bow_instruction
from vla_cartpole.utils.visualization import plot_episode_summary


def _resolve_weights_path(weights: str | None, checkpoint_step: int | None) -> Path | None:
    if weights and checkpoint_step:
        raise ValueError("Use either --weights or --checkpoint-step, not both.")
    if checkpoint_step is not None:
        return Path("checkpoints") / f"model_step_{checkpoint_step}.pth"
    if weights:
        return Path(weights)
    return Path("model.pth")


def _save_gif(frames: list[np.ndarray], path: Path, fps: int = 20):
    if not frames:
        raise ValueError("No frames to save.")
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame) for frame in frames]
    duration_ms = int(1000 / max(fps, 1))
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )


def main(argv: list[str] | None = None):
    """Run a full episode rollout and plot metrics/frames."""
    parser = argparse.ArgumentParser(description="Visualize trained VLA CartPole behavior.")
    parser.add_argument("--weights", type=str, default=None, help="Path to .pth weights file (default: model.pth)")
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="Convenience for loading checkpoints/model_step_<N>.pth",
    )
    parser.add_argument("--instruction", type=str, default="keep the pole upright")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-plot", type=str, default=None, help="Path to save the summary plot PNG")
    parser.add_argument("--save-gif", type=str, default=None, help="Path to save an episode GIF")
    parser.add_argument("--gif-fps", type=int, default=20)
    parser.add_argument("--no-show", action="store_true", help="Do not open matplotlib windows")
    args = parser.parse_args(argv)

    weights_path = _resolve_weights_path(args.weights, args.checkpoint_step)

    # Configuration
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    instruction = args.instruction
    
    # Create environment and model
    env = MiniCartPoleVisionEnv()
    model = MiniVLA().to(device)
    
    # Load trained weights (if available)
    try:
        if weights_path is None:
            raise FileNotFoundError
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded trained model from '{weights_path}'")
    except FileNotFoundError:
        print("Warning: No trained model found. Using random weights.")
    
    model.eval()
    bow = make_bow_instruction(instruction).unsqueeze(0).to(device)
    
    # Optional quick scalar evaluation
    print("Quick eval (deterministic):")
    results = evaluate_model(
        env=MiniCartPoleVisionEnv(),
        model=model,
        instruction=instruction,
        num_episodes=5,
        max_steps=args.max_steps,
        device=device,
        verbose=True,
    )
    print(
        f"Avg Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f} | "
        f"Avg Length: {results['avg_length']:.1f} ± {results['std_length']:.1f}"
    )

    # Run a full episode rollout and capture everything
    obs, _ = env.reset(seed=args.seed)

    frames: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    values: list[float] = []
    probs_left: list[float] = []
    probs_right: list[float] = []
    xs: list[float] = []
    thetas: list[float] = []

    for step in range(args.max_steps):
        frames.append(obs)
        img_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            logits, value_t = model.get_action_and_value(img_t, bow)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            action = int(np.argmax(probs))
            value = float(value_t.squeeze(0).item())

        obs, reward, terminated, truncated, info = env.step(action)

        actions.append(action)
        rewards.append(float(reward))
        values.append(value)
        probs_left.append(float(probs[0]))
        probs_right.append(float(probs[1]))
        xs.append(float(info.get("x", np.nan)))
        thetas.append(float(info.get("theta", np.nan)))

        if terminated or truncated:
            print(f"Episode finished at step {step+1}")
            break

    if args.save_gif:
        _save_gif(frames, Path(args.save_gif), fps=args.gif_fps)
        print(f"Saved GIF to '{args.save_gif}'")

    fig = plot_episode_summary(
        frames=frames,
        rewards=rewards,
        actions=actions,
        probs_left=probs_left,
        probs_right=probs_right,
        values=values,
        xs=xs,
        thetas=thetas,
        title=f"VLA CartPole rollout ({weights_path})",
    )

    if args.save_plot:
        out_path = Path(args.save_plot)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot to '{out_path}'")

    if not args.no_show:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
