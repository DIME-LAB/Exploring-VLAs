"""Mini CartPole environment with visual observations.

This environment is intentionally simple and pixel-based. For training speed it uses a
NumPy renderer (no PIL) and implements Gymnasium-style seeding + time-limit truncation.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding


class MiniCartPoleVisionEnv(gym.Env):
    """A simplified CartPole environment that renders visual observations.
    
    The environment provides RGB image observations (64x256x3) showing
    the cart position and pole angle. Actions are discrete: 0 (left) or 1 (right).
    
    Physics: Simplified inverted pendulum where cart acceleration affects pole angle.
    """
    
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, *, max_episode_steps: int = 200, render_mode: str | None = "rgb_array"):
        super().__init__()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            0, 255, shape=(64, 256, 3), dtype=np.uint8
        )
        self.max_episode_steps = int(max_episode_steps)
        self.render_mode = render_mode
        # State variables
        self.x = 0.0           # Cart position
        self.x_dot = 0.0       # Cart velocity
        self.theta = 0.0       # Pole angle (radians, 0 = upright)
        self.theta_dot = 0.0   # Pole angular velocity
        
        # Physics constants
        self.gravity = 9.8
        self.cart_mass = 1.0
        self.pole_mass = 0.1
        self.pole_length = 0.5  # Half the pole length
        self.force_mag = 10.0
        self.dt = 0.02  # Time step
        
        # Bounds
        self.x_threshold = 1.2  # Cart position limit (for visual bounds)
        self.theta_threshold = 0.5  # ~29 degrees
        self._step_count = 0

        self.np_random = None
        self._blank = np.full((64, 256, 3), 255, dtype=np.uint8)

    def reset(self, *, seed=None, options=None):
        """Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
            
        Returns:
            observation: Initial RGB image observation
            info: Empty info dict
        """
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random, _ = seeding.np_random(seed)
        self._step_count = 0
        
        # Start with small random perturbations
        self.x = float(self.np_random.uniform(-0.02, 0.02))
        self.x_dot = float(self.np_random.uniform(-0.02, 0.02))
        self.theta = float(self.np_random.uniform(-0.05, 0.05))
        self.theta_dot = float(self.np_random.uniform(-0.05, 0.05))
        
        return self._get_obs(), {}

    def step(self, action):
        """Execute one step in the environment.
        
        Args:
            action: 0 (push left) or 1 (push right)
            
        Returns:
            observation: RGB image observation
            reward: +1 for staying alive, with bonus for being upright and centered
            terminated: True if pole falls or cart goes out of bounds
            truncated: True if time limit reached
            info: Dict with state info
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be 0 or 1.")

        # Apply force based on action
        force = self.force_mag if action == 1 else -self.force_mag
        
        # Physics calculations (simplified inverted pendulum)
        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)
        
        total_mass = self.cart_mass + self.pole_mass
        pole_mass_length = self.pole_mass * self.pole_length
        
        # Angular acceleration
        temp = (force + pole_mass_length * self.theta_dot**2 * sin_theta) / total_mass
        theta_acc = (self.gravity * sin_theta - cos_theta * temp) / (
            self.pole_length * (4.0/3.0 - self.pole_mass * cos_theta**2 / total_mass)
        )
        
        # Linear acceleration
        x_acc = temp - pole_mass_length * theta_acc * cos_theta / total_mass
        
        # Update state using Euler integration
        self.x += self.dt * self.x_dot
        self.x_dot += self.dt * x_acc
        self.theta += self.dt * self.theta_dot
        self.theta_dot += self.dt * theta_acc
        
        # Check termination
        terminated = (
            abs(self.theta) > self.theta_threshold or
            abs(self.x) > self.x_threshold
        )
        self._step_count += 1
        truncated = self._step_count >= self.max_episode_steps
        
        # Reward shaping:
        # - Base reward of +1 for each step alive
        # - Bonus for keeping pole upright (max +0.5 when perfectly vertical)
        # - Bonus for keeping cart centered (max +0.5 when at center)
        if not terminated:
            alive_reward = 1.0
            angle_bonus = 0.5 * (1.0 - abs(self.theta) / self.theta_threshold)
            position_bonus = 0.5 * (1.0 - abs(self.x) / self.x_threshold)
            reward = alive_reward + angle_bonus + position_bonus
        else:
            reward = 0.0  # No reward on terminal state
        
        info = {
            'x': self.x,
            'x_dot': self.x_dot,
            'theta': self.theta,
            'theta_dot': self.theta_dot
        }
        
        return self._get_obs(), reward, terminated, truncated, info

    def _render_rgb_array(self) -> np.ndarray:
        """Render the current state as an RGB image (rgb_array).

        Returns:
            numpy array of shape (64, 256, 3) with RGB values 0-255
        """
        img = self._blank.copy()

        # Draw ground line across the full width for visual reference
        img[52:54, :] = (180, 180, 180)

        # Scale cart position to fit in image
        # Cart can move +/- x_threshold, map to +/- 96 pixels from center
        x_scale = 96.0 / self.x_threshold
        cx = int(128 + self.x * x_scale)
        cx = int(np.clip(cx, 16, 240))  # Keep cart visible

        # Draw cart (black rectangle) — 32px wide, 11px tall
        x0 = max(cx - 16, 0)
        x1 = min(cx + 16, 255)
        img[40:51, x0 : x1 + 1] = 0

        # Draw pole (red line from cart top) — 30px long, 4px thick
        pole_length_pixels = 30
        px = cx + int(pole_length_pixels * np.sin(self.theta))
        py = 40 - int(pole_length_pixels * np.cos(self.theta))
        px = int(np.clip(px, 0, 255))
        py = int(np.clip(py, 0, 63))
        self._draw_line(img, cx, 40, px, py, color=(255, 0, 0), thickness=4)

        return img

    def _get_obs(self) -> np.ndarray:
        return self._render_rgb_array()

    def render(self):
        if self.render_mode not in (None, "rgb_array"):
            raise NotImplementedError(f"render_mode={self.render_mode!r} not supported.")
        return self._render_rgb_array()

    @staticmethod
    def _draw_line(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, *, color, thickness: int = 1):
        """Draw a line by sampling points; fast enough for small 64x64 images."""
        dx = x1 - x0
        dy = y1 - y0
        steps = int(max(abs(dx), abs(dy), 1))
        xs = np.linspace(x0, x1, steps + 1).round().astype(int)
        ys = np.linspace(y0, y1, steps + 1).round().astype(int)

        r, g, b = color
        half = max(int(thickness // 2), 0)
        for x, y in zip(xs, ys):
            x_min = max(x - half, 0)
            x_max = min(x + half, img.shape[1] - 1)
            y_min = max(y - half, 0)
            y_max = min(y + half, img.shape[0] - 1)
            img[y_min : y_max + 1, x_min : x_max + 1, 0] = r
            img[y_min : y_max + 1, x_min : x_max + 1, 1] = g
            img[y_min : y_max + 1, x_min : x_max + 1, 2] = b

    def close(self):
        return None
