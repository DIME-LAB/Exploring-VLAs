"""Mini CartPole environment with visual observations."""

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw


class MiniCartPoleVisionEnv(gym.Env):
    """A simplified CartPole environment that renders visual observations.
    
    The environment provides RGB image observations (64x64x3) showing
    the cart position and pole angle. Actions are discrete: 0 (left) or 1 (right).
    
    Physics: Simplified inverted pendulum where cart acceleration affects pole angle.
    """
    
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            0, 255, shape=(64, 64, 3), dtype=np.uint8
        )
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
        self.x_threshold = 0.3  # Cart position limit (for visual bounds)
        self.theta_threshold = 0.5  # ~29 degrees

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
            
        Returns:
            observation: Initial RGB image observation
            info: Empty info dict
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Start with small random perturbations
        self.x = np.random.uniform(-0.02, 0.02)
        self.x_dot = np.random.uniform(-0.02, 0.02)
        self.theta = np.random.uniform(-0.05, 0.05)
        self.theta_dot = np.random.uniform(-0.05, 0.05)
        
        return self.render(), {}

    def step(self, action):
        """Execute one step in the environment.
        
        Args:
            action: 0 (push left) or 1 (push right)
            
        Returns:
            observation: RGB image observation
            reward: +1 for staying alive, with bonus for being upright and centered
            terminated: True if pole falls or cart goes out of bounds
            truncated: Always False
            info: Dict with state info
        """
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
        truncated = False
        
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
        
        return self.render(), reward, terminated, truncated, info

    def render(self):
        """Render the current state as an RGB image.
        
        Returns:
            numpy array of shape (64, 64, 3) with RGB values 0-255
        """
        img = Image.new("RGB", (64, 64), "white")
        d = ImageDraw.Draw(img)

        # Scale cart position to fit in image
        # Cart can move ±x_threshold, map to ±24 pixels from center
        x_scale = 24.0 / self.x_threshold
        cx = int(32 + self.x * x_scale)
        cx = np.clip(cx, 8, 56)  # Keep cart visible
        
        # Draw cart (black rectangle)
        d.rectangle([cx-8, 40, cx+8, 50], fill="black")

        # Draw pole (red line from cart top)
        pole_length_pixels = 20
        px = cx + int(pole_length_pixels * np.sin(self.theta))
        py = 40 - int(pole_length_pixels * np.cos(self.theta))
        d.line((cx, 40, px, py), fill="red", width=3)

        return np.array(img)
