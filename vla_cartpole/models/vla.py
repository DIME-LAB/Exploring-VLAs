"""Vision-Language-Action (VLA) model implementation with Actor-Critic."""

import torch
import torch.nn as nn


class MiniVLA(nn.Module):
    """Actor-Critic Vision-Language-Action model for instruction-following.
    
    The model combines visual observations with text instructions to
    predict action probabilities (actor) and state values (critic).
    
    Args:
        num_actions: Number of possible actions (default: 2)
        vocab_size: Size of vocabulary for bag-of-words encoding (default: 1000)
        embed_dim: Dimension of text embedding (default: 32)
    """
    
    def __init__(self, num_actions=2, vocab_size=1000, embed_dim=32):
        super().__init__()

        # Shared vision encoder
        self.vision = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),   # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # 16x16 -> 8x8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
        )

        # Language encoder
        self.text_embed = nn.Sequential(
            nn.Linear(vocab_size, embed_dim),
            nn.ReLU(),
        )

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(256 + embed_dim, 64),
            nn.Tanh(),
            nn.Linear(64, num_actions)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(256 + embed_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using orthogonal initialization (good for RL)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Smaller init for policy output (encourages exploration)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)
        
        # Standard init for value output
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.zeros_(self.critic[-1].bias)

    def forward(self, image, bow_text, return_logits=False):
        """Forward pass - returns policy logits/probs and optionally value."""
        # Normalize and convert image to CHW format
        v = image / 255.0
        v = v.permute(0, 3, 1, 2)  # NHWC -> NCHW
        features = self.vision(v)

        # Encode text
        t = self.text_embed(bow_text)

        # Fuse vision and language
        fused = torch.cat([features, t], dim=-1)
        
        # Get policy logits
        logits = self.actor(fused)
        
        if return_logits:
            return logits
        
        action_prob = torch.softmax(logits, dim=-1)
        return action_prob
    
    def get_action_and_value(self, image, bow_text):
        """Get action distribution and value estimate."""
        # Normalize and convert image to CHW format
        v = image / 255.0
        v = v.permute(0, 3, 1, 2)  # NHWC -> NCHW
        features = self.vision(v)

        # Encode text
        t = self.text_embed(bow_text)

        # Fuse vision and language
        fused = torch.cat([features, t], dim=-1)
        
        # Get policy and value
        logits = self.actor(fused)
        value = self.critic(fused)
        
        return logits, value.squeeze(-1)
    
    def get_value(self, image, bow_text):
        """Get value estimate only."""
        v = image / 255.0
        v = v.permute(0, 3, 1, 2)
        features = self.vision(v)
        t = self.text_embed(bow_text)
        fused = torch.cat([features, t], dim=-1)
        return self.critic(fused).squeeze(-1)
