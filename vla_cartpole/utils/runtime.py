"""Runtime utilities (device selection, seeding).

Keep this module lightweight: avoid importing heavy deps at import time.
"""

from __future__ import annotations


def pick_device():
    """Pick the best available torch device: cuda -> mps -> cpu."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int | None):
    """Seed Python, NumPy, and Torch RNGs (best-effort).

    Notes:
    - This doesn't force deterministic kernels; it just sets RNG seeds.
    - Environment seeding should be handled via `env.reset(seed=...)`.
    """
    if seed is None:
        return

    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

