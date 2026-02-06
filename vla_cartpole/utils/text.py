"""Text processing utilities."""

import hashlib
import torch


def make_bow_instruction(text, vocab_size=1000):
    """Convert text instruction to bag-of-words encoding.
    
    Args:
        text: Input text string
        vocab_size: Size of vocabulary for hashing
        
    Returns:
        Tensor of shape (vocab_size,) with word counts
    """
    bow = torch.zeros(vocab_size)
    for w in text.lower().split():
        # Use a stable hash so indices are reproducible across runs/machines.
        # (Python's built-in `hash()` is intentionally randomized unless
        # PYTHONHASHSEED is set at interpreter startup.)
        h = hashlib.sha256(w.encode("utf-8")).digest()
        idx = int.from_bytes(h[:8], byteorder="little", signed=False) % vocab_size
        bow[idx] += 1
    return bow
