"""Text processing utilities."""

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
        idx = abs(hash(w)) % vocab_size
        bow[idx] += 1
    return bow
