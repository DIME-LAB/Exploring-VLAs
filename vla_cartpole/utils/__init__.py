"""Utility functions."""

__all__ = ["make_bow_instruction"]


def __getattr__(name: str):
    # Lazy import to keep package import light (avoid importing torch unless needed).
    if name == "make_bow_instruction":
        from vla_cartpole.utils.text import make_bow_instruction

        return make_bow_instruction
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
