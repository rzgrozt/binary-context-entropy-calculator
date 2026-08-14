"""Small deterministic composition for multi-line scientific UI copy."""


def joined_text(parts: tuple[str, ...]) -> str:
    """Join source-wrapped copy without introducing rendered whitespace."""
    return "".join(parts)
