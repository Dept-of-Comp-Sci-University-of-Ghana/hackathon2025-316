import re


def normalize_text(s: str) -> str:
    """Normalize text by stripping, lowering, and collapsing whitespace.

    Args:
        s (str): The input string (can be None).

    Returns:
        str: Normalized lowercase text with single spaces.
    """
    if s is None:
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s