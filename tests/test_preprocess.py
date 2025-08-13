from project.utils import normalize_text


def test_normalize_text_basic() -> None:
    assert normalize_text("  Hello  WORLD  ") == "hello world"


def test_normalize_text_empty() -> None:
    assert normalize_text(None) == ""
    assert normalize_text("") == ""