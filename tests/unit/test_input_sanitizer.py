import pytest
from pydantic import ValidationError
from src.api.functions.input_sanitizer import InputSanitizer

def test_sanitize_valid_input():
    sanitizer = InputSanitizer()
    input_data = {
        "content": "  test content  ",
        "priority": "HIGH",
        "lang": "FR"
    }
    result = sanitizer.sanitize(input_data)
    assert result["content"] == "test content"
    assert result["priority"] == "high"
    assert result["lang"] == "fr"


def test_sanitize_defaults():
    sanitizer = InputSanitizer()
    input_data = {"content": "test"}
    result = sanitizer.sanitize(input_data)
    assert result["content"] == "test"
    assert result["priority"] == "low"
    assert result["lang"] == "en"


def test_sanitize_missing_content():
    sanitizer = InputSanitizer()
    input_data = {"priority": "high"}
    with pytest.raises(ValidationError):
        sanitizer.sanitize(input_data)
