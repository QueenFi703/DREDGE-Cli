"""Tests for the print command."""
import io
import sys
from dredge.cli import main


def test_print_with_text(capsys):
    """Test print command with text argument."""
    result = main(["print", "I'M FREE!"])
    assert result == 0
    
    captured = capsys.readouterr()
    assert captured.out == "I'M FREE!\n"


def test_print_with_no_arguments(capsys):
    """Test print command with no arguments (clean newline)."""
    result = main(["print"])
    assert result == 0
    
    captured = capsys.readouterr()
    assert captured.out == "\n"


def test_print_with_multiword_text(capsys):
    """Test print command with multi-word text."""
    result = main(["print", "Hello World!"])
    assert result == 0
    
    captured = capsys.readouterr()
    assert captured.out == "Hello World!\n"


def test_print_help(capsys):
    """Test print command help."""
    import pytest
    with pytest.raises(SystemExit) as exc_info:
        main(["print", "--help"])
    
    # --help should exit with 0 (success)
    assert exc_info.value.code == 0
    
    # Verify help message was printed
    captured = capsys.readouterr()
    assert "Print a message or newline" in captured.out or "Text to print" in captured.out
