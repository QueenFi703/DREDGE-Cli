"""Tests for mobile optimization features in DREDGE CLI."""
import os
import subprocess
import sys
from unittest.mock import patch
from dredge.cli import _detect_mobile_context


def test_detect_mobile_context_termux():
    """Test mobile detection when TERMUX_VERSION is set."""
    with patch.dict(os.environ, {"TERMUX_VERSION": "0.118"}):
        with patch("platform.uname") as mock_uname:
            mock_uname.return_value = type('obj', (object,), {'release': 'Linux'})()
            ctx = _detect_mobile_context()
            assert ctx["is_termux"] is True
            assert ctx["is_mobile"] is True


def test_detect_mobile_context_ish():
    """Test mobile detection for iSH environment."""
    with patch.dict(os.environ, {}, clear=True):
        with patch("platform.uname") as mock_uname:
            mock_uname.return_value = type('obj', (object,), {'release': 'alpine-ish'})()
            ctx = _detect_mobile_context()
            assert ctx["is_ish"] is True
            assert ctx["is_mobile"] is True


def test_detect_mobile_context_non_mobile():
    """Test mobile detection on non-mobile environment."""
    with patch.dict(os.environ, {}, clear=True):
        with patch("platform.uname") as mock_uname:
            mock_uname.return_value = type('obj', (object,), {'release': 'Linux-5.10.0'})()
            ctx = _detect_mobile_context()
            assert ctx["is_termux"] is False
            assert ctx["is_ish"] is False
            assert ctx["is_mobile"] is False


def test_mobile_terminal_width_capped():
    """Test that terminal width is capped at 80 on mobile."""
    with patch.dict(os.environ, {"TERMUX_VERSION": "0.118"}):
        with patch("platform.uname") as mock_uname:
            mock_uname.return_value = type('obj', (object,), {'release': 'Linux'})()
            with patch("shutil.get_terminal_size") as mock_size:
                mock_size.return_value = type('obj', (object,), {'columns': 120})()
                ctx = _detect_mobile_context()
                assert ctx["term_width"] == 80


def test_cli_no_spinner_flag():
    """Test that --no-spinner flag is available."""
    result = subprocess.run(
        [sys.executable, "-m", "dredge", "--help"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "--no-spinner" in result.stdout


def test_cli_threads_flag_serve():
    """Test that --threads flag is available for serve command."""
    result = subprocess.run(
        [sys.executable, "-m", "dredge", "serve", "--help"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "--threads" in result.stdout


def test_cli_threads_flag_mcp():
    """Test that --threads flag is available for mcp command."""
    result = subprocess.run(
        [sys.executable, "-m", "dredge", "mcp", "--help"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "--threads" in result.stdout
