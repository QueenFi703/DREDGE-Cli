"""Nebius Token Factory integration for DREDGE.

Uses Nebius' OpenAI-compatible inference API without adding a new SDK
 dependency. Configuration is supplied through environment variables.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b"


class NebiusConfigurationError(RuntimeError):
    """Raised when the Nebius integration is not configured."""


def get_nebius_config() -> Dict[str, str]:
    """Return sanitized Nebius configuration from the environment."""
    return {
        "api_key": os.getenv("NEBIUS_API_KEY", ""),
        "base_url": os.getenv("NEBIUS_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
        "model": os.getenv("NEBIUS_MODEL", DEFAULT_MODEL),
    }


def nebius_configured() -> bool:
    """Return True when an API key is available."""
    return bool(get_nebius_config()["api_key"])


def chat(
    messages: List[Dict[str, Any]],
    *,
    temperature: float = 0.2,
    max_tokens: Optional[int] = 1200,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """Run a DREDGE inference request through Nebius Token Factory."""
    config = get_nebius_config()
    if not config["api_key"]:
        raise NebiusConfigurationError("NEBIUS_API_KEY is not configured")

    payload: Dict[str, Any] = {
        "model": config["model"],
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    response = requests.post(
        f"{config['base_url']}/chat/completions",
        headers={
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def dredge_reason(prompt: str, *, context: Optional[str] = None) -> Dict[str, Any]:
    """Ask Nemotron to reason over a DREDGE insight or task."""
    system = (
        "You are the DREDGE reasoning layer. Distill the request, recall the "
        "relevant context, emerge with a useful answer, detect risks, guide "
        "the next action, and evolve the working plan. Be concise and explicit."
    )
    user = prompt if not context else f"Context:\n{context}\n\nRequest:\n{prompt}"
    return chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
