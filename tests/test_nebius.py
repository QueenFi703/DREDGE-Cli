import pytest

from dredge.nebius import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    NebiusConfigurationError,
    dredge_reason,
    get_nebius_config,
    nebius_configured,
)


def test_defaults(monkeypatch):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    monkeypatch.delenv("NEBIUS_BASE_URL", raising=False)
    monkeypatch.delenv("NEBIUS_MODEL", raising=False)

    config = get_nebius_config()

    assert config["api_key"] == ""
    assert config["base_url"] == DEFAULT_BASE_URL
    assert config["model"] == DEFAULT_MODEL
    assert nebius_configured() is False


def test_reason_requires_api_key(monkeypatch):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)

    with pytest.raises(NebiusConfigurationError):
        dredge_reason("distill this insight")


def test_reason_uses_nemotron(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    monkeypatch.setenv("NEBIUS_MODEL", "nvidia/nemotron-3-super-120b-a12b")

    captured = {}

    def fake_chat(messages, **kwargs):
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return {
            "model": "nvidia/nemotron-3-super-120b-a12b",
            "choices": [{"message": {"content": "distilled"}}],
        }

    monkeypatch.setattr("dredge.nebius.chat", fake_chat)

    response = dredge_reason("distill this insight", context="memory")

    assert response["choices"][0]["message"]["content"] == "distilled"
    assert "DREDGE reasoning layer" in captured["messages"][0]["content"]
    assert "Context:\nmemory" in captured["messages"][1]["content"]
