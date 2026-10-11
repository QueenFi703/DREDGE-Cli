"""Readiness never takes a URL/route/header from the caller."""
import io

import httpx
import pytest

from dredge import chatgpt_tunnel_health as health


def test_fixed_readiness_probes(monkeypatch):
    calls = []
    class Client:
        def __init__(self, **kwargs):
            assert kwargs['trust_env'] is False
            assert kwargs['follow_redirects'] is False
            assert kwargs['timeout'] == 1.0
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, url, **kwargs):
            calls.append((url, kwargs))
            return httpx.Response(200)
    monkeypatch.setattr(health.httpx, 'Client', Client)
    assert health.ready('https://dredge.example.test')
    assert calls == [('http://dredge.example.test/health', {'headers':{'X-Forwarded-Proto':'https'}}),
                     ('http://127.0.0.1:2000/ready', {})]


@pytest.mark.parametrize('status', [301,302,401,404,500,503])
def test_unready_or_redirect_never_success(monkeypatch, status):
    class Client:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, *args, **kwargs): return httpx.Response(status)
    monkeypatch.setattr(health.httpx, 'Client', Client)
    assert health.ready('https://dredge.example.test') is False


@pytest.mark.parametrize('issuer', ['', 'http://example.test','https://example.test/','https://user@example.test','https://example.test:443','https://example.test?x=1'])
def test_invalid_probe_configuration(issuer):
    with pytest.raises(RuntimeError): health.validate_issuer(issuer)


def test_health_handler_never_proxies_input(monkeypatch):
    calls = []
    monkeypatch.setattr(health, 'ready', lambda issuer: calls.append(issuer) or True)
    cls = health.handler_for('https://dredge.example.test')
    instance = object.__new__(cls)
    statuses = []
    instance.send_response = statuses.append
    instance.send_header = lambda *args: None
    instance.end_headers = lambda: None
    for path in ['/chatgpt/mcp','/health?target=http://attacker.test','http://attacker.test/health']:
        instance.path = path
        instance.do_GET()
    assert statuses == [404,404,404]
    assert not calls
    instance.path = '/health'
    instance.do_GET()
    assert statuses[-1] == 200
    assert calls == ['https://dredge.example.test']
