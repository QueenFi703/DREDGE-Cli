"""Keep the existing production gateway route/process contract when opting in."""
import importlib

from starlette.testclient import TestClient
from test_chatgpt_transport import app, oauth_token, rpc, ISSUER


def test_enabled_gateway_keeps_existing_routes_and_runs_mcp_lifespan(app):
    from dredge import orion_gateway
    gateway = importlib.reload(orion_gateway)
    assert any(getattr(route, 'path', None) == '/mcp' and 'POST' in getattr(route, 'methods', set())
               for route in gateway.app.routes)
    token = oauth_token(app, 'github:owner')
    with TestClient(gateway.app, base_url=ISSUER) as client:
        health = client.get('/health')
        assert health.status_code == 200
        assert client.get('/preview').status_code == 200
        assert client.get('/.well-known/oauth-protected-resource/chatgpt/mcp').status_code == 200
        response = rpc(client,token,'tools/call',{'name':'get_studio_activity_report'})
        assert response.status_code == 200, response.text
        assert response.json()['result']['structuredContent']['proposed_runs'] == 0
        # Existing gateway route remains registered ahead of the fallback.
        assert client.get('/admin/stats').status_code == 422
    # Lifecycle shuts down normally; no SDK task group is left running.


def test_disabled_gateway_keeps_legacy_fallback(app, monkeypatch):
    monkeypatch.setenv('DREDGE_CHATGPT_ENABLED','false')
    from dredge import orion_gateway
    gateway = importlib.reload(orion_gateway)
    with TestClient(gateway.app, base_url=ISSUER) as client:
        assert client.get('/health').status_code == 200
        assert client.get('/preview').status_code == 200
        assert client.post('/chatgpt/mcp',json={}).status_code == 404
        assert client.get('/.well-known/oauth-authorization-server').status_code == 404
