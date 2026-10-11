"""Dedicated tunnel entry has no TCP/legacy API fallback or header identity trust."""
import os
import socket

import pytest
from starlette.testclient import TestClient
from test_chatgpt_transport import app, ISSUER

from dredge.chatgpt_tunnel import SocketIngress, create_tunnel_app


@pytest.fixture
def tunnel(app, monkeypatch, tmp_path):
    # In-process protocol tests do not emulate OS socket authorization.
    # The separate real-socket test below covers the launch boundary.
    monkeypatch.setattr('dredge.chatgpt_tunnel._verify_socket', lambda: None)
    monkeypatch.setenv('DREDGE_CHATGPT_TUNNEL_ENABLED', 'true')
    yield create_tunnel_app()


def headers(**extra):
    return {'X-Forwarded-Proto':'https', **extra}


def test_socket_discovery_and_only_readonly_surface(tunnel):
    with TestClient(tunnel, base_url=ISSUER, client=None) as c:
        assert c.get('/health', headers=headers()).status_code == 200
        assert c.get('/.well-known/oauth-authorization-server', headers=headers()).status_code == 200
        assert c.get('/.well-known/oauth-protected-resource/chatgpt/mcp', headers=headers()).status_code == 200
        assert c.post('/chatgpt/mcp', json={}, headers=headers()).status_code == 401
        for path in ['/mcp','/api/studio/report','/api/studio/runs','/billing','/preview','/advanced','/docs']:
            assert c.get(path, headers=headers()).status_code == 404
        login = c.get('/auth/login', headers=headers())
        assert login.status_code == 200
        assert 'Sign-in is not configured' in login.text
        assert '/billing' not in login.text
        assert '/advanced' not in login.text


@pytest.mark.parametrize('bad', [None, '', 'http', 'https,http', 'HTTPS', ' https', 'https '])
def test_bad_protocol_fails_closed(tunnel, bad):
    with TestClient(tunnel, base_url=ISSUER, client=None) as c:
        h = {} if bad is None else {'X-Forwarded-Proto':bad}
        assert c.get('/.well-known/oauth-authorization-server', headers=h).status_code == 400


def test_duplicates_wrong_host_and_tcp_fail_closed(tunnel):
    with TestClient(tunnel, base_url=ISSUER, client=None) as c:
        for h in [[('host','dredge.example.test'),('host','dredge.example.test'),('x-forwarded-proto','https')],
                  [('x-forwarded-proto','https'),('x-forwarded-proto','https')]]:
            assert c.get('/health', headers=h).status_code == 400
        assert c.get('/health', headers=headers(Host='attacker.test')).status_code == 400
    # A new app is unnecessary here: call the adapter directly without lifespan.
    c = TestClient(tunnel, base_url=ISSUER, client=('127.0.0.1', 1234))
    assert c.get('/health', headers=headers()).status_code == 403


@pytest.mark.anyio
async def test_forwarded_identity_and_scope_not_mutated():
    received = []
    async def downstream(scope, receive, send):
        received.append(scope)
    adapter = SocketIngress(downstream, ISSUER)
    scope = {'type':'http','client':None,'scheme':'http','path':'/health','method':'GET',
             'headers':[(b'host',b'dredge.example.test'),(b'x-forwarded-proto',b'https'),
                        (b'x-forwarded-for',b'1.2.3.4'),(b'x-forwarded-host',b'attacker.test')],
             'server':None,'root_path':''}
    await adapter(scope, None, None)
    assert scope['scheme'] == 'http'
    assert received[0]['scheme'] == 'https'
    assert received[0]['client'] is None
    assert received[0]['headers'] == scope['headers']


def test_startup_requires_optin_and_real_socket(app, monkeypatch, tmp_path):
    monkeypatch.delenv('DREDGE_CHATGPT_TUNNEL_ENABLED', raising=False)
    with pytest.raises(RuntimeError):
        create_tunnel_app()
    monkeypatch.setenv('DREDGE_CHATGPT_TUNNEL_ENABLED','true')
    monkeypatch.setenv('DREDGE_TUNNEL_SOCKET_FD','123456789')
    with pytest.raises(RuntimeError):
        create_tunnel_app()
    try:
        tcp = socket.socket()
    except PermissionError:
        pytest.skip('Executor denies socket creation; real socket validation requires CI/container.')
    with tcp:
        tcp.bind(('127.0.0.1',0)); tcp.listen()
        monkeypatch.setenv('DREDGE_TUNNEL_SOCKET_FD',str(tcp.fileno()))
        with pytest.raises(RuntimeError):
            create_tunnel_app()


def test_real_unix_socket_factory(app, monkeypatch, tmp_path):
    from dredge.chatgpt_tunnel import _verify_socket
    try:
        sock = socket.socket(socket.AF_UNIX)
    except PermissionError:
        pytest.skip('Executor denies AF_UNIX; real socket validation requires CI/container.')
    with sock:
        path = str(tmp_path / 'gateway.sock')
        sock.bind(path); sock.listen()
        monkeypatch.setattr('dredge.chatgpt_tunnel.SOCKET_PATH',path)
        monkeypatch.setenv('DREDGE_TUNNEL_SOCKET_FD',str(sock.fileno()))
        monkeypatch.setenv('DREDGE_CHATGPT_TUNNEL_ENABLED','true')
        _verify_socket()
        with TestClient(create_tunnel_app(), base_url=ISSUER, client=None) as c:
            assert c.get('/health',headers=headers()).status_code == 200


def test_both_read_tools_still_owner_bound(tunnel):
    from test_chatgpt_transport import oauth_token, rpc
    flask_app = tunnel.app.state.dredge_flask
    token = oauth_token(flask_app, 'github:tunnel-owner')
    with TestClient(tunnel, base_url=ISSUER, client=None) as c:
        for name in ['get_dredge_access','get_studio_activity_report']:
            result = rpc(c, token, 'tools/call', {'name':name}, **{'X-Forwarded-Proto':'https'})
            assert result.status_code == 200
            assert result.json()['result']['isError'] is False
        assert rpc(c, token, 'tools/call', {'name':'execute_run'}, **{'X-Forwarded-Proto':'https'}).json()['result']['isError'] is True


def test_real_uvicorn_inherited_unix_socket(app, monkeypatch, tmp_path):
    """Run in CI/container: this is not replaced with a fake when IPC is denied."""
    import subprocess
    import sys
    import time
    import httpx
    try:
        sock = socket.socket(socket.AF_UNIX)
    except PermissionError:
        pytest.skip('Executor denies AF_UNIX; real Uvicorn socket test requires CI/container.')
    with sock:
        path = str(tmp_path / 'live.sock')
        sock.bind(path); sock.listen()
        env = {k:v for k,v in os.environ.items() if k in {
            'PATH','PYTHONPATH','DREDGE_CHATGPT_ENABLED','CHATGPT_ISSUER','CHATGPT_RESOURCE',
            'CHATGPT_CLIENT_ID','CHATGPT_REDIRECT_URI','SECRET_KEY','STUDIO_DB_PATH'}}
        env.update(DREDGE_CHATGPT_TUNNEL_ENABLED='true', DREDGE_TUNNEL_SOCKET_FD=str(sock.fileno()))
        script = ('import uvicorn; import dredge.chatgpt_tunnel as t; '
                  f't.SOCKET_PATH={path!r}; '
                  f'uvicorn.run(t.create_tunnel_app,factory=True,fd={sock.fileno()},'
                  'proxy_headers=False,access_log=False,log_level="warning")')
        process = subprocess.Popen([sys.executable,'-c',script],env=env,pass_fds=(sock.fileno(),),
                                   stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        try:
            with httpx.Client(transport=httpx.HTTPTransport(uds=path),timeout=0.25,trust_env=False) as c:
                local_origin = ISSUER.replace('https://','http://',1)
                deadline = time.monotonic()+10
                while True:
                    try:
                        response = c.get(local_origin+'/health',headers=headers())
                        break
                    except httpx.HTTPError:
                        if time.monotonic()>deadline or process.poll() is not None:
                            raise
                assert response.status_code == 200
                assert c.get(local_origin+'/.well-known/oauth-authorization-server',headers=headers()).status_code==200
                assert c.post(local_origin+'/chatgpt/mcp',headers=headers(),json={}).status_code==401
                assert c.get(local_origin+'/health').status_code==400
                assert c.get(local_origin+'/billing',headers=headers()).status_code==404
        finally:
            process.terminate()
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=5)
