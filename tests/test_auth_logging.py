"""Provider failures must not print credentials, raw exceptions, or profiles."""
from types import SimpleNamespace

import pytest
from flask import Flask

from dredge import auth
from dredge.server import create_app

SENTINEL = 'fictional-sensitive-provider-token-and-profile'


class FailingProvider:
    def __repr__(self): return SENTINEL
    def authorize_redirect(self, *args, **kwargs): raise ValueError(SENTINEL)
    def authorize_access_token(self): raise ValueError(SENTINEL)


def test_provider_registration_error_redacted(monkeypatch, capsys, caplog):
    class Registry:
        def __init__(self, app): pass
        def register(self, **kwargs): raise ValueError(SENTINEL)
    monkeypatch.setattr(auth, 'OAuth', Registry)
    for name in ['GOOGLE_CLIENT_ID','GOOGLE_CLIENT_SECRET','GITHUB_CLIENT_ID','GITHUB_CLIENT_SECRET']:
        monkeypatch.setenv(name,SENTINEL)
    auth.init_auth(Flask(__name__))
    captured = capsys.readouterr()
    assert SENTINEL not in captured.out + captured.err + caplog.text
    assert 'Failed to register' in caplog.text


@pytest.mark.parametrize('path,provider', [('/auth/google','google'),('/auth/github','github'),('/auth/github/callback','github'),('/auth/google/callback','google')])
def test_provider_request_exception_redacted(monkeypatch, tmp_path, capsys, caplog, path, provider):
    monkeypatch.setenv('STUDIO_DB_PATH',str(tmp_path/'state.sqlite3'))
    monkeypatch.setenv('DREDGE_CHATGPT_ENABLED','false')
    application=create_app()
    monkeypatch.setattr(auth,'get_oauth',lambda:SimpleNamespace(**{provider:FailingProvider()}))
    result=application.test_client().get(path)
    assert result.status_code==302
    assert 'auth_failed' in result.location
    captured=capsys.readouterr()
    assert SENTINEL not in captured.out+captured.err+caplog.text


def test_successful_profile_is_not_logged(monkeypatch,tmp_path,capsys,caplog):
    monkeypatch.setenv('STUDIO_DB_PATH',str(tmp_path/'state.sqlite3'))
    monkeypatch.setenv('DREDGE_CHATGPT_ENABLED','false')
    application=create_app()
    class Response:
        def raise_for_status(self): pass
        def json(self): return {'id':'fictional-profile','name':SENTINEL,'email':'fictional@example.test'}
    provider=SimpleNamespace(authorize_access_token=lambda:None,token='fictional-token',get=lambda *a,**k:Response())
    monkeypatch.setattr(auth,'get_oauth',lambda:SimpleNamespace(github=provider))
    result=application.test_client().get('/auth/github/callback')
    assert result.status_code==302
    captured=capsys.readouterr()
    assert SENTINEL not in captured.out+captured.err+caplog.text
    auth._users.pop('github:fictional-profile',None)
