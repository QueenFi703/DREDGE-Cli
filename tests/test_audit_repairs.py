"""Regression coverage for the Studio safety and transport audit findings."""
import hashlib
import io
import json

import pytest
from PIL import Image
from test_studio import app, client_for
from test_casework import enable, create
from test_client_portal import assigned_case


@pytest.mark.parametrize('original,terms,expected', [
    ('Name Alice.', ['Alice', 'RED', 'E'], 'Name [REDACTED].'),
    ('E', ['E'] * 8, '[REDACTED]'),
    ('[a].* + Alice alice José', ['[a].*', '+', 'Alice', 'José'],
     '[REDACTED] [REDACTED] [REDACTED] alice [REDACTED]'),
    ('abcd', ['ab', 'bcd'], '[REDACTED]'),
])
def test_redaction_matches_original_once(app, original, terms, expected):
    _, _, client, headers, case_id = assigned_case(app)
    path = f'/api/client/cases/{case_id}/documents'
    uploaded = client.post(path, data={
        'file': (io.BytesIO(original.encode()), 'fictional.txt'), 'consent': 'true'
    }, headers=headers).get_json()
    response = client.post(path + '/' + uploaded['id'] + '/redact',
                           json={'terms': terms}, headers=headers)
    assert response.status_code == 201
    copy = response.get_json()
    assert copy['preview'] == expected
    assert client.get(path + '/' + copy['id']).data == expected.encode()
    assert client.get(path + '/' + uploaded['id']).data == original.encode()
    listed = client.get(path).get_json()['documents']
    assert next(d for d in listed if d['id'] == copy['id'])['sha256'] == hashlib.sha256(expected.encode()).hexdigest()


def test_repeated_redaction_copies_cannot_exceed_document_bounds(app):
    _, _, client, headers, case_id = assigned_case(app)
    path = f'/api/client/cases/{case_id}/documents'
    original = b'E' * 1000
    source = client.post(path, data={
        'file': (io.BytesIO(original), 'fictional.txt'), 'consent': 'true'
    }, headers=headers).get_json()['id']
    original_id = source
    for expected_length in (10000, 28000, 64000):
        response = client.post(path + '/' + source + '/redact', json={'terms': ['E']}, headers=headers)
        assert response.status_code == 201
        assert len(response.get_json()['preview']) == expected_length
        source = response.get_json()['id']
    response = client.post(path + '/' + source + '/redact', json={'terms': ['E']}, headers=headers)
    assert response.status_code == 400
    assert '100,000' in response.get_json()['error']
    assert len(client.get(path).get_json()['documents']) == 4
    assert client.get(path + '/' + original_id).data == original


def test_redaction_copies_respect_existing_document_quota(app):
    _, _, client, headers, case_id = assigned_case(app)
    path = f'/api/client/cases/{case_id}/documents'
    source = client.post(path, data={
        'file': (io.BytesIO(b'Fictional Alice'), 'fictional.txt'), 'consent': 'true'
    }, headers=headers).get_json()['id']
    with app.extensions['studio_store'].connect() as db:
        row = db.execute('SELECT * FROM client_documents WHERE id=?', (source,)).fetchone()
        for number in range(49):
            db.execute('INSERT INTO client_documents VALUES(?,?,?,?,?)',
                       (f'fictional-copy-{number}', case_id, row['owner'], row['data'], row['created']))
    response = client.post(path + '/' + source + '/redact', json={'terms': ['Alice']}, headers=headers)
    assert response.status_code == 409
    assert len(client.get(path).get_json()['documents']) == 50


@pytest.mark.parametrize('kind', ['analysis', 'discernment', 'legal_draft'])
@pytest.mark.parametrize('include', [None, False, True])
def test_explanation_transfer_requires_explicit_opt_in(app, monkeypatch, kind, include):
    staff, headers, client, client_headers, case_id = assigned_case(app)
    uploaded = client.post(f'/api/client/cases/{case_id}/documents', data={
        'file': (io.BytesIO(b'Fictional evidence'), 'fictional.txt'), 'consent': 'true'
    }, headers=client_headers).get_json()
    for number in range(6):
        assert client.post(f'/api/client/cases/{case_id}/explanations',
                           json={'text': f'Fictional explanation {number}'},
                           headers=client_headers).status_code == 201
    calls = []
    monkeypatch.setattr('dredge.casework.provider',
                        lambda payload: calls.append(payload) or {'blocks': [], 'usage': {}})
    body = dict(kind=kind, question='Summarize', case_id=case_id,
                file_ids=[uploaded['id']], consent=True)
    if include is not None:
        body['include_client_explanations'] = include
    response = staff.post('/api/casework/ai', json=body, headers=headers)
    assert response.status_code == 200
    sent = json.loads(calls[0]['input'])
    expected = [f'Fictional explanation {n}' for n in range(5, 0, -1)] if include else []
    assert sent.get('client_explanations', []) == expected
    assert 'tools' not in calls[0]
    assert len(staff.get(f'/api/client/staff/cases/{case_id}').get_json()['explanations']) == 6


@pytest.mark.parametrize('include', ['true', 1, None, []])
def test_malformed_explanation_opt_in_never_calls_provider(app, monkeypatch, include):
    enable(app)
    client, headers = client_for(app)
    calls = []
    monkeypatch.setattr('dredge.casework.provider', lambda p: calls.append(p))
    response = client.post('/api/casework/ai', json=dict(
        kind='analysis', question='Summarize', consent=True,
        include_client_explanations=include), headers=headers)
    assert response.status_code == 400
    assert not calls


def test_public_research_rejects_explanation_opt_in(app, monkeypatch):
    enable(app)
    client, headers = client_for(app)
    calls = []
    monkeypatch.setattr('dredge.casework.provider', lambda p: calls.append(p))
    response = client.post('/api/casework/ai', json=dict(
        kind='research', question='Public procedure', consent=True,
        public_question_confirmed=True, include_client_explanations=True), headers=headers)
    assert response.status_code == 400
    assert not calls


@pytest.mark.parametrize('character', ['x', 'é', '😀'])
def test_ocr_review_accepts_100000_characters_but_stays_bounded(app, monkeypatch, character):
    enable(app)
    client, headers = client_for(app)
    case_id = create(client, headers)
    monkeypatch.setattr('dredge.document_text.ocr', lambda *a: ([(1, 'Draft text')], 'pending_review'))
    image = io.BytesIO()
    Image.new('RGB', (10, 10), 'white').save(image, format='PNG')
    image.seek(0)
    path = f'/api/casework/cases/{case_id}/files'
    uploaded = client.post(path, data={'file': (image, 'fictional.png')}, headers=headers)
    assert uploaded.status_code == 201
    endpoint = path + '/' + uploaded.get_json()['id'] + '/text'
    draft = client.get(endpoint).get_json()
    body = dict(text=character * 100000, verified=True, revision=draft['revision'])
    assert client.post(endpoint, json=body, headers=headers).status_code == 200
    reviewed = client.get(endpoint).get_json()
    assert reviewed['text'] == body['text']
    assert client.post(endpoint, json={**body, 'text': character * 100001,
                       'revision': reviewed['revision']}, headers=headers).status_code == 400
    assert client.post(endpoint, json={**body, 'padding': 'x' * 1300000},
                       headers=headers).status_code == 413
    assert client.post('/api/casework/pages', json={'title': 'Fictional', 'body': 'x' * 70000},
                       headers=headers).status_code == 413
