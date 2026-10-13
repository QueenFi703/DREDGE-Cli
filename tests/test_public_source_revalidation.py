import hashlib
import json
import time
import pytest
from test_studio import app,client_for,proposal
from dredge.public_source_collector import VERSION,SourceUnavailable


@pytest.fixture
def collector(monkeypatch):
    state={'text':'Official fixture public source.','calls':[],'failure':False}
    def collect(key):
        state['calls'].append(key)
        if state['failure']: raise SourceUnavailable('Public source unavailable.')
        return dict(catalog_id=key,url='https://dss.mo.gov/child-support',title='Missouri public source',
                    digest=hashlib.sha256(state['text'].encode()).hexdigest(),text=state['text'],
                    observed_at=time.time(),collector_version=VERSION,bytes_observed=len(state['text']))
    monkeypatch.setattr('dredge.public_source_revalidation.collect',collect)
    return state


def run(app,c,h):
    value=proposal(c,h,evidence=[{'title':'Public DSS','url':'https://dss.mo.gov/child-support','excerpt':'Supplied older text'}])
    result=value.json['id']
    with app.extensions['studio_store'].connect() as db:
        db.execute('UPDATE runs SET status=?,result=? WHERE id=?',('completed',json.dumps(
            {'answer':'Public finding [source-1].\n\nUnsupported sentence.'}),result))
    return result


def check(c,h,run_id):
    return c.post('/api/studio/research/revalidate',headers=h,json={'run_id':run_id,'source_ids':['source-1']})


def test_actual_observations_persist_and_drive_targeted_plan(app,collector):
    c,h=client_for(app);run_id=run(app,c,h)
    first=check(c,h,run_id)
    assert first.status_code==200 and first.json['sources'][0]['state']=='first_observation'
    assert first.json['sections'][0]['action']=='blocked'
    second=check(c,h,run_id)
    assert second.json['sources'][0]['state']=='unchanged'
    assert second.json['sections'][0]['action']=='unchanged_source_review'
    collector['text']='Updated public policy fixture.'
    third=check(c,h,run_id)
    assert third.json['sources'][0]['state']=='changed'
    assert third.json['sections'][0]['action']=='refresh_review'
    assert third.json['sections'][1]['action']=='blocked'
    assert third.json['provider_calls']==0 and not third.json['automatic_reuse_allowed']
    assert 'observed_at' in third.json['sources'][0]
    assert c.get(f'/api/studio/runs/{run_id}/source-observations').json['checks'][0]['result']['check_id']==third.json['check_id']
    with app.extensions['studio_store'].connect() as db:
        assert db.execute('SELECT COUNT(*) FROM public_source_observations').fetchone()[0]==3


def test_failure_preserves_prior_but_never_calls_it_current(app,collector):
    c,h=client_for(app);run_id=run(app,c,h);check(c,h,run_id);collector['failure']=True
    result=check(c,h,run_id)
    assert result.json['sources'][0]['state']=='unknown'
    assert result.json['sources'][0]['previous_observation_id']
    assert result.json['sections'][0]['action']=='blocked'
    with app.extensions['studio_store'].connect() as db:
        assert db.execute('SELECT COUNT(*) FROM public_source_observations').fetchone()[0]==1


def test_auth_csrf_scope_and_url_injection(app,collector):
    c,h=client_for(app);run_id=run(app,c,h)
    assert check(c,{},run_id).status_code==403
    other,oh=client_for(app,'test:admin')
    assert check(other,oh,run_id).status_code==404
    assert other.get(f'/api/studio/runs/{run_id}/source-observations').status_code==404
    assert c.post('/api/studio/research/revalidate',headers=h,json={'run_id':run_id,'source_ids':['source-1'],
        'url':'https://127.0.0.1'}).status_code==400
    unsafe=proposal(c,h,evidence=[{'title':'not catalog','url':'https://example.org/?case=secret','excerpt':'private'}]).json['id']
    assert check(c,h,unsafe).status_code==400
    assert not collector['calls']


def test_pending_rate_limit_and_old_attempt_recovery(app,collector):
    c,h=client_for(app);run_id=run(app,c,h)
    with app.extensions['studio_store'].connect() as db:
        db.execute('INSERT INTO public_source_checks VALUES(?,?,?,?,?,NULL)',('pending','test:op',run_id,time.time(),'pending'))
    assert check(c,h,run_id).status_code==409
    with app.extensions['studio_store'].connect() as db:
        db.execute('UPDATE public_source_checks SET created=? WHERE id=?',(time.time()-121,'pending'))
    assert check(c,h,run_id).status_code==200
    for _ in range(8): assert check(c,h,run_id).status_code==200
    assert check(c,h,run_id).status_code==429


def test_catalog_and_page_do_not_fetch(app,collector):
    c,h=client_for(app)
    response=c.get('/api/studio/research/public-sources')
    assert response.status_code==200 and response.headers['Cache-Control']=='no-store'
    assert len(response.json['sources'])==5
    assert not collector['calls']
