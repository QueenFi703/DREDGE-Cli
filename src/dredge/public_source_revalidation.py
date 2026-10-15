"""Own-run public source observations and conservative paragraph refresh plans."""
import json
import re
import time
import uuid
from pathlib import Path

from flask import Blueprint, jsonify, request, send_file
from flask_login import current_user, login_required

from .public_source_collector import CATALOG, VERSION, SourceUnavailable, catalog_key, collect
from .studio_research_controls import own_run, stored_payload

bp = Blueprint('public_source_revalidation', __name__)


def initialize_schema(db):
    db.executescript('''
        CREATE TABLE IF NOT EXISTS public_source_checks (
            id TEXT PRIMARY KEY, owner TEXT NOT NULL, run_id TEXT NOT NULL,
            created REAL NOT NULL, status TEXT NOT NULL, result TEXT);
        CREATE INDEX IF NOT EXISTS public_source_checks_owner ON public_source_checks(owner,created);
        CREATE TABLE IF NOT EXISTS public_source_observations (
            id TEXT PRIMARY KEY, owner TEXT NOT NULL, run_id TEXT NOT NULL,
            source_id TEXT NOT NULL, observed REAL NOT NULL, data TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS public_source_observations_scope
            ON public_source_observations(owner,run_id,source_id,observed);
    ''')


def register(app):
    with app.extensions['studio_store'].connect() as db:
        initialize_schema(db)
    app.register_blueprint(bp)


@bp.before_request
def protect():
    from .studio import protect_mutations
    return protect_mutations()


@bp.after_request
def private_response(response):
    response.headers['Cache-Control'] = 'no-store'
    return response


@bp.get('/api/studio/research/public-sources')
@login_required
def catalog():
    return jsonify(sources=[dict(id=key, title=title, url=url) for key, (title, url) in CATALOG.items()],
                   note='Only these exact public pages can be checked. No account portals, case records, '
                        'query strings, redirects or other URLs are fetched.')


def source_plan(run, results):
    """Citation syntax guides review, never proves dependency completeness or truth."""
    try:
        value = json.loads(run['result']) if run['result'] and len(run['result']) <= 256000 else {}
        answer = value.get('answer') if isinstance(value, dict) else None
    except (ValueError, TypeError):
        answer = None
    if run['status'] != 'completed' or not isinstance(answer, str) or len(answer) > 64000:
        return []
    paragraphs = re.split(r'\n\s*\n', answer.strip())
    if len(paragraphs) > 100:
        return []
    states = {item['source_id']: item['state'] for item in results}
    sections = []
    for index, paragraph in enumerate(paragraphs):
        sources = sorted(set(re.findall(r'\[(source-\d+)\]', paragraph)))
        unknown = not sources or any(states.get(source) not in {'changed', 'unchanged'} for source in sources)
        action = ('blocked' if unknown else 'refresh_review' if any(states[source] == 'changed' for source in sources)
                  else 'unchanged_source_review')
        sections.append(dict(id=f'section-{index+1}', text=paragraph, source_ids=sources, action=action))
    return sections


@bp.post('/api/studio/research/revalidate')
@login_required
def revalidate():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or set(data) != {'run_id', 'source_ids'}:
        return jsonify(error='Provide run_id and source_ids only.'), 400
    return check_sources_for_owner(data['run_id'], data['source_ids'])


def check_sources_for_owner(run_id, source_ids):
    """Internal service seam: caller must bind an authenticated verified owner.

    Website callers use session/CSRF; MCP callers independently require a scoped
    verified token. Return the existing Flask response/status; never infer consent.
    """
    from .studio import store, role
    if not current_user.is_authenticated:
        return jsonify(error='Authentication is required.'), 401
    if role() not in {'operator', 'admin'}:
        return jsonify(error='An operator role is required.'), 403
    data = {'run_id': run_id, 'source_ids': source_ids}
    selected = data['source_ids']
    if (not isinstance(selected, list) or not 1 <= len(selected) <= 3
            or any(not isinstance(item, str) for item in selected) or len(set(selected)) != len(selected)):
        return jsonify(error='Select one to three distinct source IDs.'), 400
    owner, check_id, now = current_user.get_id(), str(uuid.uuid4()), time.time()
    with store().connect() as db:
        db.execute('BEGIN IMMEDIATE')
        run = own_run(db, data['run_id'])
        if not run:
            return jsonify(error='Run not found.'), 404
        try:
            payload = stored_payload(run['payload'])
        except ValueError:
            return jsonify(error='Saved source records are unavailable.'), 409
        sources = {source['id']: source for source in payload['evidence']}
        if any(source not in sources or not catalog_key(sources[source]['url']) for source in selected):
            return jsonify(error='Every selected source must use an exact approved public catalog URL.'), 400
        # Read-only collection is retryable after an interrupted bounded attempt.
        db.execute("UPDATE public_source_checks SET status='interrupted' WHERE owner=? AND status='pending' AND created<?",
                   (owner, now-120))
        if db.execute("SELECT 1 FROM public_source_checks WHERE owner=? AND status='pending'", (owner,)).fetchone():
            return jsonify(error='A source check is already running. Inspect its history before retrying.'), 409
        if db.execute('SELECT COUNT(*) FROM public_source_checks WHERE owner=? AND created>?',
                      (owner, now-3600)).fetchone()[0] >= 10:
            return jsonify(error='Limit of ten public source checks per hour reached.'), 429
        db.execute('INSERT INTO public_source_checks VALUES(?,?,?,?,?,NULL)',
                   (check_id, owner, run['id'], now, 'pending'))
        previous = {}
        for source_id in selected:
            row = db.execute('SELECT id,data FROM public_source_observations WHERE owner=? AND run_id=? '
                             'AND source_id=? ORDER BY observed DESC,id DESC LIMIT 1',
                             (owner, run['id'], source_id)).fetchone()
            if row:
                try:
                    previous[source_id] = (row['id'], json.loads(row['data']))
                except ValueError:
                    pass
        store().audit(db, owner, 'public_sources_check_started', run['id'])
    results, fetched = [], {}
    for source_id in selected:
        key = catalog_key(sources[source_id]['url'])
        try:
            if key not in fetched:
                try:
                    fetched[key] = collect(key)
                except SourceUnavailable as exc:
                    fetched[key] = exc
            if isinstance(fetched[key], SourceUnavailable):
                raise fetched[key]
            observation = fetched[key]
            old = previous.get(source_id)
            compatible = bool(old and isinstance(old[1], dict)
                              and old[1].get('collector_version') == VERSION
                              and old[1].get('url') == observation['url'])
            state = ('unchanged' if old[1].get('digest') == observation['digest'] else 'changed') if compatible else 'first_observation'
            observation_id = str(uuid.uuid4())
            with store().connect() as db:
                db.execute('INSERT INTO public_source_observations VALUES(?,?,?,?,?,?)',
                           (observation_id, owner, run['id'], source_id, observation['observed_at'], json.dumps(observation)))
            results.append(dict(source_id=source_id, state=state, observation_id=observation_id,
                                previous_observation_id=old[0] if compatible else None,
                                **{key: observation[key] for key in ('url', 'title', 'observed_at', 'digest', 'collector_version', 'bytes_observed')},
                                excerpt=observation['text'][:4000], excerpt_truncated=len(observation['text']) > 4000))
        except SourceUnavailable as exc:
            results.append(dict(source_id=source_id, state='unknown', error=str(exc),
                                previous_observation_id=previous[source_id][0] if source_id in previous else None))
    result = dict(check_id=check_id, run_id=run['id'], sources=results, sections=source_plan(run, results),
                  provider_calls=0, automatic_reuse_allowed=False,
                  note='Successful observations describe the returned public page bytes at the stated time only. '
                       'The first observation establishes a baseline; it does not verify the source used by an older answer. '
                       'Changes may be cosmetic. Paragraph citations are provisional; human review is required. '
                       'No answer was rewritten and no legal validity or claim correctness was established.')
    with store().connect() as db:
        db.execute('UPDATE public_source_checks SET status=?,result=? WHERE id=? AND owner=?',
                   ('completed', json.dumps(result), check_id, owner))
        store().audit(db, owner, 'public_sources_checked', run['id'])
    return jsonify(result)


@bp.get('/api/studio/runs/<run_id>/source-observations')
@login_required
def history(run_id):
    from .studio import store
    with store().connect() as db:
        if not own_run(db, run_id):
            return jsonify(error='Run not found.'), 404
        rows = db.execute('SELECT id,created,status,result FROM public_source_checks '
                          'WHERE owner=? AND run_id=? ORDER BY created DESC LIMIT 30',
                          (current_user.get_id(), run_id)).fetchall()
    return jsonify(checks=[dict(id=row['id'], created=row['created'], status=row['status'],
                               result=json.loads(row['result']) if row['result'] else None) for row in rows])


@bp.get('/studio/public-source-checks')
@login_required
def page():
    return send_file(Path(__file__).parent / 'static' / 'public_source_checks.html')
