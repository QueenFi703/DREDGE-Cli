"""Own-account supplied-text comparison. No pricing, spending or remote freshness claim."""
import hashlib
import json
import re
from pathlib import Path

from flask import Blueprint, jsonify, request, send_file
from flask_login import current_user, login_required

bp = Blueprint('studio_research_controls', __name__)
VERSION = 'supplied-source-v1'


def initialize_schema(db):
    db.executescript('''
        CREATE TABLE IF NOT EXISTS studio_source_context (
            owner TEXT NOT NULL, run_id TEXT NOT NULL, signature TEXT NOT NULL,
            PRIMARY KEY(owner,run_id));
        CREATE TABLE IF NOT EXISTS studio_source_versions (
            owner TEXT NOT NULL, run_id TEXT NOT NULL, source_id TEXT NOT NULL,
            digest TEXT NOT NULL, version TEXT NOT NULL,
            PRIMARY KEY(owner,run_id,source_id));
    ''')


def register(app):
    with app.extensions['studio_store'].connect() as db:
        initialize_schema(db)
    app.register_blueprint(bp)
    from .public_source_revalidation import register as register_public_sources
    register_public_sources(app)


@bp.before_request
def protect():
    from .studio import protect_mutations
    return protect_mutations()


def stored_payload(raw):
    # Stored JSON uses ASCII escapes, so accepted UTF-8 text can expand several
    # times. Keep a bounded decode ceiling large enough for all validated fields.
    if not isinstance(raw, str) or len(raw) > 1100000:
        raise ValueError('Unavailable stored proposal')
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError) as error:
        raise ValueError('Malformed stored proposal') from error
    if not isinstance(payload, dict) or not isinstance(payload.get('evidence'), list) or len(payload['evidence']) > 10:
        raise ValueError('Malformed source records')
    for index, source in enumerate(payload['evidence']):
        if not isinstance(source, dict) or source.get('id') != f'source-{index+1}':
            raise ValueError('Malformed source identity')
        for key, maximum in [('title', 300), ('url', 4096), ('excerpt', 4000)]:
            if not isinstance(source.get(key), str) or len(source[key]) > maximum:
                raise ValueError('Malformed source content')
    return payload


def capture_sources(db, owner, run_id, payload, capture_context=True):
    """Persist server-derived hashes of validated supplied records, never web hashes."""
    if capture_context:
        from .studio_provider import INSTRUCTIONS
        context = {key: payload.get(key) for key in
                   ('query', 'model', 'execution_mode', 'pipeline_type', 'max_output_tokens')}
        context.update(instructions=INSTRUCTIONS, service_tier='default', reasoning='low',
                       version=VERSION)
        signature = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
        db.execute('INSERT OR IGNORE INTO studio_source_context VALUES(?,?,?)',
                   (owner, run_id, signature))
    for source in payload['evidence']:
        value = {key: source[key] for key in ('id', 'title', 'url', 'excerpt')}
        digest = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                           separators=(',', ':')).encode()).hexdigest()
        db.execute('INSERT OR IGNORE INTO studio_source_versions VALUES(?,?,?,?,?)',
                   (owner, run_id, source['id'], digest, VERSION))


def own_run(db, run_id):
    if not isinstance(run_id, str) or len(run_id) > 100:
        return None
    return db.execute('SELECT * FROM runs WHERE id=? AND owner=?',
                      (run_id, current_user.get_id())).fetchone()


@bp.post('/api/studio/research/refresh-plan')
@login_required
def refresh_plan():
    from .studio import store
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or set(data) != {'base_run_id', 'new_run_id'}:
        return jsonify(error='Provide base_run_id and new_run_id only.'), 400
    with store().connect() as db:
        base, new = own_run(db, data['base_run_id']), own_run(db, data['new_run_id'])
        if not base or not new:
            return jsonify(error='Run not found.'), 404
        if base['id'] == new['id'] or new['created'] < base['created']:
            return jsonify(error='Choose a separate, newer proposal.'), 400
        try:
            old_payload, new_payload = stored_payload(base['payload']), stored_payload(new['payload'])
        except ValueError:
            return jsonify(error='Saved proposal source data is unavailable or malformed.'), 409
        if base['status'] != 'completed' or not base['result']:
            return jsonify(error='The original run needs a completed answer.'), 409
        try:
            if len(base['result']) > 256000:
                raise ValueError('Oversized saved result')
            old_result = json.loads(base['result'])
            answer = old_result.get('answer') if isinstance(old_result, dict) else None
        except (TypeError, ValueError):
            answer = None
        if not isinstance(answer, str) or not answer.strip() or len(answer) > 64000:
            return jsonify(error='The original run has no text answer to plan.'), 409
        # Backfill legacy immutable supplied records locally; no remote observation.
        capture_sources(db, current_user.get_id(), base['id'], old_payload, capture_context=False)
        capture_sources(db, current_user.get_id(), new['id'], new_payload, capture_context=False)
        def versions(run_id):
            return {row['source_id']: (row['digest'], row['version']) for row in db.execute(
                'SELECT * FROM studio_source_versions WHERE owner=? AND run_id=?',
                (current_user.get_id(), run_id))}
        old_sources, new_sources = versions(base['id']), versions(new['id'])
        states = {source_id: 'unknown_version' if old_sources[source_id][1] != VERSION
                  or (source_id in new_sources and new_sources[source_id][1] != VERSION) else
                  'unchanged_supplied_record' if old_sources.get(source_id) == new_sources.get(source_id)
                  else 'changed_supplied_record' if source_id in new_sources else 'missing'
                  for source_id in old_sources}
        plan_compatible = all(old_payload.get(key) == new_payload.get(key) for key in
                              ('query', 'model', 'execution_mode', 'pipeline_type', 'max_output_tokens'))
        contexts = [db.execute('SELECT signature FROM studio_source_context WHERE owner=? AND run_id=?',
                               (current_user.get_id(), run_id)).fetchone()
                    for run_id in (base['id'], new['id'])]
        plan_compatible = bool(plan_compatible and all(contexts) and contexts[0][0] == contexts[1][0])
        source_set_changed = old_sources.keys() != new_sources.keys()
        sections = []
        paragraphs = re.split(r'\n\s*\n', answer.strip())
        if len(paragraphs) > 100:
            return jsonify(error='Saved answer has too many sections for bounded comparison.'), 409
        for index, paragraph in enumerate(paragraphs):
            dependencies = sorted(set(re.findall(r'\[(source-\d+)\]', paragraph)))
            unknown = not dependencies or any(source not in states or states[source] in {'missing', 'unknown_version'}
                                              for source in dependencies)
            action = ('blocked' if not plan_compatible or source_set_changed or unknown else
                      'changed_input_review' if any(states[source] == 'changed_supplied_record' for source in dependencies)
                      else 'unchanged_input_review')
            sections.append(dict(id=f'section-{index+1}', text=paragraph,
                                 source_ids=dependencies, action=action))
        store().audit(db, current_user.get_id(), 'supplied_source_refresh_planned', base['id'])
    return jsonify(base_run_id=base['id'], new_run_id=new['id'], source_states=states,
                   sections=sections, proposal_context_matches=plan_compatible,
                   source_set_changed=source_set_changed, web_freshness='not_verified',
                   automatic_reuse_allowed=False, execution_context_verified=False, provider_calls=0,
                   note='This compares saved user-supplied records only. Websites were not fetched '
                        'or revalidated. Matching proposal metadata does not verify the instructions actually used during execution. Section citations are provisional dependencies; a person '
                        'must check completeness and uncited claims before any reuse or rewrite.')


@bp.get('/studio/research-controls')
@login_required
def controls_page():
    response = send_file(Path(__file__).parent / 'static' / 'studio_research_controls.html')
    response.headers['Cache-Control'] = 'no-store'
    return response
