"""Opt-in, owner-bound budget and public catalog controls. Never execute AI."""
import json
import re
from copy import deepcopy

from flask import current_app
from flask_login import current_user

from .chatgpt_runs import RESULT_SCHEMA, _run_uuid, _finite_number, _stored_object
from .public_source_collector import CATALOG, catalog_key

RESEARCH_SCOPE = 'studio:research:write'
BUDGET_SCOPE = 'studio:budget:write'
BUDGET_READ_SCOPE = 'studio:budget:read'
EMPTY_SCHEMA = {'type': 'object', 'properties': {}, 'additionalProperties': False}
BUDGET_SCHEMA = {'type': 'object', 'properties': {
    'currency': {'type': 'string', 'enum': ['USD']},
    'enabled': {'type': 'boolean'},
    'limit_micro_usd': {'type': 'integer', 'minimum': 0, 'maximum': 1_000_000_000_000,
                        'description': 'Cumulative integer micro-USD limit; 1 USD = 1,000,000 micro-USD.'}},
    'required': ['currency', 'enabled', 'limit_micro_usd'], 'additionalProperties': False}
CHECK_SCHEMA = deepcopy(RESULT_SCHEMA)
CHECK_SCHEMA['properties']['catalog_ids'] = {'type': 'array', 'minItems': 1, 'maxItems': 3,
    'uniqueItems': True, 'items': {'type': 'string', 'enum': list(CATALOG)},
    'description': 'Approved catalog IDs already cited by this owned public_or_fictional run.'}
CHECK_SCHEMA['required'].append('catalog_ids')
SPECS = (
    ('get_my_studio_budget', 'Read your cumulative protected-generation budget, reservations and unknown actual billing. No payment or execution.', EMPTY_SCHEMA, BUDGET_READ_SCOPE),
    ('update_my_studio_budget', 'Change your cumulative protected-generation USD budget or enable/disable it. Requires operator/admin. Disabling removes this protection for subsequent generation. No reset, payment or AI execution.', BUDGET_SCHEMA, BUDGET_SCOPE),
    ('list_approved_public_sources', 'List the fixed public institutional source catalog. No network fetch or private run data.', EMPTY_SCHEMA, None),
    ('check_my_studio_public_sources', 'Fetch one to three selected approved catalog pages already cited by your public_or_fictional run and save observations. Requires operator/admin. Returns hashes and section actions only, no source or answer text. No AI execution or automatic answer rewrite.', CHECK_SCHEMA, RESEARCH_SCOPE),
    ('get_my_studio_source_summary', 'Read up to ten saved public-source checks for your public_or_fictional run: version/hash states and section actions only. No excerpts, answer paragraphs, raw case data, source fetch or AI execution.', RESULT_SCHEMA, None),
)
CONTROL_NAMES = frozenset(spec[0] for spec in SPECS)


def required_scopes(name):
    scopes = ['studio:read']
    if name in {'get_my_studio_budget', 'update_my_studio_budget'}:
        scopes.append(BUDGET_READ_SCOPE)
    if name == 'update_my_studio_budget':
        scopes.append(BUDGET_SCOPE)
    if name in {'check_my_studio_public_sources', 'get_my_studio_source_summary'}:
        scopes.append('studio:runs:read')
    if name == 'check_my_studio_public_sources':
        scopes.append(RESEARCH_SCOPE)
    return scopes


def tool_definitions():
    return [{'name': name, 'description': description, 'inputSchema': deepcopy(schema),
             'annotations': {'readOnlyHint': scope not in {BUDGET_SCOPE, RESEARCH_SCOPE},
                             'destructiveHint': name == 'update_my_studio_budget',
                             'idempotentHint': name != 'check_my_studio_public_sources',
                             'openWorldHint': name == 'check_my_studio_public_sources'},
             'securitySchemes': [{'type': 'oauth2', 'scopes': required_scopes(name)}]}
            for name, description, schema, scope in SPECS]


def validate_control_arguments(name, arguments):
    """Pure semantic validation, before any consent challenge or data read."""
    if not isinstance(arguments, dict) or name not in CONTROL_NAMES:
        raise ValueError('Invalid controls request.')
    fields = ({'currency', 'enabled', 'limit_micro_usd'} if name == 'update_my_studio_budget' else
              {'run_id', 'catalog_ids'} if name == 'check_my_studio_public_sources' else
              {'run_id'} if name == 'get_my_studio_source_summary' else set())
    if set(arguments) != fields:
        raise ValueError('Invalid controls fields.')
    if 'run_id' in fields and not _run_uuid(arguments['run_id']):
        raise ValueError('Invalid run ID.')
    if name == 'update_my_studio_budget':
        if (arguments['currency'] != 'USD' or type(arguments['enabled']) is not bool
                or type(arguments['limit_micro_usd']) is not int
                or not 0 <= arguments['limit_micro_usd'] <= 1_000_000_000_000):
            raise ValueError('Invalid USD budget settings.')
    if 'catalog_ids' in fields:
        ids = arguments['catalog_ids']
        if (not isinstance(ids, list) or not 1 <= len(ids) <= 3
                or any(not isinstance(item, str) or item not in CATALOG for item in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError('Select one to three distinct approved catalog IDs.')


def _owned_public_payload(db, run_id):
    from .studio_research_controls import own_run, stored_payload
    run = own_run(db, run_id)
    if not run:
        raise PermissionError('Owned run unavailable.')
    payload = stored_payload(run['payload'])
    if payload.get('data_scope') != 'public_or_fictional':
        raise PermissionError('Public or fictional labeling is required.')
    return payload


def _summary(result):
    """Explicit allowlist: never return backend excerpts, paragraphs or errors."""
    if not isinstance(result, dict):
        return None
    def identifier(value, pattern):
        return value if isinstance(value, str) and re.fullmatch(pattern, value) else None
    raw_sources, raw_sections = result.get('sources'), result.get('sections')
    sources = []
    for item in (raw_sources if isinstance(raw_sources, list) else [])[:3]:
        if not isinstance(item, dict):
            continue
        url = item.get('url')
        key = catalog_key(url) if isinstance(url, str) and len(url) <= 4096 else None
        state = item.get('state')
        sources.append(dict(source_id=identifier(item.get('source_id'), r'source-[1-9][0-9]?'),
            state=state if isinstance(state, str) and state in {'changed', 'unchanged', 'first_observation', 'unknown'} else 'unknown',
            observation_id=_run_uuid(item.get('observation_id')),
            previous_observation_id=_run_uuid(item.get('previous_observation_id')),
            observed_at=_finite_number(item.get('observed_at'), 253402300799),
            digest=identifier(item.get('digest'), r'[a-f0-9]{64}'),
            collector_version=identifier(item.get('collector_version'), r'[a-zA-Z0-9._-]{1,80}'),
            bytes_observed=_finite_number(item.get('bytes_observed'), 524288), catalog_id=key))
    sections = []
    for item in (raw_sections if isinstance(raw_sections, list) else [])[:100]:
        if not isinstance(item, dict):
            continue
        action = item.get('action')
        sections.append(dict(id=identifier(item.get('id'), r'section-(?:[1-9][0-9]?|100)'),
                             action=action if isinstance(action, str) and action in {
                                 'blocked', 'refresh_review', 'unchanged_source_review'} else 'blocked'))
    return dict(check_id=_run_uuid(result.get('check_id')), run_id=_run_uuid(result.get('run_id')), sources=sources,
                sections=sections, provider_calls=0, automatic_reuse_allowed=False,
                note='First observations establish a baseline only. Changes may be cosmetic. '
                     'Section citations are provisional; human review is required. No answer was rewritten.')


def call_control_tool(name, arguments, *, scopes):
    from .studio import role, store
    from .studio_budgets import budget_status, set_budget
    validate_control_arguments(name, arguments)
    if (current_app.config.get('DREDGE_CHATGPT_CONTROLS_ENABLED') is not True
            or not current_user.is_authenticated or not current_user.get_id()
            or not isinstance(scopes, frozenset) or 'studio:read' not in scopes):
        raise PermissionError('Controls authorization is required.')
    required = next(spec[3] for spec in SPECS if spec[0] == name)
    if not frozenset(required_scopes(name)) <= scopes:
        raise PermissionError('The required controls scopes are missing.')
    if required in {BUDGET_SCOPE, RESEARCH_SCOPE} and role() not in {'operator', 'admin'}:
        raise PermissionError('Scoped operator authorization is required.')
    owner = current_user.get_id()
    if name == 'list_approved_public_sources':
        return dict(sources=[dict(id=key, title=title, url=url) for key, (title, url) in CATALOG.items()])
    with store().connect() as db:
        if name == 'get_my_studio_budget':
            return budget_status(db, owner)
        if name == 'update_my_studio_budget':
            db.execute('BEGIN IMMEDIATE')
            result = set_budget(db, owner, enabled=arguments['enabled'], limit_micro_usd=arguments['limit_micro_usd'])
            store().audit(db, owner, 'studio_budget_updated', detail={
                'enabled': result['enabled'], 'limit_micro_usd': result['limit_micro_usd']})
            return result
        payload = _owned_public_payload(db, arguments['run_id'])
        if name == 'get_my_studio_source_summary':
            rows = db.execute('SELECT id,created,status,result FROM public_source_checks '
                              'WHERE owner=? AND run_id=? ORDER BY created DESC,id DESC LIMIT 10',
                              (owner, arguments['run_id'])).fetchall()
            return dict(checks=[dict(id=row['id'], created=row['created'], status=row['status'],
                                    summary=_summary(_stored_object(row['result'])) if row['result'] else None)
                               for row in rows])
        selected = [source['id'] for source in payload['evidence']
                    if catalog_key(source['url']) in arguments['catalog_ids']]
        found = {catalog_key(source['url']) for source in payload['evidence'] if source['id'] in selected}
        if found != set(arguments['catalog_ids']) or not 1 <= len(selected) <= 3:
            raise ValueError('Selected catalog sources must already be cited by this run (at most three records).')
    from .public_source_revalidation import check_sources_for_owner
    response = check_sources_for_owner(arguments['run_id'], selected)
    if isinstance(response, tuple) or response.status_code != 200:
        raise ValueError('The source check is unavailable or invalid.')
    return _summary(response.get_json())
