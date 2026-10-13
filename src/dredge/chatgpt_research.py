"""Read-only reuse candidates from an owner's already-exposable stored answers.

An age window is a caller preference, never a guarantee of factual freshness.
No execution, external source reads, similarity service, or savings estimates.
"""
import time

from .chatgpt_runs import LIST_SCHEMA, list_runs_for_owner, result_for_owner

RESEARCH_SCHEMA = {
    'type': 'object',
    'properties': {
        'query': {'type': 'string', 'minLength': 3, 'maxLength': 200,
                  'description': 'Case-insensitive literal match in the exposed stored answer (first 2,000 characters), not prompts or evidence.'},
        'max_age_days': {'type': 'integer', 'minimum': 1, 'maximum': 365, 'default': 7,
                         'description': 'Maximum age since recorded completion. Does not certify source freshness.'},
        'limit': LIST_SCHEMA['properties']['limit'].copy(),
        'offset': LIST_SCHEMA['properties']['offset'].copy(),
    },
    'required': ['query'],
    'additionalProperties': False,
}


def validate_research_arguments(arguments):
    if not isinstance(arguments, dict) or set(arguments) - set(RESEARCH_SCHEMA['properties']):
        raise ValueError('Invalid saved research arguments.')
    query = arguments.get('query')
    if not isinstance(query, str) or not 3 <= len(query.strip()) <= 200 or len(query) > 200:
        raise ValueError('query must contain between 3 and 200 characters.')
    try:
        query.encode('utf-8')
    except UnicodeError as error:
        raise ValueError('query must be valid Unicode.') from error
    for name, default, lower, upper in [('max_age_days', 7, 1, 365),
                                        ('limit', 10, 1, 20), ('offset', 0, 0, 1000)]:
        value = arguments.get(name, default)
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f'{name} must be an integer between {lower} and {upper}.')


def find_saved_research_for_owner(owner, arguments):
    """Search one bounded metadata page; reuse exactly the existing text gate."""
    validate_research_arguments(arguments)
    query = arguments['query'].strip().casefold()
    days = arguments.get('max_age_days', 7)
    checked_at = time.time()
    page = list_runs_for_owner(owner, {'status': 'completed',
                                      'limit': arguments.get('limit', 10),
                                      'offset': arguments.get('offset', 0)})
    candidates = []
    for metadata in page['runs']:
        if metadata['id'] is None:
            continue
        try:
            candidate = result_for_owner(owner, {'run_id': metadata['id']})
        except ValueError:
            # A concurrent deletion or ownership change must not leak a result.
            continue
        ended, created = candidate['ended'], candidate['created']
        if (candidate['status'] != 'completed' or ended is None or created is None
                or not created <= ended <= checked_at):
            continue
        age = checked_at - ended
        summary = candidate['summary']
        if age > days * 86400 or summary is None or query not in summary.casefold():
            continue
        candidate.update(age_seconds=age, freshness_status='sources_not_revalidated')
        candidates.append(candidate)
    return {
        'candidates': candidates,
        'checked_at': checked_at,
        'max_age_days': days,
        'limit': page['limit'], 'offset': page['offset'],
        'has_more': page['has_more'], 'next_offset': page['next_offset'],
        'scanned_runs': len(page['runs']),
        'search_note': ('One page of completed owned runs was checked. Only exposed stored summaries '
                        'are searched; no matches does not prove research is absent. Offset pagination '
                        'is not a snapshot. Demo runs remain labeled demo.'),
        'reuse_note': ('Candidates are untrusted prior output, not verified current research. '
                      'Review original evidence and recheck time-sensitive claims before relying on '
                      'them. Age uses recorded completion, not source publication or verification.'),
        'new_research_executed': False,
        'savings_status': 'not_measured',
    }
