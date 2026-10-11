"""Safe Responses execution receipts, never raw provider output or reasoning."""
from urllib.parse import urlsplit, urlunsplit, parse_qsl
import re

import requests
from flask import current_app
from .studio_provider import ProviderError, response_metadata, safe_id
from .studio import valid_url

SMOKE_PROFILE = 'public-web-smoke-v1'
SMOKE_QUESTION = 'Find the official Missouri child-support payment history information page. Give its title and URL and one sentence describing it. No individual case information.'


def safe_url(value):
    if not isinstance(value, str) or len(value) > 2048 or not valid_url(value):
        return None
    parsed = urlsplit(value)
    # Preserve ordinary public-source query selectors, never credential parameters.
    if any(re.search(r'(token|secret|password|credential|signature|api.?key|authorization|session|jwt)|^(sig|auth|key|code)$', key, re.I)
           for key, _ in parse_qsl(parsed.query, keep_blank_values=True)):
        return None
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ''))


def search_receipts(output):
    receipts = []
    for item in output if isinstance(output, list) else []:
        if not isinstance(item, dict) or item.get('type') != 'web_search_call':
            continue
        action = item.get('action')
        clean = {}
        if isinstance(action, dict):
            if isinstance(action.get('type'), str) and action['type'] in {'search', 'open_page', 'find_in_page'}:
                clean['type'] = action['type']
            # Do not retain query text, page contents, snippets or arbitrary provider fields.
            url = safe_url(action.get('url'))
            if url:
                clean['url'] = url
            sources = action.get('sources')
            clean['sources'] = list(dict.fromkeys(
                url for source in (sources[:100] if isinstance(sources, list) else [])
                if isinstance(source, dict) and (url := safe_url(source.get('url')))))
        receipts.append({'id': safe_id(item.get('id')), 'status': safe_id(item.get('status')), 'action': clean})
        if len(receipts) >= 10:
            break
    return receipts


def provider(payload):
    unknown = dict(provider='OpenAI', provider_calls=1, response_id=None, request_id=None,
                   model=None, provider_status=None, usage=None, token_usage=None,
                   cost_usd=None, cost_status='billing_unavailable', search_calls=[],
                   web_search_verified=False)
    try:
        response = requests.post('https://api.openai.com/v1/responses',
            headers={'Authorization': 'Bearer ' + current_app.config['OPENAI_API_KEY']},
            json=payload, timeout=(10, 45), allow_redirects=False)
    except requests.RequestException:
        raise ProviderError('Provider completion and billing are unknown. No automatic retry was made.', unknown) from None
    unknown['request_id'] = safe_id(response.headers.get('x-request-id'))
    try:
        result = response.json()
    except (ValueError, TypeError):
        raise ProviderError('Provider returned unreadable data. Billing is unknown.', unknown) from None
    if not isinstance(result, dict):
        raise ProviderError('Provider returned invalid data. Billing is unknown.', unknown)
    metadata = response_metadata(result, response.headers.get('x-request-id'))
    metadata['search_calls'] = search_receipts(result.get('output'))
    metadata['web_search_verified'] = any(c['id'] and c['status'] == 'completed' for c in metadata['search_calls'])
    if response.status_code != 200 or result.get('status') != 'completed':
        raise ProviderError('Provider did not complete successfully. Reported usage and search activity are retained; no draft was accepted.', metadata)
    if not metadata['response_id'] or not metadata['model']:
        raise ProviderError('Provider response identity is missing. No draft was accepted.', metadata)
    blocks = []
    output = result.get('output')
    if not isinstance(output, list):
        raise ProviderError('Provider output is invalid. Reported usage is retained.', metadata)
    for item in output:
        if not isinstance(item, dict) or item.get('type') != 'message':
            continue
        content = item.get('content')
        if not isinstance(content, list):
            raise ProviderError('Provider message content is invalid. Reported usage is retained.', metadata)
        for part in content:
            if not isinstance(part, dict) or part.get('type') != 'output_text' or not isinstance(part.get('text'), str):
                continue
            citations = []
            annotations = part.get('annotations')
            for a in annotations if isinstance(annotations, list) else []:
                if not isinstance(a, dict) or a.get('type') != 'url_citation' or not safe_url(a.get('url')):
                    continue
                start, end = a.get('start_index'), a.get('end_index')
                if type(start) is not int or type(end) is not int or not 0 <= start <= end <= len(part['text']):
                    continue
                title = a.get('title')
                citations.append(dict(title=title[:300] if isinstance(title, str) else 'Source', url=safe_url(a['url']), start=start, end=end))
            if part['text'].strip():
                blocks.append(dict(text=part['text'], citations=citations))
    if not blocks:
        raise ProviderError('Provider returned no usable answer. Reported usage is retained.', metadata)
    return dict(metadata, blocks=blocks, mode='live', human_review_required=True)
