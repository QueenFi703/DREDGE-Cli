"""Internal read-only ChatGPT tools, used by the opt-in MCP transport.

The OAuth/MCP adapter validates tokens and binds their issuer/subject to an
existing DREDGE account before entering this layer. Never accept scopes or account identity from tool arguments.
Browser login alone is NOT OAuth authorization for ChatGPT.
"""
from copy import deepcopy

from flask import current_app
from flask_login import current_user

from .billing import configured, has_subscription, mode
from .studio import activity_report_for_owner, role

SCOPE = 'studio:read'
_TOOLS = tuple({
    'name': name,
    'description': description,
    'inputSchema': {'type': 'object', 'properties': {}, 'additionalProperties': False},
    'annotations': {'readOnlyHint': True, 'destructiveHint': False,
                    'idempotentHint': True, 'openWorldHint': False},
    'securitySchemes': [{'type': 'oauth2', 'scopes': [SCOPE]}],
} for name, description in (
    ('get_dredge_access', 'Read the signed-in account’s current DREDGE access configuration. No purchases or account changes.'),
    ('get_studio_activity_report', 'Read aggregate activity for the signed-in account’s own Studio runs. Missing usage and costs remain unknown.'),
))


def tool_definitions():
    """Proposed MCP descriptors only; returning them does not register a server."""
    return deepcopy(list(_TOOLS))


def call_read_tool(name, arguments, *, scopes):
    """Internal adapter seam; scopes MUST come from a verified OAuth access token.

    This does not validate tokens, and must not be exposed as a Flask route.
    The current principal must be resolved by the trusted auth adapter.
    """
    if not current_user.is_authenticated or not current_user.get_id():
        raise PermissionError('An authenticated DREDGE account is required.')
    if not isinstance(scopes, frozenset) or SCOPE not in scopes:
        raise PermissionError('The studio:read scope is required.')
    if not isinstance(arguments, dict) or arguments:
        raise ValueError('These read-only tools accept no arguments.')
    if name == 'get_studio_activity_report':
        return activity_report_for_owner(current_user.get_id())
    if name == 'get_dredge_access':
        return {
            'workspace_role': role(),
            'studio_report_available': True,
            'subscription_status': ('sandbox_only' if mode() != 'live' else
                                    'active' if has_subscription(current_user.get_id()) else 'inactive'),
            'subscription_requirement_enabled': bool(current_app.config['BILLING_REQUIRE_SUBSCRIPTION']),
            'billing_configured': configured(),
            'billing_environment': mode(),
        }
    raise ValueError('Unknown read-only tool.')


def register_chatgpt_foundation(app):
    """Configure the optional OAuth component; transport uses its ASGI factory."""
    from .chatgpt_transport import configure_chatgpt
    configure_chatgpt(app)
