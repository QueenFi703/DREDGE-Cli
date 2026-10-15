"""Generic subscription-access information, without commerce or account state."""
from pathlib import Path
from urllib.parse import urlsplit

from flask import Blueprint, Response, current_app


subscription_info_bp = Blueprint('subscription_info', __name__)
_PAGE = Path(__file__).with_name('static') / 'subscription_info.html'


def subscription_info_url():
    """Expose only the fixed informational path on the configured HTTPS origin."""
    # The connected ChatGPT deployment can be separate from website billing.
    # Its generic page is served on that deployment's validated issuer origin.
    origin = (current_app.config.get('CHATGPT_ISSUER')
              if current_app.config.get('DREDGE_CHATGPT_ENABLED') is True
              else current_app.config.get('BILLING_ORIGIN'))
    if not isinstance(origin, str) or not origin:
        return None
    if any(ord(char) <= 32 or ord(char) == 127 or char in '\\%?#' for char in origin):
        return None
    origin = origin.rstrip('/')
    try:
        parsed = urlsplit(origin)
        # Accessing port validates malformed and out-of-range values too.
        _ = parsed.port
    except ValueError:
        return None
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.path or parsed.query
            or parsed.fragment or parsed.username is not None or parsed.password is not None):
        return None
    return origin + '/subscription-info'


@subscription_info_bp.get('/subscription-info')
def subscription_info_page():
    # Deliberately do not read the principal, session, database or query parameters.
    response = Response(_PAGE.read_text(encoding='utf-8'), mimetype='text/html')
    response.headers['Content-Security-Policy'] = (
        "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; "
        "form-action 'none'; frame-ancestors 'none'"
    )
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'no-referrer'
    return response


def register_subscription_info(app):
    """Register a public, GET-only informational page with no account operations."""
    app.register_blueprint(subscription_info_bp)


@subscription_info_bp.get('/privacy')
def plugin_privacy():
    return _public_notice('plugin_privacy.html')


@subscription_info_bp.get('/terms')
def plugin_terms():
    return _public_notice('plugin_terms.html')


@subscription_info_bp.get('/support')
def plugin_support():
    return _public_notice('plugin_support.html')


def _public_notice(filename):
    response = Response(_PAGE.with_name(filename).read_text(encoding='utf-8'), mimetype='text/html')
    response.headers['Content-Security-Policy'] = "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'no-referrer'
    return response


@subscription_info_bp.get('/.well-known/openai-apps-challenge')
def plugin_domain_challenge():
    # Public ownership proof for this submission, never an account credential.
    return Response('hzKn4qMTN_vkhBDOmiSKusERBAOkGyAuYY6I2enYfBs', mimetype='text/plain')
