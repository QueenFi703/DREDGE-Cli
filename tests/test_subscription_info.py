"""Public information is static and has no commerce or account-data surface."""
from html.parser import HTMLParser

import pytest
from flask import Flask

from dredge.subscription_info import register_subscription_info, subscription_info_url


@pytest.fixture
def app():
    app = Flask(__name__)
    app.config.update(TESTING=True, SECRET_KEY='fictional-info-test-key')
    register_subscription_info(app)
    return app


class PageElements(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.destinations = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.destinations.extend(value for key, value in attrs
                                 if key in {'href', 'src', 'action', 'formaction'})


def test_information_is_public_static_and_describes_entitlement_limits(app):
    client = app.test_client()
    response = client.get('/subscription-info')
    assert response.status_code == 200
    assert response.mimetype == 'text/html'
    body = response.get_data(as_text=True)
    for expected in ('Subscription access', 'Workspace roles', 'Stored reads',
                     'Paid AI usage is not included', 'Sandbox'):
        assert expected in body
    # Neither request parameters nor session contents personalize this page.
    with client.session_transaction() as session:
        session['studio_profile'] = {'id': 'PRIVATE-OWNER', 'email': 'PRIVATE-EMAIL'}
        session['subscription_status'] = 'active'
    assert client.get('/subscription-info?owner=PRIVATE-OWNER&status=active&next=/billing').data == response.data
    assert 'PRIVATE-' not in body
    assert client.head('/subscription-info').status_code == 200
    for method in ('post', 'put', 'patch', 'delete'):
        assert getattr(client, method)('/subscription-info').status_code == 405


def test_information_has_no_transactions_scripts_forms_or_outbound_links(app):
    response = app.test_client().get('/subscription-info')
    body = response.get_data(as_text=True)
    parsed = PageElements()
    parsed.feed(body)
    assert not set(parsed.tags) & {'script', 'form', 'input', 'button', 'iframe', 'object', 'embed'}
    assert parsed.destinations == []
    for forbidden in ('/billing', '/checkout', '/portal', 'stripe.com', 'paypal.com',
                      'upgrade', 'subscribe now', 'buy now', '$19', 'pricing'):
        assert forbidden not in body.lower()
    assert "default-src 'none'" in response.headers['Content-Security-Policy']
    assert "form-action 'none'" in response.headers['Content-Security-Policy']
    assert response.headers['X-Content-Type-Options'] == 'nosniff'


@pytest.mark.parametrize('origin, expected', [
    ('https://dredge.example.test', 'https://dredge.example.test/subscription-info'),
    ('https://dredge.example.test/', 'https://dredge.example.test/subscription-info'),
    ('https://dredge.example.test:8443', 'https://dredge.example.test:8443/subscription-info'),
])
def test_only_configured_https_origin_can_supply_information_link(app, origin, expected):
    app.config['BILLING_ORIGIN'] = origin
    with app.app_context():
        assert subscription_info_url() == expected


@pytest.mark.parametrize('origin', [
    None, '', False, 42, 'http://dredge.example.test', '//dredge.example.test',
    'javascript:alert(1)', 'https://', 'https://dredge.example.test/billing',
    'https://dredge.example.test?next=/billing', 'https://dredge.example.test#billing',
    'https://dredge.example.test?', 'https://dredge.example.test#',
    'https://owner:password@dredge.example.test', 'https://@dredge.example.test',
    'https://dredge.example.test:invalid', 'https://dredge.example.test:99999',
    'https://dredge.example.test\n', ' https://dredge.example.test',
    'https://dredge.example.test\\other', 'https://[invalid',
])
def test_invalid_origins_never_become_links(app, origin):
    app.config['BILLING_ORIGIN'] = origin
    with app.app_context():
        assert subscription_info_url() is None


def test_connected_plugin_uses_its_deployment_not_production_billing(app):
    app.config.update(DREDGE_CHATGPT_ENABLED=True,
                      CHATGPT_ISSUER='https://chatgpt-staging.example.test',
                      BILLING_ORIGIN='https://production.example.test')
    with app.app_context():
        assert subscription_info_url() == 'https://chatgpt-staging.example.test/subscription-info'
    app.config['CHATGPT_ISSUER'] = ''
    with app.app_context():
        assert subscription_info_url() is None
    app.config['CHATGPT_ISSUER'] = 'https://chatgpt-staging.example.test/billing'
    with app.app_context():
        assert subscription_info_url() is None
