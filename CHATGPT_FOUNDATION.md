# DREDGE ChatGPT foundation and integration candidate

The first slice supplied internal read-only tools. This local candidate adds an
opt-in standards-based OAuth authorization server and official MCP HTTP transport.
It is **disabled by default and has not been deployed or connected to ChatGPT**.

Read `CHATGPT_SETUP.md` for implemented interfaces, required deployment inputs and
remaining live checks. Read `CHATGPT_COMMERCE_GATE.md` for approval-dependent sales
preparation. No sales tool exists in this candidate.

Base: `6e9019dbadb97b6f43d90bcb82b7ba4d0400eedf`.
Two tools: `get_dredge_access` and `get_studio_activity_report`, both read-only,
owner-bound, requiring `studio:read`, and accepting no arguments. Existing report
behavior and unknown usage/cost remain unchanged.

Local checks:

    PYTHONPATH=src python -m pytest -o addopts='' tests/test_chatgpt_foundation.py tests/test_chatgpt_oauth.py tests/test_chatgpt_transport.py tests/test_chatgpt_gateway.py tests/test_studio.py tests/test_studio_live.py tests/test_billing.py tests/test_server.py -q

Verification records must distinguish offline tests from actual browser-provider
login, deployed HTTPS and ChatGPT connection tests. No authorization to publish,
create live grants, or configure payments is conveyed by this source package.
