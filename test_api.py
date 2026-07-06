#!/usr/bin/env python
import sys
sys.path.insert(0, '.')
from fastapi.testclient import TestClient
from api.index import app

client = TestClient(app)

routes_to_test = [
    ('/', 'GET'),
    ('/health', 'GET'),
    ('/status', 'GET'),
    ('/api', 'GET'),
    ('/api/status', 'GET'),
    ('/api/capabilities', 'GET'),
    ('/api/dredge/status', 'GET'),
    ('/api/gordon/capabilities', 'GET'),
]

print("Testing API endpoints...\n")
all_pass = True
for route, method in routes_to_test:
    if method == 'GET':
        r = client.get(route)
    status = 'PASS' if r.status_code == 200 else 'FAIL'
    if r.status_code != 200:
        all_pass = False
    print(f'[{status}] {method:4} {route:30} -> {r.status_code}')

print()
if all_pass:
    print("SUCCESS: All tests passed!")
else:
    print("FAILURE: Some tests failed!")
    sys.exit(1)
