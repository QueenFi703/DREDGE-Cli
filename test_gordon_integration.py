#!/usr/bin/env python
import sys
sys.path.insert(0, '.')
print('Testing Gordon integration with proper startup...')
print()

# Test 1: Import and mount
try:
    from core_gateway import app, mount_adapters
    print('[PASS] Core gateway imported')
    
    # Manually mount adapters (normally done at startup)
    mount_adapters()
    print('[PASS] Adapters mounted')
except Exception as e:
    print(f'[FAIL] Setup failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Test Gordon endpoints
from fastapi.testclient import TestClient
client = TestClient(app)

print()
print('Testing Gordon endpoints...')
gordon_routes = [
    '/gordon/health',
    '/gordon/status',
    '/gordon/capabilities',
]

all_pass = True
for route in gordon_routes:
    r = client.get(route)
    status = 'PASS' if r.status_code == 200 else 'FAIL'
    if r.status_code != 200:
        all_pass = False
    print(f'[{status}] GET {route} -> {r.status_code}')

print()
if all_pass:
    print('SUCCESS! Gordon integration working correctly.')
else:
    print('Some tests failed.')
    sys.exit(1)
