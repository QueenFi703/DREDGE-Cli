# Patch to inject architecture routes into server.py
import sys

content = open(__file__[:-11] + 'server.py').read()

injection = '''    # -- Architecture Routes (Pipeline, Providers, Telemetry)
    try:
        from .architecture_routes import register_architecture_routes
        register_architecture_routes(app)
    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not load architecture routes: {e}")
'''

# Find insertion point
idx = content.find('    # -- Application routes')
if idx > 0:
    content = content[:idx] + injection + '\n' + content[idx:]
    open(__file__[:-11] + 'server.py', 'w').write(content)
    print('server.py patched')
