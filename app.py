"""
Main ASGI Application Entry Point for Vercel
Simplified entry point that handles import errors gracefully
"""

import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Try to import the unified gateway
try:
    from unified_auth_gateway import app
    print("✅ Successfully imported unified_auth_gateway")
except ImportError as e:
    print(f"❌ Failed to import unified_auth_gateway: {e}")
    print("Attempting fallback to orion_gateway_authenticated...")
    try:
        from orion_gateway_authenticated import app
        print("✅ Successfully imported orion_gateway_authenticated")
    except ImportError as e2:
        print(f"❌ Failed to import orion_gateway_authenticated: {e2}")
        print("Creating minimal FastAPI app...")
        from fastapi import FastAPI
        app = FastAPI(title="DREDGE Error", version="1.0.0")
        
        @app.get("/health")
        async def health():
            return {"status": "error", "message": "Failed to load application"}

# Ensure app is exported (Vercel requirement)
__all__ = ['app']


# Local testing entry point
if __name__ == "__main__":
    import uvicorn
    
    print("=" * 80)
    print("  DREDGE Auth Gateway - Starting")
    print("=" * 80)
    print()
    print("Starting on http://127.0.0.1:9000")
    print("Swagger UI: http://127.0.0.1:9000/docs")
    print("Health: http://127.0.0.1:9000/health")
    print()
    print("Press Ctrl+C to stop")
    print()
    print("=" * 80)
    print()
    
    try:
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=9000,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\nShutdown...")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        import traceback
        traceback.print_exc()
