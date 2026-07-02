"""
Main ASGI Application Entry Point for Vercel
Vercel recognizes: app.py, index.py, server.py, main.py, wsgi.py, or asgi.py
This file serves as the root entry point for all deployments
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the unified gateway as the primary app
try:
    from unified_auth_gateway import app as gateway_app
    logger.info("Successfully imported unified_auth_gateway")
except ImportError as e:
    logger.error(f"Failed to import unified_auth_gateway: {e}")
    raise

# Create the main ASGI application
# This is what Vercel will use
app = gateway_app

# Ensure CORS is configured
if not any(isinstance(m, type(CORSMiddleware)) for m in app.user_middleware):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Health check endpoint at root for Vercel
@app.get("/health", tags=["Health"])
async def root_health():
    """Root health check endpoint for Vercel monitoring"""
    return {
        "status": "healthy",
        "service": "dredge-auth-gateway",
        "version": "2.0.0",
        "environment": os.getenv("FLASK_ENV", "development")
    }

# Vercel requires this for serverless functions
if __name__ == "__main__":
    import uvicorn
    
    print("=" * 80)
    print("  DREDGE Auth Gateway - Main Entry Point")
    print("=" * 80)
    print()
    print("Starting on http://127.0.0.1:9000")
    print("Swagger UI: http://127.0.0.1:9000/docs")
    print("Health: http://127.0.0.1:9000/health")
    print()
    print("For Vercel deployment:")
    print("  This file (app.py) is the recognized entry point")
    print("  Deploy with: vercel --prod")
    print()
    print("=" * 80)
    print()
    
    # Run with uvicorn
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=9000,
        reload=False,
        log_level="info"
    )
