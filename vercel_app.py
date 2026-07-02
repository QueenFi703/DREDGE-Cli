"""
Vercel ASGI Entry Point
This file serves as the entry point for Vercel's serverless deployment
It works with both unified_auth_gateway and orion_gateway_authenticated
"""

from unified_auth_gateway import app as gateway_app
from orion_gateway_authenticated import app as orion_app

# Export for Vercel
# Vercel will use this as the ASGI application
app = gateway_app

# Make both available if needed
__all__ = ['app', 'gateway_app', 'orion_app']
