"""Vercel entrypoint for the DREDGE Flask application."""

from dredge.server import create_app

# Vercel Python runtime looks for a top-level WSGI application object.
app = create_app()
