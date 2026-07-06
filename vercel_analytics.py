"""
Vercel Web Analytics Integration for Python FastAPI Backend

This module provides integration with Vercel Web Analytics for FastAPI applications
that serve HTML content. Since @vercel/analytics is a JavaScript package and this is
a Python backend, we use the static site approach by injecting the analytics script
into HTML responses.

Usage:
    1. Add the analytics script middleware to your FastAPI app
    2. Any HTML responses will automatically include Vercel Analytics tracking

Documentation:
    https://vercel.com/docs/analytics/quickstart
"""

from fastapi import Request, Response
from fastapi.responses import HTMLResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable
import re


class VercelAnalyticsMiddleware(BaseHTTPMiddleware):
    """
    Middleware to inject Vercel Analytics script into HTML responses.
    
    This middleware automatically adds the Vercel Web Analytics tracking code
    to any HTML response served by the application.
    """
    
    # Vercel Analytics script to inject
    ANALYTICS_SCRIPT = """
    <!-- Vercel Web Analytics -->
    <script>
        window.va = window.va || function () { (window.vaq = window.vaq || []).push(arguments); };
    </script>
    <script defer src="/_vercel/insights/script.js"></script>
"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and inject analytics into HTML responses.
        
        Args:
            request: The incoming request
            call_next: The next middleware or route handler
            
        Returns:
            Response with analytics injected if it's HTML
        """
        response = await call_next(request)
        
        # Only process HTML responses
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type:
            # Read the response body
            body = b""
            async for chunk in response.body_iterator:
                body += chunk
            
            # Decode and inject analytics
            html_content = body.decode("utf-8")
            
            # Check if analytics is already present
            if "/_vercel/insights/script.js" not in html_content:
                # Try to inject before </head> tag
                if "</head>" in html_content:
                    html_content = html_content.replace(
                        "</head>",
                        f"{self.ANALYTICS_SCRIPT}</head>",
                        1
                    )
                # Fallback: inject before </body> tag
                elif "</body>" in html_content:
                    html_content = html_content.replace(
                        "</body>",
                        f"{self.ANALYTICS_SCRIPT}</body>",
                        1
                    )
                # Fallback: append to end of document
                else:
                    html_content += self.ANALYTICS_SCRIPT
            
            # Return modified response
            return HTMLResponse(
                content=html_content,
                status_code=response.status_code,
                headers=dict(response.headers)
            )
        
        return response


def inject_vercel_analytics(html_content: str) -> str:
    """
    Inject Vercel Analytics script into HTML content.
    
    This is a utility function that can be used to manually inject
    analytics into HTML content if you're not using the middleware.
    
    Args:
        html_content: The HTML content to modify
        
    Returns:
        HTML content with Vercel Analytics script injected
        
    Example:
        >>> html = "<html><head></head><body>Content</body></html>"
        >>> html_with_analytics = inject_vercel_analytics(html)
    """
    analytics_script = """
    <!-- Vercel Web Analytics -->
    <script>
        window.va = window.va || function () { (window.vaq = window.vaq || []).push(arguments); };
    </script>
    <script defer src="/_vercel/insights/script.js"></script>
"""
    
    # Check if analytics is already present
    if "/_vercel/insights/script.js" in html_content:
        return html_content
    
    # Try to inject before </head> tag
    if "</head>" in html_content:
        return html_content.replace("</head>", f"{analytics_script}</head>", 1)
    
    # Fallback: inject before </body> tag
    if "</body>" in html_content:
        return html_content.replace("</body>", f"{analytics_script}</body>", 1)
    
    # Fallback: append to end of document
    return html_content + analytics_script


def get_analytics_head_tags() -> str:
    """
    Get the Vercel Analytics script tags for manual inclusion.
    
    Returns:
        HTML script tags for Vercel Analytics
        
    Example:
        In your HTML template:
        <head>
            {{ analytics_tags }}
        </head>
    """
    return """<script>
        window.va = window.va || function () { (window.vaq = window.vaq || []).push(arguments); };
    </script>
    <script defer src="/_vercel/insights/script.js"></script>"""
