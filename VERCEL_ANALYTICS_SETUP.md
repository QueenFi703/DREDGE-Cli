# Vercel Web Analytics Integration

This document describes the Vercel Web Analytics integration for the DREDGE CLI project.

## Overview

Vercel Web Analytics has been integrated into this Python FastAPI backend application. Since `@vercel/analytics` is a JavaScript/Node.js package and this is a Python project, we've implemented the **static site approach** from the official Vercel documentation.

## Implementation

### 1. Analytics Middleware

A custom FastAPI middleware (`VercelAnalyticsMiddleware`) has been created in `vercel_analytics.py` that:

- Automatically injects the Vercel Analytics script into all HTML responses
- Works transparently without requiring changes to individual route handlers
- Handles edge cases (missing head/body tags)
- Avoids duplicate script injection

### 2. Integration Points

The analytics middleware has been added to:

- `core_gateway.py` - Main application gateway
- `full_web_server.py` - DREDGE Studio web server
- `api/index.py` - API index endpoint

### 3. HTML Templates

Created base HTML templates with analytics pre-configured:

- `templates/base.html` - Base template for extending
- `templates/index.html` - Landing page example with analytics

## How It Works

### Automatic Injection

Any HTML response served by the application will automatically include:

```html
<!-- Vercel Web Analytics -->
<script>
    window.va = window.va || function () { (window.vaq = window.vaq || []).push(arguments); };
</script>
<script defer src="/_vercel/insights/script.js"></script>
```

### Manual Usage

If you need to manually add analytics to a specific HTML file:

```python
from vercel_analytics import inject_vercel_analytics

html_content = "<html><head></head><body>Content</body></html>"
html_with_analytics = inject_vercel_analytics(html_content)
```

Or get the script tags directly:

```python
from vercel_analytics import get_analytics_head_tags

analytics_tags = get_analytics_head_tags()
# Use in your template
```

## Setup on Vercel

To complete the setup:

1. **Enable Analytics in Vercel Dashboard**
   - Go to your project in the Vercel dashboard
   - Navigate to the "Analytics" tab
   - Click "Enable Web Analytics"

2. **Deploy the Application**
   ```bash
   git push origin master
   # Or
   vercel deploy
   ```

3. **Verify Installation**
   - After deployment, visit your application
   - Open browser DevTools → Network tab
   - Look for requests to `/_vercel/insights/view`
   - Check the Vercel dashboard for analytics data (appears after ~24 hours)

## Testing Locally

To test the analytics integration locally:

1. Start the application:
   ```bash
   python core_gateway.py
   # Or
   uvicorn api.deployment:app --reload
   ```

2. Visit any HTML page served by the application

3. Check the page source to verify the analytics script is present

4. Note: Analytics data will only be collected when deployed on Vercel, not during local development

## File Structure

```
.
├── vercel_analytics.py          # Analytics middleware and utilities
├── templates/
│   ├── base.html               # Base template with analytics
│   └── index.html              # Example landing page
├── core_gateway.py             # Main gateway (analytics enabled)
├── full_web_server.py          # Web server (analytics enabled)
└── api/
    └── index.py                # API entry (analytics enabled)
```

## Technical Details

### Middleware Approach

The middleware operates on the response pipeline:

1. Request comes in
2. Route handler generates response
3. Middleware checks if response is HTML
4. If HTML, injects analytics script before `</head>` or `</body>`
5. Returns modified response

### Performance Impact

- Minimal: Script injection happens in-memory before response is sent
- No external dependencies at build time
- Analytics script loads asynchronously (deferred)
- No impact on server-side rendering performance

### Framework Compatibility

This approach works with:

- ✅ FastAPI (Python)
- ✅ Flask (Python)
- ✅ Any ASGI/WSGI Python framework that serves HTML

## Troubleshooting

### Analytics Not Showing Up

1. **Verify script is present**: View page source and search for `/_vercel/insights/script.js`
2. **Check Vercel deployment**: Analytics only works on Vercel-deployed sites
3. **Wait for data**: Initial data can take 24-48 hours to appear
4. **Enable in dashboard**: Ensure Web Analytics is enabled in your Vercel project

### Script Injected Multiple Times

The middleware includes duplicate detection. If you see multiple scripts:
- Check if you're manually including the script in HTML files
- Remove manual script tags; let the middleware handle it

### Middleware Not Loading

Check the startup logs for:
- `[Analytics] Vercel Web Analytics middleware enabled` (success)
- `[Analytics] Vercel Analytics middleware not available` (module not found)
- `[Analytics] Failed to add Vercel Analytics middleware` (error)

## References

- [Vercel Web Analytics Quickstart](https://vercel.com/docs/analytics/quickstart)
- [Vercel Web Analytics for Static Sites](https://vercel.com/docs/analytics/quickstart#static-sites)
- [FastAPI Middleware](https://fastapi.tiangolo.com/tutorial/middleware/)

## Support

For issues related to:
- **Analytics integration**: Check this document and the middleware code
- **Vercel platform**: Consult [Vercel documentation](https://vercel.com/docs)
- **Analytics data**: Contact Vercel support

---

**Last Updated**: July 6, 2026
**Version**: 1.0.0
