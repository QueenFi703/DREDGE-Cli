"""
DREDGE x Dolly Server
A lightweight web server for the DREDGE x Dolly integration.
"""
import hashlib
import os
import sys
import logging
import time
from functools import lru_cache
from pathlib import Path
from flask import Flask, Request, jsonify, request, send_file, redirect, url_for, session
from flask_login import login_required, current_user, logout_user

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

from . import __version__
from .config import load_config


def setup_logging(debug: bool = False):
    """Setup logging configuration."""
    config = load_config()
    log_config = config.get("logging", {})
    
    # Safely get log level with validation
    level_name = log_config.get("level", "INFO")
    try:
        level = logging.DEBUG if debug else getattr(logging, level_name, logging.INFO)
    except AttributeError:
        level = logging.INFO
    
    log_format = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


@lru_cache(maxsize=1024)
def _compute_insight_hash(insight_text: str) -> str:
    """Compute SHA256 hash of insight text with caching for repeated insights."""
    return hashlib.sha256(insight_text.encode()).hexdigest()


class BoundedRequest(Request):
    """Support endpoint-specific limits on Flask releases before 3.1 too."""
    _route_max_content_length = None

    @property
    def max_content_length(self):
        if self._route_max_content_length is not None:
            return self._route_max_content_length
        return super().max_content_length

    @max_content_length.setter
    def max_content_length(self, value):
        self._route_max_content_length = value


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.request_class = BoundedRequest

    # -- Session secret key
    secret_key = os.environ.get("SECRET_KEY", "")
    if not secret_key:
        import secrets as _secrets
        secret_key = _secrets.token_hex(32)
        logging.getLogger(__name__).warning(
            "SECRET_KEY not set - using a random key. "
            "Sessions will not survive restarts. Set SECRET_KEY in your environment."
        )
    app.secret_key = secret_key

    # -- Flask-Login configuration
    from flask_login import LoginManager
    login_manager = LoginManager()
    login_manager.login_view = "auth.login"
    login_manager.login_message = "Please sign in to access this page."
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        from .auth import _users, User
        profile = session.get('studio_profile', {})
        if profile.get('id') == user_id:
            return User(user_id, profile['name'], profile['email'], profile['provider'], profile.get('avatar', ''))
        return _users.get(user_id)

    # -- OAuth / login
    from .auth import init_auth
    init_auth(app)

    from .studio import register_studio, role
    register_studio(app)
    from .billing import register_billing
    register_billing(app)
    from .casework import register_casework
    register_casework(app)
    from .client_portal import register_clients
    register_clients(app)
    app.config['MAX_CONTENT_LENGTH'] = 65536
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
    if os.environ.get('OAUTH_REDIRECT_BASE', '').startswith('https://'):
        app.config['SESSION_COOKIE_SECURE'] = True

    @login_manager.unauthorized_handler
    def unauthorized():
        if request.path == '/advanced':
            from uuid import UUID
            try:
                session['studio_return_run'] = str(UUID(request.args.get('run', '')))
            except (ValueError, TypeError, AttributeError):
                session.pop('studio_return_run', None)
        if request.path.startswith('/api/') or request.is_json:
            return jsonify(error='Your session has expired. Sign in again.', code='session_expired',
                           login_url='/auth/login?reason=session_expired'), 401
        return redirect('/auth/login?reason=session_expired')

    @app.before_request
    def studio_session_guard():
        if current_user.is_authenticated:
            now = time.time()
            started = session.setdefault('studio_started', now)
            last = session.setdefault('studio_last_activity', now)
            if now-started > app.config['STUDIO_SESSION_SECONDS'] or now-last > app.config['STUDIO_IDLE_SECONDS']:
                logout_user()
                for key in ('studio_profile','studio_started','studio_last_activity','studio_csrf'):
                    session.pop(key, None)
                if request.path.startswith('/api/') or request.path in {'/advanced','/advanced/toolkit','/lift'}:
                    return unauthorized()
            else:
                session['studio_last_activity'] = now
        if request.path.startswith(('/api/advanced/', '/api/architecture/', '/api/gordon/', '/api/dependabot/')):
            if request.path == '/api/architecture/health':
                return None
            if not current_user.is_authenticated:
                return unauthorized()
            if request.method not in {'GET','HEAD','OPTIONS'}:
                if role() not in {'operator','admin'}:
                    return jsonify(error='An operator role is required.'), 403
                import hmac
                token = session.get('studio_csrf', '')
                if not token or not hmac.compare_digest(token, request.headers.get('X-CSRF-Token','')):
                    return jsonify(error='Refresh the workspace before submitting.', code='csrf_failed'), 403
                if request.path == '/api/architecture/pipeline/execute':
                    return jsonify(error='Use the Studio proposal and review workflow before pipeline execution.'), 409

    @app.after_request
    def studio_headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        response.headers['X-Frame-Options'] = 'SAMEORIGIN'
        if request.path.startswith(('/api/studio/', '/api/billing/', '/api/casework/', '/api/client/', '/client', '/casework', '/billing', '/auth/')) or request.path.startswith('/advanced'):
            response.headers['Cache-Control'] = 'no-store'
        if request.path.startswith(('/api/advanced/', '/api/dependabot/')) and response.is_json:
            data = response.get_json()
            if isinstance(data, dict) and 'error' not in data:
                data['operation_mode'] = 'simulated'
                data['disclosure'] = 'Prototype demonstration response; not verified live execution.'
                response.set_data(app.json.dumps(data))
        return response

    # -- Advanced Features
    try:
        from .advanced_features import register_advanced_features
        register_advanced_features(app)
    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not load advanced features: {e}")

    # -- Architecture Routes (Pipeline, Providers, Telemetry)
    try:
        from .architecture_routes import register_architecture_routes
        register_architecture_routes(app)
    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not load architecture routes: {e}")


    # -- Gordon Integration Routes
    try:
        from .gordon_routes import register_gordon_routes
        register_gordon_routes(app)
    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not load Gordon routes: {e}")
    # -- Application routes

    @app.route('/')
    def index():
        """Root endpoint with API information."""
        from flask_login import current_user
        
        if not current_user.is_authenticated:
            return redirect(url_for('auth.login'))
        
        return jsonify({
            "name": "DREDGE x Dolly",
            "version": __version__,
            "description": "GPU-CPU Lifter - Save - Files - Print",
            "user": {
                "name":     current_user.name,
                "email":    current_user.email,
                "provider": current_user.provider,
            },
            "endpoints": {
                "/":              "API information (this page)",
                "/health":        "Health check (public)",
                "/lift":          "Lift an insight (POST, authenticated)",
                "/quasimoto-gpu": "Quasimoto GPU visualization (authenticated)",
                "/advanced":      "Advanced features dashboard (authenticated)",
                "/api/advanced/*": "Advanced feature endpoints",
                "/auth/login":    "Sign-in page",
                "/auth/logout":   "Sign out",
                "/auth/me":       "Current user profile (JSON)",
                "/auth/status":   "Authentication status (public)",
            }
        })

    @app.route('/health')
    def health():
        """Health check endpoint (public - no auth required)."""
        return jsonify({"status": "healthy", "version": __version__})

    @app.route('/lift', methods=['POST'])
    @login_required
    def lift_insight():
        """
        Lift an insight with Dolly integration.

        Expected JSON payload:
        {
            "insight_text": "Your insight text here"
        }
        """
        data = request.get_json()
        
        if not data or 'insight_text' not in data:
            return jsonify({
                "error": "Missing required field: insight_text"
            }), 400
        
        insight_text = data['insight_text']
        
        # Optimised: use cached hash computation for duplicate insights
        insight_id = _compute_insight_hash(insight_text)
        
        result = {
            "id": insight_id,
            "text": insight_text,
            "lifted": True,
            "message": "Insight processed (full GPU acceleration requires PyTorch/Dolly setup)"
        }
        
        return jsonify(result)

    @app.route('/advanced')
    @login_required
    def advanced_dashboard():
        """Serve the advanced features dashboard."""
        static_dir = Path(__file__).parent / 'static'
        # Prefer the governed workspace; retain older dashboards as a fallback.
        html_file = static_dir / 'studio_workspace.html'
        if not html_file.exists():
            html_file = static_dir / 'dashboard_simple.html'
        if not html_file.exists():
            html_file = static_dir / 'advanced_dashboard.html'
        
        if not html_file.exists():
            return jsonify({"error": "Dashboard file not found"}), 404
        
        return send_file(html_file, mimetype='text/html')

    @app.route('/advanced/toolkit')
    @login_required
    def demonstration_toolkit():
        return send_file(Path(__file__).parent / 'static' / 'frontend_complete_swift.html', mimetype='text/html')

    @app.route('/quasimoto-gpu')
    @login_required
    def quasimoto_gpu():
        """Serve the Quasimoto GPU visualization page."""
        static_dir = Path(__file__).parent / 'static'
        html_file = static_dir / 'quasimoto-gpu.html'
        
        if not html_file.exists():
            return jsonify({"error": "Visualization file not found"}), 404
        
        return send_file(html_file, mimetype='text/html')

    return app


def run (host='0.0.0.0', port=8001, debug=False):
    """
    Run the DREDGE x Dolly server.

    Args:
        host:  Host to bind to (default: 0.0.0.0)
        port:  Port to listen on (default: 3000)
        debug: Enable debug mode (default: False)
    """
    logger = setup_logging(debug)

    logger.info(f"Starting DREDGE x Dolly Server v{__version__}")
    logger.info(f"Host: {host}, Port: {port}, Debug: {debug}")

    app = create_app()

    # Use ASCII-safe output
    if sys.stdout.encoding and 'utf' not in sys.stdout.encoding.lower():
        sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)
    
    print(f"Starting DREDGE x Dolly server on http://{host}:{port}")
    print(f"API Version: {__version__}")
    print(f"Dashboard: http://localhost:{port}/advanced")
    print(f"API Endpoints: http://localhost:{port}/api/advanced/")
    print(f"Sign in at: http://localhost:{port}/auth/login")
    print(f"Debug mode: {debug}")
    print("Server ready. Press CTRL+C to stop.")

    logger.info("Starting Flask app...")
    try:
        app.run(host=host, port=port, debug=debug, use_reloader=False)
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == '__main__':
   run()

