"""
DREDGE x Dolly Server
A lightweight web server for the DREDGE x Dolly integration.
"""
import hashlib
import os
import sys
import logging
from functools import lru_cache
from pathlib import Path
from flask import Flask, jsonify, request, send_file, redirect, url_for
from flask_login import login_required, current_user

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


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)

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
        from .auth import _users
        return _users.get(user_id)

    # -- OAuth / login
    from .auth import init_auth
    init_auth(app)

    # -- Advanced Features
    try:
        from .advanced_features import register_advanced_features
        register_advanced_features(app)
    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not load advanced features: {e}")

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
        html_file = static_dir / 'advanced_dashboard.html'
        
        if not html_file.exists():
            return jsonify({"error": "Dashboard file not found"}), 404
        
        return send_file(html_file, mimetype='text/html')

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


def run_server(host='0.0.0.0', port=3000, debug=False):
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
    run_server()
