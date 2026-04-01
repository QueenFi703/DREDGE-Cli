"""
OAuth2 authentication for DREDGE — Google and GitHub login.

Required environment variables:
    SECRET_KEY             — Flask session signing key (required, no default)
    GOOGLE_CLIENT_ID       — Google OAuth2 client ID
    GOOGLE_CLIENT_SECRET   — Google OAuth2 client secret
    GITHUB_CLIENT_ID       — GitHub OAuth app client ID
    GITHUB_CLIENT_SECRET   — GitHub OAuth app client secret
    OAUTH_REDIRECT_BASE    — Base URL for OAuth callbacks, e.g. https://myapp.example.com
                             Defaults to http://localhost:3000

Callback URLs to register with each provider:
    Google:  {OAUTH_REDIRECT_BASE}/auth/google/callback
    GitHub:  {OAUTH_REDIRECT_BASE}/auth/github/callback
"""
import os
import logging
from functools import wraps

from flask import (
    Blueprint,
    redirect,
    url_for,
    session,
    request,
    jsonify,
    render_template_string,
)
from authlib.integrations.flask_client import OAuth
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    login_required,
    current_user,
)

logger = logging.getLogger(__name__)

# ── Blueprint ─────────────────────────────────────────────────────────────────
auth_bp = Blueprint("auth", __name__, url_prefix="/auth")

# ── Minimal in-memory user store ──────────────────────────────────────────────
# In production, replace with a database-backed user store.
_users: dict[str, "User"] = {}


class User(UserMixin):
    """Lightweight user object stored in memory for the session lifetime."""

    def __init__(self, user_id: str, name: str, email: str, provider: str, avatar: str = ""):
        self.id = user_id
        self.name = name
        self.email = email
        self.provider = provider
        self.avatar = avatar

    def get_id(self) -> str:
        return self.id


# ── Module-level singletons (populated by init_auth) ─────────────────────────
oauth: OAuth | None = None
login_manager: LoginManager | None = None


def init_auth(app) -> None:
    """
    Register OAuth providers, the LoginManager, and the auth blueprint
    with the Flask application.

    Call this from create_app() after setting app.secret_key.
    """
    global oauth, login_manager

    # ── Flask-Login ───────────────────────────────────────────────────────────
    login_manager = LoginManager()
    login_manager.login_view = "auth.login"
    login_manager.login_message = "Please sign in to access this page."
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id: str):
        return _users.get(user_id)

    # ── Authlib OAuth registry ────────────────────────────────────────────────
    oauth = OAuth(app)

    _redirect_base = os.environ.get("OAUTH_REDIRECT_BASE", "http://localhost:3000").rstrip("/")

    # Google
    google_id = os.environ.get("GOOGLE_CLIENT_ID", "")
    google_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "")
    if google_id and google_secret:
        oauth.register(
            name="google",
            client_id=google_id,
            client_secret=google_secret,
            server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
            client_kwargs={"scope": "openid email profile"},
            redirect_uri=f"{_redirect_base}/auth/google/callback",
        )
        logger.info("Google OAuth provider registered.")
    else:
        logger.warning("Google OAuth not configured (GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET missing).")

    # GitHub
    github_id = os.environ.get("GITHUB_CLIENT_ID", "")
    github_secret = os.environ.get("GITHUB_CLIENT_SECRET", "")
    if github_id and github_secret:
        oauth.register(
            name="github",
            client_id=github_id,
            client_secret=github_secret,
            access_token_url="https://github.com/login/oauth/access_token",
            authorize_url="https://github.com/login/oauth/authorize",
            api_base_url="https://api.github.com/",
            client_kwargs={"scope": "read:user user:email"},
            redirect_uri=f"{_redirect_base}/auth/github/callback",
        )
        logger.info("GitHub OAuth provider registered.")
    else:
        logger.warning("GitHub OAuth not configured (GITHUB_CLIENT_ID / GITHUB_CLIENT_SECRET missing).")

    app.register_blueprint(auth_bp)


# ── Login page ────────────────────────────────────────────────────────────────

@auth_bp.route("/login")
def login():
    """Render the login page (served from static/login.html)."""
    from flask import send_from_directory
    from pathlib import Path
    static_dir = Path(__file__).parent / "static"
    return send_from_directory(str(static_dir), "login.html")


# ── Google OAuth routes ───────────────────────────────────────────────────────

@auth_bp.route("/google")
def google_login():
    """Redirect the user to Google's OAuth consent screen."""
    if not (oauth and hasattr(oauth, "google")):
        return jsonify({"error": "Google OAuth is not configured."}), 503
    redirect_uri = url_for("auth.google_callback", _external=True)
    return oauth.google.authorize_redirect(redirect_uri)


@auth_bp.route("/google/callback")
def google_callback():
    """Handle the Google OAuth callback and create a session."""
    if not (oauth and hasattr(oauth, "google")):
        return jsonify({"error": "Google OAuth is not configured."}), 503
    try:
        token = oauth.google.authorize_access_token()
        user_info = token.get("userinfo") or oauth.google.userinfo()
    except Exception as exc:
        logger.error("Google OAuth callback error: %s", exc)
        return redirect(url_for("auth.login") + "?error=google_auth_failed")

    user_id = f"google:{user_info['sub']}"
    user = User(
        user_id=user_id,
        name=user_info.get("name", ""),
        email=user_info.get("email", ""),
        provider="google",
        avatar=user_info.get("picture", ""),
    )
    _users[user_id] = user
    login_user(user, remember=True)
    logger.info("User logged in via Google: %s", user.email)
    return redirect(url_for("index"))


# ── GitHub OAuth routes ───────────────────────────────────────────────────────

@auth_bp.route("/github")
def github_login():
    """Redirect the user to GitHub's OAuth consent screen."""
    if not (oauth and hasattr(oauth, "github")):
        return jsonify({"error": "GitHub OAuth is not configured."}), 503
    redirect_uri = url_for("auth.github_callback", _external=True)
    return oauth.github.authorize_redirect(redirect_uri)


@auth_bp.route("/github/callback")
def github_callback():
    """Handle the GitHub OAuth callback and create a session."""
    if not (oauth and hasattr(oauth, "github")):
        return jsonify({"error": "GitHub OAuth is not configured."}), 503
    try:
        oauth.github.authorize_access_token()
        resp = oauth.github.get("user")
        resp.raise_for_status()
        user_info = resp.json()

        # Fetch primary verified email if not in the user profile
        email = user_info.get("email") or ""
        if not email:
            emails_resp = oauth.github.get("user/emails")
            if emails_resp.status_code == 200:
                for entry in emails_resp.json():
                    if entry.get("primary") and entry.get("verified"):
                        email = entry["email"]
                        break
    except Exception as exc:
        logger.error("GitHub OAuth callback error: %s", exc)
        return redirect(url_for("auth.login") + "?error=github_auth_failed")

    user_id = f"github:{user_info['id']}"
    user = User(
        user_id=user_id,
        name=user_info.get("name") or user_info.get("login", ""),
        email=email,
        provider="github",
        avatar=user_info.get("avatar_url", ""),
    )
    _users[user_id] = user
    login_user(user, remember=True)
    logger.info("User logged in via GitHub: %s", user.email or user.name)
    return redirect(url_for("index"))


# ── Logout ────────────────────────────────────────────────────────────────────

@auth_bp.route("/logout")
@login_required
def logout():
    user_id = current_user.id
    logout_user()
    _users.pop(user_id, None)
    return redirect(url_for("auth.login"))


# ── Current-user API endpoint ─────────────────────────────────────────────────

@auth_bp.route("/me")
@login_required
def me():
    """Return the current authenticated user's profile as JSON."""
    return jsonify({
        "id":       current_user.id,
        "name":     current_user.name,
        "email":    current_user.email,
        "provider": current_user.provider,
        "avatar":   current_user.avatar,
    })


# ── Auth status (public — useful for the SPA to check login state) ────────────

@auth_bp.route("/status")
def status():
    if current_user.is_authenticated:
        return jsonify({
            "authenticated": True,
            "name":     current_user.name,
            "email":    current_user.email,
            "provider": current_user.provider,
            "avatar":   current_user.avatar,
        })
    return jsonify({"authenticated": False})
