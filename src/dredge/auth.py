"""
OAuth2 authentication for DREDGE — Google and GitHub login.

Required environment variables:
    SECRET_KEY             — Flask session signing key (required, no default)
    GOOGLE_CLIENT_ID       — Google OAuth2 client ID
    GOOGLE_CLIENT_SECRET   — Google OAuth2 client secret
    GITHUB_CLIENT_ID       — GitHub OAuth app client ID
    GITHUB_CLIENT_SECRET   — GitHub OAuth app client secret
    OAUTH_REDIRECT_BASE    — Base URL for OAuth callbacks (default: http://localhost:3000)

Callback URLs to register with each provider:
    Google:  {OAUTH_REDIRECT_BASE}/auth/google/callback
    GitHub:  {OAUTH_REDIRECT_BASE}/auth/github/callback
"""
from __future__ import annotations
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


# ── Module-level singleton (populated by init_auth) ──────────────────────────
oauth: OAuth | None = None


def init_auth(app) -> None:
    """
    Register OAuth providers and the auth blueprint with the Flask application.

    Call this from create_app() after setting app.secret_key and initializing LoginManager.
    """
    global oauth

    # ── Authlib OAuth registry ────────────────────────────────────────────────
    oauth = OAuth(app)

    _redirect_base = os.environ.get("OAUTH_REDIRECT_BASE", "http://localhost:3000").rstrip("/")

    # ── Google OAuth ──────────────────────────────────────────────────────────
    google_id = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
    google_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()
    
    if google_id and google_secret:
        try:
            oauth.register(
                name="google",
                client_id=google_id,
                client_secret=google_secret,
                server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
                client_kwargs={"scope": "openid email profile"},
            )
            logger.info("✅ Google OAuth provider registered.")
            print("✅ Google OAuth provider registered.")
        except Exception as e:
            logger.error(f"Failed to register Google OAuth: {e}")
            print(f"❌ Failed to register Google OAuth: {e}")
    else:
        logger.warning("⚠️  Google OAuth not configured (GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET missing).")
        print("⚠️  Google OAuth not configured (GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET missing).")

    # ── GitHub OAuth ──────────────────────────────────────────────────────────
    github_id = os.environ.get("GITHUB_CLIENT_ID", "").strip()
    github_secret = os.environ.get("GITHUB_CLIENT_SECRET", "").strip()
    
    if github_id and github_secret:
        try:
            oauth.register(
                name="github",
                client_id=github_id,
                client_secret=github_secret,
                access_token_url="https://github.com/login/oauth/access_token",
                authorize_url="https://github.com/login/oauth/authorize",
                api_base_url="https://api.github.com/",
                client_kwargs={"scope": "user:email"},
            )
            logger.info("✅ GitHub OAuth provider registered.")
            print("✅ GitHub OAuth provider registered.")
        except Exception as e:
            logger.error(f"Failed to register GitHub OAuth: {e}")
            print(f"❌ Failed to register GitHub OAuth: {e}")
    else:
        logger.warning("⚠️  GitHub OAuth not configured (GITHUB_CLIENT_ID / GITHUB_CLIENT_SECRET missing).")
        print("⚠️  GitHub OAuth not configured (GITHUB_CLIENT_ID / GITHUB_CLIENT_SECRET missing).")

    app.register_blueprint(auth_bp)
    logger.info(f"OAuth redirect base: {_redirect_base}")


# ── Login page ────────────────────────────────────────────────────────────────

LOGIN_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DREDGE Studio - Sign In</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .container {
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            padding: 60px 40px;
            max-width: 400px;
            width: 90%;
        }
        h1 {
            text-align: center;
            margin-bottom: 10px;
            color: #333;
            font-size: 28px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 40px;
            font-size: 14px;
        }
        .login-buttons {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .btn {
            padding: 14px 20px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 500;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            transition: all 0.3s ease;
            text-decoration: none;
        }
        .btn-github {
            background: #333;
            color: white;
        }
        .btn-github:hover {
            background: #222;
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
        }
        .btn-google {
            background: white;
            color: #333;
            border: 2px solid #ddd;
        }
        .btn-google:hover {
            background: #f9f9f9;
            border-color: #999;
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
        }
        .icon {
            font-size: 20px;
        }
        .error {
            background: #fee;
            color: #c00;
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 20px;
            font-size: 14px;
        }
        .status {
            text-align: center;
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            font-size: 14px;
            color: #666;
        }
        .status-item {
            padding: 8px 0;
        }
        .status-icon {
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔐 DREDGE Studio</h1>
        <p class="subtitle">Sign in to continue</p>
        
        {% if error %}
        <div class="error">
            ❌ Authentication failed. Please try again.
        </div>
        {% endif %}
        
        <div class="login-buttons">
            <a href="/auth/github" class="btn btn-github">
                <span class="icon">🐙</span>
                Sign in with GitHub
            </a>
            <a href="/auth/google" class="btn btn-google">
                <span class="icon">🔵</span>
                Sign in with Google
            </a>
        </div>
        
        <div class="status">
            <div class="status-item">
                <span class="status-icon">✅</span> OAuth is configured
            </div>
            <div class="status-item">
                <span class="status-icon">🔒</span> Secure authentication
            </div>
            <div class="status-item" style="font-size: 12px; margin-top: 10px; color: #999;">
                Version: 2.0.0
            </div>
        </div>
    </div>
</body>
</html>
"""


@auth_bp.route("/login")
def login():
    """Render the login page with OAuth options."""
    error = request.args.get('error')
    return render_template_string(LOGIN_HTML, error=error)


# ── Google OAuth routes ───────────────────────────────────────────────────────

@auth_bp.route("/google")
def google_login():
    """Redirect the user to Google's OAuth consent screen."""
    if not oauth or not hasattr(oauth, "google"):
        return jsonify({"error": "Google OAuth is not configured."}), 503
    
    redirect_uri = url_for("auth.google_callback", _external=True)
    logger.info(f"Google OAuth redirect: {redirect_uri}")
    
    return oauth.google.authorize_redirect(redirect_uri)


@auth_bp.route("/google/callback")
def google_callback():
    """Handle the Google OAuth callback and create a session."""
    if not oauth or not hasattr(oauth, "google"):
        logger.error("Google OAuth not configured for callback")
        return redirect(url_for("auth.login") + "?error=google_not_configured")
    
    try:
        token = oauth.google.authorize_access_token()
        user_info = token.get("userinfo")
        
        if not user_info:
            user_info = oauth.google.userinfo()
        
        logger.info(f"Google OAuth success for: {user_info.get('email')}")
    except Exception as exc:
        logger.error(f"Google OAuth callback error: {exc}")
        return redirect(url_for("auth.login") + "?error=google_auth_failed")

    user_id = f"google:{user_info['sub']}"
    user = User(
        user_id=user_id,
        name=user_info.get("name", "Google User"),
        email=user_info.get("email", ""),
        provider="google",
        avatar=user_info.get("picture", ""),
    )
    _users[user_id] = user
    login_user(user, remember=True)
    
    logger.info(f"User logged in via Google: {user.email}")
    print(f"✅ User logged in via Google: {user.email}")
    
    return redirect(url_for("index"))


# ── GitHub OAuth routes ───────────────────────────────────────────────────────

@auth_bp.route("/github")
def github_login():
    """Redirect the user to GitHub's OAuth consent screen."""
    if not oauth or not hasattr(oauth, "github"):
        return jsonify({"error": "GitHub OAuth is not configured."}), 503
    
    redirect_uri = url_for("auth.github_callback", _external=True)
    logger.info(f"GitHub OAuth redirect: {redirect_uri}")
    
    return oauth.github.authorize_redirect(redirect_uri)


@auth_bp.route("/github/callback")
def github_callback():
    """Handle the GitHub OAuth callback and create a session."""
    if not oauth or not hasattr(oauth, "github"):
        logger.error("GitHub OAuth not configured for callback")
        return redirect(url_for("auth.login") + "?error=github_not_configured")
    
    try:
        oauth.github.authorize_access_token()
        resp = oauth.github.get("user", token=oauth.github.token)
        resp.raise_for_status()
        user_info = resp.json()

        # Fetch primary verified email if not in the user profile
        email = user_info.get("email") or ""
        if not email:
            try:
                emails_resp = oauth.github.get("user/emails", token=oauth.github.token)
                if emails_resp.status_code == 200:
                    for entry in emails_resp.json():
                        if entry.get("primary") and entry.get("verified"):
                            email = entry["email"]
                            break
            except Exception as e:
                logger.warning(f"Could not fetch GitHub emails: {e}")
        
        logger.info(f"GitHub OAuth success for: {user_info.get('login')}")
    except Exception as exc:
        logger.error(f"GitHub OAuth callback error: {exc}")
        return redirect(url_for("auth.login") + "?error=github_auth_failed")

    user_id = f"github:{user_info['id']}"
    user = User(
        user_id=user_id,
        name=user_info.get("name") or user_info.get("login", "GitHub User"),
        email=email,
        provider="github",
        avatar=user_info.get("avatar_url", ""),
    )
    _users[user_id] = user
    login_user(user, remember=True)
    
    logger.info(f"User logged in via GitHub: {user.email or user.name}")
    print(f"✅ User logged in via GitHub: {user.email or user.name}")
    
    return redirect(url_for("index"))


# ── Logout ────────────────────────────────────────────────────────────────────

@auth_bp.route("/logout")
@login_required
def logout():
    """Log out the current user."""
    user_id = current_user.id
    logout_user()
    _users.pop(user_id, None)
    logger.info(f"User logged out: {user_id}")
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
    """Check authentication status (public endpoint)."""
    if current_user.is_authenticated:
        return jsonify({
            "authenticated": True,
            "name":     current_user.name,
            "email":    current_user.email,
            "provider": current_user.provider,
            "avatar":   current_user.avatar,
        })
    return jsonify({"authenticated": False})
