"""
OAuth2 authentication for DREDGE — Google and GitHub login.
"""
from __future__ import annotations
import os
import logging
import time
from uuid import UUID
from pathlib import Path

from flask import (
    Blueprint,
    redirect,
    url_for,
    request,
    jsonify,
    render_template_string,
    session,
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

# Blueprint
auth_bp = Blueprint("auth", __name__, url_prefix="/auth")

# Minimal in-memory user store
_users: dict[str, "User"] = {}

# Global OAuth object (will be set by init_auth)
_oauth_instance: OAuth | None = None


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


def get_oauth():
    """Get the OAuth instance. Must call init_auth first."""
    return _oauth_instance


def init_auth(app) -> None:
    """
    Register OAuth providers and the auth blueprint with the Flask application.
    """
    global _oauth_instance

    # Authlib OAuth registry
    _oauth_instance = OAuth(app)

    _redirect_base = os.environ.get("OAUTH_REDIRECT_BASE", "http://localhost:3000").rstrip("/")
    
    print(f"\n[Auth Module] Initializing OAuth providers")
    print(f"  Redirect base: {_redirect_base}")

    # Google OAuth
    google_id = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
    google_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()
    
    print(f"  Google ID set: {bool(google_id)}")
    print(f"  Google Secret set: {bool(google_secret)}")
    
    if google_id and google_secret:
        try:
            _oauth_instance.register(
                name="google",
                client_id=google_id,
                client_secret=google_secret,
                server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
                client_kwargs={"scope": "openid email profile"},
            )
            logger.info("[+] Google OAuth provider registered.")
            print("[+] Google OAuth provider registered.")
        except Exception as e:
            logger.error(f"Failed to register Google OAuth: {e}")
            print(f"[-] Failed to register Google OAuth: {e}")
    else:
        logger.warning("[!] Google OAuth not configured.")
        print("[!] Google OAuth not configured.")

    # GitHub OAuth
    github_id = os.environ.get("GITHUB_CLIENT_ID", "").strip()
    github_secret = os.environ.get("GITHUB_CLIENT_SECRET", "").strip()
    
    print(f"  GitHub ID set: {bool(github_id)}")
    print(f"  GitHub Secret set: {bool(github_secret)}")
    
    if github_id and github_secret:
        try:
            _oauth_instance.register(
                name="github",
                client_id=github_id,
                client_secret=github_secret,
                access_token_url="https://github.com/login/oauth/access_token",
                authorize_url="https://github.com/login/oauth/authorize",
                api_base_url="https://api.github.com/",
                client_kwargs={"scope": "user:email"},
            )
            logger.info("[+] GitHub OAuth provider registered.")
            print("[+] GitHub OAuth provider registered.")
            
            # Verify registration
            print(f"  OAuth object: {_oauth_instance}")
            print(f"  Has github attr: {hasattr(_oauth_instance, 'github')}")
            if hasattr(_oauth_instance, 'github'):
                print(f"  GitHub provider: {_oauth_instance.github}")
            
        except Exception as e:
            logger.error(f"Failed to register GitHub OAuth: {e}")
            print(f"[-] Failed to register GitHub OAuth: {e}")
            import traceback
            traceback.print_exc()
    else:
        logger.warning("[!] GitHub OAuth not configured.")
        print("[!] GitHub OAuth not configured.")

    app.register_blueprint(auth_bp)
    logger.info(f"OAuth redirect base: {_redirect_base}")
    print(f"[Auth Module] Initialization complete\n")


# Login and public architecture use the shared, responsive Studio template.


@auth_bp.route("/login")
def login():
    """Render the login page with OAuth options."""
    messages = {
        'github_not_configured': 'GitHub sign-in is unavailable. Try another enabled method.',
        'google_not_configured': 'Google sign-in is unavailable. Try another enabled method.',
        'github_auth_failed': 'GitHub could not complete sign-in. Try again; your workspace has not been changed.',
        'google_auth_failed': 'Google could not complete sign-in. Try again; your workspace has not been changed.',
        'session_expired': 'Your session expired. Sign in again to return to your saved workspace.'
    }
    code = request.args.get('error') or request.args.get('reason')
    error = messages.get(code, 'Sign-in could not be completed. Try again.') if code else None
    oauth = get_oauth()
    template = (Path(__file__).parent / 'static' / 'studio_public.html').read_text()
    return render_template_string(template, page='login', error=error,
                                  github_enabled=bool(oauth and oauth.create_client('github')),
                                  google_enabled=bool(oauth and oauth.create_client('google')))


def studio_return_path():
    """Restore only a canonical Studio run ID, never an arbitrary redirect URL."""
    value = session.pop('studio_return_run', None)
    try:
        run_id = str(UUID(value))
    except (ValueError, TypeError, AttributeError):
        return '/advanced'
    return '/advanced?run=' + run_id


def remember_studio_session(user):
    # Only the signed session stores this profile. Never accept profile or role from a request body.
    session['studio_profile'] = dict(id=user.id, name=user.name, email=user.email,
                                    provider=user.provider, avatar=user.avatar)
    session['studio_started'] = session['studio_last_activity'] = time.time()
    session.pop('studio_csrf', None)


@auth_bp.route("/github")
def github_login():
    """Redirect to GitHub OAuth."""
    try:
        oauth = get_oauth()
        print(f"\n[GitHub Login] OAuth object: {oauth}")
        print(f"[GitHub Login] Has github: {hasattr(oauth, 'github') if oauth else 'oauth is None'}")
        
        if not oauth:
            print("[GitHub Login] ERROR: oauth is None")
            return redirect(url_for('auth.login', error='github_auth_failed'))
        
        if not hasattr(oauth, "github"):
            print("[GitHub Login] ERROR: oauth has no github attribute")
            return redirect(url_for('auth.login', error='github_not_configured'))
        
        base = os.environ.get('OAUTH_REDIRECT_BASE', '').rstrip('/')
        redirect_uri = base + '/auth/github/callback' if base else url_for("auth.github_callback", _external=True)
        print(f"[GitHub Login] Redirect URI: {redirect_uri}\n")
        
        return oauth.github.authorize_redirect(redirect_uri)
        
    except Exception as e:
        print(f"[GitHub Login] Exception: {e}\n")
        import traceback
        traceback.print_exc()
        return redirect(url_for('auth.login', error='github_auth_failed'))


@auth_bp.route("/github/callback")
def github_callback():
    """Handle GitHub OAuth callback."""
    try:
        print("[GitHub Callback] Processing...")
        
        oauth = get_oauth()
        if not oauth or not hasattr(oauth, "github"):
            print("[GitHub Callback] ERROR: GitHub not configured")
            return redirect(url_for("auth.login") + "?error=github_not_configured")
        
        oauth.github.authorize_access_token()
        resp = oauth.github.get("user", token=oauth.github.token)
        resp.raise_for_status()
        user_info = resp.json()

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
        
        user_id = f"github:{user_info['id']}"
        user = User(
            user_id=user_id,
            name=user_info.get("name") or user_info.get("login", "GitHub User"),
            email=email,
            provider="github",
            avatar=user_info.get("avatar_url", ""),
        )
        _users[user_id] = user
        login_user(user, remember=False)
        remember_studio_session(user)
        
        print(f"[GitHub Callback] User logged in: {user.name}")
        print(f"[GitHub Callback] Redirecting to /advanced\n")
        
        # Redirect to advanced dashboard (this route doesn't require the blueprint prefix)
        return redirect(studio_return_path())
        
    except Exception as e:
        print(f"[GitHub Callback] Exception: {e}\n")
        import traceback
        traceback.print_exc()
        logger.error(f"GitHub OAuth callback error: {e}")
        return redirect(url_for("auth.login") + "?error=github_auth_failed")


@auth_bp.route("/google")
def google_login():
    """Redirect to Google OAuth."""
    oauth = get_oauth()
    if not oauth or not hasattr(oauth, "google"):
        return redirect(url_for('auth.login', error='google_not_configured'))
    
    base = os.environ.get('OAUTH_REDIRECT_BASE', '').rstrip('/')
    redirect_uri = base + '/auth/google/callback' if base else url_for("auth.google_callback", _external=True)
    return oauth.google.authorize_redirect(redirect_uri)


@auth_bp.route("/google/callback")
def google_callback():
    """Handle Google OAuth callback."""
    oauth = get_oauth()
    if not oauth or not hasattr(oauth, "google"):
        return redirect(url_for("auth.login") + "?error=google_not_configured")
    
    try:
        token = oauth.google.authorize_access_token()
        user_info = token.get("userinfo")
        
        if not user_info:
            user_info = oauth.google.userinfo()
        
        user_id = f"google:{user_info['sub']}"
        user = User(
            user_id=user_id,
            name=user_info.get("name", "Google User"),
            email=user_info.get("email", ""),
            provider="google",
            avatar=user_info.get("picture", ""),
        )
        _users[user_id] = user
        login_user(user, remember=False)
        remember_studio_session(user)
        
        # Redirect to advanced dashboard
        return redirect(studio_return_path())
        
    except Exception as e:
        logger.error(f"Google OAuth callback error: {e}")
        return redirect(url_for("auth.login") + "?error=google_auth_failed")


@auth_bp.route("/logout")
@login_required
def logout():
    """Log out the current user."""
    user_id = current_user.id
    logout_user()
    for key in ('studio_profile','studio_started','studio_last_activity','studio_csrf','studio_return_run'):
        session.pop(key, None)
    _users.pop(user_id, None)
    return redirect(url_for("auth.login"))


@auth_bp.route("/me")
@login_required
def me():
    """Return current user profile."""
    return jsonify({
        "id": current_user.id,
        "name": current_user.name,
        "email": current_user.email,
        "provider": current_user.provider,
        "avatar": current_user.avatar,
    })


@auth_bp.route("/status")
def status():
    """Check authentication status."""
    if current_user.is_authenticated:
        return jsonify({
            "authenticated": True,
            "name": current_user.name,
            "email": current_user.email,
            "provider": current_user.provider,
            "avatar": current_user.avatar,
        })
    return jsonify({"authenticated": False})

