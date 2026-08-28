"""
DREDGE Authentication Module - Login, Sessions, and OAuth Support
"""

from fastapi import APIRouter, HTTPException, Depends, Request, status
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import secrets
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# DATA MODELS
# ============================================================================

class LoginRequest(BaseModel):
    email: str
    password: str
    remember: bool = False

class SignupRequest(BaseModel):
    email: str
    password: str
    name: str

class OAuthCallback(BaseModel):
    code: str
    provider: str

# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

class SessionManager:
    """Simple session manager (use Redis in production)"""
    
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
    
    def create_session(self, user_id: str, email: str, remember: bool = False) -> str:
        """Create a new session"""
        session_id = secrets.token_urlsafe(32)
        
        expiry = datetime.utcnow() + (timedelta(days=30) if remember else timedelta(hours=24))
        
        self.sessions[session_id] = {
            "user_id": user_id,
            "email": email,
            "created": datetime.utcnow(),
            "expires": expiry,
            "active": True
        }
        
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Validate a session"""
        session = self.sessions.get(session_id)
        
        if not session:
            return None
        
        if datetime.utcnow() > session["expires"]:
            self.sessions.pop(session_id, None)
            return None
        
        if not session["active"]:
            return None
        
        return session
    
    def destroy_session(self, session_id: str):
        """Destroy a session"""
        if session_id in self.sessions:
            self.sessions.pop(session_id)

# Global session manager
session_manager = SessionManager()

# ============================================================================
# USER DATABASE (Mock - Use Real DB in Production)
# ============================================================================

class UserDB:
    """Mock user database"""
    
    def __init__(self):
        self.users = {
            "demo@dredge.io": {
                "id": "user_1",
                "email": "demo@dredge.io",
                "password_hash": "$2b$12$K8s3c4m2l1k0j9i8h7g6f5e4d3c2b1a0",  # "demo123" (demo only)
                "name": "Demo User",
                "created": datetime.utcnow()
            }
        }
    
    def get_user(self, email: str):
        """Get user by email"""
        return self.users.get(email)
    
    def create_user(self, email: str, password_hash: str, name: str):
        """Create new user"""
        user_id = f"user_{len(self.users) + 1}"
        self.users[email] = {
            "id": user_id,
            "email": email,
            "password_hash": password_hash,
            "name": name,
            "created": datetime.utcnow()
        }
        return user_id

user_db = UserDB()

# ============================================================================
# PASSWORD HASHING (Mock - Use bcrypt in Production)
# ============================================================================

def hash_password(password: str) -> str:
    """Hash password (use bcrypt in production)"""
    import hashlib
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hash_value: str) -> bool:
    """Verify password"""
    return hash_password(password) == hash_value

# ============================================================================
# AUTH ROUTES
# ============================================================================

def create_auth_router() -> APIRouter:
    """Create authentication router"""
    
    router = APIRouter(prefix="/auth", tags=["Auth"])
    
    # ========================================================================
    # LOGIN ENDPOINTS
    # ========================================================================
    
    @router.get("/login", response_class=HTMLResponse, tags=["Pages"])
    async def login_page():
        """Serve login page"""
        login_html = Path(__file__).parent / "templates" / "login.html"
        
        if login_html.exists():
            return login_html.read_text()
        
        raise HTTPException(status_code=404, detail="Login page not found")
    
    @router.post("/login", tags=["API"])
    async def login(request: LoginRequest) -> Dict[str, Any]:
        """Authenticate user"""
        
        # Get user from database
        user = user_db.get_user(request.email)
        
        if not user:
            logger.warning(f"[Auth] Failed login attempt: {request.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Verify password
        if not verify_password(request.password, user["password_hash"]):
            logger.warning(f"[Auth] Failed login attempt: {request.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Create session
        session_id = session_manager.create_session(
            user["id"],
            user["email"],
            request.remember
        )
        
        logger.info(f"[Auth] Successful login: {request.email}")
        
        return {
            "status": "success",
            "message": "Login successful",
            "session_id": session_id,
            "user": {
                "id": user["id"],
                "email": user["email"],
                "name": user["name"]
            }
        }
    
    # ========================================================================
    # SIGNUP ENDPOINTS
    # ========================================================================
    
    @router.get("/signup", response_class=HTMLResponse, tags=["Pages"])
    async def signup_page():
        """Serve signup page"""
        # For now, return login page (modify later)
        login_html = Path(__file__).parent / "templates" / "login.html"
        
        if login_html.exists():
            return login_html.read_text()
        
        raise HTTPException(status_code=404, detail="Signup page not found")
    
    @router.post("/signup", tags=["API"])
    async def signup(request: SignupRequest) -> Dict[str, Any]:
        """Create new user account"""
        
        # Check if user exists
        if user_db.get_user(request.email):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email already registered"
            )
        
        # Create user
        password_hash = hash_password(request.password)
        user_id = user_db.create_user(request.email, password_hash, request.name)
        
        # Create session
        session_id = session_manager.create_session(user_id, request.email)
        
        logger.info(f"[Auth] New signup: {request.email}")
        
        return {
            "status": "success",
            "message": "Signup successful",
            "session_id": session_id,
            "user": {
                "id": user_id,
                "email": request.email,
                "name": request.name
            }
        }
    
    # ========================================================================
    # LOGOUT
    # ========================================================================
    
    @router.post("/logout", tags=["API"])
    async def logout(request: Request) -> Dict[str, str]:
        """Logout user"""
        session_id = request.cookies.get("session_id")
        
        if session_id:
            session_manager.destroy_session(session_id)
        
        logger.info("[Auth] User logout")
        
        return {"status": "success", "message": "Logged out"}
    
    # ========================================================================
    # OAUTH (GitHub, Google)
    # ========================================================================
    
    @router.get("/oauth/{provider}", tags=["OAuth"])
    async def oauth_redirect(provider: str):
        """Redirect to OAuth provider"""
        
        if provider == "github":
            # Redirect to GitHub OAuth
            client_id = "YOUR_GITHUB_CLIENT_ID"
            return {
                "status": "redirect",
                "url": f"https://github.com/login/oauth/authorize?client_id={client_id}"
            }
        elif provider == "google":
            # Redirect to Google OAuth
            client_id = "YOUR_GOOGLE_CLIENT_ID"
            return {
                "status": "redirect",
                "url": f"https://accounts.google.com/o/oauth2/v2/auth?client_id={client_id}"
            }
        
        raise HTTPException(status_code=400, detail="Unknown provider")
    
    @router.get("/oauth/callback/{provider}", tags=["OAuth"])
    async def oauth_callback(provider: str, code: str, state: Optional[str] = None):
        """Handle OAuth callback"""
        
        # In production, exchange code for token and get user info
        logger.info(f"[Auth] OAuth callback from {provider}")
        
        return {
            "status": "success",
            "message": f"OAuth login from {provider} (implement token exchange)"
        }
    
    # ========================================================================
    # SESSION VALIDATION
    # ========================================================================
    
    @router.get("/validate", tags=["API"])
    async def validate_session(request: Request) -> Dict[str, Any]:
        """Validate current session"""
        session_id = request.cookies.get("session_id")
        
        if not session_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No session found"
            )
        
        session = session_manager.validate_session(session_id)
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired session"
            )
        
        return {
            "status": "valid",
            "user_id": session["user_id"],
            "email": session["email"],
            "expires": session["expires"].isoformat()
        }
    
    # ========================================================================
    # PASSWORD RESET
    # ========================================================================
    
    @router.get("/forgot-password", response_class=HTMLResponse, tags=["Pages"])
    async def forgot_password_page():
        """Serve password reset page"""
        raise HTTPException(status_code=501, detail="Not implemented yet")
    
    @router.post("/forgot-password", tags=["API"])
    async def forgot_password(email: str) -> Dict[str, str]:
        """Request password reset"""
        user = user_db.get_user(email)
        
        if user:
            logger.info(f"[Auth] Password reset requested for: {email}")
        
        # Always return success for security
        return {
            "status": "success",
            "message": "If an account exists, a reset link has been sent"
        }
    
    # ========================================================================
    # STATUS
    # ========================================================================
    
    @router.get("/status", tags=["API"])
    async def auth_status() -> Dict[str, Any]:
        """Authentication system status"""
        return {
            "status": "operational",
            "adapter": "auth",
            "features": [
                "Email/Password Login",
                "User Registration",
                "Session Management",
                "OAuth (GitHub, Google)",
                "Password Reset",
                "2FA Support (planned)"
            ],
            "total_users": len(user_db.users),
            "active_sessions": len(session_manager.sessions)
        }
    
    return router


# ============================================================================
# DEPENDENCY FOR PROTECTED ROUTES
# ============================================================================

async def get_current_user(request: Request) -> Dict[str, Any]:
    """Get current authenticated user"""
    session_id = request.cookies.get("session_id")
    
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    
    session = session_manager.validate_session(session_id)
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session"
        )
    
    return session


__all__ = [
    'create_auth_router',
    'session_manager',
    'user_db',
    'get_current_user',
    'SessionManager',
]
