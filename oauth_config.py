#!/usr/bin/env python3
"""
DREDGE OAuth Configuration Helper

Load and validate OAuth credentials from environment or .env file.
Generates SECRET_KEY if not present.
"""

import os
import sys
import secrets
from pathlib import Path
from typing import Optional, Dict, Tuple

def load_env_file(path: str = ".env") -> None:
    """Load environment variables from .env file if it exists."""
    env_path = Path(path)
    if env_path.exists():
        print(f"📄 Loading from {path}")
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        os.environ[key.strip()] = value.strip().strip('"\'')
        print(f"✅ Loaded environment from {path}")
    else:
        print(f"⚠️  {path} not found (creating .env.example)")


def generate_secret_key() -> str:
    """Generate a secure SECRET_KEY."""
    return secrets.token_hex(32)


def validate_config() -> Tuple[bool, Dict[str, any]]:
    """
    Validate OAuth configuration.
    Returns: (is_valid, config_dict)
    """
    config = {}
    errors = []
    warnings = []

    # ── Flask Configuration ──────────────────────────────────────────────────
    secret_key = os.environ.get("SECRET_KEY", "").strip()
    if not secret_key:
        secret_key = generate_secret_key()
        print("🔑 Generated SECRET_KEY (hidden)")
        print("   ⚠️  Add to .env to persist between restarts")
    elif len(secret_key) < 32:
        warnings.append("SECRET_KEY is short (recommended: 32+ characters)")
    
    config["SECRET_KEY"] = secret_key

    # ── GitHub Configuration ─────────────────────────────────────────────────
    github_id = os.environ.get("GITHUB_CLIENT_ID", "").strip()
    github_secret = os.environ.get("GITHUB_CLIENT_SECRET", "").strip()

    if github_id and github_secret:
        config["github"] = {
            "client_id": github_id,
            "client_secret": github_secret[:10] + "...",  # Hide secret
            "status": "✅ CONFIGURED"
        }
    else:
        config["github"] = {
            "status": "⚠️  NOT CONFIGURED",
            "help": "See OAUTH_CONFIGURATION.md for setup"
        }
        warnings.append("GitHub OAuth not configured (optional)")

    # ── Google Configuration ─────────────────────────────────────────────────
    google_id = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
    google_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()

    if google_id and google_secret:
        config["google"] = {
            "client_id": google_id[:20] + "...",  # Show partial ID
            "client_secret": google_secret[:10] + "...",  # Hide secret
            "status": "✅ CONFIGURED"
        }
    else:
        config["google"] = {
            "status": "⚠️  NOT CONFIGURED",
            "help": "See OAUTH_CONFIGURATION.md for setup"
        }
        warnings.append("Google OAuth not configured (optional)")

    # ── OAuth Redirect Base ──────────────────────────────────────────────────
    redirect_base = os.environ.get("OAUTH_REDIRECT_BASE", "http://localhost:3000").strip()
    config["redirect_base"] = redirect_base

    # ── Flask Environment ────────────────────────────────────────────────────
    flask_env = os.environ.get("FLASK_ENV", "development").strip()
    config["flask_env"] = flask_env

    # ── Summary ──────────────────────────────────────────────────────────────
    is_valid = len(errors) == 0
    config["valid"] = is_valid
    config["errors"] = errors
    config["warnings"] = warnings

    return is_valid, config


def print_config_report(is_valid: bool, config: Dict) -> None:
    """Print configuration report to stdout."""
    print("\n" + "=" * 80)
    print("DREDGE OAUTH CONFIGURATION REPORT")
    print("=" * 80 + "\n")

    # ── Flask Settings ───────────────────────────────────────────────────────
    print("🔧 Flask Configuration:")
    print(f"   Environment:     {config.get('flask_env', 'unknown')}")
    print(f"   SECRET_KEY:      {'[SET]' if config.get('SECRET_KEY') else '[NOT SET]'}")
    print(f"   Redirect Base:   {config.get('redirect_base', 'N/A')}")
    print()

    # ── GitHub ───────────────────────────────────────────────────────────────
    print("🐙 GitHub OAuth:")
    github = config.get("github", {})
    if github.get("status") == "✅ CONFIGURED":
        print(f"   Status:  {github['status']}")
        print(f"   ID:      {github['client_id']}")
        print(f"   Secret:  {github['client_secret']}")
    else:
        print(f"   Status:  {github.get('status', 'N/A')}")
        if "help" in github:
            print(f"   Help:    {github['help']}")
    print()

    # ── Google ───────────────────────────────────────────────────────────────
    print("🔵 Google OAuth:")
    google = config.get("google", {})
    if google.get("status") == "✅ CONFIGURED":
        print(f"   Status:  {google['status']}")
        print(f"   ID:      {google['client_id']}")
        print(f"   Secret:  {google['client_secret']}")
    else:
        print(f"   Status:  {google.get('status', 'N/A')}")
        if "help" in google:
            print(f"   Help:    {google['help']}")
    print()

    # ── Warnings & Errors ────────────────────────────────────────────────────
    if config.get("warnings"):
        print("⚠️  Warnings:")
        for warning in config["warnings"]:
            print(f"   - {warning}")
        print()

    if config.get("errors"):
        print("❌ Errors:")
        for error in config["errors"]:
            print(f"   - {error}")
        print()

    # ── Status ───────────────────────────────────────────────────────────────
    print("=" * 80)
    if is_valid:
        print("✅ Configuration Valid - Ready to Start Server")
    else:
        print("❌ Configuration Invalid - Fix Errors Above")
    print("=" * 80 + "\n")


def main():
    """Main CLI entry point."""
    # Check for help flag
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help", "help"]:
        print(__doc__)
        print("\nUsage:")
        print("  python oauth_config.py               # Check current config")
        print("  python oauth_config.py generate      # Generate SECRET_KEY")
        print("  python oauth_config.py load          # Load from .env")
        print("  python oauth_config.py validate      # Validate config")
        return

    # Load .env if it exists
    load_env_file()
    print()

    # Check for subcommand
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "generate":
            print("🔐 Generating SECRET_KEY:")
            key = generate_secret_key()
            print("   [HIDDEN]")
            print("\n   Add to .env:")
            print("   SECRET_KEY=<generated-secret>")
            return
        
        elif command == "validate":
            is_valid, config = validate_config()
            print_config_report(is_valid, config)
            sys.exit(0 if is_valid else 1)
    
    # Default: show config report
    is_valid, config = validate_config()
    print_config_report(is_valid, config)

    # Provide next steps
    print("📋 Next Steps:")
    if not all([
        os.environ.get("GITHUB_CLIENT_ID"),
        os.environ.get("GOOGLE_CLIENT_ID"),
    ]):
        print("   1. Generate OAuth credentials:")
        print("      - GitHub: https://github.com/settings/developers")
        print("      - Google: https://console.developers.google.com/")
        print("   2. Add to .env file")
        print("   3. Run: python oauth_config.py validate")
    
    print("   4. Start DREDGE:")
    print("      python -m dredge.server")
    print()


if __name__ == "__main__":
    main()
