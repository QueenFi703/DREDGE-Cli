"""
API Key Management System for Orion Gateway
Secure generation, hashing, validation, and tracking
"""

import os
import secrets
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from enum import Enum
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class KeyStatus(str, Enum):
    """API key status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    REVOKED = "revoked"
    EXPIRED = "expired"


class KeyTier(str, Enum):
    """API key tier/subscription level"""
    FREE = "free"          # 100 requests/month
    STARTER = "starter"    # 1,000 requests/month
    PRO = "pro"            # 10,000 requests/month
    ENTERPRISE = "enterprise"  # Unlimited


class KeyMode(str, Enum):
    """API key capability modes"""
    READ_ONLY = "read_only"      # /health, /usage
    INVOKE_ONLY = "invoke_only"  # /invoke only
    FULL_ACCESS = "full_access"  # All endpoints


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class APIKeyMetadata:
    """API key metadata"""
    key_id: str                  # Unique key identifier (e.g., "key_123abc")
    name: str                    # Human-readable name
    key_hash: str                # SHA-256 hash of the key (stored in DB)
    created_at: datetime
    expires_at: Optional[datetime]
    last_used_at: Optional[datetime]
    status: KeyStatus
    tier: KeyTier
    mode: KeyMode
    requests_this_month: int
    requests_limit: int         # Monthly limit based on tier
    usage_percent: float
    created_by: str             # User/admin who created it
    environment: str            # "development", "staging", "production"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict"""
        return {
            "key_id": self.key_id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "status": self.status.value,
            "tier": self.tier.value,
            "mode": self.mode.value,
            "requests_this_month": self.requests_this_month,
            "requests_limit": self.requests_limit,
            "usage_percent": self.usage_percent,
            "created_by": self.created_by,
            "environment": self.environment
        }


@dataclass
class APIKeyUsageRecord:
    """API key usage tracking"""
    key_id: str
    endpoint: str
    method: str
    timestamp: datetime
    response_code: int
    tokens_used: int
    duration_ms: float
    ip_address: str
    user_agent: str


# ============================================================================
# API KEY GENERATION & HASHING
# ============================================================================

class APIKeyGenerator:
    """Generate and manage API keys"""
    
    PREFIX_LENGTH = 4
    KEY_LENGTH = 32  # 256-bit key when base64 encoded
    
    @staticmethod
    def generate_key(tier: KeyTier = KeyTier.FREE) -> tuple[str, str]:
        """
        Generate a new API key
        
        Returns:
            (plaintext_key, key_id)
            
        Format: orion_key_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx (production prefix)
                dev_key_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx (development prefix)
        """
        # Generate random bytes
        random_bytes = secrets.token_bytes(APIKeyGenerator.KEY_LENGTH)
        key_part = secrets.token_urlsafe(APIKeyGenerator.KEY_LENGTH)
        
        # Create prefix based on tier
        prefix = "orion"
        
        # Format: prefix_key_<random>
        plaintext_key = f"{prefix}_key_{key_part}"
        
        # Generate key ID (first 8 chars after prefix for display)
        key_id = f"key_{key_part[:12]}"
        
        return plaintext_key, key_id
    
    @staticmethod
    def hash_key(plaintext_key: str, salt: Optional[str] = None) -> str:
        """
        Hash API key using SHA-256 + salt
        
        Args:
            plaintext_key: The plaintext key to hash
            salt: Optional salt (generated if not provided)
            
        Returns:
            Hash in format: $2$<salt>$<hash> (like bcrypt format)
        """
        if salt is None:
            salt = secrets.token_hex(16)
        
        # Create hash with salt
        hash_obj = hashlib.sha256(f"{salt}{plaintext_key}".encode())
        key_hash = hash_obj.hexdigest()
        
        # Return in format similar to bcrypt: $algorithm$salt$hash
        return f"$2${salt}${key_hash}"
    
    @staticmethod
    def verify_key(plaintext_key: str, stored_hash: str) -> bool:
        """
        Verify a plaintext key against stored hash
        
        Args:
            plaintext_key: The key to verify
            stored_hash: The stored hash (format: $2$salt$hash)
            
        Returns:
            True if key matches, False otherwise
        """
        try:
            # Extract salt from stored hash
            parts = stored_hash.split("$")
            if len(parts) != 4 or parts[1] != "2":
                return False
            
            salt = parts[2]
            
            # Compute hash with extracted salt
            computed_hash = APIKeyGenerator.hash_key(plaintext_key, salt)
            
            # Constant-time comparison to prevent timing attacks
            return hmac.compare_digest(computed_hash, stored_hash)
        except Exception as e:
            logger.error(f"Key verification error: {e}")
            return False


# ============================================================================
# API KEY STORAGE & MANAGEMENT
# ============================================================================

class APIKeyStore:
    """Store and manage API keys (in-memory or file-based)"""
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize API key store
        
        Args:
            storage_path: Path to JSON file for persistence
                         If None, uses in-memory only
        """
        self.storage_path = storage_path
        self.keys: Dict[str, APIKeyMetadata] = {}  # key_hash -> metadata
        self.usage: Dict[str, List[APIKeyUsageRecord]] = {}  # key_id -> records
        
        # Load from file if provided
        if storage_path and os.path.exists(storage_path):
            self._load_from_file()
    
    def create_key(
        self,
        name: str,
        tier: KeyTier = KeyTier.FREE,
        mode: KeyMode = KeyMode.INVOKE_ONLY,
        created_by: str = "system",
        environment: str = "development",
        expires_in_days: Optional[int] = 90
    ) -> tuple[str, APIKeyMetadata]:
        """
        Create a new API key
        
        Returns:
            (plaintext_key, metadata)
            Note: plaintext_key is only returned once!
        """
        # Generate key
        plaintext_key, key_id = APIKeyGenerator.generate_key(tier)
        
        # Hash key for storage
        key_hash = APIKeyGenerator.hash_key(plaintext_key)
        
        # Get request limit for tier
        tier_limits = {
            KeyTier.FREE: 100,
            KeyTier.STARTER: 1_000,
            KeyTier.PRO: 10_000,
            KeyTier.ENTERPRISE: 1_000_000
        }
        
        # Create metadata
        metadata = APIKeyMetadata(
            key_id=key_id,
            name=name,
            key_hash=key_hash,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(days=expires_in_days) if expires_in_days else None,
            last_used_at=None,
            status=KeyStatus.ACTIVE,
            tier=tier,
            mode=mode,
            requests_this_month=0,
            requests_limit=tier_limits[tier],
            usage_percent=0.0,
            created_by=created_by,
            environment=environment
        )
        
        # Store
        self.keys[key_hash] = metadata
        self.usage[key_id] = []
        
        # Persist
        self._save_to_file()
        
        logger.info(f"Created API key: {key_id} (tier: {tier}, mode: {mode})")
        
        return plaintext_key, metadata
    
    def validate_key(self, plaintext_key: str) -> Optional[APIKeyMetadata]:
        """
        Validate a plaintext key
        
        Returns:
            APIKeyMetadata if valid, None otherwise
        """
        # Try to find key by validating against all stored hashes
        for key_hash, metadata in self.keys.items():
            if APIKeyGenerator.verify_key(plaintext_key, key_hash):
                # Check if key is active and not expired
                if metadata.status != KeyStatus.ACTIVE:
                    logger.warning(f"Key {metadata.key_id} is {metadata.status}")
                    return None
                
                if metadata.expires_at and datetime.utcnow() > metadata.expires_at:
                    logger.warning(f"Key {metadata.key_id} is expired")
                    metadata.status = KeyStatus.EXPIRED
                    self._save_to_file()
                    return None
                
                # Update last used
                metadata.last_used_at = datetime.utcnow()
                self._save_to_file()
                
                return metadata
        
        logger.warning("Invalid API key provided")
        return None
    
    def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key"""
        for metadata in self.keys.values():
            if metadata.key_id == key_id:
                metadata.status = KeyStatus.REVOKED
                self._save_to_file()
                logger.info(f"Revoked API key: {key_id}")
                return True
        return False
    
    def record_usage(
        self,
        key_id: str,
        endpoint: str,
        method: str,
        response_code: int,
        tokens_used: int = 0,
        duration_ms: float = 0.0,
        ip_address: str = "",
        user_agent: str = ""
    ) -> None:
        """Record API key usage"""
        record = APIKeyUsageRecord(
            key_id=key_id,
            endpoint=endpoint,
            method=method,
            timestamp=datetime.utcnow(),
            response_code=response_code,
            tokens_used=tokens_used,
            duration_ms=duration_ms,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        if key_id not in self.usage:
            self.usage[key_id] = []
        
        self.usage[key_id].append(record)
        
        # Update metadata
        for metadata in self.keys.values():
            if metadata.key_id == key_id:
                metadata.requests_this_month += 1
                metadata.usage_percent = (
                    (metadata.requests_this_month / metadata.requests_limit) * 100
                    if metadata.requests_limit > 0 else 0
                )
                break
        
        self._save_to_file()
    
    def get_usage_stats(self, key_id: str) -> Optional[Dict[str, Any]]:
        """Get usage statistics for a key"""
        for metadata in self.keys.values():
            if metadata.key_id == key_id:
                records = self.usage.get(key_id, [])
                
                # Calculate stats
                if not records:
                    return asdict(metadata)
                
                stats = asdict(metadata)
                stats["total_requests_all_time"] = len(records)
                stats["last_request"] = max(r.timestamp for r in records).isoformat()
                
                # Requests by endpoint
                by_endpoint = {}
                for record in records:
                    ep = record.endpoint
                    by_endpoint[ep] = by_endpoint.get(ep, 0) + 1
                stats["requests_by_endpoint"] = by_endpoint
                
                # Average response time
                if records:
                    avg_duration = sum(r.duration_ms for r in records) / len(records)
                    stats["avg_response_time_ms"] = round(avg_duration, 2)
                
                return stats
        
        return None
    
    def list_keys(self, environment: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all API keys (without plaintext keys)"""
        keys_list = []
        for metadata in self.keys.values():
            if environment and metadata.environment != environment:
                continue
            keys_list.append(metadata.to_dict())
        return keys_list
    
    def _save_to_file(self) -> None:
        """Persist to JSON file"""
        if not self.storage_path:
            return
        
        try:
            data = {
                "keys": {
                    k: asdict(v) for k, v in self.keys.items()
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Serialize with datetime handling
            def json_serializer(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Type {type(obj)} not serializable")
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2, default=json_serializer)
        except Exception as e:
            logger.error(f"Failed to save API keys to file: {e}")
    
    def _load_from_file(self) -> None:
        """Load from JSON file"""
        if not self.storage_path:
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            for key_hash, key_data in data.get("keys", {}).items():
                metadata = APIKeyMetadata(
                    key_id=key_data["key_id"],
                    name=key_data["name"],
                    key_hash=key_hash,
                    created_at=datetime.fromisoformat(key_data["created_at"]),
                    expires_at=datetime.fromisoformat(key_data["expires_at"]) if key_data.get("expires_at") else None,
                    last_used_at=datetime.fromisoformat(key_data["last_used_at"]) if key_data.get("last_used_at") else None,
                    status=KeyStatus(key_data["status"]),
                    tier=KeyTier(key_data["tier"]),
                    mode=KeyMode(key_data["mode"]),
                    requests_this_month=key_data["requests_this_month"],
                    requests_limit=key_data["requests_limit"],
                    usage_percent=key_data["usage_percent"],
                    created_by=key_data["created_by"],
                    environment=key_data["environment"]
                )
                self.keys[key_hash] = metadata
        except Exception as e:
            logger.error(f"Failed to load API keys from file: {e}")


# ============================================================================
# USAGE TRACKING
# ============================================================================

class UsageTracker:
    """Track API usage and billing"""
    
    def __init__(self, key_store: APIKeyStore):
        self.key_store = key_store
    
    def check_rate_limit(self, key_id: str) -> tuple[bool, str]:
        """
        Check if key has exceeded rate limit
        
        Returns:
            (is_allowed, message)
        """
        # Find metadata
        for metadata in self.key_store.keys.values():
            if metadata.key_id == key_id:
                if metadata.usage_percent >= 100:
                    return False, f"Rate limit exceeded ({metadata.requests_this_month}/{metadata.requests_limit})"
                
                remaining = metadata.requests_limit - metadata.requests_this_month
                return True, f"OK - {remaining} requests remaining"
        
        return False, "Key not found"
    
    def get_usage_report(self, key_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed usage report"""
        stats = self.key_store.get_usage_stats(key_id)
        if not stats:
            return None
        
        return {
            "key_id": key_id,
            "usage_this_month": {
                "requests": stats["requests_this_month"],
                "limit": stats["requests_limit"],
                "percent": stats["usage_percent"]
            },
            "tier": stats["tier"],
            "status": stats["status"],
            "reset_date": self._get_reset_date().isoformat(),
            "created_at": stats["created_at"],
            "last_used_at": stats.get("last_used_at"),
            "by_endpoint": stats.get("requests_by_endpoint", {})
        }
    
    def _get_reset_date(self) -> datetime:
        """Get next usage reset date (1st of next month)"""
        now = datetime.utcnow()
        if now.month == 12:
            return datetime(now.year + 1, 1, 1)
        return datetime(now.year, now.month + 1, 1)


# ============================================================================
# INITIALIZATION
# ============================================================================

def init_api_key_system(storage_path: str = "./data/api_keys.json") -> tuple[APIKeyStore, UsageTracker]:
    """Initialize the complete API key system"""
    # Create storage directory
    storage_dir = os.path.dirname(storage_path)
    if storage_dir and not os.path.exists(storage_dir):
        os.makedirs(storage_dir, exist_ok=True)
    
    # Initialize stores
    key_store = APIKeyStore(storage_path)
    tracker = UsageTracker(key_store)
    
    return key_store, tracker


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("API KEY SYSTEM - EXAMPLE USAGE")
    print("=" * 80)
    
    # Initialize
    key_store, tracker = init_api_key_system()
    
    # Create a test key
    print("\n1. Creating API key...")
    plaintext_key, metadata = key_store.create_key(
        name="Test API Key",
        tier=KeyTier.PRO,
        mode=KeyMode.FULL_ACCESS,
        created_by="admin",
        environment="development"
    )
    print(f"   Plaintext Key: {plaintext_key}")
    print(f"   Key ID: {metadata.key_id}")
    print(f"   Status: {metadata.status.value}")
    print(f"   Tier: {metadata.tier.value}")
    print(f"   Limit: {metadata.requests_limit} requests/month")
    
    # Validate key
    print("\n2. Validating key...")
    validated = key_store.validate_key(plaintext_key)
    print(f"   Valid: {validated is not None}")
    print(f"   Key ID: {validated.key_id if validated else 'N/A'}")
    
    # Record usage
    print("\n3. Recording usage...")
    for i in range(5):
        key_store.record_usage(
            metadata.key_id,
            "/invoke",
            "POST",
            200,
            tokens_used=150,
            duration_ms=245.3,
            ip_address="192.168.1.100"
        )
    print(f"   Recorded 5 requests")
    
    # Check rate limit
    print("\n4. Checking rate limit...")
    allowed, msg = tracker.check_rate_limit(metadata.key_id)
    print(f"   Allowed: {allowed}")
    print(f"   Message: {msg}")
    
    # Get usage stats
    print("\n5. Usage statistics...")
    stats = key_store.get_usage_stats(metadata.key_id)
    print(f"   Requests this month: {stats['requests_this_month']}/{stats['requests_limit']}")
    print(f"   Usage: {stats['usage_percent']:.1f}%")
    print(f"   Last used: {stats.get('last_used_at')}")
    
    # List keys
    print("\n6. All API keys...")
    for key in key_store.list_keys():
        print(f"   {key['key_id']}: {key['name']} ({key['tier']}) - {key['status']}")
    
    print("\n" + "=" * 80)
