"""
Security Management for Universal MCP Server
Handles encryption, authentication, rate limiting, and security monitoring
"""

import hashlib
import hmac
import secrets
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict
import logging

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import base64
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False

try:
    import jwt
    HAS_JWT = True
except ImportError:
    HAS_JWT = False

from .logger import log_security_event, log_audit_event

@dataclass
class RateLimitRule:
    """Rate limiting rule configuration."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10

@dataclass
class SecurityEvent:
    """Security event data structure."""
    event_type: str
    severity: str
    description: str
    timestamp: datetime
    ip_address: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class RateLimiter:
    """Token bucket rate limiter with multiple time windows."""
    
    def __init__(self, rules: RateLimitRule):
        self.rules = rules
        self.buckets: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: {
                'minute': [],
                'hour': [],
                'day': [],
                'burst': []
            }
        )
        self.logger = logging.getLogger(__name__)
    
    def is_allowed(self, identifier: str) -> Tuple[bool, str]:
        """
        Check if a request is allowed based on rate limits.
        
        Args:
            identifier: Unique identifier (IP address, user ID, etc.)
            
        Returns:
            Tuple of (is_allowed, reason)
        """
        now = time.time()
        bucket = self.buckets[identifier]
        
        # Clean old entries
        self._cleanup_bucket(bucket, now)
        
        # Check burst limit
        if len(bucket['burst']) >= self.rules.burst_limit:
            return False, "Burst limit exceeded"
        
        # Check per-minute limit
        if len(bucket['minute']) >= self.rules.requests_per_minute:
            return False, "Per-minute limit exceeded"
        
        # Check per-hour limit
        if len(bucket['hour']) >= self.rules.requests_per_hour:
            return False, "Per-hour limit exceeded"
        
        # Check per-day limit
        if len(bucket['day']) >= self.rules.requests_per_day:
            return False, "Per-day limit exceeded"
        
        # Add request to all buckets
        bucket['burst'].append(now)
        bucket['minute'].append(now)
        bucket['hour'].append(now)
        bucket['day'].append(now)
        
        return True, "Request allowed"
    
    def _cleanup_bucket(self, bucket: Dict[str, List[float]], now: float):
        """Remove expired entries from rate limit bucket."""
        # Burst: 1 second window
        bucket['burst'] = [t for t in bucket['burst'] if now - t < 1]
        
        # Minute: 60 seconds
        bucket['minute'] = [t for t in bucket['minute'] if now - t < 60]
        
        # Hour: 3600 seconds
        bucket['hour'] = [t for t in bucket['hour'] if now - t < 3600]
        
        # Day: 86400 seconds
        bucket['day'] = [t for t in bucket['day'] if now - t < 86400]
    
    def get_status(self, identifier: str) -> Dict[str, Any]:
        """Get current rate limit status for an identifier."""
        now = time.time()
        bucket = self.buckets[identifier]
        self._cleanup_bucket(bucket, now)
        
        return {
            "identifier": identifier,
            "limits": {
                "burst": {"used": len(bucket['burst']), "limit": self.rules.burst_limit},
                "minute": {"used": len(bucket['minute']), "limit": self.rules.requests_per_minute},
                "hour": {"used": len(bucket['hour']), "limit": self.rules.requests_per_hour},
                "day": {"used": len(bucket['day']), "limit": self.rules.requests_per_day}
            },
            "reset_times": {
                "burst": int(now + 1),
                "minute": int(now + 60),
                "hour": int(now + 3600),
                "day": int(now + 86400)
            }
        }

class Encryptor:
    """Handles encryption and decryption of sensitive data."""
    
    def __init__(self, password: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        if not HAS_CRYPTOGRAPHY:
            self.logger.warning("Cryptography library not available. Encryption disabled.")
            self.cipher = None
            return
        
        if password is None:
            password = secrets.token_urlsafe(32)
            self.logger.info("Generated random encryption password")
        
        # Derive key from password
        salt = b'universal_mcp_server_salt'  # In production, use a random salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        self.cipher = Fernet(key)
    
    def is_available(self) -> bool:
        """Check if encryption is available."""
        return self.cipher is not None
    
    def encrypt(self, data: str) -> Optional[str]:
        """Encrypt a string."""
        if not self.cipher:
            return data  # Return unencrypted if encryption not available
        
        try:
            encrypted = self.cipher.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            return None
    
    def decrypt(self, encrypted_data: str) -> Optional[str]:
        """Decrypt a string."""
        if not self.cipher:
            return encrypted_data  # Return as-is if encryption not available
        
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = self.cipher.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            return None
    
    def encrypt_dict(self, data: Dict[str, Any]) -> Optional[str]:
        """Encrypt a dictionary as JSON."""
        try:
            json_str = json.dumps(data)
            return self.encrypt(json_str)
        except Exception as e:
            self.logger.error(f"Dictionary encryption failed: {e}")
            return None
    
    def decrypt_dict(self, encrypted_data: str) -> Optional[Dict[str, Any]]:
        """Decrypt JSON back to dictionary."""
        try:
            json_str = self.decrypt(encrypted_data)
            if json_str:
                return json.loads(json_str)
            return None
        except Exception as e:
            self.logger.error(f"Dictionary decryption failed: {e}")
            return None

class APIKeyManager:
    """Manages API key generation, validation, and rotation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.keys: Dict[str, Dict[str, Any]] = {}
        self.revoked_keys: set = set()
    
    def generate_key(
        self, 
        name: str, 
        permissions: List[str] = None,
        expires_in_days: Optional[int] = None
    ) -> str:
        """Generate a new API key."""
        api_key = f"umcp_{secrets.token_urlsafe(32)}"
        
        key_data = {
            "name": name,
            "permissions": permissions or ["read", "write"],
            "created_at": datetime.now().isoformat(),
            "last_used": None,
            "usage_count": 0,
            "expires_at": None
        }
        
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)
            key_data["expires_at"] = expires_at.isoformat()
        
        self.keys[api_key] = key_data
        
        log_audit_event(
            action="api_key_generated",
            resource=f"api_key:{name}",
            details={"key_id": api_key[:12] + "..."}
        )
        
        return api_key
    
    def validate_key(self, api_key: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Validate an API key."""
        if not api_key or api_key in self.revoked_keys:
            return False, None
        
        key_data = self.keys.get(api_key)
        if not key_data:
            return False, None
        
        # Check expiration
        if key_data["expires_at"]:
            expires_at = datetime.fromisoformat(key_data["expires_at"])
            if datetime.now() > expires_at:
                return False, {"error": "API key expired"}
        
        # Update usage
        key_data["last_used"] = datetime.now().isoformat()
        key_data["usage_count"] += 1
        
        return True, key_data
    
    def revoke_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        if api_key in self.keys:
            self.revoked_keys.add(api_key)
            key_data = self.keys[api_key]
            
            log_audit_event(
                action="api_key_revoked",
                resource=f"api_key:{key_data['name']}",
                details={"key_id": api_key[:12] + "..."}
            )
            
            return True
        return False
    
    def list_keys(self) -> List[Dict[str, Any]]:
        """List all API keys (without the actual key values)."""
        result = []
        for api_key, data in self.keys.items():
            if api_key not in self.revoked_keys:
                result.append({
                    "key_id": api_key[:12] + "...",
                    "name": data["name"],
                    "permissions": data["permissions"],
                    "created_at": data["created_at"],
                    "last_used": data["last_used"],
                    "usage_count": data["usage_count"],
                    "expires_at": data["expires_at"],
                    "status": "active"
                })
        return result

class SecurityManager:
    """Main security manager that coordinates all security features."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.rate_limiter = RateLimiter(
            RateLimitRule(
                requests_per_minute=self.config.get("rate_limit_requests_per_minute", 60),
                requests_per_hour=self.config.get("rate_limit_requests_per_hour", 1000),
                requests_per_day=self.config.get("rate_limit_requests_per_day", 10000),
                burst_limit=self.config.get("rate_limit_burst", 10)
            )
        )
        
        self.encryptor = Encryptor(self.config.get("secret_key"))
        self.api_key_manager = APIKeyManager()
        
        # Security monitoring
        self.security_events: List[SecurityEvent] = []
        self.failed_auth_attempts: Dict[str, List[datetime]] = defaultdict(list)
        
        # Load API keys if provided
        api_keys = self.config.get("api_keys", [])
        for key in api_keys:
            if isinstance(key, str):
                # Simple string key
                self.api_key_manager.keys[key] = {
                    "name": "default",
                    "permissions": ["read", "write"],
                    "created_at": datetime.now().isoformat(),
                    "last_used": None,
                    "usage_count": 0,
                    "expires_at": None
                }
        
        self.logger.info("Security manager initialized")
    
    def is_encryption_enabled(self) -> bool:
        """Check if encryption is enabled."""
        return self.encryptor.is_available()
    
    def is_audit_logging_enabled(self) -> bool:
        """Check if audit logging is enabled."""
        return self.config.get("enable_audit_logging", True)
    
    def check_rate_limit(self, identifier: str) -> Tuple[bool, str]:
        """Check rate limit for an identifier."""
        allowed, reason = self.rate_limiter.is_allowed(identifier)
        
        if not allowed:
            self.log_security_event(
                "rate_limit_exceeded",
                "medium",
                f"Rate limit exceeded for {identifier}: {reason}",
                ip_address=identifier
            )
        
        return allowed, reason
    
    def authenticate_api_key(self, api_key: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Authenticate an API key."""
        valid, key_data = self.api_key_manager.validate_key(api_key)
        
        if not valid:
            self.log_security_event(
                "invalid_api_key",
                "high",
                f"Invalid API key used: {api_key[:12]}...",
                metadata={"key_prefix": api_key[:12] if api_key else "empty"}
            )
        
        return valid, key_data
    
    def check_authentication_rate_limit(self, identifier: str) -> bool:
        """Check if authentication attempts are within limits."""
        now = datetime.now()
        attempts = self.failed_auth_attempts[identifier]
        
        # Clean old attempts (older than 1 hour)
        cutoff = now - timedelta(hours=1)
        attempts[:] = [attempt for attempt in attempts if attempt > cutoff]
        
        # Check if too many failed attempts
        if len(attempts) >= 5:  # Max 5 failed attempts per hour
            self.log_security_event(
                "authentication_rate_limit_exceeded",
                "high",
                f"Too many failed authentication attempts for {identifier}",
                ip_address=identifier
            )
            return False
        
        return True
    
    def record_failed_auth(self, identifier: str):
        """Record a failed authentication attempt."""
        self.failed_auth_attempts[identifier].append(datetime.now())
        
        self.log_security_event(
            "authentication_failed",
            "medium",
            f"Failed authentication attempt from {identifier}",
            ip_address=identifier
        )
    
    def log_security_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        ip_address: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log a security event."""
        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            description=description,
            timestamp=datetime.now(),
            ip_address=ip_address,
            user_id=user_id,
            metadata=metadata
        )
        
        self.security_events.append(event)
        
        # Also log using the security logger
        log_security_event(
            event_type=event_type,
            severity=severity,
            description=description,
            ip_address=ip_address,
            user_id=user_id,
            **(metadata or {})
        )
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get a summary of security status."""
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        
        recent_events = [
            event for event in self.security_events
            if event.timestamp > last_24h
        ]
        
        event_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for event in recent_events:
            event_counts[event.event_type] += 1
            severity_counts[event.severity] += 1
        
        return {
            "encryption_enabled": self.is_encryption_enabled(),
            "audit_logging_enabled": self.is_audit_logging_enabled(),
            "total_events_24h": len(recent_events),
            "event_types": dict(event_counts),
            "severity_distribution": dict(severity_counts),
            "active_api_keys": len(self.api_key_manager.list_keys()),
            "rate_limiting_enabled": True,
            "last_updated": now.isoformat()
        }
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate a cryptographically secure token."""
        return secrets.token_urlsafe(length)
    
    def hash_password(self, password: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """Hash a password with salt."""
        if salt is None:
            salt = secrets.token_hex(16)
        
        # Use PBKDF2 for password hashing
        pwdhash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # 100,000 iterations
        )
        
        return pwdhash.hex(), salt
    
    def verify_password(self, password: str, hash_value: str, salt: str) -> bool:
        """Verify a password against its hash."""
        pwdhash, _ = self.hash_password(password, salt)
        return hmac.compare_digest(pwdhash, hash_value)
    
    def create_jwt_token(
        self,
        payload: Dict[str, Any],
        secret_key: Optional[str] = None,
        expires_in_hours: int = 24
    ) -> Optional[str]:
        """Create a JWT token."""
        if not HAS_JWT:
            self.logger.warning("JWT library not available")
            return None
        
        secret_key = secret_key or self.config.get("secret_key", "default_secret")
        
        # Add expiration
        payload["exp"] = datetime.utcnow() + timedelta(hours=expires_in_hours)
        payload["iat"] = datetime.utcnow()
        
        try:
            token = jwt.encode(payload, secret_key, algorithm="HS256")
            return token
        except Exception as e:
            self.logger.error(f"JWT creation failed: {e}")
            return None
    
    def verify_jwt_token(
        self,
        token: str,
        secret_key: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token."""
        if not HAS_JWT:
            return None
        
        secret_key = secret_key or self.config.get("secret_key", "default_secret")
        
        try:
            payload = jwt.decode(token, secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            self.log_security_event(
                "jwt_token_expired",
                "low",
                "JWT token has expired"
            )
            return None
        except jwt.InvalidTokenError as e:
            self.log_security_event(
                "jwt_token_invalid",
                "medium",
                f"Invalid JWT token: {str(e)}"
            )
            return None
