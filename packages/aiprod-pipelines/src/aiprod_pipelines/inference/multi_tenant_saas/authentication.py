"""
Authentication and Token Management for Multi-Tenant SaaS.

Provides API key validation, JWT token generation/verification,
OAuth integration support, and session management.

Core Classes:
  - APIKey: API key representation
  - JWTTokenManager: Token lifecycle management
  - AuthenticationManager: Main authentication interface
  - SessionManager: Session tracking and validation
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any, Set
from enum import Enum
import time
import hashlib
import secrets
import hmac
from datetime import datetime, timedelta
import threading
import json


class TokenType(str, Enum):
    """Token type enumeration."""
    API_KEY = "api_key"
    JWT = "jwt"
    OAUTH = "oauth"
    SESSION = "session"


class AuthenticationMethod(str, Enum):
    """Supported authentication methods."""
    API_KEY = "api_key"
    JWT = "jwt"
    OAUTH_GITHUB = "oauth_github"
    OAUTH_GOOGLE = "oauth_google"
    SESSION_TOKEN = "session_token"


@dataclass
class APIKey:
    """API key for programmatic access."""
    key_id: str
    tenant_id: str
    key_hash: str  # SHA-256 hash of actual key
    key_prefix: str  # First 6 characters of key (for UI display)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    is_active: bool = True
    name: str = ""
    scopes: Set[str] = field(default_factory=lambda: {"read", "write"})
    rate_limit_per_minute: int = 100
    description: str = ""
    
    def is_expired(self) -> bool:
        """Check if key is expired."""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at
    
    def is_valid(self) -> bool:
        """Check if key is valid for use."""
        return self.is_active and not self.is_expired()
    
    def has_scope(self, scope: str) -> bool:
        """Check if key has specific scope."""
        return scope in self.scopes
    
    def update_last_used(self) -> None:
        """Update last used timestamp."""
        self.last_used_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (safe for API responses)."""
        return {
            "key_id": self.key_id,
            "key_prefix": self.key_prefix,
            "created_at": self.created_at.isoformat(),
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_active": self.is_active,
            "name": self.name,
            "scopes": list(self.scopes),
            "rate_limit_per_minute": self.rate_limit_per_minute,
        }


@dataclass
class JWTToken:
    """JWT token data."""
    token_id: str
    tenant_id: str
    user_id: str
    issued_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    scopes: Set[str] = field(default_factory=lambda: {"read", "write"})
    custom_claims: Dict[str, Any] = field(default_factory=dict)
    is_refresh_token: bool = False
    
    def is_expired(self) -> bool:
        """Check if token is expired."""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at
    
    def has_scope(self, scope: str) -> bool:
        """Check if token has specific scope."""
        return scope in self.scopes


class JWTTokenManager:
    """JWT token generation and verification."""
    
    def __init__(self, secret_key: str, issuer: str = "aiprod-saas"):
        """Initialize token manager."""
        self.secret_key = secret_key
        self.issuer = issuer
        self._token_blacklist: Set[str] = set()
        self._lock = threading.RLock()
    
    def generate_token(
        self,
        tenant_id: str,
        user_id: str,
        expires_in_hours: int = 24,
        scopes: Optional[Set[str]] = None,
    ) -> Tuple[str, JWTToken]:
        """Generate a new JWT token."""
        token_id = self._generate_token_id()
        now = datetime.utcnow()
        expires_at = now + timedelta(hours=expires_in_hours)
        
        token_data = JWTToken(
            token_id=token_id,
            tenant_id=tenant_id,
            user_id=user_id,
            issued_at=now,
            expires_at=expires_at,
            scopes=scopes or {"read", "write"},
        )
        
        # Create JWT claims
        claims = {
            "jti": token_id,
            "iss": self.issuer,
            "sub": user_id,
            "aud": tenant_id,
            "iat": int(now.timestamp()),
            "exp": int(expires_at.timestamp()),
            "scopes": list(token_data.scopes),
        }
        
        # Simple JWT encoding (in production use PyJWT)
        encoded = self._encode_jwt(claims)
        return encoded, token_data
    
    def generate_refresh_token(
        self,
        tenant_id: str,
        user_id: str,
        expires_in_days: int = 30,
    ) -> Tuple[str, JWTToken]:
        """Generate a refresh token."""
        token_id = self._generate_token_id()
        now = datetime.utcnow()
        expires_at = now + timedelta(days=expires_in_days)
        
        token_data = JWTToken(
            token_id=token_id,
            tenant_id=tenant_id,
            user_id=user_id,
            issued_at=now,
            expires_at=expires_at,
            scopes={"refresh"},
            is_refresh_token=True,
        )
        
        claims = {
            "jti": token_id,
            "iss": self.issuer,
            "sub": user_id,
            "aud": tenant_id,
            "iat": int(now.timestamp()),
            "exp": int(expires_at.timestamp()),
            "type": "refresh",
        }
        
        encoded = self._encode_jwt(claims)
        return encoded, token_data
    
    def verify_token(self, token: str) -> Optional[JWTToken]:
        """Verify JWT token validity."""
        try:
            claims = self._decode_jwt(token)
            if claims and claims["iss"] != self.issuer:
                return None
            
            token_id = claims.get("jti")
            if token_id in self._token_blacklist:
                return None
            
            token_data = JWTToken(
                token_id=token_id,
                tenant_id=claims.get("aud"),
                user_id=claims.get("sub"),
                issued_at=datetime.fromtimestamp(claims.get("iat", 0)),
                expires_at=datetime.fromtimestamp(claims.get("exp", 0)),
                scopes=set(claims.get("scopes", [])),
                is_refresh_token=claims.get("type") == "refresh",
            )
            
            if token_data.is_expired():
                return None
            
            return token_data
        except Exception:
            return None
    
    def revoke_token(self, token_id: str) -> None:
        """Add token to blacklist."""
        with self._lock:
            self._token_blacklist.add(token_id)
    
    def _generate_token_id(self) -> str:
        """Generate unique token ID."""
        return secrets.token_hex(16)
    
    def _encode_jwt(self, claims: Dict[str, Any]) -> str:
        """Simple JWT encoding."""
        import base64
        header = base64.urlsafe_b64encode(
            json.dumps({"alg": "HS256", "typ": "JWT"}).encode()
        ).decode().rstrip("=")
        payload = base64.urlsafe_b64encode(
            json.dumps(claims).encode()
        ).decode().rstrip("=")
        message = f"{header}.{payload}"
        signature = base64.urlsafe_b64encode(
            hmac.new(
                self.secret_key.encode(),
                message.encode(),
                hashlib.sha256
            ).digest()
        ).decode().rstrip("=")
        return f"{message}.{signature}"
    
    def _decode_jwt(self, token: str) -> Optional[Dict[str, Any]]:
        """Simple JWT decoding with verification."""
        import base64
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None
            
            header, payload, signature = parts
            message = f"{header}.{payload}"
            
            expected_signature = base64.urlsafe_b64encode(
                hmac.new(
                    self.secret_key.encode(),
                    message.encode(),
                    hashlib.sha256
                ).digest()
            ).decode().rstrip("=")
            
            if signature != expected_signature:
                return None
            
            decoded = base64.urlsafe_b64decode(payload + "==")
            return json.loads(decoded)
        except Exception:
            return None


class SessionManager:
    """Session token and lifecycle management."""
    
    def __init__(self, session_timeout_minutes: int = 60):
        """Initialize session manager."""
        self.session_timeout_minutes = session_timeout_minutes
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
    
    def create_session(
        self,
        tenant_id: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create new user session."""
        session_id = secrets.token_urlsafe(32)
        now = datetime.utcnow()
        
        with self._lock:
            self._sessions[session_id] = {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "created_at": now,
                "last_activity": now,
                "metadata": metadata or {},
            }
        
        return session_id
    
    def validate_session(self, session_id: str) -> bool:
        """Validate session is active and not expired."""
        with self._lock:
            if session_id not in self._sessions:
                return False
            
            session = self._sessions[session_id]
            elapsed = (datetime.utcnow() - session["last_activity"]).total_seconds()
            
            if elapsed > self.session_timeout_minutes * 60:
                del self._sessions[session_id]
                return False
            
            return True
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information."""
        if not self.validate_session(session_id):
            return None
        
        with self._lock:
            if session_id in self._sessions:
                session = self._sessions[session_id].copy()
                session["elapsed_seconds"] = (
                    datetime.utcnow() - session["last_activity"]
                ).total_seconds()
                return session
        
        return None
    
    def update_session_activity(self, session_id: str) -> bool:
        """Update last activity timestamp."""
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id]["last_activity"] = datetime.utcnow()
                return True
        return False
    
    def revoke_session(self, session_id: str) -> bool:
        """Revoke a session."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
        return False
    
    def revoke_user_sessions(self, tenant_id: str, user_id: str) -> int:
        """Revoke all sessions for a user."""
        revoked = 0
        with self._lock:
            session_ids_to_revoke = [
                sid for sid, sess in self._sessions.items()
                if sess["tenant_id"] == tenant_id and sess["user_id"] == user_id
            ]
            for session_id in session_ids_to_revoke:
                del self._sessions[session_id]
                revoked += 1
        return revoked


class AuthenticationManager:
    """Main authentication interface for the SaaS platform."""
    
    def __init__(self, jwt_secret: str):
        """Initialize authentication manager."""
        self.jwt_manager = JWTTokenManager(jwt_secret)
        self.session_manager = SessionManager()
        self._api_keys: Dict[str, APIKey] = {}  # key_hash -> APIKey
        self._tenant_keys: Dict[str, Set[str]] = {}  # tenant_id -> set of key_hashes
        self._lock = threading.RLock()
    
    def create_api_key(
        self,
        tenant_id: str,
        name: str = "",
        expires_in_days: Optional[int] = None,
    ) -> Tuple[str, APIKey]:
        """Create new API key for tenant."""
        key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        key_prefix = key[:6]
        key_id = secrets.token_hex(8)
        
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        api_key = APIKey(
            key_id=key_id,
            tenant_id=tenant_id,
            key_hash=key_hash,
            key_prefix=key_prefix,
            expires_at=expires_at,
            name=name,
        )
        
        with self._lock:
            self._api_keys[key_hash] = api_key
            if tenant_id not in self._tenant_keys:
                self._tenant_keys[tenant_id] = set()
            self._tenant_keys[tenant_id].add(key_hash)
        
        return key, api_key
    
    def verify_api_key(self, api_key: str) -> Optional[APIKey]:
        """Verify API key validity."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        with self._lock:
            if key_hash in self._api_keys:
                api_key_obj = self._api_keys[key_hash]
                if api_key_obj.is_valid():
                    api_key_obj.update_last_used()
                    return api_key_obj
        
        return None
    
    def revoke_api_key(self, key_hash: str) -> bool:
        """Revoke an API key."""
        with self._lock:
            if key_hash in self._api_keys:
                api_key = self._api_keys[key_hash]
                api_key.is_active = False
                return True
        return False
    
    def list_tenant_api_keys(self, tenant_id: str) -> list:
        """List all API keys for a tenant."""
        with self._lock:
            if tenant_id in self._tenant_keys:
                return [
                    self._api_keys[key_hash].to_dict()
                    for key_hash in self._tenant_keys[tenant_id]
                    if key_hash in self._api_keys
                ]
        return []
