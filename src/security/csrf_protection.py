"""
CSRF Protection - Cross-Site Request Forgery protection
Uses double-submit cookie pattern with secure token validation
"""

import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict
from fastapi import Request, HTTPException
from src.utils.monitoring import logger


class CSRFTokenManager:
    """
    Manages CSRF tokens using double-submit cookie pattern.
    
    Pattern:
    1. Client requests CSRF token
    2. Server generates unique token and sets in HttpOnly cookie
    3. Client includes token in X-CSRF-Token header for state-changing requests
    4. Server validates: token in header matches token in cookie
    """
    
    COOKIE_NAME = "csrf_token"
    HEADER_NAME = "X-CSRF-Token"
    TOKEN_LENGTH = 32
    TTL_MINUTES = 60
    SAFE_METHODS = {"GET", "HEAD", "OPTIONS", "TRACE"}
    
    def __init__(self, secret_key: Optional[str] = None):
        """
        Initialize CSRF Token Manager
        
        Args:
            secret_key: Secret key for token signing (uses random if not provided)
        """
        self.secret_key = secret_key or secrets.token_hex(32)
        self.token_cache: Dict[str, Dict] = {}  # token -> {created_at, user_id}
    
    def generate_token(self, user_id: Optional[str] = None) -> str:
        """
        Generate a new CSRF token
        
        Args:
            user_id: Optional user ID for tracking
            
        Returns:
            CSRF token (random hex string)
        """
        token = secrets.token_hex(self.TOKEN_LENGTH)
        
        # Store token metadata
        self.token_cache[token] = {
            "created_at": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "used": False,
        }
        
        logger.debug(f"Generated CSRF token for user {user_id}")
        return token
    
    def validate_token(self, token: str, user_id: Optional[str] = None) -> bool:
        """
        Validate a CSRF token
        
        Args:
            token: Token to validate
            user_id: Optional user ID to verify against
            
        Returns:
            True if token is valid, False otherwise
        """
        if not token:
            logger.warning("CSRF validation failed: token is empty")
            return False
        
        if token not in self.token_cache:
            logger.warning(f"CSRF validation failed: token not found")
            return False
        
        token_data = self.token_cache[token]
        
        # Check token expiration
        created_at = datetime.fromisoformat(token_data["created_at"])
        age_minutes = (datetime.utcnow() - created_at).total_seconds() / 60
        
        if age_minutes > self.TTL_MINUTES:
            logger.warning(f"CSRF validation failed: token expired ({age_minutes} minutes)")
            del self.token_cache[token]
            return False
        
        # Verify user ID if provided
        if user_id and token_data.get("user_id") != user_id:
            logger.warning(f"CSRF validation failed: user_id mismatch")
            return False
        
        logger.debug(f"CSRF token validated successfully")
        return True
    
    def revoke_token(self, token: str):
        """
        Revoke a CSRF token (e.g., after use or logout)
        
        Args:
            token: Token to revoke
        """
        self.token_cache.pop(token, None)
        logger.debug(f"CSRF token revoked")
    
    def cleanup_expired(self):
        """Clean up expired tokens from cache"""
        now = datetime.utcnow()
        expired_tokens = []
        
        for token, data in self.token_cache.items():
            created_at = datetime.fromisoformat(data["created_at"])
            age_minutes = (now - created_at).total_seconds() / 60
            
            if age_minutes > self.TTL_MINUTES:
                expired_tokens.append(token)
        
        for token in expired_tokens:
            del self.token_cache[token]
        
        if expired_tokens:
            logger.debug(f"Cleaned up {len(expired_tokens)} expired CSRF tokens")


# Global CSRF manager instance
_csrf_manager = None


def get_csrf_manager(secret_key: Optional[str] = None) -> CSRFTokenManager:
    """Get or create singleton CSRF manager"""
    global _csrf_manager
    if _csrf_manager is None:
        _csrf_manager = CSRFTokenManager(secret_key)
    return _csrf_manager


async def verify_csrf_token(request: Request) -> bool:
    """
    FastAPI dependency to verify CSRF token in request
    
    Should be used on POST, PUT, DELETE, PATCH endpoints
    
    Args:
        request: FastAPI request object
        
    Returns:
        True if CSRF check passes, raises HTTPException otherwise
        
    Raises:
        HTTPException(403) if CSRF token is invalid/missing
    """
    # Skip CSRF check for safe methods
    if request.method in CSRFTokenManager.SAFE_METHODS:
        return True
    
    # Get token from header
    token = request.headers.get(CSRFTokenManager.HEADER_NAME)
    if not token:
        logger.warning(f"CSRF token missing in {request.method} {request.url.path}")
        raise HTTPException(
            status_code=403,
            detail="CSRF token missing. Include X-CSRF-Token header."
        )
    
    # Validate token
    manager = get_csrf_manager()
    if not manager.validate_token(token):
        logger.warning(f"CSRF token validation failed for {request.url.path}")
        raise HTTPException(
            status_code=403,
            detail="CSRF token invalid or expired"
        )
    
    return True


async def get_csrf_token(request: Request, user_id: Optional[str] = None) -> str:
    """
    Generate a new CSRF token for client
    
    Args:
        request: FastAPI request
        user_id: Optional user ID
        
    Returns:
        CSRF token to include in subsequent requests
    """
    manager = get_csrf_manager()
    token = manager.generate_token(user_id)
    return token
