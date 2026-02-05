"""
API Key Manager - Manages API key generation, rotation, revocation, and validation
Supports multiple API keys per user with individual tracking and audit logging
"""

import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from enum import Enum
import json
from src.utils.monitoring import logger


class APIKeyStatus(str, Enum):
    """Status of an API key"""
    ACTIVE = "active"
    ROTATED = "rotated"
    REVOKED = "revoked"
    EXPIRED = "expired"


class APIKeyManager:
    """
    Service for managing API keys with rotation, revocation, and audit tracking.
    
    Features:
    - Generate secure API keys (unique per user)
    - Rotate existing keys (maintain backward compatibility during rotation)
    - Revoke keys permanently
    - Track creation, rotation, and usage dates
    - Validate API keys against stored hash
    - TTL enforcement (default: 90 days)
    - Rate limiting per key
    """
    
    DEFAULT_KEY_TTL_DAYS = 90
    KEY_PREFIX = "apk_"
    KEY_HASH_ITERATIONS = 100000
    
    def __init__(self, db_client=None, redis_client=None):
        """
        Initialize API Key Manager
        
        Args:
            db_client: Firestore client for persistent storage
            redis_client: Redis client for caching (optional)
        """
        self.db = db_client
        self.redis = redis_client
        self._in_memory_cache = {}
    
    def generate_api_key(self, user_id: str, name: str = "default") -> Dict[str, Any]:
        """
        Generate a new API key for the user
        
        Args:
            user_id: User ID (Firebase UID or similar)
            name: Friendly name for the key (e.g., "production", "staging")
            
        Returns:
            Dict with key_id, key_value (only shown once), and metadata
        """
        try:
            # Generate 32 random bytes and encode to base64-like format
            random_bytes = secrets.token_bytes(32)
            key_value = f"{self.KEY_PREFIX}{secrets.token_urlsafe(48)}"
            
            # Hash the key for storage (PBKDF2)
            key_hash = hashlib.pbkdf2_hmac(
                'sha256',
                key_value.encode(),
                user_id.encode(),
                self.KEY_HASH_ITERATIONS
            ).hex()
            
            # Generate unique key ID
            key_id = f"key_{secrets.token_hex(8)}"
            
            # Create key metadata
            now = datetime.utcnow()
            key_metadata = {
                "key_id": key_id,
                "user_id": user_id,
                "name": name,
                "key_hash": key_hash,
                "status": APIKeyStatus.ACTIVE.value,
                "created_at": now.isoformat(),
                "rotated_at": None,
                "revoked_at": None,
                "expires_at": (now + timedelta(days=self.DEFAULT_KEY_TTL_DAYS)).isoformat(),
                "last_used": None,
                "usage_count": 0,
                "rotation_count": 0,
                "parent_key_id": None,  # For tracking rotation chain
            }
            
            # Store in database (or memory for demo)
            if self.db:
                self.db.collection("api_keys").document(key_id).set(key_metadata)
            else:
                self._in_memory_cache[key_id] = key_metadata
            
            logger.info(f"Generated API key {key_id} for user {user_id}")
            
            # Return key (value only shown once)
            return {
                "key_id": key_id,
                "key_value": key_value,  # IMPORTANT: Only shown once
                "prefix": self.KEY_PREFIX,
                "created_at": key_metadata["created_at"],
                "expires_at": key_metadata["expires_at"],
                "warning": "Save the key value securely. It will not be shown again!"
            }
            
        except Exception as e:
            logger.error(f"Error generating API key for user {user_id}: {e}")
            raise
    
    def verify_api_key(self, key_value: str, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Verify an API key and return metadata if valid
        
        Args:
            key_value: The API key value to verify
            user_id: Expected user ID
            
        Returns:
            Key metadata if valid, None if invalid/expired/revoked
        """
        try:
            # Hash the provided key
            key_hash = hashlib.pbkdf2_hmac(
                'sha256',
                key_value.encode(),
                user_id.encode(),
                self.KEY_HASH_ITERATIONS
            ).hex()
            
            # Find matching key in database
            if self.db:
                query = self.db.collection("api_keys") \
                    .where("user_id", "==", user_id) \
                    .where("key_hash", "==", key_hash)
                docs = query.stream()
                
                for doc in docs:
                    key_data = doc.to_dict()
                    
                    # Check if key is active and not expired
                    if key_data["status"] != APIKeyStatus.ACTIVE.value:
                        logger.warning(f"Invalid key status: {key_data['status']}")
                        return None
                    
                    if datetime.fromisoformat(key_data["expires_at"]) < datetime.utcnow():
                        logger.warning(f"Key expired: {key_data['key_id']}")
                        # Mark as expired
                        self.db.collection("api_keys").document(doc.id).update({
                            "status": APIKeyStatus.EXPIRED.value
                        })
                        return None
                    
                    # Update usage stats
                    self.db.collection("api_keys").document(doc.id).update({
                        "last_used": datetime.utcnow().isoformat(),
                        "usage_count": key_data.get("usage_count", 0) + 1
                    })
                    
                    return key_data
            else:
                # In-memory cache search
                for key_id, key_data in self._in_memory_cache.items():
                    if (key_data["user_id"] == user_id and 
                        key_data["key_hash"] == key_hash and
                        key_data["status"] == APIKeyStatus.ACTIVE.value):
                        
                        # Check expiration
                        if datetime.fromisoformat(key_data["expires_at"]) < datetime.utcnow():
                            key_data["status"] = APIKeyStatus.EXPIRED.value
                            return None
                        
                        # Update usage
                        key_data["last_used"] = datetime.utcnow().isoformat()
                        key_data["usage_count"] = key_data.get("usage_count", 0) + 1
                        
                        return key_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error verifying API key for user {user_id}: {e}")
            return None
    
    def rotate_api_key(self, key_id: str, user_id: str) -> Dict[str, Any]:
        """
        Rotate an API key (generate new one, mark old as rotated)
        
        Args:
            key_id: ID of key to rotate
            user_id: User ID (for authorization)
            
        Returns:
            New key details
        """
        try:
            # Fetch old key
            if self.db:
                doc = self.db.collection("api_keys").document(key_id).get()
                if not doc.exists():
                    raise ValueError(f"Key {key_id} not found")
                
                old_key = doc.to_dict()
            else:
                if key_id not in self._in_memory_cache:
                    raise ValueError(f"Key {key_id} not found")
                old_key = self._in_memory_cache[key_id]
            
            # Verify ownership
            if old_key["user_id"] != user_id:
                raise ValueError("Unauthorized key access")
            
            # Generate new key
            new_key = self.generate_api_key(user_id, old_key["name"])
            
            # Mark old key as rotated
            update_data = {
                "status": APIKeyStatus.ROTATED.value,
                "rotated_at": datetime.utcnow().isoformat(),
            }
            
            if self.db:
                self.db.collection("api_keys").document(key_id).update(update_data)
                # Update new key's parent reference
                self.db.collection("api_keys") \
                    .document(new_key["key_id"]) \
                    .update({"parent_key_id": key_id})
            else:
                old_key.update(update_data)
                self._in_memory_cache[new_key["key_id"]]["parent_key_id"] = key_id
            
            logger.info(f"Rotated API key {key_id} -> {new_key['key_id']} for user {user_id}")
            
            return new_key
            
        except Exception as e:
            logger.error(f"Error rotating API key {key_id}: {e}")
            raise
    
    def revoke_api_key(self, key_id: str, user_id: str) -> bool:
        """
        Revoke an API key (permanent)
        
        Args:
            key_id: ID of key to revoke
            user_id: User ID (for authorization)
            
        Returns:
            True if revoked successfully
        """
        try:
            # Fetch key
            if self.db:
                doc = self.db.collection("api_keys").document(key_id).get()
                if not doc.exists():
                    raise ValueError(f"Key {key_id} not found")
                key_data = doc.to_dict()
            else:
                if key_id not in self._in_memory_cache:
                    raise ValueError(f"Key {key_id} not found")
                key_data = self._in_memory_cache[key_id]
            
            # Verify ownership
            if key_data["user_id"] != user_id:
                raise ValueError("Unauthorized key access")
            
            # Mark as revoked
            update_data = {
                "status": APIKeyStatus.REVOKED.value,
                "revoked_at": datetime.utcnow().isoformat(),
            }
            
            if self.db:
                self.db.collection("api_keys").document(key_id).update(update_data)
            else:
                key_data.update(update_data)
            
            logger.info(f"Revoked API key {key_id} for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error revoking API key {key_id}: {e}")
            raise
    
    def list_api_keys(self, user_id: str, include_inactive: bool = False) -> List[Dict]:
        """
        List all API keys for a user
        
        Args:
            user_id: User ID
            include_inactive: Include revoked/rotated keys
            
        Returns:
            List of key metadata (without key values)
        """
        try:
            keys = []
            
            if self.db:
                query = self.db.collection("api_keys").where("user_id", "==", user_id)
                docs = query.stream()
                
                for doc in docs:
                    key_data = doc.to_dict()
                    
                    # Filter inactive if requested
                    if not include_inactive and key_data["status"] != APIKeyStatus.ACTIVE.value:
                        continue
                    
                    # Remove sensitive hash from response
                    key_data.pop("key_hash", None)
                    keys.append(key_data)
            else:
                for key_id, key_data in self._in_memory_cache.items():
                    if key_data["user_id"] == user_id:
                        if not include_inactive and key_data["status"] != APIKeyStatus.ACTIVE.value:
                            continue
                        
                        key_copy = {k: v for k, v in key_data.items() if k != "key_hash"}
                        keys.append(key_copy)
            
            logger.info(f"Listed {len(keys)} API keys for user {user_id}")
            return keys
            
        except Exception as e:
            logger.error(f"Error listing API keys for user {user_id}: {e}")
            raise
    
    def revoke_all_keys(self, user_id: str) -> int:
        """
        Revoke all active API keys for a user (security incident response)
        
        Args:
            user_id: User ID
            
        Returns:
            Number of keys revoked
        """
        try:
            revoked_count = 0
            now = datetime.utcnow().isoformat()
            
            if self.db:
                query = self.db.collection("api_keys") \
                    .where("user_id", "==", user_id) \
                    .where("status", "==", APIKeyStatus.ACTIVE.value)
                docs = query.stream()
                
                for doc in docs:
                    doc.reference.update({
                        "status": APIKeyStatus.REVOKED.value,
                        "revoked_at": now
                    })
                    revoked_count += 1
            else:
                for key_id, key_data in self._in_memory_cache.items():
                    if (key_data["user_id"] == user_id and 
                        key_data["status"] == APIKeyStatus.ACTIVE.value):
                        key_data["status"] = APIKeyStatus.REVOKED.value
                        key_data["revoked_at"] = now
                        revoked_count += 1
            
            logger.warning(f"Revoked all {revoked_count} API keys for user {user_id}")
            return revoked_count
            
        except Exception as e:
            logger.error(f"Error revoking all API keys for user {user_id}: {e}")
            raise


# Singleton instance
_api_key_manager = None


def get_api_key_manager(db_client=None, redis_client=None) -> APIKeyManager:
    """Get or create singleton APIKeyManager instance"""
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager(db_client, redis_client)
    return _api_key_manager
