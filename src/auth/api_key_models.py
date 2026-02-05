"""
Pydantic models for API Key management
Handles validation and serialization of API key requests/responses
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime


class CreateAPIKeyRequest(BaseModel):
    """Request to create a new API key"""
    name: str = Field(
        default="default",
        min_length=1,
        max_length=50,
        description="Friendly name for the key (e.g., 'production', 'staging')"
    )
    
    model_config = ConfigDict(json_schema_extra={
        "example": {"name": "production-key"}
    })


class APIKeyResponse(BaseModel):
    """Response for newly created API key (key value shown only once)"""
    key_id: str = Field(description="Unique key identifier")
    key_value: str = Field(description="API key value (ONLY shown when created)")
    prefix: str = Field(description="Key prefix (e.g., 'apk_')")
    created_at: str = Field(description="ISO timestamp of creation")
    expires_at: str = Field(description="ISO timestamp of expiration")
    warning: str = Field(default="Save the key value securely. It will not be shown again!")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "key_id": "key_a1b2c3d4",
            "key_value": "apk_abc123def456xyz789...",
            "prefix": "apk_",
            "created_at": "2026-02-05T12:00:00",
            "expires_at": "2026-05-05T12:00:00",
            "warning": "Save the key value securely. It will not be shown again!"
        }
    })


class APIKeyMetadata(BaseModel):
    """Metadata for an existing API key (no key value)"""
    key_id: str = Field(description="Unique key identifier")
    name: str = Field(description="Friendly name")
    status: str = Field(description="Status: active, rotated, revoked, expired")
    created_at: str = Field(description="ISO timestamp of creation")
    rotated_at: Optional[str] = Field(default=None, description="ISO timestamp of last rotation")
    revoked_at: Optional[str] = Field(default=None, description="ISO timestamp of revocation")
    expires_at: str = Field(description="ISO timestamp of expiration")
    last_used: Optional[str] = Field(default=None, description="ISO timestamp of last use")
    usage_count: int = Field(default=0, description="Number of times this key was used")
    rotation_count: int = Field(default=0, description="Number of times this key was rotated")
    parent_key_id: Optional[str] = Field(default=None, description="ID of key this was rotated from")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "key_id": "key_a1b2c3d4",
            "name": "production-key",
            "status": "active",
            "created_at": "2026-02-05T12:00:00",
            "rotated_at": None,
            "revoked_at": None,
            "expires_at": "2026-05-05T12:00:00",
            "last_used": "2026-02-05T12:30:00",
            "usage_count": 150,
            "rotation_count": 0,
            "parent_key_id": None
        }
    })


class ListAPIKeysResponse(BaseModel):
    """Response for listing API keys"""
    keys: List[APIKeyMetadata] = Field(description="List of API keys")
    total: int = Field(description="Total number of keys")
    active_count: int = Field(description="Number of active keys")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "keys": [],
            "total": 2,
            "active_count": 1
        }
    })


class RotateAPIKeyRequest(BaseModel):
    """Request to rotate an API key"""
    key_id: str = Field(description="ID of key to rotate")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {"key_id": "key_a1b2c3d4"}
    })


class RevokeAPIKeyRequest(BaseModel):
    """Request to revoke an API key"""
    key_id: str = Field(description="ID of key to revoke")
    reason: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Optional reason for revocation"
    )
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "key_id": "key_a1b2c3d4",
            "reason": "Key leaked in public repository"
        }
    })


class RevokeAllKeysRequest(BaseModel):
    """Request to revoke all API keys for a user"""
    confirm: bool = Field(
        description="Must be true to confirm revoking all keys (security paranoia)"
    )
    reason: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Optional reason for mass revocation"
    )
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "confirm": True,
            "reason": "Account security audit"
        }
    })


class RevokeAllKeysResponse(BaseModel):
    """Response for revoking all API keys"""
    revoked_count: int = Field(description="Number of keys revoked")
    timestamp: str = Field(description="ISO timestamp of revocation")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "revoked_count": 5,
            "timestamp": "2026-02-05T12:35:00"
        }
    })


class APIKeyStatsResponse(BaseModel):
    """Statistics about API keys"""
    total_keys: int = Field(description="Total number of keys")
    active_keys: int = Field(description="Number of active keys")
    rotated_keys: int = Field(description="Number of rotated keys")
    revoked_keys: int = Field(description="Number of revoked keys")
    expired_keys: int = Field(description="Number of expired keys")
    next_expiration: Optional[str] = Field(
        default=None,
        description="ISO timestamp of next key expiration"
    )
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "total_keys": 5,
            "active_keys": 2,
            "rotated_keys": 1,
            "revoked_keys": 1,
            "expired_keys": 1,
            "next_expiration": "2026-03-05T12:00:00"
        }
    })


class APIKeyHealthCheck(BaseModel):
    """Health status of API keys"""
    has_active_keys: bool = Field(description="User has at least one active key")
    expiration_warning: bool = Field(description="At least one key expiring soon (< 7 days)")
    unused_keys: int = Field(description="Number of keys not used in 30 days")
    recommendations: List[str] = Field(description="Security recommendations")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "has_active_keys": True,
            "expiration_warning": False,
            "unused_keys": 1,
            "recommendations": [
                "Consider rotating keys quarterly",
                "Revoke unused keys (key_id_xyz not used in 30 days)"
            ]
        }
    })
