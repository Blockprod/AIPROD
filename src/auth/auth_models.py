"""
Modèles Pydantic pour les endpoints d'authentification.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from datetime import datetime


class RefreshTokenRequest(BaseModel):
    """Requête pour rafraîchir un access token."""

    refresh_token: str = Field(
        ..., description="Token de refresh obtenu lors de l'authentification"
    )

    model_config = ConfigDict(json_schema_extra={"example": {"refresh_token": "eyJ0eXAi..."}})


class TokenResponse(BaseModel):
    """Réponse avec tokens d'authentification."""

    access_token: str = Field(..., description="Nouveau access token (JWT)")
    refresh_token: str = Field(..., description="Token de refresh pour les appels futurs")
    token_type: str = Field(default="Bearer", description="Type de token")
    expires_in: int = Field(..., description="Durée de vie de l'access token en secondes")
    issued_at: datetime = Field(..., description="Moment de génération du token")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
                "refresh_token": "eyJ0eXAiOiJSZWZyZXNoIi...",
                "token_type": "Bearer",
                "expires_in": 900,
                "issued_at": "2026-02-05T12:00:00",
            }
        }
    )


class TokenInfoResponse(BaseModel):
    """Informations sur un token."""

    user_id: str
    expires_at: datetime
    created_at: datetime
    version: int


class RevokeTokenRequest(BaseModel):
    """Requête pour révoquer un token."""

    refresh_token: str = Field(..., description="Token à révoquer")


class LoginResponse(BaseModel):
    """Réponse successful login."""

    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int
    user_id: str
    email: Optional[str] = None
