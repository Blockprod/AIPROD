"""
Middleware d'authentification FastAPI.
Protège les endpoints critiques avec JWT Bearer tokens.
"""

import logging
from typing import Optional, Callable, Any
from functools import wraps

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from src.auth.firebase_auth import (
    get_firebase_authenticator,
    extract_token_from_header,
)

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)


async def verify_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """
    Dépendance FastAPI pour vérifier le token Bearer.

    Utilisation :
        @app.get("/protected")
        async def protected_route(user: dict = Depends(verify_token)):
            return {"message": f"Hello {user['email']}"}
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    authenticator = get_firebase_authenticator()
    if not authenticator.enabled:
        logger.warning("Authentication disabled, allowing request")
        return {"uid": "anonymous", "email": "anonymous@localhost"}

    user = authenticator.get_user_from_token(credentials.credentials)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


async def optional_verify_token(
    request: Request,
) -> Optional[dict]:
    """
    Optionnel : Vérifie le token s'il est présent.
    Accepte les requêtes sans token (pour endpoints publics).

    Utilisation :
        @app.get("/public-with-auth")
        async def public_route(user: dict = Depends(optional_verify_token)):
            return {"user": user}
    """
    auth_header = request.headers.get("authorization")
    if not auth_header:
        return None

    token = extract_token_from_header(auth_header)
    if not token:
        return None

    authenticator = get_firebase_authenticator()
    if not authenticator.enabled:
        return None

    user = authenticator.get_user_from_token(token)
    return user


def require_auth(required_roles: Optional[list] = None):
    """
    Décorateur pour protéger une fonction avec l'authentification.

    Utilisation :
        @app.post("/admin-only")
        @require_auth(required_roles=["admin"])
        async def admin_route(user: dict = Depends(verify_token)):
            return {"message": "Admin only"}
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            user = kwargs.get("user")

            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                )

            if required_roles:
                user_roles = user.get("custom_claims", {}).get("roles", [])
                if not any(role in user_roles for role in required_roles):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Insufficient permissions",
                    )

            return (
                await func(*args, **kwargs)
                if hasattr(func, "__await__")
                else func(*args, **kwargs)
            )

        return wrapper

    return decorator


class AuthMiddleware:
    """
    ASGI Middleware pour logger les requêtes authentifiées.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope.get("headers", []))
            auth_header = headers.get(b"authorization", b"").decode()

            if auth_header:
                token = extract_token_from_header(auth_header)
                if token:
                    authenticator = get_firebase_authenticator()
                    user = authenticator.get_user_from_token(token)
                    if user:
                        scope["user"] = user
                        logger.info(
                            f"Authenticated request from {user.get('email')} "
                            f"to {scope['path']}"
                        )

        await self.app(scope, receive, send)
