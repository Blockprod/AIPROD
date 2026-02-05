"""
RBAC middleware and decorators for FastAPI
Handles role-based access control for API endpoints
"""

from typing import Callable, Optional, List, Any
from functools import wraps
from fastapi import Request, HTTPException, status, Depends
from starlette.middleware.base import BaseHTTPMiddleware
import logging

from src.infra.rbac import (
    Role, Permission, UserContext, RBACManager, 
    RBACConfig, get_rbac_manager
)

logger = logging.getLogger(__name__)


class RBACMiddleware(BaseHTTPMiddleware):
    """Middleware to attach user context to requests"""
    
    def __init__(self, app, rbac_manager: Optional[RBACManager] = None):
        super().__init__(app)
        self.rbac_manager = rbac_manager or get_rbac_manager()
    
    async def dispatch(self, request: Request, call_next):
        """Attach user context from token to request state"""
        # Extract token from Authorization header
        auth_header = request.headers.get("authorization", "")
        
        # Try to extract and validate claims (in real implementation, verify JWT)
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            # In production, validate JWT token here
            # For now, we'll extract claims from token payload
            try:
                # This would be actual JWT validation with Firebase Admin SDK
                # claims = verify_firebase_token(token)
                # For testing, we'll extract from custom header
                claims = request.headers.get("X-User-Claims", {})
                if isinstance(claims, str):
                    import json
                    claims = json.loads(claims)
                
                user_context = UserContext.from_token_claims(claims)
                request.state.user = user_context
                request.state.rbac_manager = self.rbac_manager
            except Exception as e:
                logger.warning(f"Failed to extract user context: {e}")
                request.state.user = None
        else:
            request.state.user = None
        
        response = await call_next(request)
        return response


# Dependency for use in FastAPI route handlers
async def get_current_user(request: Request) -> UserContext:
    """Dependency to get current user from request"""
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    return user


def require_role(*roles: Role):
    """
    Decorator to require specific roles
    
    Usage:
        @router.get("/admin")
        @require_role(Role.ADMIN)
        async def admin_endpoint(user: UserContext = Depends(get_current_user)):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get request from kwargs or args
            request = kwargs.get("request") or (args[0] if args and isinstance(args[0], Request) else None)
            
            # Get user context
            user: Optional[UserContext] = None
            for arg in args:
                if isinstance(arg, UserContext):
                    user = arg
                    break
            
            if not user:
                for value in kwargs.values():
                    if isinstance(value, UserContext):
                        user = value
                        break
            
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Not authenticated"
                )
            
            # Check if user has one of required roles
            if user.role not in roles:
                logger.warning(
                    f"Access denied for user {user.user_id}: "
                    f"required roles {[r.value for r in roles]}, "
                    f"got {user.role.value}"
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Required roles: {[r.value for r in roles]}"
                )
            
            return await func(*args, **kwargs) if hasattr(func, '__await__') else func(*args, **kwargs)
        
        return wrapper
    return decorator


def require_permission(*permissions: Permission):
    """
    Decorator to require specific permissions
    
    Usage:
        @router.delete("/jobs/{job_id}")
        @require_permission(Permission.DELETE_JOB)
        async def delete_job(job_id: str, user: UserContext = Depends(get_current_user)):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get user context
            user: Optional[UserContext] = None
            for arg in args:
                if isinstance(arg, UserContext):
                    user = arg
                    break
            
            if not user:
                for value in kwargs.values():
                    if isinstance(value, UserContext):
                        user = value
                        break
            
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Not authenticated"
                )
            
            # Check if user has all required permissions
            if not user.has_all_permissions(list(permissions)):
                logger.warning(
                    f"Access denied for user {user.user_id}: "
                    f"required permissions {[p.value for p in permissions]}"
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Required permissions: {[p.value for p in permissions]}"
                )
            
            return await func(*args, **kwargs) if hasattr(func, '__await__') else func(*args, **kwargs)
        
        return wrapper
    return decorator


def require_any_permission(*permissions: Permission):
    """
    Decorator to require any of specific permissions
    
    Usage:
        @router.get("/results")
        @require_any_permission(Permission.READ_RESULTS, Permission.MANAGE_USERS)
        async def get_results(user: UserContext = Depends(get_current_user)):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get user context
            user: Optional[UserContext] = None
            for arg in args:
                if isinstance(arg, UserContext):
                    user = arg
                    break
            
            if not user:
                for value in kwargs.values():
                    if isinstance(value, UserContext):
                        user = value
                        break
            
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Not authenticated"
                )
            
            # Check if user has any of the required permissions
            if not user.has_any_permission(list(permissions)):
                logger.warning(
                    f"Access denied for user {user.user_id}: "
                    f"required any of {[p.value for p in permissions]}"
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Required any of: {[p.value for p in permissions]}"
                )
            
            return await func(*args, **kwargs) if hasattr(func, '__await__') else func(*args, **kwargs)
        
        return wrapper
    return decorator
