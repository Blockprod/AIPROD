"""
CORS (Cross-Origin Resource Sharing) security configuration.
Implements strict CORS policies for production.
"""

import os
from typing import List

# Get allowed origins from environment
ALLOWED_ORIGINS = os.getenv(
    "CORS_ALLOWED_ORIGINS",
    "https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app,https://aiprod-dashboard.example.com"
).split(",")

# Production domains only
ALLOWED_ORIGINS = [origin.strip() for origin in ALLOWED_ORIGINS if origin.strip()]

# Ensure no wildcards in production
if "*" in ALLOWED_ORIGINS:
    import warnings
    warnings.warn(
        "⚠️  WARNING: Wildcard '*' in CORS_ALLOWED_ORIGINS detected! "
        "This is a security risk in production. Remove it immediately."
    )

CORS_CONFIG = {
    "allow_origins": ALLOWED_ORIGINS,
    "allow_credentials": True,
    "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "allow_headers": [
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "X-CSRF-Token",
        "Accept",
    ],
    "expose_headers": [
        "Content-Length",
        "Content-Range",
        "X-Total-Count",
        "X-Request-ID",
    ],
    "max_age": 600,  # 10 minutes max-age for preflight cache
}

# Security headers
SECURITY_HEADERS = {
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": (
        "geolocation=(), microphone=(), camera=(), payment=(), usb=(), "
        "magnetometer=(), gyroscope=(), accelerometer=()"
    ),
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self' data:; "
        "connect-src 'self' https:"
    ),
}
