"""
Input validation and request size limiting configuration.
Implements protections against oversized and malformed requests.
"""

from fastapi import Request, HTTPException
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)

# Request size limits (in bytes)
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB
MAX_JSON_BODY_SIZE = 5 * 1024 * 1024   # 5 MB
MAX_FORM_BODY_SIZE = 5 * 1024 * 1024   # 5 MB

# Request timeout (in seconds)
REQUEST_TIMEOUT = 30

# Input validation rules
VALIDATION_RULES = {
    # Pipeline input
    "content": {
        "min_length": 10,
        "max_length": 10000,
        "required": True,
    },
    # Duration
    "duration_sec": {
        "min": 5,
        "max": 300,  # 5 minutes max
        "required": False,
    },
    # Preset
    "preset": {
        "allowed_values": ["quick_social", "brand_campaign", "premium_spot"],
        "required": False,
    },
    # Priority
    "priority": {
        "allowed_values": ["low", "normal", "high"],
        "required": False,
    },
    # Language
    "lang": {
        "allowed_values": ["en", "fr", "es", "de", "it", "pt", "ja", "zh", "ko"],
        "required": False,
    },
}


async def validate_request_size(request: Request) -> bool:
    """
    Validate that request content length doesn't exceed limit.
    
    Args:
        request: FastAPI Request object
        
    Returns:
        bool: True if valid, raises HTTPException if too large
        
    Raises:
        HTTPException: 413 if content too large
    """
    content_length = request.headers.get("content-length")
    
    if content_length:
        try:
            size = int(content_length)
            if size > MAX_CONTENT_LENGTH:
                logger.warning(
                    f"Request size {size} bytes exceeds limit {MAX_CONTENT_LENGTH}"
                )
                raise HTTPException(
                    status_code=413,
                    detail=f"Request body too large (max {MAX_CONTENT_LENGTH} bytes)"
                )
        except ValueError:
            logger.warning(f"Invalid Content-Length header: {content_length}")
            raise HTTPException(
                status_code=400,
                detail="Invalid Content-Length header"
            )
    
    return True


async def validate_input_field(
    field_name: str, 
    value: Any, 
    rules: dict
) -> bool:
    """
    Validate a single input field against rules.
    
    Args:
        field_name: Name of field being validated
        value: Value to validate
        rules: Validation rules for the field
        
    Returns:
        bool: True if valid
        
    Raises:
        HTTPException: 422 if validation fails
    """
    # Check required
    if rules.get("required", False) and value is None:
        raise HTTPException(
            status_code=422,
            detail=f"Field '{field_name}' is required"
        )
    
    if value is None:
        return True
    
    # String validations
    if "min_length" in rules and len(str(value)) < rules["min_length"]:
        raise HTTPException(
            status_code=422,
            detail=f"Field '{field_name}' must be at least {rules['min_length']} characters"
        )
    
    if "max_length" in rules and len(str(value)) > rules["max_length"]:
        raise HTTPException(
            status_code=422,
            detail=f"Field '{field_name}' must be at most {rules['max_length']} characters"
        )
    
    # Numeric validations
    if "min" in rules and isinstance(value, (int, float)) and value < rules["min"]:
        raise HTTPException(
            status_code=422,
            detail=f"Field '{field_name}' must be at least {rules['min']}"
        )
    
    if "max" in rules and isinstance(value, (int, float)) and value > rules["max"]:
        raise HTTPException(
            status_code=422,
            detail=f"Field '{field_name}' must be at most {rules['max']}"
        )
    
    # Allowed values
    if "allowed_values" in rules and value not in rules["allowed_values"]:
        raise HTTPException(
            status_code=422,
            detail=f"Field '{field_name}' must be one of: {', '.join(rules['allowed_values'])}"
        )
    
    logger.debug(f"Field '{field_name}' validation passed")
    return True


async def validate_request_payload(payload: dict, schema_rules: dict) -> bool:
    """
    Validate entire request payload against schema rules.
    
    Args:
        payload: Request payload dictionary
        schema_rules: Validation rules for the schema
        
    Returns:
        bool: True if all fields valid
        
    Raises:
        HTTPException: 422 if any field validation fails
    """
    for field_name, rules in schema_rules.items():
        value = payload.get(field_name)
        await validate_input_field(field_name, value, rules)
    
    return True
