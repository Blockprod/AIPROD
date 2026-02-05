"""
InputSanitizer pour AIPROD
Nettoyage et prétraitement des entrées utilisateur.
"""
from typing import Any, Dict
from pydantic import BaseModel, ValidationError, ConfigDict
from src.utils.monitoring import logger

class InputSchema(BaseModel):
    """Schéma de validation des entrées utilisateur."""
    content: str
    priority: str = "low"
    lang: str = "en"
    
    model_config = ConfigDict(extra="allow")

class InputSanitizer:
    """
    Sanitize et prétraite les entrées utilisateur du pipeline.
    """
    def __init__(self):
        pass

    def sanitize(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Nettoie et valide les entrées utilisateur.
        Args:
            user_input (Dict[str, Any]): Entrées brutes de l'utilisateur.
        Returns:
            Dict[str, Any]): Entrées sanitizées et validées.
        """
        logger.info(f"InputSanitizer: start sanitization")
        try:
            # Validation avec Pydantic
            validated = InputSchema(**user_input)
            sanitized = validated.model_dump()
            # Nettoyage supplémentaire (trim, lowercase, etc)
            sanitized["content"] = sanitized.get("content", "").strip()
            sanitized["priority"] = sanitized.get("priority", "low").lower()
            sanitized["lang"] = sanitized.get("lang", "en").lower()
            logger.info(f"InputSanitizer: sanitized={sanitized}")
            return sanitized
        except ValidationError as e:
            logger.error(f"InputSanitizer: validation error {e}")
            raise
