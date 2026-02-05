"""
SchemaValidator pour AIPROD
Valide les données contre le memorySchema défini dans AIPROD.json
"""
from typing import Any, Dict, List
from pydantic import BaseModel, ValidationError, ConfigDict
from src.utils.monitoring import logger

class SchemaValidator:
    """
    Validateur de schéma pour la mémoire AIPROD.
    Basé sur le memorySchema du JSON.
    """
    
    def __init__(self):
        self.required_fields = {
            "sanitized_input": {"required": True},
            "production_manifest": {"required": True, "condition": "pipeline_mode == 'full'"},
            "consistency_markers": {"required": True, "condition": "pipeline_mode == 'full'"},
            "prompt_bundle": {"required": True},
            "optimized_backend_selection": {"required": True},
            "cost_certification": {"required": True},
            "generated_assets": {"required": True},
            "technical_validation_report": {"required": True},
            "consistency_report": {"required": True},
            "final_approval": {"required": True},
            "delivery_manifest": {"required": True}
        }
    
    def validate_field(self, field_name: str, value: Any, memory: Dict[str, Any]) -> bool:
        """
        Valide un champ spécifique.
        
        Args:
            field_name (str): Nom du champ
            value (Any): Valeur à valider
            memory (Dict[str, Any]): Mémoire complète pour évaluer les conditions
            
        Returns:
            bool: True si valide
        """
        if field_name not in self.required_fields:
            return True
        
        field_config = self.required_fields[field_name]
        
        # Vérifie la condition si présente
        if "condition" in field_config:
            condition = field_config["condition"]
            if condition == "pipeline_mode == 'full'" and memory.get("pipeline_mode") != "full":
                return True  # Pas requis pour ce mode
        
        # Vérifie que le champ n'est pas None
        if field_config.get("required") and value is None:
            logger.error(f"SchemaValidator: required field {field_name} is None")
            return False
        
        return True
    
    def validate_memory(self, memory: Dict[str, Any]) -> bool:
        """
        Valide la mémoire complète.
        
        Args:
            memory (Dict[str, Any]): Mémoire à valider
            
        Returns:
            bool: True si toute la mémoire est valide
        """
        errors = []
        for field_name, field_config in self.required_fields.items():
            value = memory.get(field_name)
            if not self.validate_field(field_name, value, memory):
                errors.append(f"Field {field_name} validation failed")
        
        if errors:
            for err in errors:
                logger.error(f"SchemaValidator: {err}")
            return False
        
        logger.info("SchemaValidator: memory validation passed")
        return True
