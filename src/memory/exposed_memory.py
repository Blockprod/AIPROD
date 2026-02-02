"""
ExposedMemory pour AIPROD V33
Gère la mémoire exposée à l'ICC (Interface Client Collaboratif)
Basé sur exposedMemory du JSON AIPROD_V33.json
"""
from typing import Any, Dict
from src.utils.monitoring import logger

class ExposedMemory:
    """
    Gère la mémoire exposée à l'ICC avec permissions d'édition.
    Selon AIPROD_V33.json exposedMemory.
    """
    
    def __init__(self):
        self.exposed_config = {
            "production_manifest": {
                "editable": True,
                "description": "ICC: User can view and edit the creative plan before rendering"
            },
            "consistency_report": {
                "editable": False,
                "description": "ICC: User can view the semantic QA report with visual highlights"
            },
            "cost_certification": {
                "editable": False,
                "description": "ICC: User can view cost breakdown and approve/reject"
            },
            "delivery_manifest": {
                "editable": False,
                "description": "ICC: Final delivery package with download links"
            }
        }
        self._exposed_data: Dict[str, Any] = {}
    
    def expose(self, key: str, value: Any) -> bool:
        """
        Expose une donnée si elle est dans la configuration.
        
        Args:
            key (str): Clé de la donnée
            value (Any): Valeur à exposer
            
        Returns:
            bool: True si exposée avec succès
        """
        if key in self.exposed_config:
            self._exposed_data[key] = value
            logger.info(f"ExposedMemory: exposed {key}")
            return True
        return False
    
    def get_exposed(self, key: str) -> Any:
        """
        Récupère une donnée exposée.
        """
        return self._exposed_data.get(key)
    
    def update(self, key: str, value: Any) -> bool:
        """
        Met à jour une donnée si éditable.
        
        Args:
            key (str): Clé de la donnée
            value (Any): Nouvelle valeur
            
        Returns:
            bool: True si mis à jour
        """
        if key not in self.exposed_config:
            logger.warning(f"ExposedMemory: {key} not in exposed config")
            return False
        
        if not self.exposed_config[key].get("editable"):
            logger.warning(f"ExposedMemory: {key} is not editable")
            return False
        
        self._exposed_data[key] = value
        logger.info(f"ExposedMemory: updated {key}")
        return True
    
    def get_all_exposed(self) -> Dict[str, Any]:
        """
        Retourne toutes les données exposées avec métadonnées.
        """
        return {
            key: {
                "value": self._exposed_data.get(key),
                "editable": self.exposed_config[key].get("editable"),
                "description": self.exposed_config[key].get("description")
            }
            for key in self.exposed_config
        }
    
    def is_editable(self, key: str) -> bool:
        """
        Vérifie si une clé est éditable.
        """
        return self.exposed_config.get(key, {}).get("editable", False)
