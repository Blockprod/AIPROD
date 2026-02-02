# memory_manager.py
# Gestionnaire de mémoire pour AIPROD V33, basé sur le memorySchema du JSON

from typing import Any, Dict, Optional
from pydantic import BaseModel, ValidationError, ConfigDict, RootModel
from src.utils.cache_manager import CacheManager
from src.utils.monitoring import logger
from src.memory.consistency_cache import ConsistencyCache, get_consistency_cache

# Schéma mémoire basé sur la section 'memorySchema' du JSON
class MemorySchema(BaseModel):
    sanitized_input: Any
    production_manifest: Optional[Any] = None
    consistency_markers: Optional[Any] = None
    prompt_bundle: Any
    optimized_backend_selection: Any
    cost_certification: Any
    generated_assets: Any
    technical_validation_report: Any
    consistency_report: Any
    final_approval: Any
    delivery_manifest: Any

    model_config = ConfigDict(extra="allow")

class MemoryManager:
    """
    Gère la mémoire partagée entre les agents et fonctions du pipeline AIPROD V33.
    Valide et expose les données selon le memorySchema.
    Gère la mémoire exposée (exposed_memory) et fournit une interface ICC.
    Utilise un cache de cohérence avec TTL (168h).

    Attributes:
        _memory (Dict[str, Any]): Mémoire principale du pipeline.
        _exposed_memory (Dict[str, Any]): Mémoire exposée à l'ICC.
        cache (CacheManager): Cache de cohérence avec TTL.
        consistency_cache (ConsistencyCache): Cache de cohérence marque (7 jours).
    """
    def __init__(self):
        self._memory: Dict[str, Any] = {}
        self._exposed_memory: Dict[str, Any] = {}
        self.cache = CacheManager(ttl_hours=168)
        self.consistency_cache = get_consistency_cache()
        self._current_brand_id: Optional[str] = None

    def write(self, key: str, value: Any, expose: bool = False, cache: bool = False) -> None:
        """
        Écrit une valeur en mémoire. Si expose=True, ajoute à la mémoire exposée.
        Si cache=True, stocke la valeur dans le cache de cohérence.

        Args:
            key (str): Clé de la donnée.
            value (Any): Valeur à stocker.
            expose (bool): Expose la donnée à l'ICC.
            cache (bool): Stocke la donnée dans le cache de cohérence.
        """
        self._memory[key] = value
        logger.info(f"Write: key={key}, expose={expose}, cache={cache}")
        if expose:
            self._exposed_memory[key] = value
            logger.info(f"Expose: key={key}")
        if cache:
            self.cache.set(key, value)
            logger.info(f"Cache set: key={key}")

    def read(self, key: str, exposed: bool = False, cache: bool = False) -> Any:
        """
        Lit une valeur depuis la mémoire, la mémoire exposée ou le cache.

        Args:
            key (str): Clé de la donnée.
            exposed (bool): Cherche dans la mémoire exposée.
            cache (bool): Cherche dans le cache de cohérence.

        Returns:
            Any: Valeur trouvée ou None.
        """
        if cache:
            val = self.cache.get(key)
            logger.info(f"Cache read: key={key}, value={val}")
            return val
        if exposed:
            val = self._exposed_memory.get(key)
            logger.info(f"Read exposed: key={key}, value={val}")
            return val
        val = self._memory.get(key)
        logger.info(f"Read: key={key}, value={val}")
        return val

    def set(self, key: str, value: Any, cache: bool = False) -> None:
        """
        Alias de write.
        """
        self.write(key, value, cache=cache)

    def get(self, key: str, cache: bool = False) -> Any:
        """
        Alias de read.
        """
        return self.read(key, cache=cache)

    def validate_data(self) -> bool:
        """
        Vérifie les contraintes métier du schéma mémoire (ex: scores entre 0 et 1, coût, etc).
        Retourne True si tout est conforme, False sinon.
        """
        data = self._memory
        errors = []
        # Vérification du score de complexité
        complexity = data.get('complexity_score')
        if complexity is not None:
            if not (0 <= complexity <= 1):
                errors.append(f"complexity_score hors bornes : {complexity}")
        # Vérification du coût (exemple)
        cost = data.get('cost_certification')
        if cost is not None:
            try:
                cost_val = float(cost)
                if cost_val < 0:
                    errors.append(f"cost_certification négatif : {cost}")
            except Exception:
                errors.append(f"cost_certification non numérique : {cost}")
        # Ajoute ici d'autres règles métier selon le schéma V33
        # ...
        if errors:
            for err in errors:
                logger.error(f"Validation data error: {err}")
            return False
        logger.info("Validation data OK")
        return True

    def validate(self) -> bool:
        """
        Valide la mémoire courante selon le schéma.
        """
        try:
            MemorySchema(**self._memory)
            logger.info("Memory schema validation OK")
            return True
        except ValidationError as e:
            logger.error(f"Memory validation error: {e}")
            return False

    def export(self) -> Dict[str, Any]:
        """
        Exporte la mémoire courante.
        """
        logger.info("Export memory")
        return dict(self._memory)

    def export_exposed(self) -> Dict[str, Any]:
        """
        Exporte la mémoire exposée.
        """
        logger.info("Export exposed memory")
        return dict(self._exposed_memory)

    def clear(self) -> None:
        """
        Efface la mémoire et le cache.
        """
        logger.info("Clear memory and cache")
        self._memory.clear()
        self._exposed_memory.clear()
        self.cache.clear()

    def get_icc_data(self) -> Dict[str, Any]:
        """
        Retourne les données ICC (interface client) extraites de la mémoire exposée.
        À adapter selon la structure ICC réelle, ici on expose tout.
        """
        logger.info("Get ICC data")
        return self.export_exposed()

    # ========================================
    # PHASE 1.3: CONSISTENCY CACHE (7 jours)
    # ========================================
    
    def set_brand_id(self, brand_id: str) -> None:
        """
        Définit l'identifiant de marque pour le job courant.
        Permet la réutilisation du cache de cohérence entre jobs.
        
        Args:
            brand_id: Identifiant unique de la marque
        """
        self._current_brand_id = brand_id
        self._memory["brand_id"] = brand_id
        logger.info(f"MemoryManager: brand_id set to {brand_id}")
    
    def get_cached_consistency_markers(
        self, 
        content: str,
        style_markers: Optional[Dict] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Récupère les marqueurs de cohérence depuis le cache.
        Utilise le brand_id courant pour la recherche.
        
        Args:
            content: Contenu pour générer le hash de style
            style_markers: Marqueurs de style optionnels
            
        Returns:
            Dict avec consistency_markers si trouvé en cache, None sinon
        """
        brand_id = self._current_brand_id or "default"
        markers = self.consistency_cache.get_consistency_markers(
            brand_id=brand_id,
            content=content,
            style_markers=style_markers
        )
        
        if markers:
            # Stocker en mémoire pour accès rapide
            self._memory["consistency_markers"] = markers
            self._memory["consistency_cache_hit"] = True
            logger.info(f"MemoryManager: consistency markers CACHE HIT for brand={brand_id}")
        else:
            self._memory["consistency_cache_hit"] = False
            
        return markers
    
    def store_consistency_markers(
        self,
        content: str,
        markers: Dict[str, Any],
        style_markers: Optional[Dict] = None
    ) -> bool:
        """
        Stocke les marqueurs de cohérence dans le cache (local + GCS).
        Persiste pendant 7 jours pour réutilisation entre jobs.
        
        Args:
            content: Contenu pour générer le hash de style
            markers: Marqueurs de cohérence à stocker
            style_markers: Marqueurs de style optionnels
            
        Returns:
            True si stockage réussi
        """
        brand_id = self._current_brand_id or "default"
        success = self.consistency_cache.set_consistency_markers(
            brand_id=brand_id,
            content=content,
            markers=markers,
            style_markers=style_markers
        )
        
        if success:
            self._memory["consistency_markers"] = markers
            logger.info(f"MemoryManager: consistency markers STORED for brand={brand_id}")
            
        return success
    
    def get_consistency_cache_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache de cohérence."""
        return self.consistency_cache.get_cache_stats()

