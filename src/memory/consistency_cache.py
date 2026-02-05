"""
AIPROD - Consistency Cache avec stockage GCS
Cache de cohérence pour maintenir le style/caractères entre projets d'une même marque.
TTL: 7 jours (168 heures)
"""
import os
import json
import hashlib
from typing import Any, Dict, Optional
from datetime import datetime, timedelta
from src.utils.monitoring import logger

# Essayer d'importer google.cloud.storage, fallback en mode mock
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    logger.warning("google-cloud-storage not installed, using local cache only")


class ConsistencyCache:
    """
    Cache de cohérence pour maintenir les marqueurs de style entre projets.
    Supporte le stockage local et GCS pour persistance.
    
    Features:
    - TTL de 7 jours (168 heures)
    - Stockage GCS pour persistance cross-instances
    - Réutilisation entre jobs d'un même brand_id
    - Hash de style pour détection de similarité
    """
    
    def __init__(self, ttl_hours: int = 168, bucket_name: Optional[str] = None):
        """
        Initialise le cache de cohérence.
        
        Args:
            ttl_hours: Durée de vie du cache en heures (défaut: 168 = 7 jours)
            bucket_name: Nom du bucket GCS (optionnel, utilise env var sinon)
        """
        self.ttl_hours = ttl_hours
        self.bucket_name = bucket_name or os.getenv("GCS_BUCKET_NAME", "aiprod-484120-assets")
        self._local_cache: Dict[str, Dict[str, Any]] = {}
        self._gcs_client = None
        self._bucket = None
        
        # Initialiser GCS si disponible
        if GCS_AVAILABLE:
            try:
                self._gcs_client = storage.Client()
                self._bucket = self._gcs_client.bucket(self.bucket_name)
                logger.info(f"ConsistencyCache: GCS initialized with bucket {self.bucket_name}")
            except Exception as e:
                logger.warning(f"ConsistencyCache: GCS init failed: {e}, using local only")
    
    def _generate_cache_key(self, brand_id: str, style_hash: str) -> str:
        """Génère une clé de cache unique pour brand+style."""
        combined = f"{brand_id}:{style_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def _generate_style_hash(self, content: str, style_markers: Optional[Dict] = None) -> str:
        """
        Génère un hash de style pour détecter les similarités.
        
        Args:
            content: Contenu de base
            style_markers: Marqueurs de style optionnels
        """
        # Extraire les mots-clés principaux pour le hash
        words = content.lower().split()[:10]  # 10 premiers mots
        base = " ".join(sorted(words))
        
        if style_markers:
            base += json.dumps(style_markers, sort_keys=True)
        
        return hashlib.md5(base.encode()).hexdigest()[:8]
    
    def _is_expired(self, timestamp: str) -> bool:
        """Vérifie si un cache est expiré."""
        try:
            cached_time = datetime.fromisoformat(timestamp)
            expiry_time = cached_time + timedelta(hours=self.ttl_hours)
            return datetime.now() > expiry_time
        except:
            return True
    
    def get_consistency_markers(
        self, 
        brand_id: str, 
        content: str,
        style_markers: Optional[Dict] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Récupère les marqueurs de cohérence pour une marque.
        
        Args:
            brand_id: Identifiant de la marque
            content: Contenu pour hash de style
            style_markers: Marqueurs de style optionnels
            
        Returns:
            Dict avec les consistency_markers ou None si pas en cache
        """
        style_hash = self._generate_style_hash(content, style_markers)
        cache_key = self._generate_cache_key(brand_id, style_hash)
        
        # 1. Chercher en cache local
        if cache_key in self._local_cache:
            cached = self._local_cache[cache_key]
            if not self._is_expired(cached.get("timestamp", "")):
                logger.info(f"ConsistencyCache HIT (local): brand={brand_id}")
                return cached.get("markers")
            else:
                del self._local_cache[cache_key]
        
        # 2. Chercher en GCS
        if self._bucket:
            try:
                blob_path = f"cache/{brand_id}/consistency_{cache_key}.json"
                blob = self._bucket.blob(blob_path)
                
                if blob.exists():
                    content_bytes = blob.download_as_bytes()
                    cached = json.loads(content_bytes)
                    
                    if not self._is_expired(cached.get("timestamp", "")):
                        # Stocker en local pour accès rapide
                        self._local_cache[cache_key] = cached
                        logger.info(f"ConsistencyCache HIT (GCS): brand={brand_id}")
                        return cached.get("markers")
                    else:
                        # Supprimer cache expiré
                        blob.delete()
                        logger.info(f"ConsistencyCache EXPIRED: brand={brand_id}")
            except Exception as e:
                logger.warning(f"ConsistencyCache GCS read error: {e}")
        
        logger.info(f"ConsistencyCache MISS: brand={brand_id}")
        return None
    
    def set_consistency_markers(
        self,
        brand_id: str,
        content: str,
        markers: Dict[str, Any],
        style_markers: Optional[Dict] = None
    ) -> bool:
        """
        Stocke les marqueurs de cohérence pour une marque.
        
        Args:
            brand_id: Identifiant de la marque
            content: Contenu pour hash de style
            markers: Marqueurs de cohérence à stocker
            style_markers: Marqueurs de style optionnels
            
        Returns:
            True si stockage réussi
        """
        style_hash = self._generate_style_hash(content, style_markers)
        cache_key = self._generate_cache_key(brand_id, style_hash)
        
        cache_data = {
            "brand_id": brand_id,
            "style_hash": style_hash,
            "markers": markers,
            "timestamp": datetime.now().isoformat(),
            "ttl_hours": self.ttl_hours
        }
        
        # 1. Stocker en local
        self._local_cache[cache_key] = cache_data
        
        # 2. Stocker en GCS pour persistance
        if self._bucket:
            try:
                blob_path = f"cache/{brand_id}/consistency_{cache_key}.json"
                blob = self._bucket.blob(blob_path)
                blob.upload_from_string(
                    json.dumps(cache_data, indent=2),
                    content_type="application/json"
                )
                logger.info(f"ConsistencyCache SET (GCS): brand={brand_id}, key={cache_key}")
                return True
            except Exception as e:
                logger.warning(f"ConsistencyCache GCS write error: {e}")
                return False
        
        logger.info(f"ConsistencyCache SET (local only): brand={brand_id}")
        return True
    
    def invalidate_brand_cache(self, brand_id: str) -> int:
        """
        Invalide tout le cache d'une marque.
        
        Args:
            brand_id: Identifiant de la marque
            
        Returns:
            Nombre d'entrées invalidées
        """
        count = 0
        
        # 1. Invalider local
        keys_to_delete = [k for k, v in self._local_cache.items() 
                         if v.get("brand_id") == brand_id]
        for key in keys_to_delete:
            del self._local_cache[key]
            count += 1
        
        # 2. Invalider GCS
        if self._bucket:
            try:
                blobs = list(self._bucket.list_blobs(prefix=f"cache/{brand_id}/"))
                for blob in blobs:
                    blob.delete()
                    count += 1
                logger.info(f"ConsistencyCache INVALIDATE: brand={brand_id}, count={count}")
            except Exception as e:
                logger.warning(f"ConsistencyCache GCS invalidate error: {e}")
        
        return count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Retourne des statistiques sur le cache."""
        return {
            "local_entries": len(self._local_cache),
            "ttl_hours": self.ttl_hours,
            "gcs_enabled": self._bucket is not None,
            "bucket_name": self.bucket_name if self._bucket else None
        }


# Instance globale pour réutilisation
_consistency_cache: Optional[ConsistencyCache] = None

def get_consistency_cache() -> ConsistencyCache:
    """Retourne l'instance globale du cache de cohérence."""
    global _consistency_cache
    if _consistency_cache is None:
        _consistency_cache = ConsistencyCache()
    return _consistency_cache
