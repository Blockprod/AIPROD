"""
Tests unitaires pour le Consistency Cache AIPROD.
"""
import pytest
from unittest.mock import MagicMock, patch
from src.memory.consistency_cache import ConsistencyCache, get_consistency_cache


class TestConsistencyCache:
    """Tests pour le module consistency_cache."""
    
    def test_init_default(self):
        """Test initialisation par défaut."""
        cache = ConsistencyCache(ttl_hours=168)
        assert cache.ttl_hours == 168
        assert cache._local_cache == {}
    
    def test_generate_style_hash(self):
        """Test génération du hash de style."""
        cache = ConsistencyCache()
        
        hash1 = cache._generate_style_hash("Test content")
        hash2 = cache._generate_style_hash("Test content")
        hash3 = cache._generate_style_hash("Different content")
        
        # Même contenu = même hash
        assert hash1 == hash2
        # Contenu différent = hash différent
        assert hash1 != hash3
    
    def test_generate_style_hash_with_markers(self):
        """Test génération hash avec marqueurs de style."""
        cache = ConsistencyCache()
        
        markers1 = {"style": "modern"}
        markers2 = {"style": "classic"}
        
        hash1 = cache._generate_style_hash("Test", markers1)
        hash2 = cache._generate_style_hash("Test", markers2)
        
        # Marqueurs différents = hash différent
        assert hash1 != hash2
    
    def test_generate_cache_key(self):
        """Test génération de la clé de cache."""
        cache = ConsistencyCache()
        
        key1 = cache._generate_cache_key("brand_a", "hash_1")
        key2 = cache._generate_cache_key("brand_a", "hash_1")
        key3 = cache._generate_cache_key("brand_b", "hash_1")
        
        # Même brand + hash = même clé
        assert key1 == key2
        # Brand différente = clé différente
        assert key1 != key3
        # Clé de longueur fixe
        assert len(key1) == 16
    
    def test_set_get_consistency_markers_local(self):
        """Test stockage/récupération local sans GCS."""
        cache = ConsistencyCache(ttl_hours=168)
        cache._bucket = None  # Désactiver GCS
        
        markers = {
            "style": "modern",
            "color_palette": ["#FF0000", "#00FF00"],
            "mood": "dynamic"
        }
        
        # Stocker
        result = cache.set_consistency_markers(
            brand_id="test_brand",
            content="Test video content",
            markers=markers
        )
        assert result is True
        
        # Récupérer
        retrieved = cache.get_consistency_markers(
            brand_id="test_brand",
            content="Test video content"
        )
        assert retrieved is not None
        assert retrieved["style"] == "modern"
        assert retrieved["mood"] == "dynamic"
    
    def test_cache_miss(self):
        """Test cache miss retourne None."""
        cache = ConsistencyCache()
        cache._bucket = None
        
        result = cache.get_consistency_markers(
            brand_id="nonexistent",
            content="Some content"
        )
        assert result is None
    
    def test_different_brands_isolated(self):
        """Test que les caches de différentes marques sont isolés."""
        cache = ConsistencyCache()
        cache._bucket = None
        
        markers_a = {"style": "modern"}
        markers_b = {"style": "classic"}
        
        # Stocker pour brand_a
        cache.set_consistency_markers("brand_a", "content", markers_a)
        # Stocker pour brand_b
        cache.set_consistency_markers("brand_b", "content", markers_b)
        
        # Vérifier isolation
        result_a = cache.get_consistency_markers("brand_a", "content")
        result_b = cache.get_consistency_markers("brand_b", "content")
        
        assert result_a is not None, "Cache miss for brand_a"
        assert result_b is not None, "Cache miss for brand_b"
        assert result_a["style"] == "modern"
        assert result_b["style"] == "classic"
    
    def test_invalidate_brand_cache(self):
        """Test invalidation du cache d'une marque."""
        cache = ConsistencyCache()
        cache._bucket = None
        
        # Stocker plusieurs entrées
        cache.set_consistency_markers("brand_x", "content1", {"key": "val1"})
        cache.set_consistency_markers("brand_x", "content2", {"key": "val2"})
        cache.set_consistency_markers("brand_y", "content1", {"key": "val3"})
        
        # Invalider brand_x
        count = cache.invalidate_brand_cache("brand_x")
        assert count >= 2
        
        # brand_x invalidée
        assert cache.get_consistency_markers("brand_x", "content1") is None
        # brand_y toujours valide
        assert cache.get_consistency_markers("brand_y", "content1") is not None
    
    def test_get_cache_stats(self):
        """Test statistiques du cache."""
        cache = ConsistencyCache()
        cache._bucket = None
        
        # Ajouter des entrées
        cache.set_consistency_markers("brand", "c1", {"k": "v1"})
        cache.set_consistency_markers("brand", "c2", {"k": "v2"})
        
        stats = cache.get_cache_stats()
        
        assert stats["local_entries"] == 2
        assert stats["ttl_hours"] == 168
        assert stats["gcs_enabled"] is False
    
    def test_is_expired(self):
        """Test détection expiration."""
        cache = ConsistencyCache(ttl_hours=1)
        
        from datetime import datetime, timedelta
        
        # Timestamp récent = pas expiré
        recent = datetime.now().isoformat()
        assert cache._is_expired(recent) is False
        
        # Timestamp ancien = expiré
        old = (datetime.now() - timedelta(hours=2)).isoformat()
        assert cache._is_expired(old) is True
        
        # Timestamp invalide = expiré
        assert cache._is_expired("invalid") is True
    
    def test_global_cache_singleton(self):
        """Test que get_consistency_cache retourne un singleton."""
        cache1 = get_consistency_cache()
        cache2 = get_consistency_cache()
        
        assert cache1 is cache2
    
    def test_style_hash_deterministic(self):
        """Test que le hash de style est déterministe."""
        cache = ConsistencyCache()
        
        # Même contenu plusieurs fois
        hashes = [cache._generate_style_hash("Consistent content") for _ in range(5)]
        
        assert all(h == hashes[0] for h in hashes)
    
    def test_cache_with_complex_markers(self):
        """Test cache avec marqueurs complexes."""
        cache = ConsistencyCache()
        cache._bucket = None
        
        complex_markers = {
            "style": "cinematic",
            "color_palette": ["#FF5733", "#33FF57", "#3357FF", "#F033FF"],
            "mood": "epic",
            "lighting": "dramatic",
            "character_style": "photorealistic",
            "camera_movements": ["dolly", "pan", "zoom"],
            "audio_style": {
                "music": "orchestral",
                "sfx": "immersive"
            }
        }
        
        cache.set_consistency_markers("premium_brand", "premium content", complex_markers)
        retrieved = cache.get_consistency_markers("premium_brand", "premium content")
        
        assert retrieved is not None
        assert retrieved["style"] == "cinematic"
        assert len(retrieved["color_palette"]) == 4
        assert retrieved["audio_style"]["music"] == "orchestral"
