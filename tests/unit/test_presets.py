"""
Tests unitaires pour le système de presets AIPROD.
"""
import pytest
from src.api.presets import (
    get_preset, get_all_presets, apply_preset_to_request,
    estimate_cost_for_preset, PresetTier, PRESETS
)


class TestPresets:
    """Tests pour le module presets."""
    
    def test_get_preset_quick_social(self):
        """Test récupération preset quick_social."""
        preset = get_preset("quick_social")
        assert preset is not None
        assert preset.name == "Quick Social"
        assert preset.pipeline_mode == "fast"
        assert preset.quality_threshold == 0.6
        assert preset.max_duration_sec == 30
        assert preset.estimated_cost == 0.30
    
    def test_get_preset_brand_campaign(self):
        """Test récupération preset brand_campaign."""
        preset = get_preset("brand_campaign")
        assert preset is not None
        assert preset.name == "Brand Campaign"
        assert preset.pipeline_mode == "full"
        assert preset.quality_threshold == 0.8
        assert preset.allow_icc is True
        assert preset.consistency_cache is True
    
    def test_get_preset_premium_spot(self):
        """Test récupération preset premium_spot."""
        preset = get_preset("premium_spot")
        assert preset is not None
        assert preset.name == "Premium Spot"
        assert preset.pipeline_mode == "full"
        assert preset.quality_threshold == 0.9
        assert preset.multi_review is True
    
    def test_get_preset_invalid(self):
        """Test preset invalide retourne None."""
        preset = get_preset("invalid_preset")
        assert preset is None
    
    def test_get_all_presets(self):
        """Test récupération de tous les presets."""
        all_presets = get_all_presets()
        assert len(all_presets) == 3
        assert "quick_social" in all_presets
        assert "brand_campaign" in all_presets
        assert "premium_spot" in all_presets
        
        # Vérifier structure
        for name, config in all_presets.items():
            assert "name" in config
            assert "description" in config
            assert "pipeline_mode" in config
            assert "quality_threshold" in config
            assert "estimated_cost" in config
    
    def test_apply_preset_to_request(self):
        """Test application d'un preset à une requête."""
        request = {"content": "Test video", "priority": "low"}
        enriched = apply_preset_to_request(request, "brand_campaign")
        
        assert enriched["_preset"] == "brand_campaign"
        assert "_preset_config" in enriched
        assert enriched["_preset_config"]["pipeline_mode"] == "full"
        assert enriched["_preset_config"]["quality_threshold"] == 0.8
        assert enriched["priority"] == "medium"  # Override par le preset
    
    def test_apply_preset_preserves_priority(self):
        """Test que la priorité utilisateur est préservée si != low."""
        request = {"content": "Test video", "priority": "high"}
        enriched = apply_preset_to_request(request, "quick_social")
        
        assert enriched["priority"] == "high"  # Préservée
    
    def test_apply_invalid_preset(self):
        """Test application preset invalide retourne requête originale."""
        request = {"content": "Test video"}
        enriched = apply_preset_to_request(request, "invalid")
        
        assert enriched == request
        assert "_preset" not in enriched
    
    def test_estimate_cost_quick_social(self):
        """Test estimation coût pour quick_social."""
        estimate = estimate_cost_for_preset("quick_social", 30)
        
        assert estimate["preset"] == "quick_social"
        assert estimate["duration_sec"] == 30
        assert estimate["aiprod_optimized"] == 0.25  # 30s * $0.50/min / 60
        assert estimate["runway_alone"] == 1.25  # 30s * $2.50/min / 60
        assert estimate["savings"] == 1.0
        assert estimate["savings_percent"] == 80.0
        assert estimate["quality_guarantee"] == 0.6
    
    def test_estimate_cost_brand_campaign(self):
        """Test estimation coût pour brand_campaign."""
        estimate = estimate_cost_for_preset("brand_campaign", 60)
        
        assert estimate["preset"] == "brand_campaign"
        assert estimate["duration_sec"] == 60
        assert estimate["aiprod_optimized"] == 0.95  # 60s * $0.95/min / 60
        assert estimate["runway_alone"] == 2.5  # 60s * $2.50/min / 60
        assert estimate["savings"] == 1.55
        assert estimate["quality_guarantee"] == 0.8
    
    def test_estimate_cost_invalid_preset(self):
        """Test estimation coût pour preset invalide."""
        estimate = estimate_cost_for_preset("invalid", 30)
        
        assert "error" in estimate
        assert estimate["error"] == "Preset not found"
    
    def test_preset_tier_enum(self):
        """Test enum PresetTier."""
        assert PresetTier.QUICK_SOCIAL.value == "quick_social"
        assert PresetTier.BRAND_CAMPAIGN.value == "brand_campaign"
        assert PresetTier.PREMIUM_SPOT.value == "premium_spot"
    
    def test_presets_have_correct_structure(self):
        """Test structure des presets."""
        for name, preset in PRESETS.items():
            assert hasattr(preset, 'name')
            assert hasattr(preset, 'description')
            assert hasattr(preset, 'pipeline_mode')
            assert hasattr(preset, 'quality_threshold')
            assert hasattr(preset, 'max_duration_sec')
            assert hasattr(preset, 'max_cost_per_minute')
            assert hasattr(preset, 'allow_icc')
            assert hasattr(preset, 'consistency_cache')
            assert hasattr(preset, 'multi_review')
            assert hasattr(preset, 'priority')
            assert hasattr(preset, 'estimated_cost')
            
            # Vérifier cohérence
            assert 0 <= preset.quality_threshold <= 1
            assert preset.max_duration_sec > 0
            assert preset.max_cost_per_minute > 0
