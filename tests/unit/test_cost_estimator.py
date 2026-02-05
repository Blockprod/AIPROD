"""
Tests unitaires pour l'estimateur de coûts AIPROD.
"""
import pytest
from src.api.cost_estimator import (
    estimate_gemini_cost, estimate_runway_cost, estimate_gcs_cost,
    estimate_cloud_run_cost, get_full_cost_estimate, get_job_actual_costs,
    PRICING
)


class TestCostEstimator:
    """Tests pour le module cost_estimator."""
    
    def test_estimate_gemini_cost_low(self):
        """Test coût Gemini complexité low."""
        cost = estimate_gemini_cost("low")
        assert cost > 0
        assert cost < 0.01  # Gemini est très peu cher
    
    def test_estimate_gemini_cost_standard(self):
        """Test coût Gemini complexité standard."""
        cost = estimate_gemini_cost("standard")
        low_cost = estimate_gemini_cost("low")
        assert cost > low_cost  # Standard > low
    
    def test_estimate_gemini_cost_high(self):
        """Test coût Gemini complexité high."""
        cost = estimate_gemini_cost("high")
        standard_cost = estimate_gemini_cost("standard")
        assert cost > standard_cost  # High > standard
    
    def test_estimate_runway_cost_full(self):
        """Test coût Runway mode full (gen3)."""
        cost = estimate_runway_cost(30, "full")
        assert cost > 0
        # 30s * $0.02/s + $0.05 image = $0.65
        assert 0.5 < cost < 1.0
    
    def test_estimate_runway_cost_fast(self):
        """Test coût Runway mode fast (gen4_turbo)."""
        cost = estimate_runway_cost(30, "fast")
        full_cost = estimate_runway_cost(30, "full")
        assert cost < full_cost  # Fast moins cher
        # 30s * $0.008/s + $0.05 = $0.29
        assert 0.2 < cost < 0.5
    
    def test_estimate_runway_cost_scales_with_duration(self):
        """Test que le coût Runway augmente avec la durée."""
        cost_30s = estimate_runway_cost(30, "full")
        cost_60s = estimate_runway_cost(60, "full")
        assert cost_60s > cost_30s
    
    def test_estimate_gcs_cost(self):
        """Test coût GCS storage/egress."""
        cost = estimate_gcs_cost(30)
        assert cost >= 0
        assert cost < 0.1  # GCS très peu cher
    
    def test_estimate_cloud_run_cost(self):
        """Test coût Cloud Run compute."""
        cost = estimate_cloud_run_cost(60)
        assert cost >= 0
        assert cost < 0.01  # Cloud Run peu cher
    
    def test_get_full_cost_estimate_basic(self):
        """Test estimation complète basique."""
        estimate = get_full_cost_estimate(
            content="Test video",
            duration_sec=30
        )
        
        assert "aiprod_optimized" in estimate
        assert "runway_alone" in estimate
        assert "savings" in estimate
        assert "savings_percent" in estimate
        assert "breakdown" in estimate
        assert "competitors" in estimate
        assert "value_proposition" in estimate
        
        # Vérifier cohérence
        assert estimate["savings"] == round(
            estimate["runway_alone"] - estimate["aiprod_optimized"], 2
        )
    
    def test_get_full_cost_estimate_quick_social(self):
        """Test estimation avec preset quick_social."""
        estimate = get_full_cost_estimate(
            content="Social video",
            duration_sec=30,
            preset="quick_social"
        )
        
        assert estimate["preset"] == "quick_social"
        assert estimate["quality_guarantee"] == 0.6
        assert estimate["complexity"] == "low"
        assert "gen4_turbo" in estimate["backend_selected"]
    
    def test_get_full_cost_estimate_brand_campaign(self):
        """Test estimation avec preset brand_campaign."""
        estimate = get_full_cost_estimate(
            content="Brand video",
            duration_sec=60,
            preset="brand_campaign"
        )
        
        assert estimate["preset"] == "brand_campaign"
        assert estimate["quality_guarantee"] == 0.8
    
    def test_get_full_cost_estimate_premium_spot(self):
        """Test estimation avec preset premium_spot."""
        estimate = get_full_cost_estimate(
            content="Premium video",
            duration_sec=30,
            preset="premium_spot"
        )
        
        assert estimate["preset"] == "premium_spot"
        assert estimate["quality_guarantee"] == 0.9
        assert estimate["complexity"] == "high"
    
    def test_get_full_cost_estimate_breakdown(self):
        """Test breakdown des coûts."""
        estimate = get_full_cost_estimate(
            content="Test",
            duration_sec=30
        )
        
        breakdown = estimate["breakdown"]
        assert "gemini_api" in breakdown
        assert "runway_api" in breakdown
        assert "gcs_storage" in breakdown
        assert "cloud_run" in breakdown
        
        # Somme du breakdown = total
        total_breakdown = sum(breakdown.values())
        assert abs(total_breakdown - estimate["aiprod_optimized"]) < 0.05
    
    def test_get_full_cost_estimate_competitors(self):
        """Test comparaison concurrents."""
        estimate = get_full_cost_estimate(
            content="Test",
            duration_sec=60
        )
        
        competitors = estimate["competitors"]
        assert "runway_direct" in competitors
        assert "synthesia" in competitors
        assert "pictory" in competitors
        assert "heygen" in competitors
        
        # Vérifier que AIPROD est moins cher
        assert estimate["aiprod_optimized"] < competitors["runway_direct"]
    
    def test_get_full_cost_estimate_savings_positive(self):
        """Test que les économies sont positives."""
        estimate = get_full_cost_estimate(
            content="Test",
            duration_sec=30
        )
        
        assert estimate["savings"] > 0
        assert estimate["savings_percent"] > 0
    
    def test_get_job_actual_costs(self):
        """Test calcul coûts réels d'un job."""
        job_data = {
            "render": {
                "status": "rendered",
                "duration_seconds": 5
            },
            "_cost_estimate": 1.50
        }
        
        costs = get_job_actual_costs(job_data)
        
        assert "estimated" in costs
        assert "actual" in costs
        assert "variance" in costs
        assert "variance_percent" in costs
        assert "breakdown" in costs
        assert "within_budget" in costs
        
        assert costs["estimated"] == 1.50
    
    def test_get_job_actual_costs_breakdown(self):
        """Test breakdown coûts réels."""
        job_data = {
            "render": {"duration_seconds": 10},
            "_cost_estimate": 1.00
        }
        
        costs = get_job_actual_costs(job_data)
        breakdown = costs["breakdown"]
        
        assert "gemini" in breakdown
        assert "runway" in breakdown
        assert "storage" in breakdown
        assert "compute" in breakdown
    
    def test_get_job_actual_costs_within_budget(self):
        """Test vérification SLA ±20%."""
        # Variance acceptable
        job_data = {
            "render": {"duration_seconds": 5},
            "_cost_estimate": 0.50
        }
        costs = get_job_actual_costs(job_data)
        # Si variance < 20%, within_budget = True
        assert "within_budget" in costs
    
    def test_pricing_structure(self):
        """Test structure des tarifs."""
        assert "gemini" in PRICING
        assert "runway" in PRICING
        assert "gcs" in PRICING
        assert "cloud_run" in PRICING
        assert "competitors" in PRICING
        
        # Vérifier les concurrents
        assert PRICING["competitors"]["runway_direct"] == 2.50
