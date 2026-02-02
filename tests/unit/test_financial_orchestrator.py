import pytest
from src.api.functions.financial_orchestrator import FinancialOrchestrator

def test_optimize_base_cost():
    fo = FinancialOrchestrator()
    manifest = {"complexity_score": 0.5}
    result = fo.optimize(manifest)
    assert result["optimized_cost"] == 1.25  # 1.0 + 0.5*0.5
    assert result["quality"] == "optimal"
    assert "CERT-" in result["certification"]


def test_audit_trail():
    fo = FinancialOrchestrator()
    manifest = {"complexity_score": 0.3}
    fo.optimize(manifest)
    trail = fo.get_audit_trail()
    assert len(trail) == 1
    assert trail[0]["action"] == "optimize"


def test_update_pricing():
    fo = FinancialOrchestrator(update_interval_hours=0)  # Force update
    import time
    time.sleep(0.1)
    fo.update_pricing()
    assert fo.pricing_rules["base"] > 1.0
