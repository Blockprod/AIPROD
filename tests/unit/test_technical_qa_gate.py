import pytest
from src.api.functions.technical_qa_gate import TechnicalQAGate

def test_validate_success():
    gate = TechnicalQAGate()
    manifest = {
        "assets": ["img1", "img2"],
        "complexity_score": 0.5,
        "cost": 1.5,
        "quality_score": 0.8
    }
    report = gate.validate(manifest)
    assert report["technical_valid"] is True
    assert all(report["checks"].values())


def test_validate_missing_assets():
    gate = TechnicalQAGate()
    manifest = {
        "assets": [],
        "complexity_score": 0.5,
        "cost": 1.5,
        "quality_score": 0.8
    }
    report = gate.validate(manifest)
    assert report["technical_valid"] is False
    assert report["checks"]["asset_count"] is False


def test_validate_low_quality():
    gate = TechnicalQAGate()
    manifest = {
        "assets": ["img1"],
        "complexity_score": 0.5,
        "cost": 1.5,
        "quality_score": 0.2
    }
    report = gate.validate(manifest)
    assert report["technical_valid"] is False
