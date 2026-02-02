"""
Tests pour le Supervisor Agent
"""
import pytest
import asyncio
from src.agents.supervisor import Supervisor


@pytest.mark.asyncio
async def test_supervisor_approved():
    """Test approbation avec qualité et budget OK."""
    supervisor = Supervisor()
    
    inputs = {
        "consistency_report": {"consistency_level": 0.85},
        "cost_certification": {"total_cost": 50.0, "breakdown": {}},
        "technical_validation_report": {"technical_score": 0.9},
        "quality_score": 0.8,
        "client_budget": 100.0
    }
    
    result = await supervisor.run(inputs)
    
    assert result["final_approval"] is True
    assert result["decision"] == "APPROVED"
    assert result["delivery_manifest"]["approved"] is True
    assert "client_report" in result
    assert result["client_report"]["status"] == "APPROVED"


@pytest.mark.asyncio
async def test_supervisor_rejected_low_quality():
    """Test rejet avec qualité insuffisante."""
    supervisor = Supervisor()
    
    inputs = {
        "consistency_report": {"consistency_level": 0.3},
        "cost_certification": {"total_cost": 30.0, "breakdown": {}},
        "technical_validation_report": {"technical_score": 0.7},  # < 0.8 threshold
        "quality_score": 0.35,  # < 0.4 threshold
        "client_budget": 100.0
    }
    
    result = await supervisor.run(inputs)
    
    assert result["final_approval"] is False
    assert result["decision"] == "REJECTED"
    assert result["delivery_manifest"]["approved"] is False


@pytest.mark.asyncio
async def test_supervisor_escalate():
    """Test escalade avec qualité modérée."""
    supervisor = Supervisor()
    
    inputs = {
        "consistency_report": {"consistency_level": 0.6},
        "cost_certification": {"total_cost": 40.0, "breakdown": {}},
        "technical_validation_report": {"technical_score": 0.85},
        "quality_score": 0.55,  # Entre 0.4 et 0.7
        "client_budget": 100.0
    }
    
    result = await supervisor.run(inputs)
    
    assert result["decision"] == "ESCALATE"
    assert result["client_report"]["status"] == "PENDING_REVIEW"


@pytest.mark.asyncio
async def test_supervisor_budget_exceeded():
    """Test dépassement de budget malgré bonne qualité."""
    supervisor = Supervisor()
    
    inputs = {
        "consistency_report": {"consistency_level": 0.8},
        "cost_certification": {"total_cost": 150.0, "breakdown": {}},
        "technical_validation_report": {"technical_score": 0.9},
        "quality_score": 0.8,  # >= 0.7
        "client_budget": 100.0  # Cost dépasse
    }
    
    result = await supervisor.run(inputs)
    
    assert result["decision"] == "REVIEW"
    assert "exceeds budget" in result["client_report"]["message"].lower()


def test_supervisor_initialization():
    """Test l'initialisation du Supervisor."""
    supervisor = Supervisor()
    assert supervisor.name == "AIPROD Supervisor"
    assert supervisor.llm_model == "gemini-1.5-pro"
