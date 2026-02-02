import pytest
import asyncio
from src.agents.semantic_qa import SemanticQA


def test_run_semantic_validation():
    """Test SemanticQA returns valid validation report."""
    agent = SemanticQA()
    outputs = {"result": "some output"}
    report = asyncio.run(agent.run(outputs))

    # Check structure matches new implementation
    assert "semantic_valid" in report
    assert "overall_score" in report
    assert isinstance(report["semantic_valid"], bool)
    assert 0 <= report["overall_score"] <= 1
    assert "provider" in report  # Should be "mock" or "gemini"
