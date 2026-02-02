import pytest
import asyncio
from src.orchestrator.state_machine import StateMachine, PipelineState

def test_pipeline_fast_track():
    sm = StateMachine()
    inputs = {"priority": "high", "lang": "fr"}
    result = asyncio.run(sm.run(inputs))
    assert sm.state == PipelineState.DELIVERED
    assert "fast_track" in result
    assert result["visual_translation"]["lang"] == "fr"


def test_pipeline_fusion():
    sm = StateMachine()
    inputs = {"priority": "low", "lang": "en"}
    result = asyncio.run(sm.run(inputs))
    assert sm.state == PipelineState.DELIVERED
    assert "fusion" in result
    assert result["visual_translation"]["lang"] == "en"


def test_pipeline_error_and_retry(monkeypatch):
    sm = StateMachine()
    def fail_run(*args, **kwargs):
        raise Exception("Test error")
    monkeypatch.setattr(sm.creative_director, "run", fail_run)
    inputs = {"priority": "low"}
    result = asyncio.run(sm.run(inputs))
    assert sm.state == PipelineState.ERROR
    assert "error" in result
