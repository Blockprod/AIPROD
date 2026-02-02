import pytest
import asyncio
from src.orchestrator.state_machine import StateMachine, PipelineState

def test_initial_state():
    sm = StateMachine()
    assert sm.state == PipelineState.INIT
    assert sm.retry_count == 0


def test_transition():
    sm = StateMachine()
    sm.transition(PipelineState.INPUT_SANITIZED)
    assert sm.state == PipelineState.INPUT_SANITIZED


def test_run_success():
    sm = StateMachine()
    result = asyncio.run(sm.run({"priority": "high", "lang": "en"}))
    assert sm.state == PipelineState.DELIVERED
    assert sm.retry_count == 0
    assert result is not None


def test_run_error_and_retry(monkeypatch):
    sm = StateMachine()
    # Force une erreur sur la première transition
    def fail_run(*args, **kwargs):
        raise Exception("Test error")
    monkeypatch.setattr(sm, "creative_director", type('obj', (object,), {'run': fail_run})())
    sm.max_retries = 2
    result = asyncio.run(sm.run({"priority": "low"}))
    assert sm.state == PipelineState.ERROR
    assert "error" in result
