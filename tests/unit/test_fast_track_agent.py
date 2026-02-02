import pytest
import asyncio
from src.agents.fast_track_agent import FastTrackAgent

def test_run_priority_high():
    agent = FastTrackAgent()
    result = asyncio.run(agent.run({"priority": "high"}))
    assert result["status"] == "fast_tracked"
    assert result["inputs"]["priority"] == "high"


def test_run_priority_normal():
    agent = FastTrackAgent()
    result = asyncio.run(agent.run({"priority": "low"}))
    assert result["status"] == "normal"
    assert result["inputs"]["priority"] == "low"
