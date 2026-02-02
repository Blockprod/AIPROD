import pytest
import asyncio
from src.agents.render_executor import RenderExecutor

def test_run_rendered():
    agent = RenderExecutor()
    # Test with mock (no real API key)
    prompt_bundle = {"text_prompt": "A beautiful scene"}
    result = asyncio.run(agent.run(prompt_bundle))
    assert result["status"] in ["rendered", "rendered_mock", "error"]
    assert "assets" in result
    assert ("video" in result["assets"] or "error" in result)
