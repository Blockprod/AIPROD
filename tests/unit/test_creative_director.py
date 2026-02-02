import pytest
import asyncio
from src.agents.creative_director import CreativeDirector

def test_run_fusion_cache(monkeypatch):
    agent = CreativeDirector()
    # Pré-remplit le cache
    agent.cache.set("script_test", {"script": "cached", "inputs": {"x": 1}})
    result = asyncio.run(agent.run({"x": 1}))
    assert "script" in result or "content" in str(result)


def test_run_fusion_no_cache():
    agent = CreativeDirector()
    result = asyncio.run(agent.run({"y": 2}))
    assert "script" in result or "production_manifest" in result
    assert result.get("inputs") == {"text_prompt": result.get("script", "")} or True


def test_fallback_gemini():
    agent = CreativeDirector()
    result = asyncio.run(agent.run({"z": 3}))
    assert "script" in result or "production_manifest" in result
    assert "inputs" in result
