import pytest
import asyncio
from src.agents.visual_translator import VisualTranslator


def test_run_translation_en():
    """Test VisualTranslator handles English translation."""
    agent = VisualTranslator()
    assets = {"img": "image.png", "vid": "video.mp4"}
    result = asyncio.run(agent.run(assets, target_lang="en"))

    # Check new structure
    assert result["status"] == "adapted"
    assert result["language"] == "en"
    assert "adapted_assets" in result
    assert "readiness_score" in result
    assert 0 <= result["readiness_score"] <= 1


def test_run_translation_fr():
    """Test VisualTranslator handles French translation."""
    agent = VisualTranslator()
    assets = {"img": "image.png"}
    result = asyncio.run(agent.run(assets, target_lang="fr"))

    # Check new structure
    assert result["status"] == "adapted"
    assert result["language"] == "fr"
    assert "adapted_assets" in result
    assert "localization_metadata" in result
