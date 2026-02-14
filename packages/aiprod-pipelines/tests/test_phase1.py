"""
PHASE 1 Adapter Tests - Simplified Import Path
================================================

This test file focuses on PHASE 1 adapters without importing
the full aiprod_pipelines package (which has torch dependencies).
"""

import pytest
import asyncio
import sys
import os
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch, MagicMock
import importlib.util
import importlib.machinery

# Setup paths
project_root = Path(__file__).parent.parent.parent
aiprod_pipelines_src = project_root / "aiprod-pipelines" / "src"
aiprod_core_src = project_root / "aiprod-core" / "src"

sys.path.insert(0, str(aiprod_pipelines_src))
sys.path.insert(0, str(aiprod_core_src))

# Mock heavy dependencies before importing
sys.modules["torch"] = MagicMock()
sys.modules["triton"] = MagicMock()
sys.modules["diffusers"] = MagicMock()
sys.modules["transformers"] = MagicMock()

# Directly load api modules using importlib to bypass __init__.py
def load_module_direct(module_name, file_path):
    """Load a Python module directly without triggering __init__.py files."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"Warning: Could not fully load {module_name}: {e}")
        return module
    return None

# Load api adapters directly
base_adapter_path = aiprod_pipelines_src / "aiprod_pipelines" / "api" / "adapters" / "base.py"
input_sanitizer_path = aiprod_pipelines_src / "aiprod_pipelines" / "api" / "adapters" / "input_sanitizer.py"
creative_path = aiprod_pipelines_src / "aiprod_pipelines" / "api" / "adapters" / "creative.py"
visual_translator_path = aiprod_pipelines_src / "aiprod_pipelines" / "api" / "adapters" / "visual_translator.py"
render_path = aiprod_pipelines_src / "aiprod_pipelines" / "api" / "adapters" / "render.py"

# Create minimal module structure
sys.modules["aiprod_pipelines"] = MagicMock()
sys.modules["aiprod_pipelines.api"] = MagicMock()
sys.modules["aiprod_pipelines.api.adapters"] = MagicMock()
sys.modules["aiprod_pipelines.api.schema"] = MagicMock()

# Try direct imports with fallback
try:
    # Read and execute adapter code directly in a custom namespace
    base_namespace = {}
    with open(base_adapter_path) as f:
        exec(f.read(), base_namespace)
    BaseAdapter = base_namespace["BaseAdapter"]
    
    # Now import others
    input_namespace = {"BaseAdapter": BaseAdapter, "Context": Dict[str, Any], "Dict": Dict, "Any": Any, "Optional": type(None)}
    with open(input_sanitizer_path) as f:
        code = f.read()
        # Remove typing imports that might cause issues
        code = code.replace("from typing import", "# from typing import")
        code = code.replace("from ..schema.schemas import Context", "Context = Dict[str, Any]")
        exec(code, input_namespace)
    InputSanitizerAdapter = input_namespace["InputSanitizerAdapter"]
    
    # Creative adapter
    creative_namespace = {
        "BaseAdapter": BaseAdapter,
        "Context": Dict[str, Any],
        "Dict": Dict,
        "Any": Any,
        "Optional": type(None),
        "List": list,
        "asyncio": asyncio,
        "time": __import__("time"),
        "random": __import__("random"),
    }
    with open(creative_path) as f:
        code = f.read()
        code = code.replace("from typing import", "# from typing import")
        code = code.replace("from ..schema.schemas import Context", "Context = Dict[str, Any]")
        code = code.replace("import google.generativeai as genai", "genai = None")
        exec(code, creative_namespace)
    CreativeDirectorAdapter = creative_namespace["CreativeDirectorAdapter"]
    
    # Visual translator
    visual_namespace = {
        "BaseAdapter": BaseAdapter,
        "Context": Dict[str, Any],
        "Dict": Dict,
        "Any": Any,
        "List": list,
        "Tuple": tuple,
        "hashlib": __import__("hashlib"),
        "time": __import__("time"),
    }
    with open(visual_translator_path) as f:
        code = f.read()
        code = code.replace("from typing import", "# from typing import")
        code = code.replace("from ..schema.schemas import Context", "Context = Dict[str, Any]")
        exec(code, visual_namespace)
    VisualTranslatorAdapter = visual_namespace["VisualTranslatorAdapter"]
    
    # Render executor
    render_namespace = {
        "BaseAdapter": BaseAdapter,
        "Context": Dict[str, Any],
        "Dict": Dict,
        "Any": Any,
        "List": list,
        "Tuple": tuple,
        "Optional": type(None),
        "asyncio": asyncio,
        "time": __import__("time"),
        "random": __import__("random"),
    }
    with open(render_path) as f:
        code = f.read()
        code = code.replace("from typing import", "# from typing import")
        code = code.replace("from ..schema.schemas import Context", "Context = Dict[str, Any]")
        exec(code, render_namespace)
    RenderExecutorAdapter = render_namespace["RenderExecutorAdapter"]
    
except Exception as e:
    print(f"Failed to load adapters: {e}")
    import traceback
    traceback.print_exc()
    InputSanitizerAdapter = None
    CreativeDirectorAdapter = None
    VisualTranslatorAdapter = None
    RenderExecutorAdapter = None

# Type hints
Context = Dict[str, Any]


# ============================================================================
# Input Sanitizer Tests
# ============================================================================

class TestInputSanitizer:
    """Test InputSanitizerAdapter validation logic."""
    
    @pytest.mark.asyncio
    async def test_valid_input(self):
        """Test sanitization of valid input."""
        adapter = InputSanitizerAdapter()
        
        ctx: Context = {
            "request_id": "test-sanitizer-1",
            "state": "ANALYSIS",
            "memory": {
                "prompt": "A beautiful sunset over mountains",
                "duration_sec": 30,
                "budget": 1.5,
                "complexity": 0.6,
                "preferences": {
                    "style": "cinematic",
                    "mood": "peaceful",
                    "camera_style": "steady"
                }
            },
            "config": {}
        }
        
        result = await adapter.execute(ctx)
        assert result["memory"]["validated"] is True
        assert "errors" not in result["memory"] or len(result["memory"]["errors"]) == 0
    
    @pytest.mark.asyncio
    async def test_invalid_prompt_too_short(self):
        """Test rejection of prompt under 10 characters."""
        adapter = InputSanitizerAdapter()
        
        ctx: Context = {
            "request_id": "test-sanitizer-2",
            "state": "ANALYSIS",
            "memory": {
                "prompt": "Short",
                "duration_sec": 30,
                "budget": 1.5,
                "complexity": 0.5,
                "preferences": {}
            },
            "config": {}
        }
        
        with pytest.raises(ValueError, match="Prompt must be"):
            await adapter.execute(ctx)
    
    @pytest.mark.asyncio
    async def test_invalid_duration_range(self):
        """Test rejection of duration outside 10-300 seconds."""
        adapter = InputSanitizerAdapter()
        
        ctx: Context = {
            "request_id": "test-sanitizer-3",
            "state": "ANALYSIS",
            "memory": {
                "prompt": "A valid prompt that is long enough",
                "duration_sec": 500,  # Too long
                "budget": 1.5,
                "complexity": 0.5,
                "preferences": {}
            },
            "config": {}
        }
        
        with pytest.raises(ValueError, match="Duration must be"):
            await adapter.execute(ctx)
    
    @pytest.mark.asyncio
    async def test_invalid_budget_range(self):
        """Test rejection of budget outside $0.1-10 range."""
        adapter = InputSanitizerAdapter()
        
        ctx: Context = {
            "request_id": "test-sanitizer-4",
            "state": "ANALYSIS",
            "memory": {
                "prompt": "A valid prompt that is long enough",
                "duration_sec": 30,
                "budget": 50.0,  # Too high
                "complexity": 0.5,
                "preferences": {}
            },
            "config": {}
        }
        
        with pytest.raises(ValueError, match="Budget must be"):
            await adapter.execute(ctx)
    
    @pytest.mark.asyncio
    async def test_complexity_normalization(self):
        """Test complexity clamping to [0.0, 1.0]."""
        adapter = InputSanitizerAdapter()
        
        ctx: Context = {
            "request_id": "test-sanitizer-5",
            "state": "ANALYSIS",
            "memory": {
                "prompt": "A valid prompt that is long enough",
                "duration_sec": 30,
                "budget": 1.5,
                "complexity": 2.5,  # Out of bounds
                "preferences": {}
            },
            "config": {}
        }
        
        result = await adapter.execute(ctx)
        assert 0.0 <= result["memory"]["complexity"] <= 1.0


# ============================================================================
# Creative Director Tests
# ============================================================================

class TestCreativeDirector:
    """Test CreativeDirectorAdapter with mocked Gemini."""
    
    @pytest.mark.asyncio
    async def test_manifest_generation_with_fallback(self):
        """Test manifest generation using fallback (no Gemini)."""
        adapter = CreativeDirectorAdapter(config={"use_gemini": False})
        
        ctx: Context = {
            "request_id": "test-creative-1",
            "state": "CREATIVE_DIRECTION",
            "memory": {
                "prompt": "A majestic eagle soaring through mountain peaks at sunset",
                "duration_sec": 60,
                "budget": 3.0,
                "complexity": 0.7,
                "preferences": {}
            },
            "config": {}
        }
        
        result = await adapter.execute(ctx)
        
        # Verify manifest generated
        assert "production_manifest" in result["memory"]
        manifest = result["memory"]["production_manifest"]
        assert manifest["production_id"]
        assert manifest["title"]
        assert len(manifest["scenes"]) > 0
        assert manifest["total_duration_sec"] == 60
    
    @pytest.mark.asyncio
    async def test_manifest_caching(self):
        """Test caching of manifests."""
        adapter = CreativeDirectorAdapter(config={"use_gemini": False})
        
        # First request
        ctx1: Context = {
            "request_id": "test-creative-2a",
            "state": "CREATIVE_DIRECTION",
            "memory": {
                "prompt": "Same prompt for caching test",
                "duration_sec": 30,
                "budget": 1.0,
                "complexity": 0.5,
                "preferences": {}
            },
            "config": {}
        }
        
        result1 = await adapter.execute(ctx1)
        manifest1 = result1["memory"]["production_manifest"]
        
        # Second request with same prompt
        ctx2: Context = {
            "request_id": "test-creative-2b",
            "state": "CREATIVE_DIRECTION",
            "memory": {
                "prompt": "Same prompt for caching test",
                "duration_sec": 30,
                "budget": 1.0,
                "complexity": 0.5,
                "preferences": {}
            },
            "config": {}
        }
        
        result2 = await adapter.execute(ctx2)
        manifest2 = result2["memory"]["production_manifest"]
        
        # Should get same manifest (from cache)
        assert manifest1["production_id"] == manifest2["production_id"]


# ============================================================================
# Visual Translator Tests
# ============================================================================

class TestVisualTranslator:
    """Test VisualTranslatorAdapter scene → shot conversion."""
    
    @pytest.mark.asyncio
    async def test_scene_to_shots_conversion(self):
        """Test conversion of scenes to shot specifications."""
        adapter = VisualTranslatorAdapter()
        
        ctx: Context = {
            "request_id": "test-visual-1",
            "state": "VISUAL_TRANSLATION",
            "memory": {
                "production_manifest": {
                    "production_id": "prod-001",
                    "title": "Test",
                    "total_duration_sec": 60,
                    "scenes": [
                        {
                            "scene_id": "scene_1",
                            "duration_sec": 30,
                            "description": "An eagle soaring",
                            "camera_movement": "pan",
                            "lighting_style": "natural",
                            "mood": "dramatic",
                            "characters": ["eagle"],
                            "location": "mountains",
                            "time_of_day": "sunset"
                        },
                        {
                            "scene_id": "scene_2",
                            "duration_sec": 30,
                            "description": "Landing on rock",
                            "camera_movement": "zoom",
                            "lighting_style": "warm",
                            "mood": "peaceful",
                            "characters": ["eagle"],
                            "location": "cliff",
                            "time_of_day": "sunset"
                        }
                    ],
                    "metadata": {}
                }
            },
            "config": {}
        }
        
        result = await adapter.execute(ctx)
        
        # Verify shot list generated
        assert "shot_list" in result["memory"]
        shot_list = result["memory"]["shot_list"]
        assert len(shot_list) > 0
        
        # Verify shot structure
        for shot in shot_list:
            assert shot["shot_id"]
            assert shot["scene_id"]
            assert shot["prompt"]
            assert 0 <= shot["seed"] < 2**32
            assert shot["duration_sec"] > 0
    
    @pytest.mark.asyncio
    async def test_deterministic_seeding(self):
        """Test that seeding is deterministic for same inputs."""
        adapter = VisualTranslatorAdapter()
        
        manifest = {
            "production_id": "prod-seed-test",
            "title": "Seed Test",
            "total_duration_sec": 30,
            "scenes": [
                {
                    "scene_id": "scene_1",
                    "duration_sec": 30,
                    "description": "Test scene",
                    "camera_movement": "static",
                    "lighting_style": "neutral",
                    "mood": "neutral",
                    "characters": [],
                    "location": "studio",
                    "time_of_day": "day"
                }
            ],
            "metadata": {}
        }
        
        # First translation
        ctx1: Context = {
            "request_id": "test-seed-1",
            "state": "VISUAL_TRANSLATION",
            "memory": {"production_manifest": manifest},
            "config": {}
        }
        
        result1 = await adapter.execute(ctx1)
        shots1 = result1["memory"]["shot_list"]
        
        # Second translation (same manifest)
        ctx2: Context = {
            "request_id": "test-seed-2",
            "state": "VISUAL_TRANSLATION",
            "memory": {"production_manifest": manifest},
            "config": {}
        }
        
        result2 = await adapter.execute(ctx2)
        shots2 = result2["memory"]["shot_list"]
        
        # Seeds should be identical
        for shot1, shot2 in zip(shots1, shots2):
            assert shot1["seed"] == shot2["seed"]


# ============================================================================
# Render Executor Tests
# ============================================================================

class TestRenderExecutor:
    """Test RenderExecutorAdapter with batch processing and retry."""
    
    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test batch processing of shots."""
        adapter = RenderExecutorAdapter(config={"batch_size": 2})
        
        # Create shot list
        shots = [
            {
                "shot_id": f"shot_{i}",
                "scene_id": "scene_1",
                "prompt": f"Shot {i}",
                "duration_sec": 10,
                "seed": 12345 + i
            }
            for i in range(6)  # 6 shots → 3 batches of 2
        ]
        
        ctx: Context = {
            "request_id": "test-render-1",
            "state": "RENDER_EXECUTION",
            "memory": {
                "shot_list": shots,
                "cost_estimation": {"selected_backend": "runway_gen3"}
            },
            "config": {}
        }
        
        result = await adapter.execute(ctx)
        
        # Verify assets generated
        assert "generated_assets" in result["memory"]
        assets = result["memory"]["generated_assets"]
        assert len(assets) == 6
        
        # Verify asset structure
        for asset in assets:
            assert asset["id"]
            assert asset["url"]
            assert asset["duration_sec"] > 0
            assert asset["resolution"]
            assert asset["backend_used"]
    
    @pytest.mark.asyncio
    async def test_retry_with_exponential_backoff(self):
        """Test retry logic with backoff calculation."""
        adapter = RenderExecutorAdapter(config={"max_retries": 3})
        
        # Test backoff delay calculation
        delays = [adapter._calculate_backoff_delay(i) for i in range(3)]
        
        # Each should be positive and capped
        assert all(d > 0 for d in delays)
        assert all(d <= 30 for d in delays)  # Capped at 30s
        
        # First delay should be around 1s
        assert 0.5 <= delays[0] <= 1.5


# ============================================================================
# Pipeline Integration Tests
# ============================================================================

class TestPipeline:
    """Test full PHASE 1 pipeline integration."""
    
    @pytest.mark.asyncio
    async def test_short_video_pipeline(self):
        """Test complete pipeline execution for short video."""
        # Create adapters
        adapters = {
            "input_sanitizer": InputSanitizerAdapter(),
            "creative_director": CreativeDirectorAdapter(config={"use_gemini": False}),
            "visual_translator": VisualTranslatorAdapter(),
            "render_executor": RenderExecutorAdapter(config={"batch_size": 2})
        }
        
        # Step 1: Input Sanitization
        ctx: Context = {
            "request_id": "test-pipeline-1",
            "state": "ANALYSIS",
            "memory": {
                "prompt": "A serene lake with mountains in the background",
                "duration_sec": 30,
                "budget": 1.5,
                "complexity": 0.4,
                "preferences": {"style": "scenic", "mood": "peaceful"}
            },
            "config": {}
        }
        
        ctx = await adapters["input_sanitizer"].execute(ctx)
        assert ctx["memory"]["validated"] is True
        
        # Step 2: Creative Direction
        ctx = await adapters["creative_director"].execute(ctx)
        assert "production_manifest" in ctx["memory"]
        
        # Step 3: Visual Translation
        ctx = await adapters["visual_translator"].execute(ctx)
        assert "shot_list" in ctx["memory"]
        
        # Step 4: Render Execution
        ctx["memory"]["cost_estimation"] = {"selected_backend": "runway_gen3"}
        ctx = await adapters["render_executor"].execute(ctx)
        
        # Verify final output
        assert "generated_assets" in ctx["memory"]
        assert len(ctx["memory"]["generated_assets"]) > 0
        assert ctx["memory"]["render_stats"]["success_rate"] > 0.8


# ============================================================================
# Error Recovery Tests
# ============================================================================

class TestErrorRecovery:
    """Test error handling and recovery in adapters."""
    
    @pytest.mark.asyncio
    async def test_missing_required_fields(self):
        """Test handling of missing required context fields."""
        adapter = VisualTranslatorAdapter()
        
        # Missing production_manifest
        ctx: Context = {
            "request_id": "test-error-1",
            "state": "VISUAL_TRANSLATION",
            "memory": {},  # Empty!
            "config": {}
        }
        
        with pytest.raises(ValueError):
            await adapter.execute(ctx)
    
    @pytest.mark.asyncio
    async def test_render_partial_failure(self):
        """Test render executor handles batch processing gracefully."""
        adapter = RenderExecutorAdapter(config={"batch_size": 2})
        
        # Create shots
        shots = [
            {
                "shot_id": f"shot_{i}",
                "scene_id": "scene_1",
                "prompt": f"Shot {i}",
                "duration_sec": 10,
                "seed": 12345 + i
            }
            for i in range(4)
        ]
        
        ctx: Context = {
            "request_id": "test-render-fail",
            "state": "RENDER_EXECUTION",
            "memory": {
                "shot_list": shots,
                "cost_estimation": {"selected_backend": "runway_gen3"}
            },
            "config": {}
        }
        
        # Should complete (may have failures but should not raise)
        result = await adapter.execute(ctx)
        assert "generated_assets" in result["memory"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
