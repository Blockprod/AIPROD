"""
PHASE 0 Foundation Tests - Checkpoint, Schema, and Orchestrator
===============================================================

Comprehensive test suite for PHASE 0 deliverables:
- Checkpoint save/restore correctness
- Schema bidirectional transformation
- State machine transitions (mocked adapters)
- Failure recovery from checkpoints

Run with: pytest tests/test_foundation.py -v
"""

import pytest
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any
import sys

# Add paths to sys path BEFORE importing aiprod modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "aiprod-core" / "src"))
sys.path.insert(0, str(project_root / "aiprod-pipelines" / "src"))

# Now import mocking tools to prevent torch import
from unittest.mock import patch
import importlib.util

# Patch torch before it's imported anywhere
torch_spec = importlib.util.spec_from_loader("torch", loader=None)
sys.modules["torch"] = importlib.util.module_from_spec(torch_spec) if torch_spec else None

# Import components to test
# Use direct imports from submodules to avoid __init__.py
import aiprod_pipelines.api.checkpoint.manager as checkpoint_manager_module
import aiprod_pipelines.api.checkpoint.recovery as recovery_module
import aiprod_pipelines.api.schema.transformer as transformer_module
import aiprod_pipelines.api.schema.schemas as schemas_module
import aiprod_pipelines.api.orchestrator as orchestrator_module
import aiprod_pipelines.api.adapters.base as base_module

CheckpointManager = checkpoint_manager_module.CheckpointManager
RecoveryManager = recovery_module.RecoveryManager
RecoveryAction = recovery_module.RecoveryAction
SchemaTransformer = transformer_module.SchemaTransformer
Context = schemas_module.Context
PipelineRequest = schemas_module.PipelineRequest
Orchestrator = orchestrator_module.Orchestrator
BaseAdapter = base_module.BaseAdapter


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Create temporary directory for checkpoints."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return str(checkpoint_dir)


@pytest.fixture
def checkpoint_manager(temp_checkpoint_dir):
    """Create CheckpointManager instance."""
    return CheckpointManager(storage_path=temp_checkpoint_dir)


@pytest.fixture
def recovery_manager(checkpoint_manager):
    """Create RecoveryManager instance."""
    return RecoveryManager(checkpoint_manager, max_retries=3)


@pytest.fixture
def schema_transformer():
    """Create SchemaTransformer instance."""
    return SchemaTransformer()


@pytest.fixture
def sample_context():
    """Create sample execution context."""
    return {
        "request_id": "test-job-001",
        "state": "INIT",
        "memory": {
            "start_time": time.time(),
            "prompt": "A cat playing in a garden",
            "duration_sec": 60,
            "budget": 2.0,
            "complexity": 0.5
        },
        "config": {
            "max_retries": 3
        }
    }


@pytest.fixture
def sample_manifest():
    """Create sample AIPROD production manifest."""
    return {
        "production_id": "prod-001",
        "title": "Test Production",
        "total_duration_sec": 60,
        "scenes": [
            {
                "scene_id": "scene_1",
                "duration_sec": 30,
                "description": "A cat playing",
                "camera_movement": "pan",
                "lighting_style": "natural",
                "mood": "cheerful",
                "characters": ["cat"],
                "props": ["ball"],
                "location": "garden",
                "time_of_day": "day",
                "weather": "sunny",
                "visual_style": {"tone": "bright"}
            },
            {
                "scene_id": "scene_2",
                "duration_sec": 30,
                "description": "Cat resting",
                "camera_movement": "static",
                "lighting_style": "natural",
                "mood": "calm",
                "characters": ["cat"],
                "props": [],
                "location": "garden",
                "time_of_day": "day",
                "weather": "sunny",
                "visual_style": {"tone": "soft"}
            }
        ],
        "consistency_markers": {
            "visual_style": {
                "cinematography": "natural",
                "color_palette": ["green", "brown"],
                "lighting_style": "natural"
            },
            "character_continuity": {
                "cat": {"appearance": "orange tabby"}
            },
            "narrative_elements": {
                "pacing": "relaxed"
            }
        },
        "metadata": {
            "generator": "test"
        }
    }


# Mock adapter for testing
class MockAdapter(BaseAdapter):
    """Mock adapter for testing orchestration."""
    
    def __init__(self, config=None, should_fail=False):
        super().__init__(config)
        self.should_fail = should_fail
        self.call_count = 0
    
    async def execute(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Mock execution."""
        self.call_count += 1
        
        if self.should_fail:
            raise Exception(f"Mock adapter failure (call {self.call_count})")
        
        # Simple passthrough with marker
        ctx["memory"][f"mock_{self.name}_executed"] = True
        
        return ctx


# ============================================================================
# Checkpoint Tests
# ============================================================================

@pytest.mark.asyncio
async def test_checkpoint_save_restore(checkpoint_manager, sample_context):
    """Test checkpoint save and restore preserves context exactly."""
    job_id = sample_context["request_id"]
    state = "ANALYSIS"
    
    # Save checkpoint
    checkpoint_id = await checkpoint_manager.save_checkpoint(job_id, state, sample_context)
    
    assert checkpoint_id is not None
    assert len(checkpoint_id) > 0
    
    # Restore checkpoint
    restored = await checkpoint_manager.restore_checkpoint(checkpoint_id)
    
    # Verify context is preserved
    assert restored["request_id"] == sample_context["request_id"]
    assert restored["state"] == sample_context["state"]
    assert restored["memory"]["prompt"] == sample_context["memory"]["prompt"]
    assert restored["memory"]["duration_sec"] == sample_context["memory"]["duration_sec"]


@pytest.mark.asyncio
async def test_checkpoint_validation(checkpoint_manager, sample_context):
    """Test checkpoint context validation."""
    # Valid context
    assert await checkpoint_manager.validate_consistency(sample_context) == True
    
    # Invalid: missing required field
    invalid_ctx = {"request_id": "test"}
    assert await checkpoint_manager.validate_consistency(invalid_ctx) == False
    
    # Invalid: contradictory state
    invalid_ctx2 = {
        "request_id": "test",
        "state": "FINALIZE",
        "memory": {}  # Missing delivery_manifest
    }
    assert await checkpoint_manager.validate_consistency(invalid_ctx2) == False


@pytest.mark.asyncio
async def test_checkpoint_mark_successful(checkpoint_manager, sample_context):
    """Test marking checkpoint as successfully used."""
    checkpoint_id = await checkpoint_manager.save_checkpoint(
        sample_context["request_id"],
        "INIT",
        sample_context
    )
    
    # Mark successful
    await checkpoint_manager.mark_successful(checkpoint_id)
    
    # Verify metadata updated
    checkpoint_file = checkpoint_manager.storage_path / f"{checkpoint_id}.json"
    assert checkpoint_file.exists()
    
    with open(checkpoint_file, 'r') as f:
        data = json.load(f)
        assert data["metadata"]["used_successfully"] == True
        assert "completed_at" in data["metadata"]


@pytest.mark.asyncio
async def test_checkpoint_get_latest(checkpoint_manager, sample_context):
    """Test retrieving latest checkpoint for a job."""
    job_id = sample_context["request_id"]
    
    # Create multiple checkpoints
    ckpt1 = await checkpoint_manager.save_checkpoint(job_id, "INIT", sample_context)
    await asyncio.sleep(0.01)  # Ensure different timestamps
    ckpt2 = await checkpoint_manager.save_checkpoint(job_id, "ANALYSIS", sample_context)
    await asyncio.sleep(0.01)
    ckpt3 = await checkpoint_manager.save_checkpoint(job_id, "CREATIVE_DIRECTION", sample_context)
    
    # Get latest
    latest = await checkpoint_manager.get_latest_checkpoint(job_id)
    
    assert latest == ckpt3  # Should be the most recent


# ============================================================================
# Recovery Tests
# ============================================================================

@pytest.mark.asyncio
async def test_recovery_retry_action(recovery_manager, sample_context):
    """Test recovery manager returns RETRY action on first failure."""
    job_id = sample_context["request_id"]
    
    # Save checkpoint first
    ckpt_id = await recovery_manager.checkpoint_manager.save_checkpoint(
        job_id, "ANALYSIS", sample_context
    )
    
    # Handle failure
    action, restored_ctx = await recovery_manager.handle_failure(
        job_id=job_id,
        state="ANALYSIS",
        error=Exception("Test error"),
        attempt=1
    )
    
    assert action == RecoveryAction.RETRY
    assert restored_ctx is not None
    assert restored_ctx["request_id"] == job_id


@pytest.mark.asyncio
async def test_recovery_max_retries_exceeded(recovery_manager, sample_context):
    """Test recovery manager returns ERROR after max retries."""
    job_id = sample_context["request_id"]
    
    # Save checkpoint
    await recovery_manager.checkpoint_manager.save_checkpoint(
        job_id, "ANALYSIS", sample_context
    )
    
    # Exceed max retries
    action, restored_ctx = await recovery_manager.handle_failure(
        job_id=job_id,
        state="ANALYSIS",
        error=Exception("Test error"),
        attempt=4  # Exceeds max_retries=3
    )
    
    assert action == RecoveryAction.ERROR
    assert restored_ctx is None


@pytest.mark.asyncio
async def test_recovery_failure_history(recovery_manager, sample_context):
    """Test recovery manager records failure history."""
    job_id = sample_context["request_id"]
    
    # Record multiple failures
    await recovery_manager.handle_failure(
        job_id, "ANALYSIS", Exception("Error 1"), 1
    )
    await recovery_manager.handle_failure(
        job_id, "ANALYSIS", Exception("Error 2"), 2
    )
    
    # Check history
    history = recovery_manager.get_failure_history(job_id)
    
    assert len(history) == 2
    assert history[0]["state"] == "ANALYSIS"
    assert history[0]["error_type"] == "Exception"
    assert history[1]["attempt"] == 2


# ============================================================================
# Schema Transformation Tests
# ============================================================================

@pytest.mark.asyncio
async def test_schema_to_aiprod_conversion(schema_transformer, sample_manifest):
    """Test AIPROD manifest → AIPROD internal format conversion."""
    internal = schema_transformer.to_aiprod(sample_manifest)
    
    # Verify structure
    assert "scenes" in internal
    assert "metadata" in internal
    assert "consistency_rules" in internal
    assert "technical_params" in internal
    
    # Verify scenes converted
    assert len(internal["scenes"]) == 2
    assert internal["scenes"][0]["id"] == "scene_1"
    assert "camera" in internal["scenes"][0]
    assert "lighting" in internal["scenes"][0]


@pytest.mark.asyncio
async def test_schema_aiprod_to_manifest_conversion(schema_transformer):
    """Test AIPROD internal → AIPROD manifest conversion."""
    internal_output = {
        "job_id": "job-001",
        "scenes": [
            {
                "id": "scene_1",
                "duration": 30,
                "description": "Test scene",
                "camera": {"movement": "pan"},
                "lighting": {"style": "natural", "mood": "cheerful"},
                "environment": {"location": "garden", "weather": "sunny", "props": []},
                "subjects": {"characters": ["cat"], "actions": []},
                "style": {}
            }
        ],
        "consistency_rules": {
            "cinematography": "natural",
            "color_scheme": ["green"],
            "lighting": "natural"
        },
        "execution_time": 120,
        "cost": 1.5,
        "quality_score": 0.85,
        "backend": "runway_gen3"
    }
    
    manifest = schema_transformer.aiprod_to_manifest(internal_output)
    
    # Verify structure
    assert "production_id" in manifest
    assert "scenes" in manifest
    assert "consistency_markers" in manifest
    assert "metadata" in manifest
    
    # Verify scene conversion
    assert len(manifest["scenes"]) == 1
    assert manifest["scenes"][0]["scene_id"] == "scene_1"
    assert manifest["scenes"][0]["duration_sec"] == 30


@pytest.mark.asyncio
async def test_schema_bidirectional_equivalence(schema_transformer, sample_manifest):
    """Test round-trip transformation preserves critical information."""
    # Forward: AIPROD → internal
    internal = schema_transformer.to_aiprod(sample_manifest)
    
    # Backward: internal → AIPROD
    reconstructed = schema_transformer.aiprod_to_manifest(internal)
    
    # Check equivalence
    assert schema_transformer.schemas_equivalent(sample_manifest, reconstructed)
    
    # Verify scene count preserved
    assert len(reconstructed["scenes"]) == len(sample_manifest["scenes"])
    
    # Verify total duration preserved (within tolerance)
    orig_duration = sample_manifest["total_duration_sec"]
    recon_duration = reconstructed["total_duration_sec"]
    assert abs(orig_duration - recon_duration) / orig_duration < 0.05


@pytest.mark.asyncio
async def test_schema_validation(schema_transformer, sample_manifest):
    """Test schema validation."""
    # Valid manifest
    assert schema_transformer.validate_schema(sample_manifest) == True
    
    # Invalid: missing scenes
    invalid = {"metadata": {}}
    assert schema_transformer.validate_schema(invalid) == False
    
    # Invalid: scenes not a list
    invalid2 = {"scenes": "not a list", "metadata": {}}
    assert schema_transformer.validate_schema(invalid2) == False


# ============================================================================
# Orchestrator Tests
# ============================================================================

@pytest.mark.asyncio
async def test_orchestrator_init_to_analysis_transition(checkpoint_manager):
    """Test orchestrator transitions from INIT to ANALYSIS."""
    # Create mock adapters
    adapters = {}
    
    orchestrator = Orchestrator(
        adapters=adapters,
        checkpoint_manager=checkpoint_manager,
        max_retries=3
    )
    
    request = {
        "request_id": "test-job-001",
        "prompt": "Test video",
        "duration_sec": 60,
        "budget": 1.0,
        "complexity": 0.5,
        "preferences": {},
        "fallback_enabled": True
    }
    
    # Execute (will fail at some point due to missing adapters, but should start)
    response = await orchestrator.execute(request)
    
    # Verify execution started and created checkpoints
    assert response["job_id"] == "test-job-001"
    assert response["checkpoints_created"] > 0


@pytest.mark.asyncio
async def test_orchestrator_checkpoint_recovery(checkpoint_manager):
    """Test orchestrator recovers from failure using checkpoint."""
    # Create adapter that fails on first call, succeeds on second
    class FlakeyAdapter(BaseAdapter):
        def __init__(self):
            super().__init__()
            self.call_count = 0
        
        async def execute(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
            self.call_count += 1
            if self.call_count == 1:
                raise Exception("Simulated failure")
            ctx["memory"]["flakey_executed"] = True
            return ctx
    
    adapters = {
        "input_sanitizer": FlakeyAdapter()
    }
    
    orchestrator = Orchestrator(
        adapters=adapters,
        checkpoint_manager=checkpoint_manager,
        max_retries=3
    )
    
    request = {
        "request_id": "test-job-002",
        "prompt": "Test recovery",
        "duration_sec": 60,
        "budget": 1.0,
        "complexity": 0.8,  # Will go through ANALYSIS
        "preferences": {},
        "fallback_enabled": True
    }
    
    response = await orchestrator.execute(request)
    
    # Verify retry worked
    assert adapters["input_sanitizer"].call_count >= 2
    # First call failed, second succeeded via checkpoint recovery


@pytest.mark.asyncio
async def test_orchestrator_fast_track_path(checkpoint_manager):
    """Test orchestrator uses fast track for simple jobs."""
    adapters = {}
    
    orchestrator = Orchestrator(
        adapters=adapters,
        checkpoint_manager=checkpoint_manager
    )
    
    request = {
        "request_id": "test-job-003",
        "prompt": "Simple video",
        "duration_sec": 30,
        "budget": 0.5,
        "complexity": 0.2,  # Low complexity → fast track
        "preferences": {},
        "fallback_enabled": True
    }
    
    response = await orchestrator.execute(request)
    
    # Verify execution completed
    assert response["job_id"] == "test-job-003"


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_full_pipeline_mock_execution(checkpoint_manager):
    """Test full pipeline execution with all mock adapters."""
    # Create complete mock adapter set
    adapters = {
        "input_sanitizer": MockAdapter(),
        "creative_director": MockAdapter(),
        "visual_translator": MockAdapter(),
        "financial_orchestrator": MockAdapter(),
        "render_executor": MockAdapter(),
        "qa_technical": MockAdapter(),
        "qa_semantic": MockAdapter()
    }
    
    orchestrator = Orchestrator(
        adapters=adapters,
        checkpoint_manager=checkpoint_manager
    )
    
    request = {
        "request_id": "test-job-full",
        "prompt": "Complete pipeline test",
        "duration_sec": 60,
        "budget": 2.0,
        "complexity": 0.5,
        "preferences": {},
        "fallback_enabled": True
    }
    
    response = await orchestrator.execute(request)
    
    # Verify completed successfully
    assert response["status"] == "completed"
    assert response["checkpoints_created"] > 5  # Multiple state transitions
    assert len(response["errors"]) == 0


# ============================================================================
# PHASE 1 Adapter Tests
# ============================================================================

class TestInputSanitizer:
    """Test InputSanitizerAdapter validation logic."""
    
    @pytest.mark.asyncio
    async def test_valid_input(self):
        """Test sanitization of valid input."""
        from aiprod_pipelines.api.adapters.input_sanitizer import InputSanitizerAdapter
        
        adapter = InputSanitizerAdapter()
        
        ctx: Dict[str, Any] = {
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
        from aiprod_pipelines.api.adapters.input_sanitizer import InputSanitizerAdapter
        
        adapter = InputSanitizerAdapter()
        
        ctx: Dict[str, Any] = {
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
        from aiprod_pipelines.api.adapters.input_sanitizer import InputSanitizerAdapter
        
        adapter = InputSanitizerAdapter()
        
        ctx: Dict[str, Any] = {
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
        from aiprod_pipelines.api.adapters.input_sanitizer import InputSanitizerAdapter
        
        adapter = InputSanitizerAdapter()
        
        ctx: Dict[str, Any] = {
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
        from aiprod_pipelines.api.adapters.input_sanitizer import InputSanitizerAdapter
        
        adapter = InputSanitizerAdapter()
        
        ctx: Dict[str, Any] = {
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


class TestCreativeDirector:
    """Test CreativeDirectorAdapter with mocked Gemini."""
    
    @pytest.mark.asyncio
    async def test_manifest_generation_with_cache_miss(self):
        """Test manifest generation (cache miss path)."""
        from aiprod_pipelines.api.adapters.creative import CreativeDirectorAdapter
        
        adapter = CreativeDirectorAdapter(config={"use_gemini": False})  # Use fallback
        
        ctx: Dict[str, Any] = {
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
        from aiprod_pipelines.api.adapters.creative import CreativeDirectorAdapter
        
        adapter = CreativeDirectorAdapter(config={"use_gemini": False})
        
        # First request
        ctx1: Dict[str, Any] = {
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
        ctx2: Dict[str, Any] = {
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


class TestVisualTranslator:
    """Test VisualTranslatorAdapter scene → shot conversion."""
    
    @pytest.mark.asyncio
    async def test_scene_to_shots_conversion(self):
        """Test conversion of scenes to shot specifications."""
        from aiprod_pipelines.api.adapters.visual_translator import VisualTranslatorAdapter
        
        adapter = VisualTranslatorAdapter()
        
        ctx: Dict[str, Any] = {
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
        from aiprod_pipelines.api.adapters.visual_translator import VisualTranslatorAdapter
        
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
        ctx1: Dict[str, Any] = {
            "request_id": "test-seed-1",
            "state": "VISUAL_TRANSLATION",
            "memory": {"production_manifest": manifest},
            "config": {}
        }
        
        result1 = await adapter.execute(ctx1)
        shots1 = result1["memory"]["shot_list"]
        
        # Second translation (same manifest)
        ctx2: Dict[str, Any] = {
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


class TestRenderExecutor:
    """Test RenderExecutorAdapter with batch processing and retry."""
    
    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test batch processing of shots."""
        from aiprod_pipelines.api.adapters.render import RenderExecutorAdapter
        
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
        
        ctx: Dict[str, Any] = {
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
        from aiprod_pipelines.api.adapters.render import RenderExecutorAdapter
        
        adapter = RenderExecutorAdapter(config={"max_retries": 3})
        
        # Test backoff delay calculation
        delays = [adapter._calculate_backoff_delay(i) for i in range(3)]
        
        # Each should be roughly 2x or greater than previous (with jitter)
        assert all(d > 0 for d in delays)
        assert all(d <= 30 for d in delays)  # Capped at 30s
        
        # First delay should be around 1s, not too large
        assert 0.5 <= delays[0] <= 1.5


class TestPipeline:
    """Test full PHASE 1 pipeline integration."""
    
    @pytest.mark.asyncio
    async def test_short_video_pipeline(self):
        """Test complete pipeline execution for short video."""
        from aiprod_pipelines.api.adapters.input_sanitizer import InputSanitizerAdapter
        from aiprod_pipelines.api.adapters.creative import CreativeDirectorAdapter
        from aiprod_pipelines.api.adapters.visual_translator import VisualTranslatorAdapter
        from aiprod_pipelines.api.adapters.render import RenderExecutorAdapter
        
        # Create adapters
        adapters = {
            "input_sanitizer": InputSanitizerAdapter(),
            "creative_director": CreativeDirectorAdapter(config={"use_gemini": False}),
            "visual_translator": VisualTranslatorAdapter(),
            "render_executor": RenderExecutorAdapter(config={"batch_size": 2})
        }
        
        # Step 1: Input Sanitization
        ctx: Dict[str, Any] = {
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


class TestErrorRecovery:
    """Test error handling and recovery in adapters."""
    
    @pytest.mark.asyncio
    async def test_missing_required_fields(self):
        """Test handling of missing required context fields."""
        from aiprod_pipelines.api.adapters.visual_translator import VisualTranslatorAdapter
        
        adapter = VisualTranslatorAdapter()
        
        # Missing production_manifest
        ctx: Dict[str, Any] = {
            "request_id": "test-error-1",
            "state": "VISUAL_TRANSLATION",
            "memory": {},  # Empty!
            "config": {}
        }
        
        with pytest.raises(ValueError):
            await adapter.execute(ctx)
    
    @pytest.mark.asyncio
    async def test_render_partial_failure(self):
        """Test render executor with partial failures (some batches fail)."""
        from aiprod_pipelines.api.adapters.render import RenderExecutorAdapter
        
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
        
        ctx: Dict[str, Any] = {
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


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
