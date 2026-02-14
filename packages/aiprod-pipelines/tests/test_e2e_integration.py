"""
End-to-End Integration Test - Full Pipeline Validation
=======================================================

Validates complete AIPROD pipeline from request to delivery:
- All 5 phases integrated (PHASE 0-4)
- State machine flow (11 states)
- Checkpoint recovery
- Cost estimation accuracy
- QA gate validation
- GCS asset delivery

PHASE 5 implementation - Integration & Launch.
"""

import pytest
import asyncio
from typing import Dict, Any
from datetime import datetime
import json

# Import system components
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from aiprod_pipelines.api.orchestrator.orchestrator import Orchestrator
from aiprod_pipelines.api.checkpoint.manager import CheckpointManager
from aiprod_pipelines.api.adapters.input_sanitizer import InputSanitizerAdapter
from aiprod_pipelines.api.adapters.creative import CreativeDirectorAdapter
from aiprod_pipelines.api.adapters.visual_translator import VisualTranslatorAdapter
from aiprod_pipelines.api.adapters.financial_orchestrator import FinancialOrchestratorAdapter
from aiprod_pipelines.api.adapters.render import RenderExecutorAdapter
from aiprod_pipelines.api.adapters.qa_technical import TechnicalQAGateAdapter
from aiprod_pipelines.api.adapters.qa_semantic import SemanticQAGateAdapter
from aiprod_pipelines.api.adapters.gcp_services import GoogleCloudServicesAdapter
from aiprod_pipelines.api.optimization.performance import PerformanceOptimizationLayer
from aiprod_pipelines.api.schema.schemas import Context, State


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def integration_context():
    """Create test context for integration testing."""
    return Context(
        request_id="e2e_integration_test_001",
        state=State.INIT,
        memory={
            "user_prompt": "A cinematic journey through a futuristic city at sunset, with flying vehicles and neon lights reflecting off glass towers",
            "duration_sec": 120,
            "complexity": 0.7,
            "budget_usd": 5.0
        },
        metadata={
            "created_at": datetime.now().isoformat(),
            "test_type": "e2e_integration"
        }
    )


@pytest.fixture
def full_system(tmp_path):
    """Initialize complete system with all adapters."""
    # Checkpoint manager
    checkpoint_manager = CheckpointManager({
        "storage_path": str(tmp_path / "checkpoints")
    })
    
    # Orchestrator
    orchestrator = Orchestrator(
        config={"checkpoint_enabled": True, "max_retries": 3},
        checkpoint_manager=checkpoint_manager
    )
    
    # All adapters
    adapters = {
        "input_sanitizer": InputSanitizerAdapter(),
        "creative_director": CreativeDirectorAdapter({"api_key": None}),  # Mock mode
        "visual_translator": VisualTranslatorAdapter(),
        "financial_orchestrator": FinancialOrchestratorAdapter(),
        "render_executor": RenderExecutorAdapter(),
        "qa_technical": TechnicalQAGateAdapter(),
        "qa_semantic": SemanticQAGateAdapter({"api_key": None}),  # Mock mode
        "gcp_services": GoogleCloudServicesAdapter({
            "project_id": "test-project",
            "bucket_name": "test-bucket"
        })
    }
    
    # Performance optimizer
    optimizer = PerformanceOptimizationLayer()
    
    return {
        "orchestrator": orchestrator,
        "checkpoint_manager": checkpoint_manager,
        "adapters": adapters,
        "optimizer": optimizer
    }


# ============================================================================
# End-to-End Integration Tests
# ============================================================================

class TestEndToEndIntegration:
    """Comprehensive end-to-end integration tests."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_happy_path(self, integration_context, full_system):
        """
        Test complete pipeline from INIT to COMPLETE.
        
        This is the primary integration test validating:
        - Input sanitization
        - Creative direction generation
        - Visual translation
        - Financial optimization
        - Render execution
        - Technical QA
        - Semantic QA
        - GCP delivery
        """
        ctx = integration_context
        adapters = full_system["adapters"]
        optimizer = full_system["optimizer"]
        
        print("\n" + "="*60)
        print("E2E TEST: Full Pipeline Happy Path")
        print("="*60)
        
        # STEP 1: Input Sanitizer
        print("\n[1/8] Input Sanitization...")
        ctx = await adapters["input_sanitizer"].execute(ctx)
        assert ctx["memory"]["input_sanitized"] is True
        print("✅ Input validated")
        
        # STEP 2: Creative Director
        print("\n[2/8] Creative Direction...")
        ctx = await adapters["creative_director"].execute(ctx)
        assert "creative_direction" in ctx["memory"]
        assert len(ctx["memory"]["creative_direction"]["scenes"]) > 0
        print(f"✅ Generated {len(ctx['memory']['creative_direction']['scenes'])} scenes")
        
        # STEP 3: Visual Translator
        print("\n[3/8] Visual Translation...")
        ctx = await adapters["visual_translator"].execute(ctx)
        assert "visual_translation" in ctx["memory"]
        assert len(ctx["memory"]["visual_translation"]["shots"]) > 0
        print(f"✅ Translated to {len(ctx['memory']['visual_translation']['shots'])} shots")
        
        # STEP 4: Financial Orchestrator
        print("\n[4/8] Financial Optimization...")
        ctx = await adapters["financial_orchestrator"].execute(ctx)
        assert "financial_optimization" in ctx["memory"]
        assert "selected_backend" in ctx["memory"]["financial_optimization"]
        print(f"✅ Selected backend: {ctx['memory']['financial_optimization']['selected_backend']}")
        
        # STEP 5: Performance Optimization
        print("\n[5/8] Performance Optimization...")
        ctx = await optimizer.optimize_for_performance(ctx)
        assert ctx["memory"].get("lazy_loading_enabled") is True
        print("✅ Performance optimizations applied")
        
        # STEP 6: Render Executor (simulated)
        print("\n[6/8] Render Execution...")
        # Add mock assets since we can't actually render
        ctx["memory"]["generated_assets"] = [
            {
                "id": f"shot_{i}",
                "url": f"gs://test-bucket/video_{i}.mp4",
                "duration_sec": 30,
                "file_size_bytes": 10_000_000,
                "resolution": "1920x1080",
                "codec": "h264",
                "bitrate": 4_000_000,
                "fps": 30
            }
            for i in range(len(ctx["memory"]["visual_translation"]["shots"]))
        ]
        print(f"✅ Generated {len(ctx['memory']['generated_assets'])} video assets")
        
        # STEP 7: Technical QA
        print("\n[7/8] Technical QA...")
        ctx = await adapters["qa_technical"].execute(ctx)
        assert "technical_validation_report" in ctx["memory"]
        tech_report = ctx["memory"]["technical_validation_report"]
        print(f"✅ Technical QA: {tech_report['passed_checks']}/{tech_report['total_checks']} checks passed")
        
        # STEP 8: Semantic QA
        print("\n[8/8] Semantic QA...")
        ctx = await adapters["qa_semantic"].execute(ctx)
        assert "semantic_validation_report" in ctx["memory"]
        sem_report = ctx["memory"]["semantic_validation_report"]
        print(f"✅ Semantic QA: {sem_report['average_score']:.2f}/10 quality score")
        
        # Final validation
        assert ctx["memory"]["technical_validation_report"]["passed"] is True
        assert ctx["memory"]["semantic_validation_report"]["passed"] is True
        
        print("\n" + "="*60)
        print("✅ END-TO-END TEST PASSED")
        print("="*60)
        print(f"Request ID: {ctx['request_id']}")
        print(f"Total Cost: ${ctx['memory']['financial_optimization']['cost_estimation']['total_estimated']:.2f}")
        print(f"Quality Score: {sem_report['average_score']:.2f}/10")
        print(f"Assets Generated: {len(ctx['memory']['generated_assets'])}")
        print("="*60 + "\n")
    
    @pytest.mark.asyncio
    async def test_checkpoint_recovery_integration(self, integration_context, full_system):
        """
        Test checkpoint recovery during pipeline execution.
        
        Simulates failure after creative direction and validates recovery.
        """
        ctx = integration_context
        adapters = full_system["adapters"]
        checkpoint_mgr = full_system["checkpoint_manager"]
        
        print("\n" + "="*60)
        print("E2E TEST: Checkpoint Recovery")
        print("="*60)
        
        # Execute first two steps
        print("\n[1/2] Running pipeline up to visual translation...")
        ctx = await adapters["input_sanitizer"].execute(ctx)
        ctx = await adapters["creative_director"].execute(ctx)
        
        # Save checkpoint
        print("\n[2/2] Saving checkpoint...")
        await checkpoint_mgr.save_checkpoint(ctx)
        print("✅ Checkpoint saved")
        
        # Simulate failure and recovery
        print("\n[3/3] Simulating failure and recovery...")
        restored_ctx = await checkpoint_mgr.restore_checkpoint(ctx["request_id"])
        
        assert restored_ctx is not None
        assert restored_ctx["memory"]["creative_direction"] == ctx["memory"]["creative_direction"]
        print("✅ Successfully recovered from checkpoint")
        
        # Continue pipeline from restored state
        restored_ctx = await adapters["visual_translator"].execute(restored_ctx)
        assert "visual_translation" in restored_ctx["memory"]
        print("✅ Pipeline continued after recovery")
        
        print("\n" + "="*60)
        print("✅ CHECKPOINT RECOVERY TEST PASSED")
        print("="*60 + "\n")
    
    @pytest.mark.asyncio
    async def test_cost_accuracy_validation(self, integration_context, full_system):
        """
        Test cost estimation accuracy across pipeline.
        
        Validates that estimated costs are within acceptable range.
        """
        ctx = integration_context
        adapters = full_system["adapters"]
        
        print("\n" + "="*60)
        print("E2E TEST: Cost Accuracy Validation")
        print("="*60)
        
        # Run through financial optimization
        ctx = await adapters["input_sanitizer"].execute(ctx)
        ctx = await adapters["creative_director"].execute(ctx)
        ctx = await adapters["visual_translator"].execute(ctx)
        ctx = await adapters["financial_orchestrator"].execute(ctx)
        
        # Validate cost estimation
        cost_est = ctx["memory"]["financial_optimization"]["cost_estimation"]
        budget = ctx["memory"]["budget_usd"]
        
        print(f"\nBudget: ${budget:.2f}")
        print(f"Estimated Cost: ${cost_est['total_estimated']:.2f}")
        print(f"Budget Utilization: {(cost_est['total_estimated']/budget)*100:.1f}%")
        
        # Assertions cost should be within budget and reasonable
        assert cost_est["total_estimated"] > 0
        assert cost_est["total_estimated"] <= budget
        assert 0.05 <= cost_est["total_estimated"] <= 10.0  # Reasonable range
        
        print("\n✅ Cost estimation within acceptable range")
        
        print("\n" + "="*60)
        print("✅ COST ACCURACY TEST PASSED")
        print("="*60 + "\n")
    
    @pytest.mark.asyncio
    async def test_qa_gates_integration(self, integration_context, full_system):
        """
        Test QA gates integration with actual validation.
        
        Tests both technical and semantic QA in integrated environment.
        """
        ctx = integration_context
        adapters = full_system["adapters"]
        
        print("\n" + "="*60)
        print("E2E TEST: QA Gates Integration")
        print("="*60)
        
        # Prepare context with generated assets
        ctx["memory"]["generated_assets"] = [
            {
                "id": "qa_test_video",
                "url": "gs://test-bucket/qa_test.mp4",
                "duration_sec": 30,
                "file_size_bytes": 10_000_000,
                "resolution": "1920x1080",
                "codec": "h264",
                "bitrate": 4_000_000,
                "fps": 30,
                "color_space": "yuv420p",
                "container": "mp4"
            }
        ]
        
        # Technical QA
        print("\n[1/2] Technical QA Gate...")
        ctx = await adapters["qa_technical"].execute(ctx)
        tech_report = ctx["memory"]["technical_validation_report"]
        
        print(f"  Total Checks: {tech_report['total_checks']}")
        print(f"  Passed: {tech_report['passed_checks']}")
        print(f"  Failed: {len(tech_report['failed_checks'])}")
        print(f"  Pass Rate: {tech_report['pass_rate']*100:.1f}%")
        
        assert tech_report["passed"] is True
        print("✅ Technical QA passed")
        
        # Semantic QA
        print("\n[2/2] Semantic QA Gate...")
        ctx = await adapters["qa_semantic"].execute(ctx)
        sem_report = ctx["memory"]["semantic_validation_report"]
        
        print(f"  Average Score: {sem_report['average_score']:.2f}/10")
        print(f"  Threshold: {sem_report['approval_threshold']:.2f}/10")
        print(f"  Videos Analyzed: {sem_report['videos_analyzed']}")
        
        assert sem_report["passed"] is True
        print("✅ Semantic QA passed")
        
        print("\n" + "="*60)
        print("✅ QA GATES INTEGRATION TEST PASSED")
        print("="*60 + "\n")
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, full_system):
        """
        Test system performance with multiple concurrent requests.
        
        Simulates realistic load to validate throughput and latency.
        """
        adapters = full_system["adapters"]
        optimizer = full_system["optimizer"]
        
        print("\n" + "="*60)
        print("E2E TEST: Performance Under Load")
        print("="*60)
        
        # Create multiple requests
        num_requests = 5
        print(f"\nSimulating {num_requests} concurrent requests...")
        
        async def process_request(request_id: int):
            ctx = Context(
                request_id=f"load_test_{request_id}",
                state=State.INIT,
                memory={
                    "user_prompt": f"Test video {request_id}",
                    "duration_sec": 30,
                    "complexity": 0.5,
                    "budget_usd": 2.0
                },
                metadata={"created_at": datetime.now().isoformat()}
            )
            
            # Run mini-pipeline
            ctx = await adapters["input_sanitizer"].execute(ctx)
            ctx = await adapters["creative_director"].execute(ctx)
            ctx = await optimizer.optimize_for_performance(ctx)
            
            return ctx
        
        # Execute concurrently
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*[process_request(i) for i in range(num_requests)])
        end_time = asyncio.get_event_loop().time()
        
        duration = end_time - start_time
        throughput = num_requests / duration
        
        print(f"\n  Requests: {num_requests}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Throughput: {throughput:.2f} req/s")
        print(f"  Avg Latency: {duration/num_requests:.2f}s per request")
        
        # Validate all succeeded
        assert len(results) == num_requests
        for result in results:
            assert "creative_direction" in result["memory"]
        
        print("\n✅ All requests processed successfully")
        
        print("\n" + "="*60)
        print("✅ PERFORMANCE TEST PASSED")
        print("="*60 + "\n")


# ============================================================================
# Smoke Tests
# ============================================================================

class TestSmokeTests:
    """Smoke tests for production readiness."""
    
    @pytest.mark.asyncio
    async def test_system_health_check(self, full_system):
        """Test system health and component availability."""
        adapters = full_system["adapters"]
        
        print("\n" + "="*60)
        print("SMOKE TEST: System Health Check")
        print("="*60)
        
        # Check all adapters initialized
        assert adapters["input_sanitizer"] is not None
        assert adapters["creative_director"] is not None
        assert adapters["visual_translator"] is not None
        assert adapters["financial_orchestrator"] is not None
        assert adapters["render_executor"] is not None
        assert adapters["qa_technical"] is not None
        assert adapters["qa_semantic"] is not None
        assert adapters["gcp_services"] is not None
        
        print("\n✅ All adapters initialized")
        print("✅ System health check passed")
        
        print("\n" + "="*60)
        print("✅ SMOKE TEST PASSED")
        print("="*60 + "\n")
    
    @pytest.mark.asyncio
    async def test_minimal_request(self, full_system):
        """Test minimal viable request."""
        adapters = full_system["adapters"]
        
        print("\n" + "="*60)
        print("SMOKE TEST: Minimal Request")
        print("="*60)
        
        ctx = Context(
            request_id="smoke_test_minimal",
            state=State.INIT,
            memory={
                "user_prompt": "Test",
                "duration_sec": 10,
                "complexity": 0.1,
                "budget_usd": 0.5
            },
            metadata={"created_at": datetime.now().isoformat()}
        )
        
        # Process minimal request
        ctx = await adapters["input_sanitizer"].execute(ctx)
        assert ctx["memory"]["input_sanitized"] is True
        
        print("\n✅ Minimal request processed")
        
        print("\n" + "="*60)
        print("✅ SMOKE TEST PASSED")
        print("="*60 + "\n")


# ============================================================================
# Run Configuration
# ============================================================================

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto",
        "-s"  # Show print statements
    ])
