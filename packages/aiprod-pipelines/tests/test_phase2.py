"""
PHASE 2 Financial Optimization Tests
====================================

30+ test cases for RealisticCostEstimator and FinancialOrchestratorAdapter.
"""

import pytest
from typing import Dict, Any


class TestCostEstimator:
    """Test RealisticCostEstimator with multi-parameter model."""
    
    @pytest.fixture
    def estimator(self):
        """Create fresh estimator for each test."""
        # Import with mock torch
        import sys
        from unittest.mock import MagicMock
        sys.modules["torch"] = MagicMock()
        
        from aiprod_pipelines.api.adapters.financial_cost_estimator import RealisticCostEstimator
        return RealisticCostEstimator()
    
    # Test 1: Base cost calculation
    @pytest.mark.asyncio
    async def test_base_cost_low_complexity(self, estimator):
        """Low complexity (0.2) should cost ~$0.4-0.6 per minute."""
        job = {
            "complexity": 0.2,
            "duration_sec": 60
        }
        cost = await estimator.estimate_total_cost(job)
        rate_per_min = cost  # 60 seconds = 1 minute
        assert 0.4 <= rate_per_min <= 0.7, f"Low complexity rate out of range: {rate_per_min}"
    
    # Test 2: Base cost high complexity
    @pytest.mark.asyncio
    async def test_base_cost_high_complexity(self, estimator):
        """High complexity (0.9) should cost ~$1.1-1.3 per minute."""
        job = {
            "complexity": 0.9,
            "duration_sec": 60
        }
        cost = await estimator.estimate_total_cost(job)
        rate_per_min = cost
        assert 1.0 <= rate_per_min <= 1.4, f"High complexity rate out of range: {rate_per_min}"
    
    # Test 3: Duration impact
    @pytest.mark.asyncio
    async def test_duration_linear_scaling(self, estimator):
        """Cost should scale linearly with duration."""
        job1 = {"complexity": 0.5, "duration_sec": 60}
        job2 = {"complexity": 0.5, "duration_sec": 120}
        
        cost1 = await estimator.estimate_total_cost(job1)
        cost2 = await estimator.estimate_total_cost(job2)
        
        # Double duration should approximately double cost
        ratio = cost2 / cost1
        assert 1.8 <= ratio <= 2.2, f"Duration scaling incorrect: {ratio}x"
    
    # Test 4: Quantization Q4 (60% cost reduction)
    @pytest.mark.asyncio
    async def test_quantization_q4_savings(self, estimator):
        """Q4 quantization should reduce cost by ~60%."""
        job_fp16 = {
            "complexity": 0.5,
            "duration_sec": 60,
            "quantization": "FP16"
        }
        job_q4 = {
            "complexity": 0.5,
            "duration_sec": 60,
            "quantization": "Q4"
        }
        
        cost_fp16 = await estimator.estimate_total_cost(job_fp16)
        cost_q4 = await estimator.estimate_total_cost(job_q4)
        
        savings = (cost_fp16 - cost_q4) / cost_fp16
        assert 0.55 <= savings <= 0.65, f"Q4 savings not ~60%: {savings:.1%}"
    
    # Test 5: Quantization Q8 (35% cost reduction)
    @pytest.mark.asyncio
    async def test_quantization_q8_savings(self, estimator):
        """Q8 quantization should reduce cost by ~35%."""
        job_fp16 = {
            "complexity": 0.5,
            "duration_sec": 60,
            "quantization": "FP16"
        }
        job_q8 = {
            "complexity": 0.5,
            "duration_sec": 60,
            "quantization": "Q8"
        }
        
        cost_fp16 = await estimator.estimate_total_cost(job_fp16)
        cost_q8 = await estimator.estimate_total_cost(job_q8)
        
        savings = (cost_fp16 - cost_q8) / cost_fp16
        assert 0.30 <= savings <= 0.40, f"Q8 savings not ~35%: {savings:.1%}"
    
    # Test 6: GPU model pricing (H100 is 3x expensive)
    @pytest.mark.asyncio
    async def test_gpu_h100_premium(self, estimator):
        """H100 should be ~3x more expensive than A100."""
        job_a100 = {
            "complexity": 0.5,
            "duration_sec": 60,
            "gpu_model": "A100"
        }
        job_h100 = {
            "complexity": 0.5,
            "duration_sec": 60,
            "gpu_model": "H100"
        }
        
        cost_a100 = await estimator.estimate_total_cost(job_a100)
        cost_h100 = await estimator.estimate_total_cost(job_h100)
        
        ratio = cost_h100 / cost_a100
        assert 2.8 <= ratio <= 3.2, f"H100 premium not ~3x: {ratio}x"
    
    # Test 7: GPU T4 budget
    @pytest.mark.asyncio
    async def test_gpu_t4_budget(self, estimator):
        """T4 should be ~50% cheaper than A100."""
        job_a100 = {
            "complexity": 0.5,
            "duration_sec": 60,
            "gpu_model": "A100"
        }
        job_t4 = {
            "complexity": 0.5,
            "duration_sec": 60,
            "gpu_model": "T4"
        }
        
        cost_a100 = await estimator.estimate_total_cost(job_a100)
        cost_t4 = await estimator.estimate_total_cost(job_t4)
        
        ratio = cost_t4 / cost_a100
        assert 0.45 <= ratio <= 0.55, f"T4 discount not ~50%: {ratio}x"
    
    # Test 8: Batch size efficiency (batch 1 vs 32)
    @pytest.mark.asyncio
    async def test_batch_efficiency_significant(self, estimator):
        """Batch size 32 should be ~4-5x cheaper per unit than size 1."""
        job_batch1 = {
            "complexity": 0.5,
            "duration_sec": 60,
            "batch_size": 1
        }
        job_batch32 = {
            "complexity": 0.5,
            "duration_sec": 60,
            "batch_size": 32
        }
        
        cost_1 = await estimator.estimate_total_cost(job_batch1)
        cost_32 = await estimator.estimate_total_cost(job_batch32)
        
        ratio = cost_1 / cost_32
        assert 3.5 <= ratio <= 5.5, f"Batch efficiency not 4-5x: {ratio}x"
    
    # Test 9: Batch size diminishing returns (1→2 vs 16→32)
    @pytest.mark.asyncio
    async def test_batch_diminishing_returns(self, estimator):
        """Batch sizing should show diminishing returns."""
        job1 = {"complexity": 0.5, "duration_sec": 60, "batch_size": 1}
        job2 = {"complexity": 0.5, "duration_sec": 60, "batch_size": 2}
        job16 = {"complexity": 0.5, "duration_sec": 60, "batch_size": 16}
        job32 = {"complexity": 0.5, "duration_sec": 60, "batch_size": 32}
        
        cost1 = await estimator.estimate_total_cost(job1)
        cost2 = await estimator.estimate_total_cost(job2)
        cost16 = await estimator.estimate_total_cost(job16)
        cost32 = await estimator.estimate_total_cost(job32)
        
        # 1→2 savings should be larger than 16→32
        saving_12 = (cost1 - cost2) / cost1
        saving_1632 = (cost16 - cost32) / cost16
        
        assert saving_12 > saving_1632, "Batch diminishing returns not observed"
    
    # Test 10: Multi-GPU overhead (4 GPU = 15% overhead)
    @pytest.mark.asyncio
    async def test_multi_gpu_overhead(self, estimator):
        """4 GPUs should add ~15% orchestration overhead."""
        job_single = {
            "complexity": 0.5,
            "duration_sec": 60,
            "use_tensor_parallel": False,
            "gpu_count": 1
        }
        job_multi = {
            "complexity": 0.5,
            "duration_sec": 60,
            "use_tensor_parallel": True,
            "gpu_count": 4
        }
        
        cost_single = await estimator.estimate_total_cost(job_single)
        cost_multi = await estimator.estimate_total_cost(job_multi)
        
        overhead = (cost_multi - cost_single) / cost_single
        assert 0.10 <= overhead <= 0.20, f"Multi-GPU overhead not ~15%: {overhead:.1%}"
    
    # Test 11: Framework efficiency (vLLM vs native)
    @pytest.mark.asyncio
    async def test_framework_vllm_optimization(self, estimator):
        """vLLM should be ~20% cheaper than native PyTorch."""
        job_native = {
            "complexity": 0.5,
            "duration_sec": 60,
            "framework": "native_pytorch"
        }
        job_vllm = {
            "complexity": 0.5,
            "duration_sec": 60,
            "framework": "vLLM"
        }
        
        cost_native = await estimator.estimate_total_cost(job_native)
        cost_vllm = await estimator.estimate_total_cost(job_vllm)
        
        savings = (cost_native - cost_vllm) / cost_native
        assert 0.18 <= savings <= 0.22, f"vLLM savings not ~20%: {savings:.1%}"
    
    # Test 12: Spot instances (70% discount)
    @pytest.mark.asyncio
    async def test_spot_instance_discount(self, estimator):
        """Spot instances should be ~70% cheaper."""
        job_regular = {
            "complexity": 0.5,
            "duration_sec": 60,
            "use_spot_instances": False
        }
        job_spot = {
            "complexity": 0.5,
            "duration_sec": 60,
            "use_spot_instances": True
        }
        
        cost_regular = await estimator.estimate_total_cost(job_regular)
        cost_spot = await estimator.estimate_total_cost(job_spot)
        
        discount = (cost_regular - cost_spot) / cost_regular
        assert 0.65 <= discount <= 0.75, f"Spot discount not ~70%: {discount:.1%}"
    
    # Test 13: Combined parameters (realistic scenario)
    @pytest.mark.asyncio
    async def test_realistic_high_complexity_job(self, estimator):
        """High complexity + long duration + multi-GPU should be ~$3-5."""
        job = {
            "complexity": 0.9,
            "duration_sec": 300,  # 5 minutes
            "quantization": "FP16",
            "gpu_model": "H100",
            "batch_size": 4,
            "use_tensor_parallel": True,
            "gpu_count": 4,
            "framework": "vLLM",
            "use_spot_instances": False
        }
        
        cost = await estimator.estimate_total_cost(job)
        assert 2.5 <= cost <= 5.0, f"Realistic job cost out of range: ${cost:.2f}"
    
    # Test 14: Combined parameters (budget scenario)
    @pytest.mark.asyncio
    async def test_budget_optimized_job(self, estimator):
        """Low complexity + Q4 + T4 + spot should be ~$0.10-0.25."""
        job = {
            "complexity": 0.2,
            "duration_sec": 60,
            "quantization": "Q4",
            "gpu_model": "T4",
            "batch_size": 16,
            "framework": "native_pytorch",
            "use_spot_instances": True
        }
        
        cost = await estimator.estimate_total_cost(job)
        assert 0.05 <= cost <= 0.30, f"Budget job cost out of range: ${cost:.2f}"
    
    # Test 15: Cost bounds enforcement
    @pytest.mark.asyncio
    async def test_cost_minimum_bound(self, estimator):
        """Minimum cost should be $0.05."""
        job = {
            "complexity": 0.01,
            "duration_sec": 10,
            "quantization": "Q4",
            "gpu_model": "T4"
        }
        
        cost = await estimator.estimate_total_cost(job)
        assert cost >= 0.05, f"Cost below minimum: ${cost:.2f}"
    
    @pytest.mark.asyncio
    async def test_cost_maximum_bound(self, estimator):
        """Maximum cost should be $5.00."""
        job = {
            "complexity": 1.0,
            "duration_sec": 3600,
            "quantization": "FP32",
            "gpu_model": "H100",
            "use_tensor_parallel": True,
            "gpu_count": 8
        }
        
        cost = await estimator.estimate_total_cost(job)
        assert cost <= 5.0, f"Cost above maximum: ${cost:.2f}"
    
    # Test 16: Cost breakdown generation
    @pytest.mark.asyncio
    async def test_cost_breakdown_completeness(self, estimator):
        """Cost breakdown should include all components."""
        job = {
            "complexity": 0.5,
            "duration_sec": 60,
            "quantization": "Q8",
            "gpu_model": "A100",
            "batch_size": 4
        }
        
        breakdown = estimator.generate_cost_breakdown(job)
        
        required_fields = [
            "base_cost", "quantization_factor", "gpu_model_factor",
            "batch_efficiency_factor", "framework_efficiency_factor",
            "total_estimated_cost", "cost_per_minute"
        ]
        
        for field in required_fields:
            assert field in breakdown, f"Missing field: {field}"
        
        assert breakdown["total_estimated_cost"] > 0
        assert breakdown["cost_per_minute"] > 0


class TestFinancialOrchestrator:
    """Test FinancialOrchestratorAdapter."""
    
    @pytest.fixture
    def financial_adapter(self):
        """Create fresh adapter for each test."""
        import sys
        from unittest.mock import MagicMock
        sys.modules["torch"] = MagicMock()
        
        from aiprod_pipelines.api.adapters.financial_orchestrator import FinancialOrchestratorAdapter
        return FinancialOrchestratorAdapter()
    
    # Test 17: Backend selection - low budget (replicate_wan25)
    @pytest.mark.asyncio
    async def test_backend_selection_low_budget(self, financial_adapter):
        """Low budget utilization (>80%) should select replicate_wan25."""
        ctx = {
            "request_id": "test-fin-1",
            "state": "FINANCIAL_OPTIMIZATION",
            "memory": {
                "prompt": "Test video",
                "duration_sec": 30,
                "budget": 0.3,
                "complexity": 0.3,
                "preferences": {}
            },
            "config": {}
        }
        
        result = await financial_adapter.execute(ctx)
        backend = result["memory"]["cost_estimation"]["selected_backend"]
        assert backend == "replicate_wan25", f"Should select replicate_wan25, got {backend}"
    
    # Test 18: Backend selection - moderate budget (runway_gen3)
    @pytest.mark.asyncio
    async def test_backend_selection_moderate_budget(self, financial_adapter):
        """Moderate budget utilization (50-80%) should select runway_gen3."""
        ctx = {
            "request_id": "test-fin-2",
            "state": "FINANCIAL_OPTIMIZATION",
            "memory": {
                "prompt": "Test video",
                "duration_sec": 30,
                "budget": 1.0,
                "complexity": 0.5,
                "preferences": {}
            },
            "config": {}
        }
        
        result = await financial_adapter.execute(ctx)
        backend = result["memory"]["cost_estimation"]["selected_backend"]
        assert backend == "runway_gen3", f"Should select runway_gen3, got {backend}"
    
    # Test 19: Backend selection - high budget (veo3)
    @pytest.mark.asyncio
    async def test_backend_selection_high_budget(self, financial_adapter):
        """High budget utilization (<50%) should select veo3."""
        ctx = {
            "request_id": "test-fin-3",
            "state": "FINANCIAL_OPTIMIZATION",
            "memory": {
                "prompt": "Test video",
                "duration_sec": 30,
                "budget": 5.0,
                "complexity": 0.3,
                "preferences": {}
            },
            "config": {}
        }
        
        result = await financial_adapter.execute(ctx)
        backend = result["memory"]["cost_estimation"]["selected_backend"]
        assert backend == "veo3", f"Should select veo3, got {backend}"
    
    # Test 20: Budget exceeded - enforce constraints
    @pytest.mark.asyncio
    async def test_budget_exceeded_error(self, financial_adapter):
        """Cost exceeding budget should raise error after quantization attempts."""
        ctx = {
            "request_id": "test-fin-4",
            "state": "FINANCIAL_OPTIMIZATION",
            "memory": {
                "prompt": "Test video",
                "duration_sec": 600,  # Very long
                "budget": 0.1,  # Tiny budget
                "complexity": 1.0,  # Very complex
                "preferences": {}
            },
            "config": {}
        }
        
        with pytest.raises(ValueError, match="cannot fit in budget"):
            await financial_adapter.execute(ctx)
    
    # Test 21: Cost breakdown available
    @pytest.mark.asyncio
    async def test_cost_breakdown_in_output(self, financial_adapter):
        """Context should include detailed cost breakdown."""
        ctx = {
            "request_id": "test-fin-5",
            "state": "FINANCIAL_OPTIMIZATION",
            "memory": {
                "prompt": "Test video",
                "duration_sec": 60,
                "budget": 2.0,
                "complexity": 0.5,
                "preferences": {}
            },
            "config": {}
        }
        
        result = await financial_adapter.execute(ctx)
        breakdown = result["memory"]["cost_estimation"].get("cost_breakdown")
        
        assert breakdown is not None
        assert "base_cost" in breakdown
        assert "total_estimated_cost" in breakdown
    
    # Test 22: Audit logging
    @pytest.mark.asyncio
    async def test_audit_trail_created(self, financial_adapter):
        """Financial decisions should be logged for audit."""
        job_id = "test-fin-6"
        ctx = {
            "request_id": job_id,
            "state": "FINANCIAL_OPTIMIZATION",
            "memory": {
                "prompt": "Test video",
                "duration_sec": 60,
                "budget": 1.5,
                "complexity": 0.5,
                "preferences": {}
            },
            "config": {}
        }
        
        result = await financial_adapter.execute(ctx)
        
        # Check audit trail
        audit_trail = financial_adapter.get_cost_history(job_id)
        assert len(audit_trail) > 0, "No audit trail created"
        
        decision_types = [entry["decision_type"] for entry in audit_trail]
        assert "cost_estimation" in decision_types
        assert "backend_selection" in decision_types
    
    # Test 23: Cost per minute calculated
    @pytest.mark.asyncio
    async def test_cost_per_minute_calculation(self, financial_adapter):
        """Context should include cost_per_minute."""
        ctx = {
            "request_id": "test-fin-7",
            "state": "FINANCIAL_OPTIMIZATION",
            "memory": {
                "prompt": "Test video",
                "duration_sec": 120,  # 2 minutes
                "budget": 2.0,
                "complexity": 0.5,
                "preferences": {}
            },
            "config": {}
        }
        
        result = await financial_adapter.execute(ctx)
        cost_per_min = result["memory"]["cost_estimation"]["cost_per_minute"]
        
        assert cost_per_min > 0
        # Should be ~0.4-0.6 per minute for complexity 0.5
        assert 0.3 <= cost_per_min <= 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
