"""
Financial Orchestrator Adapter - Cost-Aware Backend Selection
==============================================================

Implements cost estimation and intelligent backend selection
based on multi-parameter cost model and client budget constraints.

PHASE 2 implementation (Weeks 7-8).
"""

from typing import Dict, Any, Tuple
import asyncio
import logging
from datetime import datetime
from .base import BaseAdapter
from .financial_cost_estimator import RealisticCostEstimator
from ..schema.schemas import Context


logger = logging.getLogger(__name__)


class AuditLogger:
    """Track all financial and backend selection decisions."""
    
    def __init__(self):
        """Initialize audit logger."""
        self.audit_log = []
    
    def log_decision(
        self, 
        job_id: str, 
        decision_type: str, 
        data: Dict[str, Any]
    ) -> None:
        """
        Log a financial decision for audit trail.
        
        Args:
            job_id: Job identifier
            decision_type: Type of decision (estimate, backend_selection, etc.)
            data: Decision details
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "job_id": job_id,
            "decision_type": decision_type,
            "data": data
        }
        self.audit_log.append(entry)
        logger.info(f"Decision logged: {decision_type} for {job_id}")
    
    def get_audit_trail(self, job_id: str) -> list:
        """Get all decisions for a job."""
        return [e for e in self.audit_log if e["job_id"] == job_id]


class FinancialOrchestratorAdapter(BaseAdapter):
    """
    Financial orchestrator with multi-parameter cost estimation.
    
    Responsibilities:
    1. Estimate cost using 8-parameter model
    2. Select backend based on cost vs budget
    3. Log all financial decisions
    4. Provide cost breakdown to client
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize financial orchestrator."""
        super().__init__(config)
        self.cost_estimator = RealisticCostEstimator(config=config)
        self.audit_logger = AuditLogger()
        
        # Default parameters (can be overridden per job)
        self.default_framework = "vLLM"
        self.default_gpu = "A100"
        self.default_quantization = "FP16"
        self.use_spot_instances = config.get("use_spot_instances", False) if config else False
    
    async def execute(self, ctx: Context) -> Context:
        """
        Execute financial optimization: estimate costs and select backend.
        
        Args:
            ctx: Context with sanitized inputs
            
        Returns:
            Context with cost_estimation and backend selection
        """
        # Validate context
        if not self.validate_context(ctx, ["prompt", "duration_sec", "budget", "complexity"]):
            raise ValueError("Missing required fields for financial optimization")
        
        # Extract relevant parameters
        job_id = ctx.get("request_id", "unknown")
        memory = ctx.get("memory", {})
        duration_sec = memory.get("duration_sec", 60)
        budget = memory.get("budget", 1.0)
        complexity = memory.get("complexity", 0.5)
        preferences = memory.get("preferences", {})
        
        # Build job configuration for cost estimation
        job_config = {
            "complexity": complexity,
            "duration_sec": duration_sec,
            "batch_size": 4,  # Default batch for video generation
            "quantization": preferences.get("quantization", self.default_quantization),
            "gpu_model": preferences.get("gpu_model", self.default_gpu),
            "framework": self.default_framework,
            "use_spot_instances": self.use_spot_instances,
            # Multi-GPU not typical for video generation (used for LLMs)
            "use_tensor_parallel": False,
            "gpu_count": 1
        }
        
        # 1. Estimate total cost
        estimated_cost = await self.cost_estimator.estimate_total_cost(job_config)
        cost_breakdown = self.cost_estimator.generate_cost_breakdown(job_config)
        
        # Log cost estimation
        self.audit_logger.log_decision(
            job_id=job_id,
            decision_type="cost_estimation",
            data={
                "estimated_cost": estimated_cost,
                "client_budget": budget,
                "budget_utilization": estimated_cost / budget if budget > 0 else 0,
                "parameters": job_config,
                "breakdown": cost_breakdown
            }
        )
        
        self.log("info", "Estimated cost", cost=estimated_cost, budget=budget)
        
        # 2. Validate cost is within budget
        if estimated_cost > budget:
            # Over budget - decide on fallback option
            self.log("warning", f"Estimated cost exceeds budget", 
                     estimated=estimated_cost, budget=budget)
            
            # Try cheaper quantization
            job_config["quantization"] = "Q8"  # Drop from FP16 to Q8
            estimated_cost = await self.cost_estimator.estimate_total_cost(job_config)
            
            if estimated_cost > budget:
                # Still over - use Q4
                job_config["quantization"] = "Q4"
                estimated_cost = await self.cost_estimator.estimate_total_cost(job_config)
            
            if estimated_cost > budget:
                # Even Q4 doesn't fit - error
                self.log("error", "Cannot fit job in budget even with max compression",
                         estimated=estimated_cost, budget=budget)
                raise ValueError(f"Job cost ${estimated_cost:.2f} cannot fit in budget ${budget:.2f}")
        
        # 3. Select backend based on cost/budget ratio
        selected_backend = self.cost_estimator.get_backend_recommendation(
            estimated_cost, budget
        )
        
        self.log("info", f"Backend selected: {selected_backend}", 
                 budget_util=(estimated_cost / budget if budget > 0 else 0))
        
        # Log backend selection
        self.audit_logger.log_decision(
            job_id=job_id,
            decision_type="backend_selection",
            data={
                "selected_backend": selected_backend,
                "estimated_cost": estimated_cost,
                "rationale": self._get_selection_rationale(estimated_cost, budget, selected_backend)
            }
        )
        
        # 4. Calculate cost per minute (for monitoring)
        cost_per_minute = (estimated_cost / (duration_sec / 60.0)) if duration_sec > 0 else 0
        
        # 5. Update context with financial information
        ctx["memory"]["cost_estimation"] = {
            "estimated_cost": estimated_cost,
            "cost_per_minute": cost_per_minute,
            "selected_backend": selected_backend,
            "quantization": job_config["quantization"],
            "gpu_model": job_config["gpu_model"],
            "framework": job_config["framework"],
            "budget_utilization": estimated_cost / budget if budget > 0 else 0,
            "confidence": 0.89,  # Multi-parameter model confidence
            "cost_breakdown": cost_breakdown,
            "cost_timestamp": datetime.utcnow().isoformat()
        }
        
        # Store audit trail reference
        ctx["memory"]["audit_trail_reference"] = job_id
        
        self.log("info", "Financial optimization complete",
                 estimated_cost=estimated_cost,
                 backend=selected_backend,
                 confidence=0.89)
        
        return ctx
    
    def _get_selection_rationale(
        self, 
        estimated_cost: float, 
        budget: float,
        selected_backend: str
    ) -> str:
        """
        Generate human-readable rationale for backend selection.
        
        Args:
            estimated_cost: Estimated cost
            budget: Client budget
            selected_backend: Selected backend
            
        Returns:
            Rationale string
        """
        utilization = estimated_cost / budget if budget > 0 else 0
        
        if utilization >= 0.8:
            return (f"Budget utilization {utilization:.1%} is high. "
                   f"Selected {selected_backend} (cheapest option).")
        elif utilization >= 0.5:
            return (f"Budget utilization {utilization:.1%} is moderate. "
                   f"Selected {selected_backend} (balanced cost/quality).")
        else:
            return (f"Budget utilization {utilization:.1%} is low. "
                   f"Selected {selected_backend} (premium quality).")
    
    async def optimize_for_constraints(
        self,
        target_cost: float,
        target_quality: str = "high"
    ) -> Dict[str, Any]:
        """
        Suggest optimizations to meet cost/quality targets.
        
        Args:
            target_cost: Target cost ceiling
            target_quality: "high", "medium", "low"
            
        Returns:
            Suggested optimizations
        """
        suggestions = {
            "high": {
                "quantization": "FP16",
                "framework": "vLLM",
                "cost_reduction_potential": "5-10%"
            },
            "medium": {
                "quantization": "Q8",
                "framework": "TensorRT",
                "cost_reduction_potential": "25-35%"
            },
            "low": {
                "quantization": "Q4",
                "framework": "native_pytorch",
                "cost_reduction_potential": "50-60%"
            }
        }
        
        return suggestions.get(target_quality, suggestions["high"])
    
    def get_cost_history(self, job_id: str) -> list:
        """Get cost decisions for a job."""
        return self.audit_logger.get_audit_trail(job_id)
