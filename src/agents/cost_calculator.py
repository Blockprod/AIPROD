"""
Cost Calculator for AIPROD - Dynamic Pricing Engine

Calculates video generation costs based on:
1. Quality tier selected (GOOD, HIGH, ULTRA)
2. Video duration
3. Complexity level
4. Optional rush delivery
5. Batch discounts
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class Complexity(Enum):
    """Video generation complexity levels"""
    SIMPLE = "simple"              # Single dialog, minimal transitions
    MODERATE = "moderate"          # Multi-scene, standard transitions
    COMPLEX = "complex"            # VFX, multiple characters, advanced effects


class RushDelivery(Enum):
    """Rush delivery options"""
    STANDARD = "standard"          # Normal delivery (30-120 sec)
    EXPRESS_6H = "express_6h"      # 6-hour delivery
    EXPRESS_2H = "express_2h"      # 2-hour delivery
    ON_DEMAND = "on_demand"        # Immediate/ASAP


@dataclass
class CostBreakdown:
    """Detailed cost breakdown"""
    tier_name: str
    base_cost_per_min: float
    duration_sec: int
    duration_min: float
    
    # Multipliers applied
    complexity_multiplier: float
    rush_multiplier: float
    batch_discount: float
    
    # Calculated costs
    base_cost: float               # tier cost × duration
    complexity_adjusted: float     # base × complexity
    with_rush: float              # complexity_adjusted × rush
    with_batch: float             # with_rush × (1 - batch_discount)
    
    # Tax and final
    subtotal_usd: float
    tax_rate: float               # 0.00-0.20
    tax_amount: float
    total_usd: float
    
    # Estimates
    estimated_delivery_sec: int
    
    def __str__(self) -> str:
        return f"${self.total_usd:.2f} (${self.subtotal_usd:.2f} + ${self.tax_amount:.2f} tax)"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier_name,
            "duration_sec": self.duration_sec,
            "base_cost_per_min": self.base_cost_per_min,
            "multipliers": {
                "complexity": self.complexity_multiplier,
                "rush_delivery": self.rush_multiplier,
                "batch_discount": f"-{self.batch_discount*100:.0f}%"
            },
            "cost_breakdown": {
                "base_cost": f"${self.base_cost:.3f}",
                "after_complexity": f"${self.complexity_adjusted:.3f}",
                "after_rush": f"${self.with_rush:.3f}",
                "after_batch_discount": f"${self.with_batch:.3f}",
            },
            "subtotal": f"${self.subtotal_usd:.2f}",
            "tax": f"${self.tax_amount:.2f}",
            "total": f"${self.total_usd:.2f}",
            "estimated_delivery_sec": self.estimated_delivery_sec
        }


class CostCalculator:
    """
    Dynamic cost calculation engine
    
    Formula: Total = Duration × TierRate × ComplexityMultiplier × RushMultiplier × (1 - BatchDiscount)
    """
    
    # Base costs per minute by tier (in USD)
    BASE_RATES = {
        "good": 0.05,      # 1080p professional
        "high": 0.15,      # 4K broadcast
        "ultra": 0.75      # 4K@60fps HDR
    }
    
    # Complexity multipliers
    COMPLEXITY_MULTIPLIERS = {
        Complexity.SIMPLE: 1.0,
        Complexity.MODERATE: 1.2,
        Complexity.COMPLEX: 1.8
    }
    
    # Rush delivery multipliers
    RUSH_MULTIPLIERS = {
        RushDelivery.STANDARD: 1.0,
        RushDelivery.EXPRESS_6H: 1.5,
        RushDelivery.EXPRESS_2H: 2.5,
        RushDelivery.ON_DEMAND: 5.0
    }
    
    # Batch discounts (cumulative)
    BATCH_DISCOUNTS = {
        1: 0.0,      # No discount
        5: 0.05,     # 5% discount for 5 videos
        10: 0.10,    # 10% for 10 videos
        25: 0.15,    # 15% for bulk orders
    }
    
    # Estimated delivery times by tier (in seconds)
    DELIVERY_TIMES = {
        "good": 35,
        "high": 60,
        "ultra": 120
    }
    
    # Tax rate (varies by region, default: GCP default is ~8%)
    DEFAULT_TAX_RATE = 0.08
    
    @classmethod
    def calculate_cost(
        cls,
        tier: str,
        duration_sec: int,
        complexity: str = "moderate",
        rush_delivery: str = "standard",
        batch_count: int = 1,
        tax_rate: Optional[float] = None
    ) -> CostBreakdown:
        """
        Calculate total cost for a video generation
        
        Args:
            tier: "good", "high", or "ultra"
            duration_sec: Video duration in seconds
            complexity: "simple", "moderate", or "complex"
            rush_delivery: "standard", "express_6h", "express_2h", "on_demand"
            batch_count: Number of videos in batch (for volume discount)
            tax_rate: Optional custom tax rate (0.0-0.20)
        
        Returns:
            CostBreakdown with detailed calculation
        """
        
        # Validate inputs
        tier = tier.lower()
        if tier not in cls.BASE_RATES:
            raise ValueError(f"Invalid tier: {tier}")
        
        if duration_sec <= 0:
            raise ValueError(f"Duration must be positive: {duration_sec}")
        
        # Get multipliers
        complexity_enum = Complexity[complexity.upper()] if isinstance(complexity, str) else complexity
        rush_enum = RushDelivery[rush_delivery.upper()] if isinstance(rush_delivery, str) else rush_delivery
        
        complexity_mult = cls.COMPLEXITY_MULTIPLIERS[complexity_enum]
        rush_mult = cls.RUSH_MULTIPLIERS[rush_enum]
        
        # Calculate batch discount
        batch_discount = cls._get_batch_discount(batch_count)
        
        # Duration in minutes
        duration_min = duration_sec / 60.0
        
        # Get base rate for tier
        base_rate = cls.BASE_RATES[tier]
        
        # Calculate costs step by step
        base_cost = duration_min * base_rate
        complexity_adjusted = base_cost * complexity_mult
        with_rush = complexity_adjusted * rush_mult
        with_batch = with_rush * (1.0 - batch_discount)
        
        # Apply tax
        tax_rate = tax_rate or cls.DEFAULT_TAX_RATE
        tax_amount = with_batch * tax_rate
        total_usd = with_batch + tax_amount
        
        # Get estimated delivery time
        estimated_delivery = cls.DELIVERY_TIMES.get(tier, 60)
        if rush_delivery != "standard":
            # Adjust delivery time for rush
            rush_time_factors = {
                "express_6h": 6 * 3600,
                "express_2h": 2 * 3600,
                "on_demand": 300  # 5 minutes
            }
            estimated_delivery = rush_time_factors.get(rush_delivery, estimated_delivery)
        
        return CostBreakdown(
            tier_name=tier,
            base_cost_per_min=base_rate,
            duration_sec=duration_sec,
            duration_min=duration_min,
            complexity_multiplier=complexity_mult,
            rush_multiplier=rush_mult,
            batch_discount=batch_discount,
            base_cost=base_cost,
            complexity_adjusted=complexity_adjusted,
            with_rush=with_rush,
            with_batch=with_batch,
            subtotal_usd=with_batch,
            tax_rate=tax_rate,
            tax_amount=tax_amount,
            total_usd=total_usd,
            estimated_delivery_sec=estimated_delivery
        )
    
    @classmethod
    def _get_batch_discount(cls, batch_count: int) -> float:
        """Get batch discount for given count"""
        if batch_count <= 1:
            return 0.0
        
        # Find applicable discount tier
        applicable_discount = 0.0
        for batch_threshold in sorted(cls.BATCH_DISCOUNTS.keys(), reverse=True):
            if batch_count >= batch_threshold:
                applicable_discount = cls.BATCH_DISCOUNTS[batch_threshold]
                break
        
        return applicable_discount
    
    @classmethod
    def get_alternatives(
        cls,
        duration_sec: int,
        complexity: str = "moderate",
        max_budget: Optional[float] = None,
        batch_count: int = 1
    ) -> List[CostBreakdown]:
        """
        Get cost estimates for all three tiers
        
        Args:
            duration_sec: Video duration
            complexity: Complexity level
            max_budget: Optional budget constraint
            batch_count: Batch size for discounts
        
        Returns:
            List of CostBreakdown objects for each tier, filtered by budget if provided
        """
        tiers = ["good", "high", "ultra"]
        estimates = []
        
        for tier in tiers:
            cost = cls.calculate_cost(
                tier=tier,
                duration_sec=duration_sec,
                complexity=complexity,
                batch_count=batch_count
            )
            
            # Only include if within budget (or no budget specified)
            if max_budget is None or cost.total_usd <= max_budget:
                estimates.append(cost)
        
        return estimates
    
    @classmethod
    def recommend_tier(
        cls,
        duration_sec: int,
        complexity: str = "moderate",
        user_budget: Optional[float] = None,
        priority: str = "quality"  # "quality", "cost", or "balanced"
    ) -> Optional[CostBreakdown]:
        """
        Recommend optimal tier based on user preferences
        
        Args:
            duration_sec: Video duration
            complexity: Complexity level
            user_budget: Optional budget constraint
            priority: "quality" = highest tier within budget
                     "cost" = lowest cost option
                     "balanced" = mid-tier best value
        
        Returns:
            Recommended CostBreakdown or None if no suitable option
        """
        
        # Get alternatives
        max_budget = user_budget * 1.5 if user_budget else None
        alternatives = cls.get_alternatives(
            duration_sec=duration_sec,
            complexity=complexity,
            max_budget=max_budget
        )
        
        if not alternatives:
            return None
        
        # Sort by total cost
        alternatives.sort(key=lambda x: x.total_usd)
        
        if priority == "quality":
            # Return highest quality (last in list)
            return alternatives[-1]
        elif priority == "cost":
            # Return lowest cost
            return alternatives[0]
        else:  # balanced
            # Return middle tier if available, else best available
            mid_idx = len(alternatives) // 2
            return alternatives[mid_idx]
    
    @classmethod
    def get_pricing_table(cls) -> Dict[str, Any]:
        """Get comprehensive pricing table for documentation"""
        return {
            "base_rates": cls.BASE_RATES,
            "complexity_multipliers": {name.value: mult for name, mult in cls.COMPLEXITY_MULTIPLIERS.items()},
            "rush_multipliers": {name.value: mult for name, mult in cls.RUSH_MULTIPLIERS.items()},
            "batch_discounts": cls.BATCH_DISCOUNTS,
            "delivery_times_sec": cls.DELIVERY_TIMES,
            "example_costs": {
                "30sec_moderate_good": cls.calculate_cost("good", 30, "moderate").total_usd,
                "30sec_moderate_high": cls.calculate_cost("high", 30, "moderate").total_usd,
                "30sec_moderate_ultra": cls.calculate_cost("ultra", 30, "moderate").total_usd,
            }
        }


if __name__ == "__main__":
    # Test calculations
    logger.info("Cost Calculator Test")
    logger.info("=" * 60)
    
    # Example 1: 30-second GOOD tier video, moderate complexity
    cost = CostCalculator.calculate_cost("good", 30, "moderate")
    logger.info(f"\n📺 Example 1: GOOD Tier (1080p)")
    logger.info(f"   Duration: 30 seconds")
    logger.info(f"   Complexity: Moderate")
    logger.info(f"   Total Cost: {cost}")
    
    # Example 2: 60-second HIGH tier with rush
    cost = CostCalculator.calculate_cost("high", 60, "moderate", rush_delivery="express_6h")
    logger.info(f"\n📺 Example 2: HIGH Tier (4K) + Rush")
    logger.info(f"   Duration: 60 seconds")
    logger.info(f"   Rush: 6-hour delivery")
    logger.info(f"   Total Cost: {cost}")
    
    # Example 3: All tiers comparison
    logger.info(f"\n📺 Example 3: All Tiers Comparison (30sec, moderate)")
    alternatives = CostCalculator.get_alternatives(30, "moderate")
    for alt in alternatives:
        logger.info(f"   {alt.tier_name.upper()}: {alt}")
