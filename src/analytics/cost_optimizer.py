"""
Cost optimization engine using ML and performance data
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class OptimizationPriority(str, Enum):
    """Optimization priorities"""
    COST = "cost"
    PERFORMANCE = "performance"
    BALANCED = "balanced"


@dataclass
class CostOpportunity:
    """Cost optimization opportunity"""
    title: str
    description: str
    potential_savings_percentage: float  # 0-100%
    implementation_effort: str  # "low", "medium", "high"
    risk_level: str  # "low", "medium", "high"
    priority_score: float  # 0-100%
    estimated_monthly_savings: float  # in dollars


@dataclass
class RegionCostAnalysis:
    """Cost analysis for a region"""
    region_id: str
    region_name: str
    monthly_cost: float
    monthly_requests: int
    cost_per_request: float
    capacity_utilization: float  # 0-100%
    error_rate: float  # 0-100%
    avg_latency_ms: float
    efficiency_score: float  # 0-100%, higher is better


@dataclass
class CostOptimizationPlan:
    """Complete cost optimization plan"""
    total_current_monthly_cost: float
    total_potential_monthly_savings: float
    savings_percentage: float
    opportunities: List[CostOpportunity] = field(default_factory=list)
    regional_analysis: List[RegionCostAnalysis] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    quick_wins: List[CostOpportunity] = field(default_factory=list)  # High priority, low effort


class CostOptimizer:
    """Cost optimization engine"""

    def __init__(self):
        self.region_costs: Dict[str, float] = {}
        self.region_metrics: Dict[str, Dict[str, Any]] = {}

    def add_region_data(
        self,
        region_id: str,
        region_name: str,
        monthly_cost: float,
        monthly_requests: int,
        capacity_utilization: float,
        error_rate: float,
        avg_latency_ms: float,
    ):
        """Add region cost and performance data"""
        self.region_costs[region_id] = monthly_cost
        self.region_metrics[region_id] = {
            "region_name": region_name,
            "monthly_cost": monthly_cost,
            "monthly_requests": monthly_requests,
            "capacity_utilization": capacity_utilization,
            "error_rate": error_rate,
            "avg_latency_ms": avg_latency_ms,
        }

    def analyze_regional_costs(self) -> List[RegionCostAnalysis]:
        """Analyze costs by region"""
        analyses = []

        for region_id, metrics in self.region_metrics.items():
            cost_per_request = (
                metrics["monthly_cost"] / metrics["monthly_requests"]
                if metrics["monthly_requests"] > 0
                else 0
            )

            # Efficiency score: high capacity utilization + low latency + low errors = high score
            capacity_score = metrics["capacity_utilization"]  # Higher is better (use what you pay for)
            latency_score = max(0, 100 - (metrics["avg_latency_ms"] / 50))  # Lower latency is better
            error_score = max(0, 100 - (metrics["error_rate"] * 2))  # Lower errors is better
            efficiency_score = (capacity_score + latency_score + error_score) / 3

            analysis = RegionCostAnalysis(
                region_id=region_id,
                region_name=metrics["region_name"],
                monthly_cost=metrics["monthly_cost"],
                monthly_requests=metrics["monthly_requests"],
                cost_per_request=round(cost_per_request, 4),
                capacity_utilization=metrics["capacity_utilization"],
                error_rate=metrics["error_rate"],
                avg_latency_ms=metrics["avg_latency_ms"],
                efficiency_score=round(efficiency_score, 2),
            )
            analyses.append(analysis)

        return analyses

    def identify_optimization_opportunities(self, priority: OptimizationPriority = OptimizationPriority.BALANCED) -> List[CostOpportunity]:
        """Identify cost optimization opportunities"""
        opportunities = []

        # Analyze regional costs
        regional_analyses = self.analyze_regional_costs()

        if not regional_analyses:
            return opportunities

        # Find underutilized regions
        avg_utilization = sum(r.capacity_utilization for r in regional_analyses) / len(regional_analyses)

        for analysis in regional_analyses:
            if analysis.capacity_utilization < avg_utilization * 0.6:  # 40% less than average
                savings = analysis.monthly_cost * 0.3  # Potential 30% savings
                opportunity = CostOpportunity(
                    title=f"Consolidate traffic from {analysis.region_name}",
                    description=f"Region {analysis.region_name} is only {analysis.capacity_utilization}% utilized. "
                    f"Consider distributing traffic to more efficient regions or downsizing capacity.",
                    potential_savings_percentage=30,
                    implementation_effort="medium",
                    risk_level="medium",
                    priority_score=80 if priority == OptimizationPriority.COST else 60,
                    estimated_monthly_savings=savings,
                )
                opportunities.append(opportunity)

        # Find high-cost regions
        avg_cost_per_request = sum(r.cost_per_request for r in regional_analyses) / len(regional_analyses)

        for analysis in regional_analyses:
            if analysis.cost_per_request > avg_cost_per_request * 1.5:  # 50% more expensive
                savings = analysis.monthly_cost * 0.2
                opportunity = CostOpportunity(
                    title=f"Optimize {analysis.region_name} instance types",
                    description=f"Region {analysis.region_name} has higher cost per request: ${analysis.cost_per_request:.4f} "
                    f"vs average ${avg_cost_per_request:.4f}. Consider using more cost-effective instance types.",
                    potential_savings_percentage=20,
                    implementation_effort="high",
                    risk_level="low",
                    priority_score=70,
                    estimated_monthly_savings=savings,
                )
                opportunities.append(opportunity)

        # Find high-error regions
        avg_error_rate = sum(r.error_rate for r in regional_analyses) / len(regional_analyses)

        for analysis in regional_analyses:
            if analysis.error_rate > avg_error_rate * 2:  # Double the error rate
                savings = analysis.monthly_cost * 0.15  # Potential 15% savings from reduced retries/incidents
                opportunity = CostOpportunity(
                    title=f"Fix reliability issues in {analysis.region_name}",
                    description=f"Region {analysis.region_name} has high error rate: {analysis.error_rate}% "
                    f"vs average {avg_error_rate}%. This causes retries and waste. Investigate root cause.",
                    potential_savings_percentage=15,
                    implementation_effort="high",
                    risk_level="medium",
                    priority_score=85,
                    estimated_monthly_savings=savings,
                )
                opportunities.append(opportunity)

        # Generic opportunities
        total_cost = sum(r.monthly_cost for r in regional_analyses)

        # Reserved capacity
        opportunities.append(
            CostOpportunity(
                title="Purchase reserved capacity",
                description="If utilization is stable, purchasing reserved capacity can save 30-50% compared to on-demand.",
                potential_savings_percentage=35,
                implementation_effort="low",
                risk_level="low",
                priority_score=90,
                estimated_monthly_savings=total_cost * 0.35,
            )
        )

        # Database optimization
        opportunities.append(
            CostOpportunity(
                title="Optimize database queries",
                description="Analysis of query performance can reduce database load by 20-40%, directly reducing costs.",
                potential_savings_percentage=25,
                implementation_effort="medium",
                risk_level="low",
                priority_score=75,
                estimated_monthly_savings=total_cost * 0.1,  # 10% of total (partial impact)
            )
        )

        # Caching improvements
        opportunities.append(
            CostOpportunity(
                title="Implement advanced caching",
                description="Multi-level caching (client, CDN, server) can reduce backend load by 30-50%.",
                potential_savings_percentage=30,
                implementation_effort="medium",
                risk_level="low",
                priority_score=80,
                estimated_monthly_savings=total_cost * 0.12,  # 12% of total (partial impact)
            )
        )

        return opportunities

    def generate_cost_plan(self, priority: OptimizationPriority = OptimizationPriority.BALANCED) -> CostOptimizationPlan:
        """Generate complete cost optimization plan"""
        opportunities = self.identify_optimization_opportunities(priority)
        regional_analyses = self.analyze_regional_costs()

        # Calculate totals
        total_current_cost = sum(r.monthly_cost for r in regional_analyses)
        total_potential_savings = sum(o.estimated_monthly_savings for o in opportunities)
        savings_percentage = (total_potential_savings / total_current_cost * 100) if total_current_cost > 0 else 0

        # Sort opportunities by priority
        opportunities.sort(key=lambda o: o.priority_score, reverse=True)

        # Identify quick wins
        quick_wins = [
            o
            for o in opportunities
            if o.implementation_effort == "low" and o.priority_score >= 75
        ]

        # Generate recommendations
        recommendations = self._generate_recommendations(opportunities, regional_analyses, priority)

        return CostOptimizationPlan(
            total_current_monthly_cost=round(total_current_cost, 2),
            total_potential_monthly_savings=round(total_potential_savings, 2),
            savings_percentage=round(savings_percentage, 2),
            opportunities=opportunities,
            regional_analysis=regional_analyses,
            recommendations=recommendations,
            quick_wins=quick_wins,
        )

    @staticmethod
    def _generate_recommendations(
        opportunities: List[CostOpportunity],
        regional_analyses: List[RegionCostAnalysis],
        priority: OptimizationPriority,
    ) -> List[str]:
        """Generate prioritized recommendations"""
        recommendations = []

        # Quick wins first
        quick_wins = [o for o in opportunities if o.implementation_effort == "low"]
        if quick_wins:
            recommendations.append(f"🚀 Quick wins: Implement {', '.join([o.title for o in quick_wins[:2]])} for immediate savings")

        # Priority-specific recommendations
        if priority == OptimizationPriority.COST:
            recommendations.append("💰 Focus on high-impact, cost-heavy optimizations: database queries and reserved capacity")
            recommendations.append("⚠️ Monitor performance during optimizations to avoid SLA violations")
        elif priority == OptimizationPriority.PERFORMANCE:
            recommendations.append("⚡ Prioritize reliability improvements: fix high-error regions and optimize queries")
            recommendations.append("📊 Cost savings will follow from reduced errors and better efficiency")
        else:  # Balanced
            recommendations.append("⚖️ Implement low-risk, quick-win opportunities first")
            recommendations.append("📈 Monitor utilization after each change and adjust strategy accordingly")

        # Regional recommendations
        underutilized = [r for r in regional_analyses if r.capacity_utilization < 50]
        if underutilized:
            recommendations.append(f"🗑️ Review underutilized regions: {', '.join([r.region_name for r in underutilized])}")

        # Risk warnings
        high_risk_opps = [o for o in opportunities if o.risk_level == "high"]
        if high_risk_opps:
            recommendations.append(f"⚠️ High-risk changes require careful testing: {', '.join([o.title for o in high_risk_opps[:2]])}")

        return recommendations

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost optimization summary"""
        plan = self.generate_cost_plan()

        return {
            "total_monthly_cost": plan.total_current_monthly_cost,
            "potential_monthly_savings": plan.total_potential_monthly_savings,
            "savings_percentage": plan.savings_percentage,
            "number_of_opportunities": len(plan.opportunities),
            "quick_wins_available": len(plan.quick_wins),
            "regions_analyzed": len(plan.regional_analysis),
            "average_region_efficiency": round(sum(r.efficiency_score for r in plan.regional_analysis) / len(plan.regional_analysis), 2) if plan.regional_analysis else 0,
            "top_recommendation": plan.recommendations[0] if plan.recommendations else "No recommendations available",
        }
