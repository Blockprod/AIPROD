"""
Advanced Analytics Dashboard

Real-time usage monitoring, insights, and reporting:
- Metric aggregation (requests, generations, latency)
- Cost breakdown by user/model
- Performance trending  
- User cohort analysis
- Anomaly detection
- Export capabilities (CSV, PDF)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np


@dataclass
class GenerationMetrics:
    """Metrics for single generation"""
    generation_id: str
    user_id: str
    timestamp: float
    prompt_length: int
    model_name: str
    completion_time_sec: float
    tokens_generated: int
    cost_usd: float
    quality_score: float  # 0-1
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class UserMetrics:
    """Aggregated metrics for user"""
    user_id: str
    total_generations: int = 0
    total_cost_usd: float = 0.0
    avg_completion_time_sec: float = 0.0
    avg_quality_score: float = 0.0
    common_models: List[str] = field(default_factory=list)
    cost_by_model: Dict[str, float] = field(default_factory=dict)
    last_active: Optional[float] = None


@dataclass
class SystemMetrics:
    """System-wide metrics"""
    timestamp: float
    active_users: int
    total_generations_today: int
    avg_latency_ms: float
    success_rate: float  # 0-1
    peak_concurrent_requests: int
    total_cost_today_usd: float


class AnalyticsDashboard:
    """
    Advanced analytics dashboard for monitoring AIPROD system.
    
    Tracks:
    - Real-time generation metrics
    - User behavior and cost
    - System performance and health
    - Anomalies and errors
    """
    
    def __init__(self):
        """Initialize analytics"""
        self.generation_metrics: List[GenerationMetrics] = []
        self.user_metrics: Dict[str, UserMetrics] = defaultdict(
            lambda: UserMetrics(user_id="")
        )
        
        # Time windows for aggregation
        self.metrics_window_sec = 3600  # 1 hour
    
    async def record_generation(
        self,
        generation_id: str,
        user_id: str,
        prompt_length: int,
        model_name: str,
        completion_time_sec: float,
        tokens_generated: int,
        cost_usd: float,
        quality_score: float,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> Dict:
        """
        Record generation metrics.
        
        Args:
            generation_id: Unique generation ID
            user_id: User ID
            prompt_length: Length of prompt
            model_name: Model used
            completion_time_sec: Time to complete (seconds)
            tokens_generated: Number of tokens
            cost_usd: Cost in USD
            quality_score: Quality 0-1
            success: Whether generation succeeded
            error_message: Error if failed
            
        Returns:
            Dict with recording status
        """
        import time
        
        metrics = GenerationMetrics(
            generation_id=generation_id,
            user_id=user_id,
            timestamp=time.time(),
            prompt_length=prompt_length,
            model_name=model_name,
            completion_time_sec=completion_time_sec,
            tokens_generated=tokens_generated,
            cost_usd=cost_usd,
            quality_score=quality_score,
            success=success,
            error_message=error_message,
        )
        
        self.generation_metrics.append(metrics)
        
        # Update user metrics
        if user_id not in self.user_metrics:
            self.user_metrics[user_id] = UserMetrics(user_id=user_id)
        
        user = self.user_metrics[user_id]
        user.total_generations += 1
        user.total_cost_usd += cost_usd
        user.last_active = time.time()
        
        # Update model tracking
        if model_name not in user.common_models:
            user.common_models.append(model_name)
        if model_name not in user.cost_by_model:
            user.cost_by_model[model_name] = 0.0
        user.cost_by_model[model_name] += cost_usd
        
        return {
            "recorded": True,
            "generation_id": generation_id,
            "total_user_generations": user.total_generations,
        }
    
    def get_dashboard_summary(self, time_window_sec: Optional[int] = None) -> Dict:
        """
        Get dashboard summary statistics.
        
        Args:
            time_window_sec: Time window to analyze (default: 3600 = 1 hour)
            
        Returns:
            Dict with summary metrics
        """
        import time
        
        if time_window_sec is None:
            time_window_sec = self.metrics_window_sec
        
        current_time = time.time()
        window_start = current_time - time_window_sec
        
        # Filter metrics in window
        recent = [
            m for m in self.generation_metrics
            if m.timestamp >= window_start
        ]
        
        if not recent:
            return {
                "time_window_sec": time_window_sec,
                "total_generations": 0,
                "active_users": 0,
                "avg_latency_ms": 0,
                "success_rate": 0,
                "total_cost_usd": 0,
            }
        
        # Compute metrics
        successful = [m for m in recent if m.success]
        completion_times = [m.completion_time_sec for m in recent]
        costs = [m.cost_usd for m in recent]
        quality_scores = [m.quality_score for m in successful]
        
        active_users = len(set(m.user_id for m in recent))
        
        return {
            "time_window_sec": time_window_sec,
            "total_generations": len(recent),
            "successful_generations": len(successful),
            "failed_generations": len(recent) - len(successful),
            "active_users": active_users,
            "avg_latency_ms": float(np.mean(completion_times) * 1000) if completion_times else 0,
            "min_latency_ms": float(min(completion_times) * 1000) if completion_times else 0,
            "max_latency_ms": float(max(completion_times) * 1000) if completion_times else 0,
            "success_rate": float(len(successful) / len(recent)) if recent else 0,
            "total_cost_usd": float(sum(costs)),
            "avg_cost_per_generation": float(np.mean(costs)) if costs else 0,
            "avg_quality_score": float(np.mean(quality_scores)) if quality_scores else 0,
            "tokens_generated": sum(m.tokens_generated for m in recent),
        }
    
    def get_user_analytics(self, user_id: str) -> Dict:
        """Get analytics for specific user"""
        if user_id not in self.user_metrics:
            return {"user_id": user_id, "status": "no_data"}
        
        user = self.user_metrics[user_id]
        user_gen = [m for m in self.generation_metrics if m.user_id == user_id]
        
        if not user_gen:
            return {"user_id": user_id, "status": "no_data"}
        
        completion_times = [m.completion_time_sec for m in user_gen]
        quality_scores = [m.quality_score for m in user_gen if m.success]
        
        return {
            "user_id": user_id,
            "total_generations": user.total_generations,
            "total_cost_usd": float(user.total_cost_usd),
            "avg_completion_time_sec": float(np.mean(completion_times)) if completion_times else 0,
            "avg_quality_score": float(np.mean(quality_scores)) if quality_scores else 0,
            "common_models": user.common_models,
            "cost_by_model": {k: float(v) for k, v in user.cost_by_model.items()},
            "last_active_timestamp": user.last_active,
            "success_rate": float(
                len([m for m in user_gen if m.success]) / len(user_gen)
            ) if user_gen else 0,
        }
    
    def get_cost_breakdown(self, time_window_sec: Optional[int] = None) -> Dict:
        """Get cost breakdown by user and model"""
        import time
        
        if time_window_sec is None:
            time_window_sec = self.metrics_window_sec
        
        current_time = time.time()
        window_start = current_time - time_window_sec
        
        recent = [
            m for m in self.generation_metrics
            if m.timestamp >= window_start
        ]
        
        cost_by_user = defaultdict(float)
        cost_by_model = defaultdict(float)
        
        for m in recent:
            cost_by_user[m.user_id] += m.cost_usd
            cost_by_model[m.model_name] += m.cost_usd
        
        return {
            "total_cost_usd": float(sum(cost_by_user.values())),
            "cost_by_user": {k: float(v) for k, v in sorted(
                cost_by_user.items(), key=lambda x: x[1], reverse=True
            )[:20]},  # Top 20 users
            "cost_by_model": {k: float(v) for k, v in cost_by_model.items()},
            "time_window_sec": time_window_sec,
        }
    
    def get_trending_metrics(self, num_periods: int = 24) -> Dict:
        """Get trending metrics over time periods"""
        import time
        
        period_sec = 3600  # 1 hour periods
        current_time = time.time()
        
        trends = []
        
        for i in range(num_periods):
            period_start = current_time - (i + 1) * period_sec
            period_end = current_time - i * period_sec
            
            period_metrics = [
                m for m in self.generation_metrics
                if period_start <= m.timestamp < period_end
            ]
            
            if period_metrics:
                successful = [m for m in period_metrics if m.success]
                completion_times = [m.completion_time_sec for m in period_metrics]
                
                trends.append({
                    "period_timestamp": period_start,
                    "generations": len(period_metrics),
                    "success_rate": float(len(successful) / len(period_metrics)),
                    "avg_latency_sec": float(np.mean(completion_times)) if completion_times else 0,
                    "total_cost_usd": float(sum(m.cost_usd for m in period_metrics)),
                })
        
        return {
            "num_periods": len(trends),
            "period_duration_sec": period_sec,
            "trends": list(reversed(trends)),
        }
    
    def detect_anomalies(self, sensitivity: float = 2.0) -> List[Dict]:
        """
        Detect anomalies in recent metrics.
        
        Args:
            sensitivity: Standard deviations for anomaly threshold
            
        Returns:
            List of detected anomalies
        """
        import time
        
        current_time = time.time()
        window_start = current_time - self.metrics_window_sec
        
        recent = [
            m for m in self.generation_metrics
            if m.timestamp >= window_start
        ]
        
        if len(recent) < 10:
            return []
        
        anomalies = []
        
        # Check latency anomalies
        latencies = [m.completion_time_sec for m in recent]
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        threshold = mean_latency + sensitivity * std_latency
        
        for m in recent:
            if m.completion_time_sec > threshold:
                anomalies.append({
                    "type": "high_latency",
                    "generation_id": m.generation_id,
                    "value": m.completion_time_sec,
                    "threshold": threshold,
                    "timestamp": m.timestamp,
                })
        
        # Check failure rate anomalies
        failure_rate = 1.0 - (len([m for m in recent if m.success]) / len(recent))
        if failure_rate > 0.1:  # >10% failure rate
            anomalies.append({
                "type": "high_failure_rate",
                "failure_rate": failure_rate,
                "timestamp": current_time,
            })
        
        # Check cost anomalies
        costs = [m.cost_usd for m in recent]
        mean_cost = np.mean(costs)
        std_cost = np.std(costs)
        threshold = mean_cost + sensitivity * std_cost
        
        for m in recent:
            if m.cost_usd > threshold:
                anomalies.append({
                    "type": "high_cost",
                    "generation_id": m.generation_id,
                    "value": m.cost_usd,
                    "threshold": threshold,
                    "timestamp": m.timestamp,
                })
        
        return anomalies
    
    async def export_metrics(
        self,
        output_format: str = "csv",
        time_window_sec: Optional[int] = None,
    ) -> str:
        """
        Export metrics to file.
        
        Args:
            output_format: "csv" or "json"
            time_window_sec: Time window to export
            
        Returns:
            Path to exported file
        """
        import time
        import json
        import csv
        from pathlib import Path
        
        if time_window_sec is None:
            time_window_sec = self.metrics_window_sec
        
        current_time = time.time()
        window_start = current_time - time_window_sec
        
        recent = [
            m for m in self.generation_metrics
            if m.timestamp >= window_start
        ]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("/tmp/aiprod_analytics")
        output_dir.mkdir(exist_ok=True)
        
        if output_format == "csv":
            output_file = output_dir / f"metrics_{timestamp}.csv"
            
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "generation_id", "user_id", "timestamp", "model_name",
                    "completion_time_sec", "cost_usd", "quality_score", "success"
                ])
                
                for m in recent:
                    writer.writerow([
                        m.generation_id, m.user_id, m.timestamp, m.model_name,
                        m.completion_time_sec, m.cost_usd, m.quality_score, m.success
                    ])
        
        else:  # json
            output_file = output_dir / f"metrics_{timestamp}.json"
            
            data = {
                "export_timestamp": datetime.now().isoformat(),
                "time_window_sec": time_window_sec,
                "metrics": [
                    {
                        "generation_id": m.generation_id,
                        "user_id": m.user_id,
                        "timestamp": m.timestamp,
                        "model_name": m.model_name,
                        "completion_time_sec": m.completion_time_sec,
                        "cost_usd": m.cost_usd,
                        "quality_score": m.quality_score,
                        "success": m.success,
                    }
                    for m in recent
                ]
            }
            
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
        
        return str(output_file)
