"""
Analytics Engine - Advanced data analysis and trend detection
Processes metrics to generate insights, anomalies, and recommendations
"""

import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Sequence
from enum import Enum
from src.utils.monitoring import logger


class AnomalyType(str, Enum):
    """Types of anomalies detected"""
    LATENCY_SPIKE = "latency_spike"
    ERROR_RATE_SPIKE = "error_rate_spike"
    THROUGHPUT_DROP = "throughput_drop"
    RESOURCE_CONSTRAINT = "resource_constraint"
    UNUSUAL_PATTERN = "unusual_pattern"


class AnalyticsEngine:
    """
    Advanced analytics for monitoring data.
    
    Features:
    - Trend analysis
    - Anomaly detection using statistical methods
    - Performance predictions
    - Resource usage optimization
    - Correlations between metrics
    """
    
    def __init__(self, stddev_threshold: float = 2.0):
        """
        Initialize analytics engine
        
        Args:
            stddev_threshold: Number of standard deviations for anomaly detection
        """
        self.stddev_threshold = stddev_threshold
        self.historical_patterns: Dict[str, List[float]] = {}
        self.anomalies: List[Dict[str, Any]] = []
    
    def detect_latency_anomalies(self, metrics: List[float]) -> List[Dict]:
        """
        Detect unusual latency patterns using statistical analysis
        
        Args:
            metrics: List of latency measurements
            
        Returns:
            List of detected anomalies
        """
        if len(metrics) < 5:
            return []
        
        anomalies = []
        
        try:
            mean = statistics.mean(metrics)
            stdev = statistics.stdev(metrics) if len(metrics) > 1 else 0
            
            # Find values more than N standard deviations from mean
            threshold = mean + (stdev * self.stddev_threshold)
            
            for i, value in enumerate(metrics[-10:]):  # Check last 10
                if value > threshold:
                    anomalies.append({
                        "type": AnomalyType.LATENCY_SPIKE,
                        "value": value,
                        "expected_range": f"{mean:.2f}±{stdev:.2f}",
                        "deviation": round((value - mean) / stdev, 2) if stdev > 0 else 0,
                    })
        
        except Exception as e:
            logger.warning(f"Error in anomaly detection: {e}")
        
        return anomalies
    
    def calculate_trend(self, values: Sequence[float], window: int = 5) -> Dict[str, Any]:
        """
        Calculate trend in a series of values
        
        Args:
            values: Time series values
            window: Window size for moving average
            
        Returns:
            Trend analysis
        """
        if len(values) < 2:
            return {"trend": "unknown", "direction": None}
        
        # Calculate moving average
        moving_avg = []
        for i in range(len(values) - window + 1):
            avg = statistics.mean(values[i:i + window])
            moving_avg.append(avg)
        
        if len(moving_avg) < 2:
            return {"trend": "insufficient_data", "direction": None}
        
        # Determine direction
        recent = moving_avg[-3:] if len(moving_avg) >= 3 else moving_avg
        avg_recent = statistics.mean(recent)
        avg_older = statistics.mean(moving_avg[:-len(recent)]) if len(moving_avg) > len(recent) else moving_avg[0]
        
        direction = "up" if avg_recent > avg_older else "down"
        change_percent = round((avg_recent - avg_older) / avg_older * 100, 2) if avg_older != 0 else 0
        
        return {
            "trend": "increasing" if direction == "up" else "decreasing",
            "direction": direction,
            "change_percent": change_percent,
            "current": recent[-1] if recent else values[-1],
            "average": statistics.mean(values),
        }
    
    def detect_correlation(self, metric1: Sequence[float], metric2: Sequence[float]) -> float:
        """
        Detect correlation between two metrics (Pearson correlation)
        
        Args:
            metric1: First metric series
            metric2: Second metric series
            
        Returns:
            Correlation coefficient (-1 to 1)
        """
        if len(metric1) != len(metric2) or len(metric1) < 2:
            return 0.0
        
        try:
            mean1 = statistics.mean(metric1)
            mean2 = statistics.mean(metric2)
            
            numerator = sum((metric1[i] - mean1) * (metric2[i] - mean2) for i in range(len(metric1)))
            
            denom1 = sum((x - mean1) ** 2 for x in metric1)
            denom2 = sum((x - mean2) ** 2 for x in metric2)
            
            if denom1 == 0 or denom2 == 0:
                return 0.0
            
            denominator = (denom1 * denom2) ** 0.5
            
            correlation = numerator / denominator
            return round(correlation, 3)
        
        except Exception as e:
            logger.warning(f"Error calculating correlation: {e}")
            return 0.0
    
    def get_performance_insights(self, stats: Dict) -> List[str]:
        """
        Generate human-readable performance insights
        
        Args:
            stats: Endpoint statistics
            
        Returns:
            List of insight messages
        """
        insights = []
        
        if not stats or stats.get("request_count", 0) == 0:
            return insights
        
        error_rate = stats.get("error_rate", 0)
        avg_latency = stats.get("avg_duration", 0)
        
        # Error rate insights
        if error_rate > 0.1:
            insights.append("CRITICAL: Error rate exceeds 10%")
        elif error_rate > 0.05:
            insights.append("WARNING: Error rate is elevated (>5%)")
        
        # Latency insights
        if avg_latency > 1000:
            insights.append(f"High average latency: {avg_latency:.0f}ms")
        elif avg_latency > 500:
            insights.append(f"Moderate latency: {avg_latency:.0f}ms")
        
        # Performance optimization
        if stats.get("max_duration", 0) > avg_latency * 3:
            insights.append("High variability in response times")
        
        if error_rate == 0 and avg_latency < 100:
            insights.append("Excellent performance")
        
        return insights
    
    def get_recommendations(self, stats: Dict, health: Dict) -> List[str]:
        """
        Get optimization recommendations
        
        Args:
            stats: Endpoint statistics
            health: System health status
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if health.get("error_rate", 0) > 5:
            recommendations.append("Investigate error sources - high error rate detected")
        
        if stats and stats.get("avg_duration", 0) > 500:
            recommendations.append("Consider caching or database optimization")
        
        if health.get("active_alerts", 0) > 5:
            recommendations.append("Address active alerts to improve system stability")
        
        if not recommendations:
            recommendations.append("System is performing well")
        
        return recommendations


# Global analytics engine instance
_analytics_engine = None


def get_analytics_engine() -> AnalyticsEngine:
    """Get or create singleton analytics engine"""
    global _analytics_engine
    if _analytics_engine is None:
        _analytics_engine = AnalyticsEngine()
    return _analytics_engine
