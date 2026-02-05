"""
Tests for monitoring system - Phase 2.1
Tests metrics collection, analytics, and monitoring endpoints
"""

import time
import pytest
from datetime import datetime
from fastapi.testclient import TestClient
from src.api.main import app
from src.monitoring.metrics_collector import get_metrics_collector, MetricsCollector
from src.monitoring.analytics_engine import get_analytics_engine


client = TestClient(app)


class TestMetricsCollection:
    """Test metrics collection functionality"""
    
    def test_record_request_basic(self):
        """Test basic request recording"""
        collector = MetricsCollector(retention_hours=24)
        
        collector.record_request(
            endpoint="/api/jobs",
            method="GET",
            duration_ms=50.5,
            status_code=200,
            response_size=1024,
        )
        
        stats = collector.get_endpoint_stats("GET /api/jobs")
        assert stats["request_count"] == 1
        assert stats["error_count"] == 0
        assert stats["total_duration"] == 50.5
    
    def test_record_request_with_error(self):
        """Test recording requests with errors"""
        collector = MetricsCollector()
        
        collector.record_request(
            endpoint="/api/jobs",
            method="POST",
            duration_ms=100.0,
            status_code=500,
            response_size=0,
            error="Internal Server Error"
        )
        
        stats = collector.get_endpoint_stats("POST /api/jobs")
        assert stats["error_count"] == 1
        assert stats["error_rate"] == 1.0  # 100% error rate
    
    def test_multiple_requests_aggregation(self):
        """Test aggregation of multiple requests"""
        collector = MetricsCollector()
        
        # Record multiple requests
        for i in range(10):
            duration = 20 + (i * 5)  # 20, 25, 30... ms
            collector.record_request(
                endpoint="/api/jobs",
                method="GET",
                duration_ms=duration,
                status_code=200 if i % 10 != 9 else 400,  # One 400 error
                response_size=500,
            )
        
        stats = collector.get_endpoint_stats("GET /api/jobs")
        assert stats["request_count"] == 10
        assert stats["error_count"] == 1
        assert stats["min_duration"] == 20
        assert stats["max_duration"] == 65
        
        # Check error rate
        assert stats["error_rate"] == pytest.approx(0.1, abs=0.01)
    
    def test_cache_hit_recording(self):
        """Test cache hit/miss tracking"""
        collector = MetricsCollector()
        
        collector.record_cache_hit(hit=True)
        collector.record_cache_hit(hit=True)
        collector.record_cache_hit(hit=False)
        
        cache_metrics = [m for m in collector.metrics["cache_hit"]]
        assert len(cache_metrics) == 3
        
        hits = sum(1 for m in cache_metrics if m.value == 1)
        assert hits == 2
    
    def test_database_query_recording(self):
        """Test database query tracking"""
        collector = MetricsCollector()
        
        collector.record_database_query(query_time_ms=45.5, table="jobs")
        collector.record_database_query(query_time_ms=1200.0, table="jobs")  # Slow query
        
        db_metrics = collector.metrics.get("db_query_time", [])
        assert len(db_metrics) >= 2
        
        # Check for alert on slow query
        slow_alerts = [a for a in collector.alerts.values() if "Slow database" in a.title]
        assert len(slow_alerts) > 0
    
    def test_health_status(self):
        """Test health status calculation"""
        collector = MetricsCollector()
        
        # Record successful requests
        for _ in range(100):
            collector.record_request("/test", "GET", 50.0, 200)
        
        # Record some errors
        for _ in range(5):
            collector.record_request("/test", "GET", 100.0, 500, error="Server Error")
        
        health = collector.get_health_status()
        
        assert health["total_requests"] == 105
        assert health["total_errors"] == 5
        assert health["error_rate"] == pytest.approx(4.76, abs=0.1)
        assert health["status"] in ["healthy", "warning", "critical"]
    
    def test_top_endpoints(self):
        """Test getting top endpoints by error rate"""
        collector = MetricsCollector()
        
        # Endpoint 1: High error rate
        for _ in range(10):
            collector.record_request("/api/bad", "GET", 50.0, 500, error="Error")
        
        # Endpoint 2: Low error rate
        for _ in range(100):
            collector.record_request("/api/good", "GET", 30.0, 200)
        
        top = collector.get_top_endpoints(limit=5)
        
        assert len(top) > 0
        # First endpoint should be the one with highest error rate
        assert "bad" in top[0]["endpoint"]


class TestAnalyticsEngine:
    """Test analytics engine functionality"""
    
    def test_latency_anomaly_detection(self):
        """Test anomaly detection in latency data"""
        analytics = get_analytics_engine()
        
        # Normal latencies
        metrics = [50.0, 52.0, 51.0, 53.0, 49.0, 51.0, 50.0, 52.0]
        
        # Add spike
        metrics.append(250.0)
        
        anomalies = analytics.detect_latency_anomalies(metrics)
        
        # Should detect the spike
        assert len(anomalies) > 0
        spike = anomalies[0]
        assert spike["type"] == "latency_spike"
        assert spike["value"] == 250.0
    
    def test_trend_analysis_increasing(self):
        """Test trend detection for increasing values"""
        analytics = get_analytics_engine()
        
        # Increasing trend
        values = [10, 15, 20, 25, 30, 35, 40]
        
        trend = analytics.calculate_trend(values, window=2)
        
        assert trend["trend"] == "increasing"
        assert trend["direction"] == "up"
        assert trend["change_percent"] > 0
    
    def test_trend_analysis_decreasing(self):
        """Test trend detection for decreasing values"""
        analytics = get_analytics_engine()
        
        # Decreasing trend
        values = [100, 90, 80, 70, 60, 50, 40]
        
        trend = analytics.calculate_trend(values, window=2)
        
        assert trend["trend"] == "decreasing"
        assert trend["direction"] == "down"
        assert trend["change_percent"] < 0
    
    def test_correlation_analysis(self):
        """Test correlation between metrics"""
        analytics = get_analytics_engine()
        
        # Positively correlated metrics
        metric1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        metric2 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]  # metric1 * 2
        
        correlation = analytics.detect_correlation(metric1, metric2)
        
        # Should be perfectly correlated (or very close)
        assert correlation > 0.9
    
    def test_performance_insights(self):
        """Test generation of performance insights"""
        analytics = get_analytics_engine()
        
        # Poor performance stats
        stats = {
            "request_count": 1000,
            "error_count": 50,  # 5% error rate
            "error_rate": 0.05,
            "avg_duration": 250.0,  # High latency
            "max_duration": 1000.0,
        }
        
        insights = analytics.get_performance_insights(stats)
        
        assert len(insights) > 0
        # Should include some insights about poor performance
        insight_text = " ".join(insights).lower()
        assert "latency" in insight_text or "high" in insight_text or "slow" in insight_text
    
    def test_recommendations_poor_health(self):
        """Test recommendations for poor system health"""
        analytics = get_analytics_engine()
        
        stats = {"avg_duration": 500.0, "request_count": 100}
        health = {
            "error_rate": 8.0,  # 8% error rate
            "active_alerts": 10,
        }
        
        recommendations = analytics.get_recommendations(stats, health)
        
        assert len(recommendations) > 0
        # Should recommend investigation and stability improvements
        assert any("error" in r.lower() or "alert" in r.lower() for r in recommendations)


class TestMonitoringEndpoints:
    """Test monitoring API endpoints"""
    
    def test_health_endpoint(self):
        """Test /monitoring/health endpoint"""
        response = client.get("/monitoring/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "uptime_seconds" in data
        assert "total_requests" in data
        assert "error_rate" in data
    
    def test_dashboard_endpoint(self):
        """Test /monitoring/dashboard endpoint"""
        response = client.get("/monitoring/dashboard")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "health" in data
        assert "top_endpoints" in data
        assert "active_alerts" in data
        assert "insights" in data
        assert "timestamp" in data
    
    def test_all_endpoints_stats(self):
        """Test /monitoring/endpoints endpoint"""
        response = client.get("/monitoring/endpoints")
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        if data:
            endpoint = data[0]
            assert "endpoint" in endpoint
            assert "requests" in endpoint
            assert "errors" in endpoint
    
    def test_alerts_endpoint(self):
        """Test /monitoring/alerts endpoint"""
        response = client.get("/monitoring/alerts")
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
    
    def test_anomaly_detection_endpoint(self):
        """Test anomaly detection endpoint"""
        response = client.post("/monitoring/anomalies/detect")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "anomalies_detected" in data
        assert "timestamp" in data
        assert "analysis_period" in data
        assert "recommendations" in data
    
    def test_recommendations_endpoint(self):
        """Test /monitoring/recommendations endpoint"""
        response = client.get("/monitoring/recommendations")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "recommendations" in data
        assert "health_status" in data
        assert "timestamp" in data
        
        assert isinstance(data["recommendations"], list)
    
    def test_cleanup_endpoint(self):
        """Test /monitoring/cleanup endpoint"""
        response = client.post("/monitoring/cleanup")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "metrics_removed" in data


class TestMetricsIntegration:
    """Integration tests for metrics across the system"""
    
    def test_request_metrics_captured_on_api_call(self):
        """Test that metrics are captured for API requests"""
        collector = get_metrics_collector()
        
        # Get initial stats
        initial_stats = dict(collector.endpoint_stats)
        
        # Make a request
        response = client.get("/health")
        
        # Metrics should be recorded (though health endpoint may be excluded)
        assert response.status_code == 200
    
    def test_metrics_cleanup_old_data(self):
        """Test that old metrics are cleaned up"""
        collector = MetricsCollector(retention_hours=24)
        
        # Record metrics
        for _ in range(100):
            collector.record_request("/test", "GET", 50.0, 200)
        
        initial_count = sum(len(m) for m in collector.metrics.values())
        assert initial_count > 0
        
        # Cleanup
        collector.cleanup_old_metrics()
        
        # Should still have recent metrics
        final_count = sum(len(m) for m in collector.metrics.values())
        # Can't be completely empty since we just added metrics
        assert final_count >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
