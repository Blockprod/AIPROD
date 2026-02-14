"""Tests for analytics dashboard"""

import pytest
import time
from aiprod_pipelines.inference.analytics import (
    AnalyticsDashboard,
    GenerationMetrics,
    UserMetrics,
)


class TestGenerationMetrics:
    """Test generation metrics"""
    
    def test_metrics_creation(self):
        """Test metrics object"""
        metrics = GenerationMetrics(
            generation_id="gen_001",
            user_id="user_001",
            timestamp=time.time(),
            prompt_length=100,
            model_name="models/ltx-2.0",
            completion_time_sec=2.5,
            tokens_generated=1024,
            cost_usd=0.05,
            quality_score=0.92,
        )
        
        assert metrics.generation_id == "gen_001"
        assert metrics.quality_score == 0.92
        assert metrics.success is True


class TestUserMetrics:
    """Test user metrics"""
    
    def test_user_metrics_creation(self):
        """Test user metrics object"""
        metrics = UserMetrics(user_id="user_001")
        
        assert metrics.user_id == "user_001"
        assert metrics.total_generations == 0
        assert metrics.total_cost_usd == 0.0


class TestAnalyticsDashboard:
    """Test analytics dashboard"""
    
    @pytest.fixture
    def dashboard(self):
        """Create dashboard"""
        return AnalyticsDashboard()
    
    def test_dashboard_initialization(self, dashboard):
        """Test dashboard creates"""
        assert dashboard is not None
        assert len(dashboard.generation_metrics) == 0
    
    @pytest.mark.asyncio
    async def test_record_generation(self, dashboard):
        """Test recording generation"""
        result = await dashboard.record_generation(
            generation_id="gen_001",
            user_id="user_001",
            prompt_length=100,
            model_name="models/ltx-2.0",
            completion_time_sec=2.5,
            tokens_generated=1024,
            cost_usd=0.05,
            quality_score=0.92,
        )
        
        assert result["recorded"] is True
        assert result["generation_id"] == "gen_001"
        assert len(dashboard.generation_metrics) == 1
    
    def test_dashboard_summary(self, dashboard):
        """Test dashboard summary"""
        # No metrics yet
        summary = dashboard.get_dashboard_summary()
        assert summary["total_generations"] == 0
        assert summary["active_users"] == 0
    
    @pytest.mark.asyncio
    async def test_multiple_recordings(self, dashboard):
        """Test multiple recordings"""
        for i in range(10):
            await dashboard.record_generation(
                generation_id=f"gen_{i:03d}",
                user_id=f"user_{i % 3:01d}",  # 3 different users
                prompt_length=100 + i * 10,
                model_name="models/ltx-2.0",
                completion_time_sec=1.0 + i * 0.1,
                tokens_generated=1000,
                cost_usd=0.05,
                quality_score=0.85 + i * 0.01,
            )
        
        assert len(dashboard.generation_metrics) == 10
        
        summary = dashboard.get_dashboard_summary()
        assert summary["total_generations"] == 10
        assert summary["active_users"] == 3  # 3 different users
    
    @pytest.mark.asyncio
    async def test_user_analytics(self, dashboard):
        """Test user-specific analytics"""
        # Record for user_001
        for i in range(5):
            await dashboard.record_generation(
                generation_id=f"gen_{i:03d}",
                user_id="user_001",
                prompt_length=100,
                model_name="models/ltx-2.0",
                completion_time_sec=2.0,
                tokens_generated=1024,
                cost_usd=0.05,
                quality_score=0.90,
            )
        
        analytics = dashboard.get_user_analytics("user_001")
        assert analytics["total_generations"] == 5
        assert analytics["total_cost_usd"] == pytest.approx(0.25)
    
    @pytest.mark.asyncio
    async def test_cost_breakdown(self, dashboard):
        """Test cost breakdown"""
        for i in range(5):
            await dashboard.record_generation(
                generation_id=f"gen_{i:03d}",
                user_id="user_001",
                prompt_length=100,
                model_name="models/ltx-2.0",
                completion_time_sec=2.0,
                tokens_generated=1024,
                cost_usd=0.10,
                quality_score=0.90,
            )
        
        for i in range(3):
            await dashboard.record_generation(
                generation_id=f"gen_{10+i:03d}",
                user_id="user_002",
                prompt_length=100,
                model_name="models/ltx-2.0-fast",
                completion_time_sec=1.0,
                tokens_generated=512,
                cost_usd=0.05,
                quality_score=0.85,
            )
        
        breakdown = dashboard.get_cost_breakdown()
        assert breakdown["total_cost_usd"] == pytest.approx(0.65)
        assert "user_001" in breakdown["cost_by_user"]
        assert "user_002" in breakdown["cost_by_user"]
    
    @pytest.mark.asyncio
    async def test_trending_metrics(self, dashboard):
        """Test trending metrics"""
        for i in range(10):
            await dashboard.record_generation(
                generation_id=f"gen_{i:03d}",
                user_id="user_001",
                prompt_length=100,
                model_name="models/ltx-2.0",
                completion_time_sec=2.0 + i * 0.1,
                tokens_generated=1024,
                cost_usd=0.05,
                quality_score=0.90,
            )
        
        trends = dashboard.get_trending_metrics(num_periods=24)
        assert "trends" in trends
        assert trends["period_duration_sec"] == 3600
    
    @pytest.mark.asyncio
    async def test_anomaly_detection(self, dashboard):
        """Test anomaly detection"""
        # Normal metrics
        for i in range(10):
            await dashboard.record_generation(
                generation_id=f"gen_{i:03d}",
                user_id="user_001",
                prompt_length=100,
                model_name="models/ltx-2.0",
                completion_time_sec=2.0,
                tokens_generated=1024,
                cost_usd=0.05,
                quality_score=0.90,
            )
        
        # One anomalous high-latency metric
        await dashboard.record_generation(
            generation_id="gen_anomaly",
            user_id="user_001",
            prompt_length=100,
            model_name="models/ltx-2.0",
            completion_time_sec=10.0,  # Much higher
            tokens_generated=1024,
            cost_usd=0.05,
            quality_score=0.90,
        )
        
        anomalies = dashboard.detect_anomalies(sensitivity=1.5)
        assert len(anomalies) > 0  # Should detect the anomaly
    
    @pytest.mark.asyncio
    async def test_export_metrics_csv(self, dashboard, tmp_path):
        """Test CSV export"""
        # Record some metrics
        for i in range(5):
            await dashboard.record_generation(
                generation_id=f"gen_{i:03d}",
                user_id="user_001",
                prompt_length=100,
                model_name="models/ltx-2.0",
                completion_time_sec=2.0,
                tokens_generated=1024,
                cost_usd=0.05,
                quality_score=0.90,
            )
        
        output_file = await dashboard.export_metrics(output_format="csv")
        assert output_file is not None
        assert "csv" in output_file
    
    @pytest.mark.asyncio
    async def test_export_metrics_json(self, dashboard):
        """Test JSON export"""
        # Record some metrics
        for i in range(5):
            await dashboard.record_generation(
                generation_id=f"gen_{i:03d}",
                user_id="user_001",
                prompt_length=100,
                model_name="models/ltx-2.0",
                completion_time_sec=2.0,
                tokens_generated=1024,
                cost_usd=0.05,
                quality_score=0.90,
            )
        
        output_file = await dashboard.export_metrics(output_format="json")
        assert output_file is not None
        assert "json" in output_file


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
