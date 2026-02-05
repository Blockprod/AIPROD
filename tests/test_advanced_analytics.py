"""
Comprehensive tests for Phase 2.4 - Advanced Analytics
"""
import pytest
from datetime import datetime, timedelta
from fastapi.testclient import TestClient

from src.api.main import app
from src.analytics.ml_models import (
    LinearRegression,
    ExponentialSmoothing,
    MovingAverage,
    STLDecomposition,
    AnomalyDetector,
    Correlation,
)
from src.analytics.anomaly_detector import AnomalyDetectionEngine, AnomalyType
from src.analytics.forecaster import PerformanceForecaster
from src.analytics.cost_optimizer import CostOptimizer, OptimizationPriority

client = TestClient(app)


class TestLinearRegression:
    """Test linear regression model"""

    def test_linear_regression_fit(self):
        """Test linear regression fitting"""
        reg = LinearRegression()
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]

        reg.fit(x, y)

        assert reg.slope == pytest.approx(2.0)
        assert reg.intercept == pytest.approx(0.0)
        assert reg.r_squared == pytest.approx(1.0)

    def test_linear_regression_predict(self):
        """Test prediction"""
        reg = LinearRegression()
        x = [1, 2, 3, 4, 5]
        y = [3, 5, 7, 9, 11]

        reg.fit(x, y)
        pred = reg.predict(6)

        assert pred == pytest.approx(13.0)

    def test_linear_regression_multiple_predict(self):
        """Test multiple predictions"""
        reg = LinearRegression()
        x = [0, 1, 2, 3]
        y = [0, 1, 2, 3]

        reg.fit(x, y)
        preds = reg.predict_multiple([0, 1, 2, 3])

        assert len(preds) == 4
        assert preds[0] == pytest.approx(0.0)


class TestExponentialSmoothing:
    """Test exponential smoothing"""

    def test_exponential_smoothing(self):
        """Test exponential smoothing"""
        smoother = ExponentialSmoothing(alpha=0.3)
        values = [10, 12, 11, 13, 12, 14]

        smoothed = smoother.smooth(values)

        assert len(smoothed) == len(values)
        assert smoothed[0] == 10  # First value unchanged

    def test_exponential_smoothing_forecast(self):
        """Test forecast after smoothing"""
        smoother = ExponentialSmoothing(alpha=0.3)
        values = [10, 12, 11, 13, 12, 14]

        smoother.smooth(values)
        forecast = smoother.forecast_next()

        assert forecast is not None
        assert isinstance(forecast, float)


class TestMovingAverage:
    """Test moving average"""

    def test_moving_average(self):
        """Test moving average calculation"""
        ma = MovingAverage(window_size=3)
        values = [1, 2, 3, 4, 5, 6]

        moving_avg = ma.calculate(values)

        assert len(moving_avg) == len(values)
        assert moving_avg[0] == 1
        assert moving_avg[2] == 2.0  # (1+2+3)/3

    def test_moving_average_latest(self):
        """Test getting latest moving average"""
        ma = MovingAverage(window_size=3)
        values = [1, 2, 3, 4, 5, 6]

        latest = ma.get_latest(values)

        assert latest == pytest.approx(5.0)  # (4+5+6)/3


class TestSTLDecomposition:
    """Test seasonal decomposition"""

    def test_stl_decomposition(self):
        """Test STL decomposition"""
        decomp = STLDecomposition(seasonal_period=7)
        values = [10] * 7 + [12] * 7 + [11] * 7

        result = decomp.decompose(values)

        assert "trend" in result
        assert "seasonal" in result
        assert "residual" in result
        assert len(result["trend"]) == len(values)


class TestAnomalyDetector:
    """Test anomaly detection methods"""

    def test_z_score_anomalies(self):
        """Test Z-score anomaly detection"""
        values = [10, 11, 10, 9, 100, 10, 11]

        anomalies = AnomalyDetector.z_score_anomalies(values, threshold=2.0)

        assert len(anomalies) == len(values)
        assert anomalies[4] is True  # 100 is anomaly

    def test_iqr_anomalies(self):
        """Test IQR anomaly detection"""
        values = [1, 2, 3, 4, 5, 100]

        anomalies = AnomalyDetector.iqr_anomalies(values)

        assert len(anomalies) == len(values)
        assert anomalies[-1] is True  # 100 is anomaly

    def test_isolation_anomalies(self):
        """Test isolation anomaly detection"""
        values = [10, 11, 9, 10, 11, 200]

        anomalies = AnomalyDetector.isolation_anomalies(values, contamination=0.2)

        assert len(anomalies) == len(values)
        assert True in anomalies


class TestCorrelation:
    """Test correlation analysis"""

    def test_pearson_correlation(self):
        """Test Pearson correlation"""
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]

        corr = Correlation.pearson_correlation(x, y)

        assert corr == pytest.approx(1.0)

    def test_correlation_interpretation(self):
        """Test correlation interpretation"""
        interp = Correlation.interpret_correlation(0.75)

        assert "strong" in interp
        assert "positive" in interp


class TestAnomalyDetectionEngine:
    """Test anomaly detection engine"""

    def test_add_data_point(self):
        """Test adding data points"""
        engine = AnomalyDetectionEngine()

        engine.add_data_point("metric1", 100)
        engine.add_data_point("metric1", 110)
        engine.add_data_point("metric1", 105)

        assert "metric1" in engine.history
        assert len(engine.history["metric1"]) == 3

    def test_detect_anomalies(self):
        """Test anomaly detection"""
        engine = AnomalyDetectionEngine()

        # Add normal values
        for i in range(20):
            engine.add_data_point("metric1", 100 + (i % 5))

        # Add anomaly
        engine.add_data_point("metric1", 500)

        report = engine.detect_anomalies("metric1", lookback_points=25)

        assert report.total_points_analyzed > 0
        assert len(report.anomalies) > 0

    def test_anomaly_summary(self):
        """Test anomaly summary"""
        engine = AnomalyDetectionEngine()

        for i in range(30):
            engine.add_data_point("metric1", 100)

        engine.add_data_point("metric1", 300)

        summary = engine.get_anomaly_summary("metric1")

        assert "metric_name" in summary
        assert "total_anomalies" in summary


class TestPerformanceForecaster:
    """Test performance forecaster"""

    def test_add_data_point(self):
        """Test adding data points"""
        forecaster = PerformanceForecaster()

        forecaster.add_data_point("latency", 50)
        forecaster.add_data_point("latency", 55)

        assert "latency" in forecaster.history
        assert len(forecaster.history["latency"]) == 2

    def test_forecast_linear(self):
        """Test linear forecasting"""
        forecaster = PerformanceForecaster()

        for i in range(20):
            forecaster.add_data_point("metric", 100 + i)

        result = forecaster.forecast_linear("metric", periods_ahead=5)

        assert result.metric_name == "metric"
        assert len(result.forecasts) == 5
        assert result.forecast_method == "linear"

    def test_forecast_exponential(self):
        """Test exponential smoothing forecast"""
        forecaster = PerformanceForecaster()

        for i in range(15):
            forecaster.add_data_point("metric", 100)

        result = forecaster.forecast_exponential_smoothing("metric", periods_ahead=5)

        assert result.metric_name == "metric"
        assert len(result.forecasts) == 5

    def test_forecast_seasonal(self):
        """Test seasonal forecasting"""
        forecaster = PerformanceForecaster()

        # Add seasonal pattern
        for cycle in range(3):
            for day in range(7):
                value = 100 + (day * 10)
                forecaster.add_data_point("metric", value)

        result = forecaster.forecast_seasonal("metric", periods_ahead=7, seasonal_period=7)

        assert result.metric_name == "metric"
        assert len(result.forecasts) == 7

    def test_ensemble_forecast(self):
        """Test ensemble forecasting"""
        forecaster = PerformanceForecaster()

        for i in range(30):
            forecaster.add_data_point("metric", 100 + (i % 10))

        result = forecaster.ensemble_forecast("metric", periods_ahead=5)

        assert result.metric_name == "metric"
        assert len(result.forecasts) == 5
        assert result.forecast_method == "ensemble"

    def test_record_prediction(self):
        """Test recording prediction accuracy"""
        forecaster = PerformanceForecaster()

        forecaster.record_prediction("metric", 100, 105)
        forecaster.record_prediction("metric", 110, 115)

        accuracy = forecaster._calculate_forecast_accuracy("metric")
        assert accuracy >= 0


class TestCostOptimizer:
    """Test cost optimizer"""

    def test_add_region_data(self):
        """Test adding region data"""
        optimizer = CostOptimizer()

        optimizer.add_region_data(
            region_id="us-east-1",
            region_name="United States East",
            monthly_cost=1000,
            monthly_requests=1000000,
            capacity_utilization=75,
            error_rate=0.5,
            avg_latency_ms=100,
        )

        assert "us-east-1" in optimizer.region_costs

    def test_analyze_regional_costs(self):
        """Test regional cost analysis"""
        optimizer = CostOptimizer()

        optimizer.add_region_data("us-east-1", "US East", 1000, 1000000, 75, 0.5, 100)
        optimizer.add_region_data("eu-west-1", "EU West", 800, 800000, 60, 1.0, 150)

        analyses = optimizer.analyze_regional_costs()

        assert len(analyses) == 2
        assert analyses[0].efficiency_score >= 0
        assert analyses[0].cost_per_request >= 0

    def test_identify_opportunities(self):
        """Test opportunity identification"""
        optimizer = CostOptimizer()

        optimizer.add_region_data("us-east-1", "US East", 1000, 1000000, 75, 0.5, 100)
        optimizer.add_region_data("eu-west-1", "EU West", 1000, 500000, 30, 1.0, 150)

        opportunities = optimizer.identify_optimization_opportunities(OptimizationPriority.BALANCED)

        assert len(opportunities) > 0

    def test_generate_cost_plan(self):
        """Test cost plan generation"""
        optimizer = CostOptimizer()

        optimizer.add_region_data("us-east-1", "US East", 1000, 1000000, 75, 0.5, 100)

        plan = optimizer.generate_cost_plan(OptimizationPriority.BALANCED)

        assert plan.total_current_monthly_cost >= 1000
        assert len(plan.opportunities) > 0
        assert len(plan.recommendations) > 0

    def test_cost_summary(self):
        """Test cost summary"""
        optimizer = CostOptimizer()

        optimizer.add_region_data("us-east-1", "US East", 1000, 1000000, 75, 0.5, 100)

        summary = optimizer.get_cost_summary()

        assert "total_monthly_cost" in summary
        assert "potential_monthly_savings" in summary
        assert "savings_percentage" in summary


class TestAnalyticsAPI:
    """Test advanced analytics API endpoints"""

    def test_health_check(self):
        """Test analytics health check endpoint"""
        response = client.get("/analytics/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "operational"
        assert "total_metrics_tracked" in data

    def test_add_metric(self):
        """Test adding metric endpoint"""
        response = client.post(
            "/analytics/metrics/add",
            json={"metric_name": "test_metric", "value": 100},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_forecast_endpoint(self):
        """Test forecast endpoint"""
        # Add data first
        for i in range(20):
            client.post("/analytics/metrics/add", json={"metric_name": "forecast_test", "value": 100 + i})

        response = client.get("/analytics/forecast/forecast_test", params={"periods_ahead": 5})

        assert response.status_code in [200, 404]  # 404 if insufficient data buffered

    def test_anomalies_endpoint(self):
        """Test anomalies endpoint"""
        # Add data first
        for i in range(20):
            client.post("/analytics/metrics/add", json={"metric_name": "anomaly_test", "value": 100})

        client.post("/analytics/metrics/add", json={"metric_name": "anomaly_test", "value": 500})

        response = client.get("/analytics/anomalies/anomaly_test")

        assert response.status_code == 200
        data = response.json()
        assert "anomalies" in data

    def test_cost_analysis_endpoint(self):
        """Test cost analysis endpoint"""
        response = client.get("/analytics/costs/analyze")

        # Should succeed even with empty data
        assert response.status_code == 200

    def test_cost_summary_endpoint(self):
        """Test cost summary endpoint"""
        response = client.get("/analytics/costs/summary")

        assert response.status_code == 200
        data = response.json()
        assert "total_monthly_cost" in data


class TestAnalyticsIntegration:
    """Integration tests for analytics"""

    def test_end_to_end_anomaly_detection(self):
        """End-to-end anomaly detection workflow"""
        engine = AnomalyDetectionEngine()

        # Add normal data
        for i in range(50):
            engine.add_data_point("sensor1", 100 + (i % 10))

        # Add anomaly
        engine.add_data_point("sensor1", 500)

        # Detect
        report = engine.detect_anomalies("sensor1", lookback_points=50)

        assert len(report.anomalies) > 0
        assert any(a.severity == "high" or a.severity == "critical" for a in report.anomalies)

    def test_end_to_end_forecasting(self):
        """End-to-end forecasting workflow"""
        forecaster = PerformanceForecaster()

        # Add trend data
        for i in range(30):
            forecaster.add_data_point("requests", 100 + i * 2)

        # Forecast
        result = forecaster.ensemble_forecast("requests", periods_ahead=10)

        assert len(result.forecasts) == 10
        assert result.trend_direction == "increasing"

    def test_end_to_end_cost_optimization(self):
        """End-to-end cost optimization workflow"""
        optimizer = CostOptimizer()

        # Add multi-regional data
        optimizer.add_region_data("us-east", "US East", 2000, 2000000, 80, 0.2, 50)
        optimizer.add_region_data("eu-west", "EU West", 1500, 1000000, 40, 1.5, 200)
        optimizer.add_region_data("ap-se", "AP SE", 1000, 500000, 20, 2.0, 300)

        # Generate plan
        plan = optimizer.generate_cost_plan(OptimizationPriority.COST)

        assert plan.total_potential_monthly_savings > 0
        assert len(plan.quick_wins) > 0
        assert len(plan.recommendations) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
