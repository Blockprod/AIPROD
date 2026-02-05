"""
Performance forecasting using time series analysis
"""
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from src.analytics.ml_models import (
    LinearRegression,
    ExponentialSmoothing,
    MovingAverage,
)

logger = logging.getLogger(__name__)


@dataclass
class Forecast:
    """Single forecast point"""
    timestamp: datetime
    predicted_value: float
    confidence_interval_lower: float
    confidence_interval_upper: float


@dataclass
class ForecastResult:
    """Complete forecast result"""
    metric_name: str
    forecast_method: str
    last_known_value: float
    forecast_horizon: int  # Number of periods forecasted
    forecasts: List[Forecast] = field(default_factory=list)
    accuracy_score: float = 0.0  # Historical accuracy (0-100%)
    trend_direction: str = "stable"  # "increasing", "decreasing", "stable"
    trend_strength: str = "weak"  # "weak", "moderate", "strong"


class PerformanceForecaster:
    """Forecast performance metrics using multiple methods"""

    def __init__(self):
        self.history: Dict[str, List[float]] = {}
        self.timestamps: Dict[str, List[datetime]] = {}
        self.past_predictions: Dict[str, List[Tuple[float, float]]] = {}  # (predicted, actual)

    def add_data_point(self, metric_name: str, value: float, timestamp: Optional[datetime] = None):
        """Add data point to metric history"""
        if metric_name not in self.history:
            self.history[metric_name] = []
            self.timestamps[metric_name] = []
            self.past_predictions[metric_name] = []

        self.history[metric_name].append(value)
        self.timestamps[metric_name].append(timestamp or datetime.utcnow())

        # Keep last 500 points
        if len(self.history[metric_name]) > 500:
            self.history[metric_name] = self.history[metric_name][-500:]
            self.timestamps[metric_name] = self.timestamps[metric_name][-500:]

    def forecast_linear(
        self, metric_name: str, periods_ahead: int = 10, lookback_periods: int = 50
    ) -> ForecastResult:
        """Forecast using linear regression"""
        if metric_name not in self.history or len(self.history[metric_name]) < 5:
            return ForecastResult(metric_name=metric_name, forecast_method="linear", last_known_value=0.0, forecast_horizon=0)

        values = self.history[metric_name][-lookback_periods:]
        timestamps = self.timestamps[metric_name][-lookback_periods:]

        if len(values) < 2:
            return ForecastResult(metric_name=metric_name, forecast_method="linear", last_known_value=values[-1] if values else 0.0, forecast_horizon=0)

        # Fit linear regression
        x = [float(i) for i in range(len(values))]
        reg = LinearRegression()
        reg.fit(x, values)

        # Generate forecasts
        forecasts = []
        current_x = len(values)
        std_dev = self._calculate_stddev(values)

        for i in range(periods_ahead):
            pred = reg.predict(current_x + i)
            # Confidence interval increases with forecast distance
            margin = std_dev * (1 + (i / periods_ahead))
            forecast = Forecast(
                timestamp=timestamps[-1] + timedelta(hours=i + 1) if timestamps else datetime.utcnow() + timedelta(hours=i + 1),
                predicted_value=pred,
                confidence_interval_lower=pred - margin,
                confidence_interval_upper=pred + margin,
            )
            forecasts.append(forecast)

        # Determine trend
        trend_direction = "increasing" if reg.slope > 0.1 else ("decreasing" if reg.slope < -0.1 else "stable")
        trend_strength = "strong" if abs(reg.slope) > 1 else ("moderate" if abs(reg.slope) > 0.5 else "weak")

        accuracy = self._calculate_forecast_accuracy(metric_name)

        return ForecastResult(
            metric_name=metric_name,
            forecast_method="linear",
            last_known_value=values[-1],
            forecast_horizon=periods_ahead,
            forecasts=forecasts,
            accuracy_score=accuracy,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
        )

    def forecast_exponential_smoothing(
        self, metric_name: str, periods_ahead: int = 10, alpha: float = 0.3, lookback_periods: int = 50
    ) -> ForecastResult:
        """Forecast using exponential smoothing"""
        if metric_name not in self.history or len(self.history[metric_name]) < 5:
            return ForecastResult(metric_name=metric_name, forecast_method="exponential_smoothing", last_known_value=0.0, forecast_horizon=0)

        values = self.history[metric_name][-lookback_periods:]
        timestamps = self.timestamps[metric_name][-lookback_periods:]

        if len(values) < 2:
            return ForecastResult(metric_name=metric_name, forecast_method="exponential_smoothing", last_known_value=values[-1] if values else 0.0, forecast_horizon=0)

        # Apply exponential smoothing
        smoother = ExponentialSmoothing(alpha=alpha)
        smoothed = smoother.smooth(values)

        # Forecast (constant after smoothing)
        last_smoothed = smoothed[-1]
        std_dev = self._calculate_stddev(values)

        forecasts = []
        for i in range(periods_ahead):
            margin = std_dev * 0.5 * (1 + (i / periods_ahead))
            forecast = Forecast(
                timestamp=timestamps[-1] + timedelta(hours=i + 1) if timestamps else datetime.utcnow() + timedelta(hours=i + 1),
                predicted_value=last_smoothed,
                confidence_interval_lower=last_smoothed - margin,
                confidence_interval_upper=last_smoothed + margin,
            )
            forecasts.append(forecast)

        # Determine trend by comparing last values vs smoothed
        recent_mean = sum(values[-5:]) / 5 if len(values) >= 5 else sum(values) / len(values)
        trend_direction = "increasing" if last_smoothed > recent_mean else ("decreasing" if last_smoothed < recent_mean else "stable")
        trend_strength = "weak"  # Exponential smoothing assumes constant trend

        accuracy = self._calculate_forecast_accuracy(metric_name)

        return ForecastResult(
            metric_name=metric_name,
            forecast_method="exponential_smoothing",
            last_known_value=values[-1],
            forecast_horizon=periods_ahead,
            forecasts=forecasts,
            accuracy_score=accuracy,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
        )

    def forecast_seasonal(self, metric_name: str, periods_ahead: int = 10, seasonal_period: int = 7, lookback_periods: int = 50) -> ForecastResult:
        """Forecast using seasonal pattern"""
        if metric_name not in self.history or len(self.history[metric_name]) < seasonal_period * 2:
            return ForecastResult(metric_name=metric_name, forecast_method="seasonal", last_known_value=0.0, forecast_horizon=0)

        values = self.history[metric_name][-lookback_periods:]
        timestamps = self.timestamps[metric_name][-lookback_periods:]

        # Calculate seasonal pattern
        pattern = self._extract_seasonal_pattern(values, seasonal_period)

        if not pattern:
            return ForecastResult(metric_name=metric_name, forecast_method="seasonal", last_known_value=values[-1] if values else 0.0, forecast_horizon=0)

        # Generate forecasts based on pattern
        forecasts = []
        base_idx = len(values)
        std_dev = self._calculate_stddev(values)

        for i in range(periods_ahead):
            pattern_idx = (base_idx + i) % len(pattern)
            pred = pattern[pattern_idx]
            margin = std_dev * 0.3
            forecast = Forecast(
                timestamp=timestamps[-1] + timedelta(hours=i + 1) if timestamps else datetime.utcnow() + timedelta(hours=i + 1),
                predicted_value=pred,
                confidence_interval_lower=pred - margin,
                confidence_interval_upper=pred + margin,
            )
            forecasts.append(forecast)

        accuracy = self._calculate_forecast_accuracy(metric_name)

        return ForecastResult(
            metric_name=metric_name,
            forecast_method="seasonal",
            last_known_value=values[-1],
            forecast_horizon=periods_ahead,
            forecasts=forecasts,
            accuracy_score=accuracy,
            trend_direction="stable",
            trend_strength="weak",
        )

    def ensemble_forecast(
        self, metric_name: str, periods_ahead: int = 10
    ) -> ForecastResult:
        """Ensemble forecast combining multiple methods"""
        if metric_name not in self.history or len(self.history[metric_name]) < 10:
            return ForecastResult(metric_name=metric_name, forecast_method="ensemble", last_known_value=0.0, forecast_horizon=0)

        # Get forecasts from different methods
        linear = self.forecast_linear(metric_name, periods_ahead)
        exponential = self.forecast_exponential_smoothing(metric_name, periods_ahead)
        seasonal = self.forecast_seasonal(metric_name, periods_ahead) if len(self.history[metric_name]) >= 14 else None

        if not linear.forecasts or not exponential.forecasts:
            return ForecastResult(metric_name=metric_name, forecast_method="ensemble", last_known_value=0.0, forecast_horizon=0)

        # Combine forecasts (weighted average)
        ensemble_forecasts = []
        methods = [linear, exponential]
        if seasonal and seasonal.forecasts:
            methods.append(seasonal)

        for i in range(periods_ahead):
            predictions = []
            for method in methods:
                if i < len(method.forecasts):
                    predictions.append(method.forecasts[i].predicted_value)

            if predictions:
                avg_pred = sum(predictions) / len(predictions)
                # Combined confidence interval
                lower_cis = [method.forecasts[i].confidence_interval_lower for method in methods if i < len(method.forecasts)]
                upper_cis = [method.forecasts[i].confidence_interval_upper for method in methods if i < len(method.forecasts)]

                forecast = Forecast(
                    timestamp=linear.forecasts[i].timestamp,
                    predicted_value=avg_pred,
                    confidence_interval_lower=sum(lower_cis) / len(lower_cis) if lower_cis else avg_pred - 10,
                    confidence_interval_upper=sum(upper_cis) / len(upper_cis) if upper_cis else avg_pred + 10,
                )
                ensemble_forecasts.append(forecast)

        # Determine trend
        values = self.history[metric_name]
        trend_direction = linear.trend_direction
        trend_strength = linear.trend_strength

        accuracy = self._calculate_forecast_accuracy(metric_name)

        return ForecastResult(
            metric_name=metric_name,
            forecast_method="ensemble",
            last_known_value=values[-1],
            forecast_horizon=periods_ahead,
            forecasts=ensemble_forecasts,
            accuracy_score=accuracy,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
        )

    @staticmethod
    def _calculate_stddev(values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return variance ** 0.5

    @staticmethod
    def _extract_seasonal_pattern(values: List[float], seasonal_period: int) -> List[float]:
        """Extract seasonal pattern"""
        if len(values) < seasonal_period:
            return []

        pattern = [0.0] * seasonal_period
        counts = [0] * seasonal_period

        for i, value in enumerate(values):
            idx = i % seasonal_period
            pattern[idx] += value
            counts[idx] += 1

        # Average
        pattern = [pattern[i] / counts[i] if counts[i] > 0 else 0 for i in range(seasonal_period)]
        return pattern

    def _calculate_forecast_accuracy(self, metric_name: str) -> float:
        """Calculate historical forecast accuracy"""
        if metric_name not in self.past_predictions or len(self.past_predictions[metric_name]) < 1:
            return 0.0

        predictions = self.past_predictions[metric_name]
        mape = sum(abs((pred - actual) / actual) if actual != 0 else 0 for pred, actual in predictions) / len(predictions) if predictions else 0

        # Convert MAPE to accuracy percentage
        accuracy = max(0, min(100, 100 - (mape * 100)))
        return accuracy

    def record_prediction(self, metric_name: str, predicted: float, actual: float):
        """Record prediction for accuracy calculation"""
        if metric_name not in self.past_predictions:
            self.past_predictions[metric_name] = []

        self.past_predictions[metric_name].append((predicted, actual))

        # Keep last 100
        if len(self.past_predictions[metric_name]) > 100:
            self.past_predictions[metric_name] = self.past_predictions[metric_name][-100:]
