"""
Machine Learning models for advanced analytics
"""
import logging
from typing import List, Dict, Tuple, Optional, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import statistics

logger = logging.getLogger(__name__)


@dataclass
class DataPoint:
    """Represents a single data point with timestamp"""
    timestamp: datetime
    value: float
    label: Optional[str] = None


@dataclass
class TrendAnalysis:
    """Results of trend analysis"""
    direction: str  # "increasing", "decreasing", "stable"
    slope: float  # Change per time unit
    r_squared: float  # Coefficient of determination (0-1)
    confidence: float  # Confidence level (0-100%)
    next_predicted_value: float


@dataclass
class CorrelationAnalysis:
    """Results of correlation analysis"""
    series1_name: str
    series2_name: str
    correlation_coefficient: float  # -1 to 1
    is_significant: bool  # p < 0.05
    interpretation: str  # "strong positive", "weak negative", etc.


class LinearRegression:
    """Simple linear regression implementation"""

    def __init__(self):
        self.slope: float = 0.0
        self.intercept: float = 0.0
        self.r_squared: float = 0.0

    def fit(self, x: Sequence[float], y: Sequence[float]) -> None:
        """Fit linear regression model"""
        if len(x) < 2 or len(y) < 2 or len(x) != len(y):
            logger.warning("Insufficient data for linear regression")
            self.slope = 0.0
            self.intercept = 0.0
            self.r_squared = 0.0
            return

        n = len(x)
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(y)

        # Calculate slope
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            self.slope = 0.0
        else:
            self.slope = numerator / denominator

        # Calculate intercept
        self.intercept = y_mean - self.slope * x_mean

        # Calculate R-squared
        ss_res = sum((y[i] - (self.slope * x[i] + self.intercept)) ** 2 for i in range(n))
        ss_tot = sum((y[i] - y_mean) ** 2 for i in range(n))

        if ss_tot == 0:
            self.r_squared = 0.0
        else:
            self.r_squared = 1 - (ss_res / ss_tot)

    def predict(self, x: float) -> float:
        """Predict value for given x"""
        return self.slope * x + self.intercept

    def predict_multiple(self, x_values: Sequence[float]) -> List[float]:
        """Predict values for multiple x values"""
        return [self.predict(x) for x in x_values]


class ExponentialSmoothing:
    """Exponential smoothing for time series smoothing"""

    def __init__(self, alpha: float = 0.3):
        """
        Initialize exponential smoothing
        alpha: smoothing factor (0-1, higher = more weight to recent values)
        """
        self.alpha = alpha
        self.last_smoothed: Optional[float] = None

    def smooth(self, values: Sequence[float]) -> List[float]:
        """Apply exponential smoothing to series"""
        if not values:
            return []

        smoothed = []
        self.last_smoothed = values[0]
        smoothed.append(self.last_smoothed)

        for value in values[1:]:
            self.last_smoothed = self.alpha * value + (1 - self.alpha) * self.last_smoothed
            smoothed.append(self.last_smoothed)

        return smoothed

    def forecast_next(self) -> Optional[float]:
        """Forecast next value based on last smoothed value"""
        return self.last_smoothed


class MovingAverage:
    """Moving average filter"""

    def __init__(self, window_size: int = 7):
        """Initialize moving average"""
        self.window_size = max(1, window_size)

    def calculate(self, values: Sequence[float]) -> List[float]:
        """Calculate moving average"""
        if not values or len(values) < self.window_size:
            return list(values)

        moving_averages = []
        for i in range(len(values)):
            start = max(0, i - self.window_size + 1)
            window = values[start : i + 1]
            moving_averages.append(statistics.mean(window))

        return moving_averages

    def get_latest(self, values: Sequence[float]) -> float:
        """Get latest moving average"""
        if not values:
            return 0.0

        window = values[-self.window_size :]
        return statistics.mean(window) if window else 0.0


class STLDecomposition:
    """Seasonal and Trend decomposition using LOESS"""

    def __init__(self, seasonal_period: int = 7):
        """Initialize STL decomposition"""
        self.seasonal_period = max(1, seasonal_period)

    def decompose(self, values: Sequence[float]) -> Dict[str, List[float]]:
        """Decompose time series into trend, seasonal, and residual"""
        if len(values) < self.seasonal_period:
            return {"trend": list(values), "seasonal": [0] * len(values), "residual": [0] * len(values)}

        # Calculate trend using moving average
        trend_filter = MovingAverage(self.seasonal_period)
        trend = trend_filter.calculate(values)

        # Calculate detrended values
        detrended = [values[i] - trend[i] for i in range(len(values))]

        # Calculate seasonal component
        seasonal = self._extract_seasonal(detrended)

        # Calculate residual
        residual = [values[i] - trend[i] - seasonal[i] for i in range(len(values))]

        return {"trend": trend, "seasonal": seasonal, "residual": residual}

    def _extract_seasonal(self, detrended: Sequence[float]) -> List[float]:
        """Extract seasonal component from detrended values"""
        seasonal = [0.0] * len(detrended)

        for i in range(len(detrended)):
            seasonal_idx = i % self.seasonal_period
            seasonal_values = [
                detrended[j]
                for j in range(len(detrended))
                if j % self.seasonal_period == seasonal_idx
            ]
            if seasonal_values:
                seasonal[i] = statistics.mean(seasonal_values)

        return seasonal


class AnomalyDetector:
    """Statistical anomaly detection using Z-score and IQR methods"""

    @staticmethod
    def z_score_anomalies(values: Sequence[float], threshold: float = 3.0) -> List[bool]:
        """Detect anomalies using Z-score method"""
        if len(values) < 2:
            return [False] * len(values)

        try:
            mean = statistics.mean(values)
            stdev = statistics.stdev(values)

            if stdev == 0:
                return [False] * len(values)

            return [abs((v - mean) / stdev) > threshold for v in values]
        except Exception as e:
            logger.warning(f"Error in Z-score anomaly detection: {e}")
            return [False] * len(values)

    @staticmethod
    def iqr_anomalies(values: Sequence[float], multiplier: float = 1.5) -> List[bool]:
        """Detect anomalies using IQR method"""
        if len(values) < 4:
            return [False] * len(values)

        try:
            sorted_values = sorted(values)
            q1_idx = len(sorted_values) // 4
            q3_idx = (3 * len(sorted_values)) // 4

            q1 = sorted_values[q1_idx]
            q3 = sorted_values[q3_idx]
            iqr = q3 - q1

            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr

            return [v < lower_bound or v > upper_bound for v in values]
        except Exception as e:
            logger.warning(f"Error in IQR anomaly detection: {e}")
            return [False] * len(values)

    @staticmethod
    def isolation_anomalies(values: Sequence[float], contamination: float = 0.1) -> List[bool]:
        """Simple isolation forest-like anomaly detection"""
        if len(values) < 2 or contamination <= 0:
            return [False] * len(values)

        # Calculate deviation from median
        median = statistics.median(values)
        deviations = [abs(v - median) for v in values]

        # Find threshold based on contamination ratio
        sorted_deviations = sorted(deviations, reverse=True)
        threshold_idx = max(0, int(len(values) * contamination))
        threshold_idx = min(threshold_idx, len(sorted_deviations) - 1)
        threshold = sorted_deviations[threshold_idx] if threshold_idx < len(sorted_deviations) else 0

        return [dev >= threshold for dev in deviations]


class Correlation:
    """Correlation analysis"""

    @staticmethod
    def pearson_correlation(x: Sequence[float], y: Sequence[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        try:
            x_mean = statistics.mean(x)
            y_mean = statistics.mean(y)

            numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x)))
            x_denom = sum((x[i] - x_mean) ** 2 for i in range(len(x)))
            y_denom = sum((y[i] - y_mean) ** 2 for i in range(len(y)))

            denominator = (x_denom * y_denom) ** 0.5

            if denominator == 0:
                return 0.0

            return numerator / denominator
        except Exception as e:
            logger.warning(f"Error calculating correlation: {e}")
            return 0.0

    @staticmethod
    def interpret_correlation(coefficient: float) -> str:
        """Interpret correlation coefficient"""
        abs_coef = abs(coefficient)
        sign = "positive" if coefficient >= 0 else "negative"

        if abs_coef < 0.1:
            return "negligible " + sign
        elif abs_coef < 0.3:
            return "weak " + sign
        elif abs_coef < 0.5:
            return "moderate " + sign
        elif abs_coef < 0.7:
            return "strong " + sign
        else:
            return "very strong " + sign
