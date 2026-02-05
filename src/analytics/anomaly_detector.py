"""
Advanced anomaly detection engine for performance monitoring
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from src.analytics.ml_models import (
    LinearRegression,
    ExponentialSmoothing,
    MovingAverage,
    STLDecomposition,
    AnomalyDetector as StatAnomalyDetector,
)

logger = logging.getLogger(__name__)


class AnomalyType(str, Enum):
    """Types of anomalies detected"""
    SPIKE = "spike"
    DROP = "drop"
    TREND_CHANGE = "trend_change"
    SEASONAL_ANOMALY = "seasonal_anomaly"
    OUTLIER = "outlier"


@dataclass
class Anomaly:
    """Detected anomaly"""
    timestamp: datetime
    value: float
    expected_value: float
    severity: str  # "low", "medium", "high", "critical"
    type: AnomalyType
    confidence: float  # 0-100%
    description: str


@dataclass
class AnomalyReport:
    """Report of anomalies detected"""
    metric_name: str
    time_period: str
    anomalies: List[Anomaly] = field(default_factory=list)
    total_points_analyzed: int = 0
    anomaly_percentage: float = 0.0
    dominant_anomaly_type: str = "none"


class AnomalyDetectionEngine:
    """Advanced anomaly detection engine using multiple methods"""

    def __init__(self):
        self.history: Dict[str, List[float]] = {}
        self.timestamps: Dict[str, List[datetime]] = {}

    def add_data_point(self, metric_name: str, value: float, timestamp: Optional[datetime] = None):
        """Add data point to metric history"""
        if metric_name not in self.history:
            self.history[metric_name] = []
            self.timestamps[metric_name] = []

        self.history[metric_name].append(value)
        self.timestamps[metric_name].append(timestamp or datetime.utcnow())

        # Keep last 1000 points
        if len(self.history[metric_name]) > 1000:
            self.history[metric_name] = self.history[metric_name][-1000:]
            self.timestamps[metric_name] = self.timestamps[metric_name][-1000:]

    def detect_anomalies(self, metric_name: str, lookback_points: int = 100) -> AnomalyReport:
        """Detect anomalies in metric using multiple methods"""
        if metric_name not in self.history or len(self.history[metric_name]) < 5:
            return AnomalyReport(metric_name=metric_name, time_period="N/A", total_points_analyzed=0)

        values = self.history[metric_name][-lookback_points:]
        timestamps = self.timestamps[metric_name][-lookback_points:]

        anomalies: List[Anomaly] = []

        # Method 1: Z-score
        z_score_anomalies = StatAnomalyDetector.z_score_anomalies(values, threshold=2.5)
        anomalies.extend(
            self._create_anomalies_from_detections(
                values, timestamps, z_score_anomalies, AnomalyType.OUTLIER, "Z-score detection", 0.8
            )
        )

        # Method 2: IQR
        iqr_anomalies = StatAnomalyDetector.iqr_anomalies(values, multiplier=1.5)
        anomalies.extend(
            self._create_anomalies_from_detections(
                values, timestamps, iqr_anomalies, AnomalyType.SPIKE, "IQR detection", 0.7
            )
        )

        # Method 3: Trend change detection
        trend_anomalies = self._detect_trend_changes(values, timestamps)
        anomalies.extend(trend_anomalies)

        # Method 4: Seasonal decomposition
        seasonal_anomalies = self._detect_seasonal_anomalies(values, timestamps)
        anomalies.extend(seasonal_anomalies)

        # Deduplicate and merge anomalies
        anomalies = self._deduplicate_anomalies(anomalies)

        # Calculate statistics
        anomaly_percentage = (len(anomalies) / len(values) * 100) if values else 0
        dominant_type = max(
            ({AnomalyType.SPIKE, AnomalyType.DROP, AnomalyType.TREND_CHANGE, AnomalyType.SEASONAL_ANOMALY, AnomalyType.OUTLIER} & {a.type for a in anomalies})
            or {AnomalyType.OUTLIER}
        )

        time_period = f"{len(values)} points" if values else "N/A"

        return AnomalyReport(
            metric_name=metric_name,
            time_period=time_period,
            anomalies=anomalies,
            total_points_analyzed=len(values),
            anomaly_percentage=round(anomaly_percentage, 2),
            dominant_anomaly_type=dominant_type,
        )

    def _create_anomalies_from_detections(
        self,
        values: List[float],
        timestamps: List[datetime],
        detections: List[bool],
        anomaly_type: AnomalyType,
        method: str,
        base_confidence: float,
    ) -> List[Anomaly]:
        """Create anomaly objects from detection results"""
        anomalies = []
        mean_val = sum(values) / len(values) if values else 0

        for i, is_anomaly in enumerate(detections):
            if is_anomaly and i < len(values):
                severity = self._calculate_severity(values[i], mean_val)
                confidence = min(100, base_confidence * 100)

                anomaly = Anomaly(
                    timestamp=timestamps[i],
                    value=values[i],
                    expected_value=mean_val,
                    severity=severity,
                    type=anomaly_type,
                    confidence=confidence,
                    description=f"{method}: value {values[i]:.2f} significantly different from expected {mean_val:.2f}",
                )
                anomalies.append(anomaly)

        return anomalies

    def _detect_trend_changes(self, values: List[float], timestamps: List[datetime]) -> List[Anomaly]:
        """Detect significant trend changes"""
        if len(values) < 10:
            return []

        anomalies = []
        mid = len(values) // 2

        # Check first half trend
        x1 = [float(i) for i in range(mid)]
        reg1 = LinearRegression()
        reg1.fit(x1, values[:mid])

        # Check second half trend
        x2 = [float(i) for i in range(len(values) - mid)]
        reg2 = LinearRegression()
        reg2.fit(x2, values[mid:])

        # If trend slopes are significantly different
        slope_diff = abs(reg1.slope - reg2.slope)
        avg_slope = (abs(reg1.slope) + abs(reg2.slope)) / 2
        avg_slope = avg_slope if avg_slope > 0 else 1

        if slope_diff > avg_slope * 0.5:  # 50% change in trend
            anomaly = Anomaly(
                timestamp=timestamps[mid],
                value=values[mid],
                expected_value=sum(values) / len(values),
                severity="medium",
                type=AnomalyType.TREND_CHANGE,
                confidence=75,
                description=f"Significant trend change detected at point {mid}. Slope changed from {reg1.slope:.3f} to {reg2.slope:.3f}",
            )
            anomalies.append(anomaly)

        return anomalies

    def _detect_seasonal_anomalies(self, values: List[float], timestamps: List[datetime]) -> List[Anomaly]:
        """Detect anomalies based on seasonal decomposition"""
        if len(values) < 14:  # Need at least 2 seasonal periods
            return []

        try:
            decomposition = STLDecomposition(seasonal_period=7)
            components = decomposition.decompose(values)

            anomalies = []
            residuals = components.get("residual", [])
            mean_residual = sum(abs(r) for r in residuals) / len(residuals) if residuals else 0

            for i, residual in enumerate(residuals):
                if abs(residual) > mean_residual * 2:  # More than 2x average residual
                    severity = "high" if abs(residual) > mean_residual * 3 else "medium"
                    anomaly = Anomaly(
                        timestamp=timestamps[i],
                        value=values[i],
                        expected_value=components["trend"][i] + components["seasonal"][i],
                        severity=severity,
                        type=AnomalyType.SEASONAL_ANOMALY,
                        confidence=70,
                        description=f"Anomaly detected: value deviates {abs(residual):.2f} from seasonal pattern",
                    )
                    anomalies.append(anomaly)

            return anomalies
        except Exception as e:
            logger.warning(f"Error in seasonal anomaly detection: {e}")
            return []

    @staticmethod
    def _calculate_severity(value: float, expected: float) -> str:
        """Calculate severity based on deviation"""
        if expected == 0:
            return "medium"

        deviation = abs(value - expected) / abs(expected)

        if deviation > 2:
            return "critical"
        elif deviation > 1:
            return "high"
        elif deviation > 0.5:
            return "medium"
        else:
            return "low"

    @staticmethod
    def _deduplicate_anomalies(anomalies: List[Anomaly]) -> List[Anomaly]:
        """Remove duplicate or overlapping anomalies"""
        if not anomalies:
            return []

        # Sort by timestamp
        anomalies.sort(key=lambda a: a.timestamp)

        deduped = []
        last_timestamp = None

        for anomaly in anomalies:
            # Skip if too close to last anomaly (within 1 minute)
            if last_timestamp and (anomaly.timestamp - last_timestamp).total_seconds() < 60:
                continue

            deduped.append(anomaly)
            last_timestamp = anomaly.timestamp

        return deduped

    def get_anomaly_summary(self, metric_name: str) -> Dict[str, Any]:
        """Get summary of all anomalies for a metric"""
        report = self.detect_anomalies(metric_name)

        anomaly_types = {}
        for anomaly in report.anomalies:
            anomaly_types[anomaly.type] = anomaly_types.get(anomaly.type, 0) + 1

        severity_counts = {}
        for anomaly in report.anomalies:
            severity_counts[anomaly.severity] = severity_counts.get(anomaly.severity, 0) + 1

        return {
            "metric_name": metric_name,
            "total_anomalies": len(report.anomalies),
            "anomaly_percentage": report.anomaly_percentage,
            "anomaly_types": anomaly_types,
            "severity_breakdown": severity_counts,
            "dominant_type": report.dominant_anomaly_type,
            "recent_anomalies": [
                {
                    "timestamp": a.timestamp.isoformat(),
                    "value": a.value,
                    "severity": a.severity,
                    "type": a.type,
                    "confidence": a.confidence,
                }
                for a in report.anomalies[-5:]
            ],
        }
