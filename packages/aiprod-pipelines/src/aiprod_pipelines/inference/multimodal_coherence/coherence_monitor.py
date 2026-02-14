"""Multimodal coherence monitoring and logging.

Provides:
- Real-time coherence tracking
- Historical metrics collection
- Degradation detection
- Reporting and logging
"""

from typing import Optional, Dict, List, Any
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class CoherenceSnapshot:
    """Snapshot of coherence at a point in time."""
    
    def __init__(
        self,
        timestamp: datetime,
        overall_coherence: float,
        components: Dict[str, float],
        issues: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize snapshot.
        
        Args:
            timestamp: When snapshot was taken
            overall_coherence: Overall coherence score 0-1
            components: Component scores
            issues: Detected issues
            metadata: Additional metadata
        """
        self.timestamp = timestamp
        self.overall_coherence = overall_coherence
        self.components = components
        self.issues = issues
        self.metadata = metadata or {}


class CoherenceMonitor:
    """Monitor multimodal coherence in real-time."""
    
    def __init__(
        self,
        window_size: int = 100,
        degradation_threshold: float = 0.1,
    ):
        """
        Initialize monitor.
        
        Args:
            window_size: Size of history window
            degradation_threshold: Threshold for degradation alert
        """
        self.window_size = window_size
        self.degradation_threshold = degradation_threshold
        
        self.history: List[CoherenceSnapshot] = []
        self.start_time = datetime.now()
    
    def record_coherence(
        self,
        coherence_metrics: Any,
        sync_metrics: Any,
        issues: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record coherence measurement.
        
        Args:
            coherence_metrics: Coherence metrics from scorer
            sync_metrics: Sync metrics from engine
            issues: Detected issues
            metadata: Additional data
        """
        # Compile components
        components = {
            "temporal_alignment": coherence_metrics.temporal_alignment,
            "event_correlation": coherence_metrics.event_correlation,
            "spectro_temporal_match": coherence_metrics.spectro_temporal_match,
            "onset_synchrony": coherence_metrics.onset_synchrony,
            "energy_correlation": coherence_metrics.energy_correlation,
        }
        
        if sync_metrics:
            components.update({
                "lip_sync": sync_metrics.lip_sync_score,
                "sync_stability": sync_metrics.sync_stability,
            })
        
        # Create snapshot
        snapshot = CoherenceSnapshot(
            timestamp=datetime.now(),
            overall_coherence=coherence_metrics.overall_coherence,
            components=components,
            issues=issues or [],
            metadata=metadata,
        )
        
        # Add to history
        self.history.append(snapshot)
        
        # Maintain window size
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        logger.debug(f"Coherence recorded: {snapshot.overall_coherence:.3f}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics from history.
        
        Returns:
            Statistics dictionary
        """
        if not self.history:
            return {}
        
        coherences = [s.overall_coherence for s in self.history]
        
        stats = {
            "mean_coherence": float(np.mean(coherences)),
            "std_coherence": float(np.std(coherences)),
            "min_coherence": float(np.min(coherences)),
            "max_coherence": float(np.max(coherences)),
            "current_coherence": float(coherences[-1]),
            "measurements": len(self.history),
        }
        
        # Component statistics
        component_stats = {}
        for component_name in self.history[0].components.keys():
            component_values = [s.components[component_name] for s in self.history]
            component_stats[component_name] = {
                "mean": float(np.mean(component_values)),
                "std": float(np.std(component_values)),
                "min": float(np.min(component_values)),
                "max": float(np.max(component_values)),
            }
        
        stats["components"] = component_stats
        
        return stats
    
    def detect_degradation(self) -> Optional[Dict[str, Any]]:
        """
        Detect coherence degradation.
        
        Returns:
            Degradation alert if detected, None otherwise
        """
        if len(self.history) < 10:
            return None
        
        # Compare recent vs earlier
        recent = np.mean([s.overall_coherence for s in self.history[-5:]])
        earlier = np.mean([s.overall_coherence for s in self.history[:-5]])
        
        # Check if degradation
        degradation = earlier - recent
        
        if degradation > self.degradation_threshold:
            return {
                "detected": True,
                "magnitude": float(degradation),
                "earlier_mean": float(earlier),
                "recent_mean": float(recent),
                "severity": "critical" if degradation > 0.2 else "warning",
            }
        
        return None
    
    def get_issue_summary(self) -> Dict[str, int]:
        """
        Get summary of issues over time.
        
        Returns:
            Count of each issue type
        """
        issue_counts = {}
        
        for snapshot in self.history:
            for issue in snapshot.issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        return issue_counts
    
    def get_health_report(self) -> Dict[str, Any]:
        """
        Get comprehensive health report.
        
        Returns:
            Health report dictionary
        """
        stats = self.get_statistics()
        degradation = self.detect_degradation()
        issues = self.get_issue_summary()
        
        # Determine overall health
        if not stats:
            health = "unknown"
        elif stats["mean_coherence"] > 0.85:
            health = "excellent"
        elif stats["mean_coherence"] > 0.75:
            health = "good"
        elif stats["mean_coherence"] > 0.65:
            health = "fair"
        else:
            health = "poor"
        
        report = {
            "health": health,
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "statistics": stats,
            "degradation": degradation,
            "issues": issues,
        }
        
        return report
    
    def export_metrics(self) -> List[Dict[str, Any]]:
        """
        Export all metrics for external analysis.
        
        Returns:
            List of metric dictionaries
        """
        export_data = []
        
        for snapshot in self.history:
            data = {
                "timestamp": snapshot.timestamp.isoformat(),
                "overall_coherence": snapshot.overall_coherence,
                "components": snapshot.components,
                "issues": snapshot.issues,
            }
            
            if snapshot.metadata:
                data.update({"metadata": snapshot.metadata})
            
            export_data.append(data)
        
        return export_data


class MultimodalCoherenceNode:
    """Graph node for multimodal coherence analysis."""
    
    def __init__(
        self,
        audio_sr: int = 16000,
        video_fps: int = 30,
    ):
        """
        Initialize node.
        
        Args:
            audio_sr: Audio sample rate
            video_fps: Video frame rate
        """
        self.audio_sr = audio_sr
        self.video_fps = video_fps
        self.monitor = CoherenceMonitor()
    
    def execute(
        self,
        audio_analysis: Any,
        video_analysis: Any,
        coherence_metrics: Any,
        sync_metrics: Any,
    ) -> Dict[str, Any]:
        """
        Execute coherence monitoring.
        
        Args:
            audio_analysis: Audio analysis result
            video_analysis: Video analysis result
            coherence_metrics: Coherence metrics
            sync_metrics: Sync metrics
            
        Returns:
            Monitoring result
        """
        # Determine issues
        issues = []
        
        if coherence_metrics.overall_coherence < 0.7:
            issues.append("Low overall coherence")
        
        if sync_metrics:
            if sync_metrics.lip_sync_score < 0.75:
                issues.append("Poor lip-sync quality")
            
            if sync_metrics.sync_stability < 0.7:
                issues.append("Unstable synchronization")
        
        # Record
        self.monitor.record_coherence(
            coherence_metrics=coherence_metrics,
            sync_metrics=sync_metrics,
            issues=issues,
            metadata={
                "num_audio_events": len(audio_analysis.events),
                "num_motion_events": len(video_analysis.motion_events),
                "num_frames": video_analysis.num_frames,
            },
        )
        
        # Get health report
        health = self.monitor.get_health_report()
        
        return {
            "coherence": coherence_metrics.overall_coherence,
            "components": {
                "temporal_alignment": coherence_metrics.temporal_alignment,
                "event_correlation": coherence_metrics.event_correlation,
                "spectro_temporal_match": coherence_metrics.spectro_temporal_match,
                "onset_synchrony": coherence_metrics.onset_synchrony,
                "energy_correlation": coherence_metrics.energy_correlation,
            },
            "sync": {
                "lip_sync": sync_metrics.lip_sync_score if sync_metrics else 0.0,
                "stability": sync_metrics.sync_stability if sync_metrics else 0.0,
                "audio_lead_ms": (sync_metrics.audio_lead * 1000) if sync_metrics else 0.0,
            },
            "issues": issues,
            "health": health,
        }


class CoherenceReporter:
    """Generate coherence reports."""
    
    def __init__(self, monitor: CoherenceMonitor):
        """Initialize reporter."""
        self.monitor = monitor
    
    def generate_summary_report(self) -> str:
        """Generate human-readable summary."""
        health = self.monitor.get_health_report()
        
        report = []
        report.append("=" * 50)
        report.append("MULTIMODAL COHERENCE REPORT")
        report.append("=" * 50)
        report.append(f"Health Status: {health['health'].upper()}")
        report.append(f"Uptime: {health['uptime_seconds']:.1f}s")
        report.append(f"Measurements: {health['statistics']['measurements']}")
        report.append("")
        
        # Overall metrics
        stats = health['statistics']
        report.append("Overall Metrics:")
        report.append(f"  Mean Coherence: {stats['mean_coherence']:.3f}")
        report.append(f"  Current: {stats['current_coherence']:.3f}")
        report.append(f"  Std Dev: {stats['std_coherence']:.3f}")
        report.append("")
        
        # Component breakdown
        report.append("Component Scores:")
        for component, comp_stats in stats['components'].items():
            report.append(f"  {component}: {comp_stats['mean']:.3f}")
        report.append("")
        
        # Issues
        issues = health['issues']
        if issues:
            report.append("Issues Detected:")
            for issue, count in issues.items():
                report.append(f"  - {issue}: {count} times")
        else:
            report.append("No issues detected")
        
        report.append("=" * 50)
        
        return "\n".join(report)
