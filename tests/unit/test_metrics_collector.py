import pytest
from src.utils.metrics_collector import MetricsCollector

def test_record_execution():
    collector = MetricsCollector()
    collector.record_execution({}, 1000, 0.5, 0.8)
    assert collector.metrics["pipeline_executions"] == 1
    assert collector.metrics["avg_latency_ms"] == 1000
    assert collector.metrics["avg_cost"] == 0.5
    assert collector.metrics["avg_quality"] == 0.8


def test_multiple_executions():
    collector = MetricsCollector()
    collector.record_execution({}, 1000, 0.5, 0.8)
    collector.record_execution({}, 2000, 1.0, 0.7)
    assert collector.metrics["pipeline_executions"] == 2
    assert collector.metrics["avg_latency_ms"] == 1500
    assert collector.metrics["avg_cost"] == 0.75
    assert collector.metrics["avg_quality"] == 0.75


def test_check_alerts_high_latency():
    collector = MetricsCollector()
    for _ in range(3):
        collector.record_execution({}, 6000, 0.5, 0.8)
    alerts = collector.check_alerts()
    assert alerts["high_latency"] is True


def test_check_alerts_low_quality():
    collector = MetricsCollector()
    for _ in range(3):
        collector.record_execution({}, 1000, 0.5, 0.5)
    alerts = collector.check_alerts()
    assert alerts["low_quality"] is True


def test_record_error():
    collector = MetricsCollector()
    collector.record_error("Test error")
    assert collector.metrics["pipeline_errors"] == 1
