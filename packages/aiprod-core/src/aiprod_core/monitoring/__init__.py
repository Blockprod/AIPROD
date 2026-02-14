"""
AIPROD Monitoring & Observability
==================================

Production-grade observability stack:
- OpenTelemetry distributed tracing (Jaeger/Tempo export)
- Prometheus metrics (latency, throughput, GPU utilization, model quality)
- Structured JSON logging (Loki/CloudWatch compatible)
- Custom model quality metrics (FID, CLIP-Score, drift detection)
- SLO monitoring with alerting hooks (PagerDuty/Grafana)

All components are opt-in: if the OpenTelemetry SDK is not installed,
the module falls back to no-op implementations.
"""
