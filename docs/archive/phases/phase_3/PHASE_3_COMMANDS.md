# 🚀 Phase 3 - Quick Command Reference

## Test Commands

### Run All Tests

```bash
python -m pytest tests/ -v
```

### Run Only Phase 3 Tests

```bash
python -m pytest tests/load/ -v
```

### Run Concurrent Job Tests

```bash
python -m pytest tests/load/test_concurrent_jobs.py -v
```

### Run Cost/Budget Tests

```bash
python -m pytest tests/load/test_cost_limits.py -v
```

### Run Tests with Coverage

```bash
python -m pytest tests/ -v --cov=src --cov-report=html
```

### Run Specific Test

```bash
python -m pytest tests/load/test_concurrent_jobs.py::TestConcurrentJobExecution::test_10_concurrent_jobs -v
```

---

## Deployment Commands

### Deploy to GCP Cloud Monitoring

```bash
gcloud monitoring policies create --policy-from-file=deployments/monitoring.yaml
```

### View Monitoring Alerts

```bash
gcloud monitoring policies list
```

### Check Alert Status

```bash
gcloud monitoring alerts-policies list --filter="displayName:Budget"
```

### Deploy Docker Container

```bash
docker-compose up -d
```

### View Docker Logs

```bash
docker-compose logs -f api
```

---

## Development Commands

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run API Server

```bash
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Run API Server (VS Code Task)

```
Ctrl+Shift+B → Select "Run API Server"
```

### Check Types with Pylance

```bash
python -m pylance check src/
```

### Format Code

```bash
black src/ tests/
```

### Sort Imports

```bash
isort src/ tests/
```

---

## Monitoring Commands

### Check Custom Metrics

```bash
gcloud monitoring time-series list \
  --filter='metric.type="custom.googleapis.com/pipeline/duration"'
```

### View Alert Policies

```bash
gcloud monitoring policies describe <POLICY_ID>
```

### Create Notification Channel

```bash
gcloud alpha monitoring channels create \
  --display-name="Email Alert" \
  --type=email \
  --channel-labels=email_address=your-email@example.com
```

---

## Debugging Commands

### Monitor Metrics in Real-time

```bash
python scripts/monitor.py
```

### Check Backend Health

```bash
curl http://localhost:8000/health
```

### Check Cost Estimate

```bash
curl -X POST http://localhost:8000/api/v1/estimate-cost \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cat running", "duration": 5}'
```

### Trigger Video Generation

```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a cat running",
    "duration": 5,
    "backend": "AUTO"
  }'
```

---

## Environment Variables

### Required for Production

```bash
export GOOGLE_CLOUD_PROJECT=aiprod-484120
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
export RUNWAY_API_KEY=your-runway-key
export REPLICATE_API_TOKEN=your-replicate-token
```

### Optional

```bash
export METRICS_MODE=cloud  # or "local" for development
export BUDGET_LIMIT_DAILY=100
export DEBUG=false
```

---

## Quick Validation

### Check All Files Are Present

```bash
# Linux/Mac
ls -la src/utils/custom_metrics.py
ls -la src/agents/render_executor.py
ls -la deployments/monitoring.yaml
ls -la tests/load/

# Windows PowerShell
Test-Path src/utils/custom_metrics.py
Test-Path src/agents/render_executor.py
Test-Path deployments/monitoring.yaml
Test-Path tests/load/
```

### Validate Python Syntax

```bash
python -m py_compile src/utils/custom_metrics.py
python -m py_compile src/agents/render_executor.py
```

### Check Requirements

```bash
pip check
```

### Validate Config File

```bash
python -c "import yaml; yaml.safe_load(open('deployments/monitoring.yaml'))"
```

---

## Production Deployment Checklist

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run all tests
python -m pytest tests/ -v

# 3. Check types
python -m pylance check src/

# 4. Build Docker image
docker build -t aiprod-v33:latest .

# 5. Deploy monitoring
gcloud monitoring policies create --policy-from-file=deployments/monitoring.yaml

# 6. Start services
docker-compose up -d

# 7. Verify deployment
curl http://localhost:8000/health
```

---

## Troubleshooting Commands

### Check Python Version

```bash
python --version  # Should be 3.13+
```

### Check Package Versions

```bash
pip list | grep -E "(google-cloud|replicate|fastapi|uvicorn)"
```

### Check GCP Authentication

```bash
gcloud auth list
gcloud config list
```

### View Application Logs

```bash
docker-compose logs -f
```

### Check Resource Usage

```bash
docker stats
```

### Test Network Connectivity

```bash
curl -v http://localhost:8000/docs
```

---

## Performance Testing

### Load Test with 10 Concurrent Jobs

```bash
python -m pytest tests/load/test_concurrent_jobs.py::TestConcurrentJobExecution::test_10_concurrent_jobs -v -s
```

### Load Test with 20 Concurrent Jobs

```bash
python -m pytest tests/load/test_concurrent_jobs.py::TestConcurrentJobExecution::test_20_concurrent_jobs -v -s
```

### Stress Test Budget Limits

```bash
python -m pytest tests/load/test_cost_limits.py::TestDailyBudgetTracking -v -s
```

### Benchmark Backend Performance

```bash
python -m pytest tests/load/test_concurrent_jobs.py::TestBackendFallback -v -s
```

---

## Integration Testing

### Test Full Video Generation Pipeline

```bash
python -m pytest tests/integration/test_full_pipeline.py -v
```

### Test with Real GCP Services

```bash
export GOOGLE_CLOUD_PROJECT=aiprod-484120
python -m pytest tests/integration/ -v
```

---

## Monitoring Verification

### Test Metrics Reporting

```python
from src.utils.custom_metrics import CustomMetricsCollector

collector = CustomMetricsCollector("aiprod-484120")
collector.report_metric("test_metric", 42.0)
print("✅ Metrics reporting working")
```

### Test Alert Configuration

```bash
gcloud monitoring policies list --filter='displayName:"Budget"'
```

---

## Cleanup Commands

### Stop All Containers

```bash
docker-compose down
```

### Remove Unused Docker Resources

```bash
docker system prune -a
```

### Clear Python Cache

```bash
find . -type d -name __pycache__ -exec rm -r {} +
find . -type f -name "*.pyc" -delete
```

### Clean Test Artifacts

```bash
rm -rf .pytest_cache
rm -rf htmlcov
rm -rf .coverage
```

---

## Documentation

### View All Documentation

```bash
ls -la PHASE_3_*.md
```

### Quick Start Guide

```bash
cat PHASE_3_QUICK_START.md
```

### Integration Guide

```bash
cat PHASE_3_INTEGRATION_GUIDE.md
```

### Completion Report

```bash
cat PHASE_3_COMPLETION.md
```

---

## Emergency Rollback

### Disable Multi-Backend (use Runway only)

```bash
# Edit src/agents/render_executor.py
# Change: backend = self._select_backend(...)
# To: backend = VideoBackend.RUNWAY
```

### Disable Metrics Reporting

```bash
# In custom_metrics.py, set:
# METRICS_MODE = "mock"
```

### Revert to Previous Version

```bash
git revert HEAD
docker-compose restart
```

---

**Last Updated**: January 15, 2026  
**Phase 3 Status**: ✅ PRODUCTION READY

For detailed information, see PHASE_3_COMPLETION.md
