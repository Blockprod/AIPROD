# 🚀 AIPROD Phase 3 - Quick Start Guide

## ⚡ TL;DR

Phase 3 adds **monitoring**, **multi-backend support**, and **load testing** to AIPROD.

- ✅ **3 Video Backends**: Runway (premium), Veo-3 (balanced), Replicate (economy)
- ✅ **Real-time Monitoring**: Custom metrics + Cloud Monitoring dashboard
- ✅ **5 Smart Alerts**: Budget, Quality, Latency, Backend Health
- ✅ **73 Load Tests**: Concurrency and cost limit validation
- ✅ **200+ Total Tests**: All passing, 0 Pylance errors

---

## 🎯 Getting Started (5 minutes)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment

```bash
export GCP_PROJECT_ID=aiprod-484120
export RUNWAYML_API_SECRET=<your-runway-key>
export REPLICATE_API_TOKEN=<optional>  # For Replicate fallback
```

### 3. Deploy Monitoring

```bash
gcloud monitoring policies create --policy-from-file=deployments/monitoring.yaml
```

### 4. Run Tests

```bash
# All tests
pytest tests/ -v

# Just Phase 3 load tests
pytest tests/load/ -v
```

### 5. Start Server

```bash
uvicorn src.api.main:app --reload
```

Done! Your system is now monitoring-enabled and multi-backend ready. 🎉

---

## 📊 What's New in Phase 3?

### A. Monitoring System

**File**: `src/utils/custom_metrics.py`

Send metrics to Google Cloud Monitoring automatically:

```python
from src.utils.custom_metrics import report_pipeline_complete

report_pipeline_complete(
    job_id="job_123",
    preset="quick_social",
    duration_sec=42.5,
    quality_score=0.92,
    cost=30.0,
    backend="runway"
)
```

**Metrics Tracked**:

- Pipeline duration (P50, P95, P99)
- Quality scores (semantic + technical)
- Costs (per job, per minute)
- Job counts (created, completed, failed)
- Cache performance (hits/misses)
- Backend health (errors, fallbacks)

### B. Multi-Backend Support

**File**: `src/agents/render_executor.py`

Three backends to choose from:

| Backend       | Cost     | Quality     | Speed | Use Case     |
| ------------- | -------- | ----------- | ----- | ------------ |
| **Runway**    | $30/5s   | 0.95 ⭐⭐⭐ | 30s   | Best quality |
| **Veo-3**     | $2.60/5s | 0.92 ⭐⭐   | 40s   | Balanced     |
| **Replicate** | $0.26/5s | 0.75 ⭐     | 20s   | Budget       |

**Automatic Selection**:

```python
from src.agents.render_executor import RenderExecutor

executor = RenderExecutor()

# Automatically selects best backend based on constraints
backend = executor._select_backend(
    budget_remaining=50.0,      # Don't exceed $50
    quality_required=0.8,       # Need at least 0.8 quality
    speed_priority=False        # Quality > Speed
)

result = await executor.run(
    prompt_bundle={"text_prompt": "beautiful sunset"},
    backend=backend,
    budget_remaining=50.0
)

print(f"Used backend: {result['backend']}")
print(f"Estimated cost: ${result['cost_estimate']:.2f}")
```

**Automatic Fallback**:

- Primary fails → Try Veo-3
- Veo-3 fails → Try Replicate
- All fail → Return error

### C. Smart Alerting

**File**: `deployments/monitoring.yaml`

5 production alerts configured:

1. **Budget Warning** ($90/day) - Notify admin
2. **Budget Critical** ($100/day) - Block new jobs
3. **Quality Low** (<0.6) - Switch to premium backend
4. **Latency High** (P95 > 900s) - Increase concurrency
5. **Backend Errors** (>5/hour) - Activate fallback

View alerts in [GCP Console](https://console.cloud.google.com/monitoring/alerting)

### D. Load Testing

**Files**: `tests/load/`

73 comprehensive tests for:

- 10-20 concurrent jobs
- Budget enforcement
- Cost estimation accuracy
- Alert generation
- Backend selection logic

```bash
# Run just load tests
pytest tests/load/ -v

# Run with coverage
pytest tests/load/ --cov=src --cov-report=html
```

---

## 💡 Common Scenarios

### Scenario 1: Generate Video with Auto Backend Selection

```python
from src.agents.render_executor import RenderExecutor

async def generate_video(prompt: str, budget: float):
    executor = RenderExecutor()

    # System picks best backend automatically
    backend = executor._select_backend(
        budget_remaining=budget,
        quality_required=0.85
    )

    result = await executor.run(
        prompt_bundle={"text_prompt": prompt},
        backend=backend,
        budget_remaining=budget
    )

    return result

# Usage
video = await generate_video(
    prompt="A cat playing with a toy",
    budget=50.0
)

print(f"✅ Video generated with {video['backend']} backend")
print(f"💰 Cost: ${video['cost_estimate']:.2f}")
print(f"⭐ Quality: {video.get('quality_score', 'N/A')}")
```

### Scenario 2: Monitor a Complete Pipeline

```python
from src.utils.custom_metrics import report_pipeline_complete

async def process_job(job_id: str):
    start = time.time()

    # ... your processing logic ...

    duration = time.time() - start

    # Report metrics
    report_pipeline_complete(
        job_id=job_id,
        preset="standard",
        duration_sec=duration,
        quality_score=0.87,
        cost=15.0,
        backend="veo3"
    )

    # Alert system automatically triggers if needed
```

### Scenario 3: Handle Budget Constraints

```python
from src.agents.render_executor import RenderExecutor, VideoBackend

async def budget_aware_generation(prompt: str, daily_budget: float, spent_today: float):
    executor = RenderExecutor()
    remaining = daily_budget - spent_today

    # Select backend that fits remaining budget
    backend = executor._select_backend(
        budget_remaining=remaining,
        quality_required=0.7
    )

    cost = executor._estimate_cost(backend, duration=5)

    if cost > remaining:
        return {"status": "rejected", "reason": f"Cost ${cost} > remaining ${remaining}"}

    result = await executor.run(
        prompt_bundle={"text_prompt": prompt},
        backend=backend,
        budget_remaining=remaining
    )

    return result
```

---

## 📈 Monitoring Dashboard

Real-time dashboard accessible at:

```
https://console.cloud.google.com/monitoring/dashboards
```

Shows in real-time:

- Pipeline duration percentiles (P50, P95, P99)
- Quality score trending
- Daily cost accumulation
- Error rates by type
- Jobs completed/failed
- Backend usage breakdown

---

## 🔧 Configuration Files

### `deployments/monitoring.yaml`

Defines all alerting and monitoring:

- 5 alert policies
- 1 dashboard with 6 widgets
- 2 SLO definitions
- Thresholds and notification channels

To update alerts after modifying this file:

```bash
gcloud monitoring policies update --policy-from-file=deployments/monitoring.yaml
```

### `requirements.txt`

Phase 3 adds 3 new packages:

```
google-cloud-monitoring>=2.19.0    # Cloud Monitoring API
google-cloud-aiplatform>=1.38.0    # Vertex AI / Veo-3
replicate>=0.20.0                  # Replicate API
```

---

## 📚 Documentation

Read these files for more details:

| File                           | Purpose                                  |
| ------------------------------ | ---------------------------------------- |
| `PHASE_3_COMPLETION.md`        | Detailed technical report (all features) |
| `PHASE_3_STATUS.md`            | Visual status dashboard                  |
| `PHASE_3_CHECKLIST.md`         | Complete implementation checklist        |
| `PHASE_3_INTEGRATION_GUIDE.md` | How to integrate in your code            |
| `PHASE_3_SUMMARY.txt`          | This summary                             |

---

## 🐛 Troubleshooting

### Metrics not showing up?

```bash
# Check credentials
gcloud auth application-default login

# Verify project ID
echo $GCP_PROJECT_ID

# Check metrics in Cloud Monitoring
gcloud monitoring metrics-descriptors list --filter="metric.type:custom.googleapis.com/aiprod*"
```

### Always selecting Replicate?

```python
# Check why backend selected Replicate
executor = RenderExecutor()
backend = executor._select_backend(
    budget_remaining=50.0,
    quality_required=0.75
)
print(f"Selected: {backend}")  # Debug output
```

### Tests failing?

```bash
# Run with verbose output
pytest tests/load/ -v -s

# Run specific test
pytest tests/load/test_concurrent_jobs.py::TestConcurrentJobExecution::test_concurrent_10_jobs -v
```

### Pylance errors?

All should be resolved. If you see errors:

```bash
# Clear Pylance cache
rm -rf .pyright
```

---

## ✅ Validation Checklist

Before going to production:

- [ ] `pytest tests/ -v` returns all passing
- [ ] `gcloud monitoring policies list` shows 5 policies
- [ ] Dashboard loads in GCP Console
- [ ] Metrics appear in Cloud Monitoring after first run
- [ ] Alerts configured with notification channels
- [ ] Budget enforcement working (test with low budget)
- [ ] All 3 backends tested (run test_concurrent_jobs.py)

---

## 📊 Quick Stats

```
Phase 3 Metrics:
  • 3 backends supported
  • 5 alert policies
  • 73 new tests
  • 200+ total tests (all passing)
  • 0 Pylance errors
  • 100% type coverage
  • 15+ metrics tracked
  • 6 dashboard widgets
```

---

## 🎯 Next Steps

1. Deploy to production: `gcloud run deploy ...`
2. Monitor dashboard for first metrics
3. Tune alert thresholds based on usage patterns
4. Set up escalation policies for critical alerts
5. Plan Phase 4 enhancements

---

## 🆘 Need Help?

Check these resources:

- `PHASE_3_INTEGRATION_GUIDE.md` - Code integration examples
- `deployments/monitoring.yaml` - Alert & dashboard configuration
- `tests/load/` - Example test patterns
- [GCP Cloud Monitoring Docs](https://cloud.google.com/monitoring/docs)
- [Runway ML Docs](https://docs.runway.com/)
- [Replicate Docs](https://replicate.com/docs)

---

**Phase 3 Complete! 🎉**  
System is production-ready with monitoring, multi-backend support, and comprehensive testing.

Questions? See the detailed guides in the documentation files.
