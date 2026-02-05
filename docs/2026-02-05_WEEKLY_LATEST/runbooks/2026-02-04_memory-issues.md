# Runbook: Memory Issues (OOM)

**Alert ID**: `alert-memory-high`  
**Severity**: 🟡 HIGH  
**Oncall**: Backend Lead  
**Last Updated**: Feb 4, 2026

---

## Quick Diagnosis

### Symptoms

- "Out of Memory" errors in logs
- Container restart loops
- API responding slowly then crashing
- Error rate suddenly increases

### Check Memory Status

```bash
# Check current memory usage
gcloud monitoring time-series list \
  --filter='metric.type="run.googleapis.com/container_memory_utilizations" AND resource.service_name="aiprod-api"' \
  --project=aiprod-484120

# Check recent revisions and their memory allocation
gcloud run revisions list \
  --service=aiprod-api \
  --region=europe-west1 \
  --project=aiprod-484120 \
  --format="table(name, memory_limit, status)"

# Check for restart events
gcloud logging read "resource.service.name=aiprod-api AND severity=WARNING" \
  --project=aiprod-484120 \
  --limit=20 \
  --format="table(timestamp, jsonPayload.message)"
```

---

## Root Causes Analysis

### 1. Memory Leak in Application

**Indicators**:

- Memory usage increases gradually over time
- Restart fixes the issue temporarily (memory resets)
- Always at same line of code in logs

**Diagnosis**:

```bash
# Check recent memory trend
gcloud monitoring time-series list \
  --filter='metric.type="run.googleapis.com/container_memory_utilizations"' \
  --project=aiprod-484120 \
  --format=json | jq '.[] | .points | .[] | {time: .interval.end_time, value: .value.double_value}'

# Find which revision introduced the leak
gcloud run revisions list \
  --service=aiprod-api \
  --region=europe-west1 \
  --project=aiprod-484120 \
  --limit=10

# Check source code for recent changes
git log --oneline -10 src/api/main.py
git log --oneline -10 src/

# Use memory profiler to identify leak
python -m memory_profiler src/api/main.py
```

**Fix**: Rollback to previous version or fix code

```bash
# Option 1: Rollback to previous revision
PREVIOUS_REV=$(gcloud run revisions list \
  --service=aiprod-api \
  --region=europe-west1 \
  --project=aiprod-484120 \
  --sort-by="~revision.name" \
  --limit=2 \
  --format="value(name)" | tail -1)

gcloud run services update-traffic aiprod-api \
  --to-revisions $PREVIOUS_REV=100 \
  --region=europe-west1 \
  --project=aiprod-484120

# Option 2: Fix the leak in code
# Common memory leaks:
# - Unclosed file handles
# - Growing caches without eviction
# - Circular references preventing garbage collection
# - Async tasks not being awaited

# Search for common patterns
grep -r "open(" src/ | grep -v "with open"  # Unclosed files
grep -r "\.append(" src/ | grep -v "list"   # Growing lists
grep -r "@lru_cache" src/ | grep -v "maxsize"  # Unbounded caches
```

---

### 2. Insufficient Memory Allocation

**Indicators**:

- Memory usage near 100% of limit
- Application runs fine with more memory (local testing)
- All requests starting to OOM simultaneously

**Diagnosis**:

```bash
# Check current memory limit
gcloud run services describe aiprod-api \
  --region=europe-west1 \
  --project=aiprod-484120 | grep -E "memory"

# Expected: 2Gi or 4Gi depending on tier

# Check memory usage pattern
gcloud logging read "jsonPayload.memory_usage_mb!=null" \
  --project=aiprod-484120 \
  --limit=50 \
  --format="table(timestamp, jsonPayload.memory_usage_mb)"
```

**Fix**: Increase memory allocation

```bash
# Option 1: Increase memory for new deployments
gcloud run services update aiprod-api \
  --memory=4Gi \
  --region=europe-west1 \
  --project=aiprod-484120

# Verify change applied
gcloud run services describe aiprod-api \
  --region=europe-west1 \
  --project=aiprod-484120 | grep memory

# Option 2: Reduce per-request memory usage
# Profile memory usage:
python -c "
import tracemalloc
tracemalloc.start()
# ... run some code ...
current, peak = tracemalloc.get_traced_memory()
print(f'Current: {current / 1024 / 1024}MB; Peak: {peak / 1024 / 1024}MB')
"
```

---

### 3. Large Cache Not Evicting

**Indicators**:

- Memory usage grows for hours/days
- Cache hit rate is high
- Problem gone after restart

**Diagnosis**:

```bash
# Find large caches in code
grep -r "@.*cache\|Cache(" src/ --include="*.py"

# Check if caches have maxsize
grep -r "maxsize=" src/

# Monitor cache size
gcloud logging read "jsonPayload.cache_size!=null" \
  --project=aiprod-484120 \
  --limit=20
```

**Fix**: Add eviction or size limits to caches

```python
# Bad: Cache grows indefinitely
my_cache = {}  # WRONG!

# Good: Bounded cache
from functools import lru_cache

@lru_cache(maxsize=128)  # Only keeps 128 items
def expensive_function(x):
    return x * 2

# Or use Redis for distributed caching
from redis import Redis
cache = Redis(host='localhost')
cache.setex('key', 3600, value)  # 1 hour expiration
```

---

### 4. Large Request/Response Bodies

**Indicators**:

- OOM happens when processing large files/requests
- Specific endpoints trigger the issue
- Memory spikes during that request

**Diagnosis**:

```bash
# Check request sizes
gcloud logging read "httpRequest.request_size!=null" \
  --project=aiprod-484120 \
  --limit=50 \
  --format="table(timestamp, httpRequest.request_size, httpRequest.requestUrl)"

# Check response sizes
gcloud logging read "httpRequest.response_size!=null" \
  --project=aiprod-484120 \
  --limit=50 \
  --format="table(timestamp, httpRequest.response_size)"

# Find large requests
gcloud logging read "httpRequest.request_size > 10485760" \  # > 10MB
  --project=aiprod-484120 \
  --limit=20
```

**Fix**: Stream large files or increase limits

```python
# Bad: Load entire file into memory
data = open('large_file.bin').read()  # OOM!

# Good: Stream the file
with open('large_file.bin', 'rb') as f:
    chunk_size = 8192
    while True:
        chunk = f.read(chunk_size)
        if not chunk:
            break
        # Process chunk

# Or limit request size
from fastapi import FastAPI
app = FastAPI()

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if file.size > 100 * 1024 * 1024:  # 100MB limit
        raise HTTPException(413, "File too large")
    # Process file
```

---

### 5. Third-Party Library Memory Leak

**Indicators**:

- Memory leak in upstream library
- Issue after updating dependencies
- Reproducible with specific library version

**Diagnosis**:

```bash
# Check recent dependency updates
git log --oneline requirements.txt | head -10

# Find which library is the culprit
pip list | grep -E "numpy|pandas|tensorflow|torch"

# Test each library in isolation
python -c "
import sys
import importlib

# Test suspect library
lib = importlib.import_module('suspect_library')
print(f'Loaded {lib}')

import gc
gc.collect()
print('Memory after import:', __import__('psutil').Process().memory_info().rss / 1024 / 1024, 'MB')
"
```

**Fix**: Update, downgrade, or replace library

```bash
# Option 1: Update to latest version
pip install --upgrade suspect_library

# Option 2: Downgrade to known-good version
pip install suspect_library==1.2.3

# Option 3: Replace with alternative
# pandas OOM? Try polars (more memory efficient)
# numpy OOM? Try numba (compiled code)

# Verify fix
python -c "
import tracemalloc
tracemalloc.start()

# ... use the library ...

current, peak = tracemalloc.get_traced_memory()
print(f'Memory: {peak / 1024 / 1024}MB')
"
```

---

## Recovery Steps

### Step 1: Assess Severity

```bash
# How many containers are OOMing?
gcloud logging read "severity=CRITICAL AND jsonPayload.error_type=OUT_OF_MEMORY" \
  --project=aiprod-484120 \
  --limit=100 | jq 'length'

# Is service still serving traffic?
gcloud monitoring time-series list \
  --filter='metric.type="run.googleapis.com/request_count"' \
  --project=aiprod-484120
```

### Step 2: Emergency Fix (Get Service Online)

```bash
# Option A: Restart with increased memory (immediate)
gcloud run services update aiprod-api \
  --memory=4Gi \
  --region=europe-west1 \
  --project=aiprod-484120

# Option B: Rollback to last known good version
gcloud run services update-traffic aiprod-api \
  --to-revisions PREVIOUS=100 \
  --region=europe-west1 \
  --project=aiprod-484120
```

### Step 3: Root Cause Analysis

See specific section above for your diagnosis

### Step 4: Permanent Fix

Apply the fix from the appropriate section

### Step 5: Verify Recovery

```bash
# Monitor memory for 10 minutes
for i in {1..10}; do
  sleep 60
  MEMORY=$(gcloud monitoring time-series list \
    --filter='metric.type="run.googleapis.com/container_memory_utilizations"' \
    --project=aiprod-484120 \
    --format=json | jq '.[0].points[0].value.double_value * 100' 2>/dev/null)

  echo "T+${i}min - Memory: ${MEMORY}%"

  # Should be stable and below 80%
done

# Check error rate
gcloud logging read "severity=ERROR" \
  --project=aiprod-484120 \
  --limit=5
```

---

## Prevention

1. **Memory limits**: Set appropriate limits and test
2. **Caching policy**: Always use bounded caches
3. **Streaming**: Use streaming for large files
4. **Testing**: Load test with realistic data sizes
5. **Monitoring**: Monitor memory trends continuously

---

## Escalation

| Time     | Action                                      |
| -------- | ------------------------------------------- |
| T+0      | Attempt memory increase or rollback         |
| T+5 min  | If not recovering, page Backend Lead        |
| T+15 min | If issue persists, notify Service Lead      |
| T+30 min | If not resolved, notify Engineering Manager |

---

## Performance Benchmarks

```
Normal Memory Usage:
- Base container: ~256 MB
- Per concurrent request: ~50 MB
- With 80 concurrent requests: ~4 GB

Warning Levels:
- 70% of limit: Monitor closely
- 85% of limit: Alert triggered
- 95% of limit: Service degrading
- 100% of limit: OOM, restart imminent

With 4Gi allocation:
- Safe concurrent requests: 60-80
- Risk zone: >80 requests
- Critical: >95 requests
```

---

## Contacts

- **Backend Lead**: backend-lead@aiprod.ai
- **DevOps Lead**: devops-lead@aiprod.ai
- **Infrastructure Lead**: infra-lead@aiprod.ai
- **Slack Channel**: #incidents

---

## Related Documentation

- [Performance Optimization](../guides/performance.md)
- [Scaling Strategy](../monitoring/scaling.md)
- [Monitoring Setup](../monitoring/metrics.md)

---

**Last Tested**: Feb 4, 2026  
**Next Review**: Mar 4, 2026
