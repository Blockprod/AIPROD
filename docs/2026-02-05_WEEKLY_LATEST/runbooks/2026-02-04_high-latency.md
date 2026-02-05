# Runbook: High Latency (P95 > 900s)

**Alert ID**: `alert-latency-p95`  
**Severity**: 🟡 HIGH  
**Oncall**: Backend Lead  
**Last Updated**: Feb 4, 2026

---

## Quick Diagnosis

### Symptoms

- API response times > 900 seconds (15 minutes)
- Alert triggered for P95 latency
- Jobs completing slower than normal
- Customers reporting slow responses

### Check Current Latency

```bash
# View latency metrics
gcloud monitoring time-series list \
  --filter='metric.type="custom.googleapis.com/pipeline/processing_time_seconds" AND resource.service_name="aiprod-api"' \
  --project=aiprod-484120 \
  --format=json | jq '.[] | .points | .[0]'

# Check recent job processing times
gcloud logging read "jsonPayload.processing_time_seconds!=null" \
  --project=aiprod-484120 \
  --limit=20 \
  --format="table(timestamp, jsonPayload.job_id, jsonPayload.processing_time_seconds)"
```

### Get Latency Details by Endpoint

```bash
# Check which endpoints are slow
gcloud logging read "httpRequest.latency!=null" \
  --project=aiprod-484120 \
  --limit=50 \
  --format="table(timestamp, httpRequest.requestUrl, httpRequest.latency)"
```

---

## Root Causes Analysis

### 1. Slow Database Queries

**Indicators**:

- Latency increases gradually
- Database query times slow
- Index usage is low

**Diagnosis**:

```bash
# Connect to database and check slow queries
gcloud sql connect aiprod-postgres \
  --user=postgres \
  --project=aiprod-484120 << EOF

-- Find slow queries
SELECT query, mean_exec_time, calls
FROM pg_stat_statements
WHERE mean_exec_time > 1000  -- > 1 second
ORDER BY mean_exec_time DESC
LIMIT 10;

-- Check index usage
SELECT schemaname, tablename, indexname, idx_scan
FROM pg_stat_user_indexes
WHERE idx_scan < 100
ORDER BY idx_scan
LIMIT 10;

-- Check table sizes
SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

EOF
```

**Fix**: Optimize queries or add indexes

```bash
# Add missing index (if identified)
gcloud sql connect aiprod-postgres \
  --user=postgres \
  --project=aiprod-484120 << EOF

CREATE INDEX CONCURRENTLY idx_jobs_created_at ON jobs(created_at DESC);

-- Verify index was created
SELECT * FROM pg_stat_user_indexes WHERE indexname = 'idx_jobs_created_at';

EOF
```

---

### 2. Overloaded Backend Service

**Indicators**:

- Latency increases with traffic volume
- CPU/memory at high levels
- Queue depth increasing

**Diagnosis**:

```bash
# Check CPU usage
gcloud monitoring time-series list \
  --filter='metric.type="run.googleapis.com/container_cpu_utilizations" AND resource.service_name="aiprod-api"' \
  --project=aiprod-484120

# Check memory usage
gcloud monitoring time-series list \
  --filter='metric.type="run.googleapis.com/container_memory_utilizations" AND resource.service_name="aiprod-api"' \
  --project=aiprod-484120

# Check request concurrency
gcloud logging read "jsonPayload.concurrent_requests" \
  --project=aiprod-484120 \
  --limit=20 \
  --format="table(timestamp, jsonPayload.concurrent_requests)"
```

**Fix**: Scale up or optimize resource usage

```bash
# Option 1: Increase Cloud Run min/max instances
gcloud run services update aiprod-api \
  --min-instances=5 \
  --max-instances=20 \
  --region=europe-west1 \
  --project=aiprod-484120

# Option 2: Increase CPU/Memory per instance
gcloud run services update aiprod-api \
  --cpu=4 \
  --memory=4Gi \
  --region=europe-west1 \
  --project=aiprod-484120

# Option 3: Optimize code (profile to find bottlenecks)
python -m cProfile -s cumtime src/api/main.py
```

---

### 3. External Service Bottleneck

**Indicators**:

- Latency correlates with specific operations (e.g., video rendering)
- External service (Runway, Veo-3) is slow
- Fallback logic not kicking in

**Diagnosis**:

```bash
# Check rendering latency by backend
gcloud logging read "jsonPayload.backend!='null'" \
  --project=aiprod-484120 \
  --limit=50 \
  --format="table(timestamp, jsonPayload.backend, jsonPayload.rendering_time_seconds)"

# Check specific backend health
gcloud logging read "jsonPayload.service=runway" \
  --project=aiprod-484120 \
  --limit=20 | jq '.[] | {timestamp: .timestamp, status: .jsonPayload.status, latency: .jsonPayload.latency}'

# Check fallback activation
gcloud logging read "jsonPayload.fallback_triggered=true" \
  --project=aiprod-484120 \
  --limit=10
```

**Fix**: Optimize or enable more aggressive fallback

```bash
# Check fallback thresholds in code
grep -r "FALLBACK_THRESHOLD\|MAX_WAIT_TIME" src/

# If Runway is slow, route to faster backend
# Edit src/utils/backend_selector.py to prefer Veo-3

# Verify change took effect after restart
gcloud run services update aiprod-api \
  --region=europe-west1 \
  --project=aiprod-484120

# Monitor latency improvement
gcloud logging read "jsonPayload.backend=veo" \
  --project=aiprod-484120 \
  --limit=20 \
  --format="table(timestamp, jsonPayload.rendering_time_seconds)"
```

---

### 4. Network or Infrastructure Issues

**Indicators**:

- Latency to external APIs high
- Network throughput reduced
- Geographic specific issues

**Diagnosis**:

```bash
# Check network latency to external services
gcloud logging read "jsonPayload.network_latency_ms!=null" \
  --project=aiprod-484120 \
  --limit=20

# Check packet loss
gcloud compute networks list --project=aiprod-484120

# Ping external service
curl -w "DNS: %{time_namelookup}, Connect: %{time_connect}, Total: %{time_total}\n" \
  -o /dev/null -s https://api.runwayml.com/health
```

**Fix**: Optimize network configuration

```bash
# Check VPC configuration
gcloud compute networks describe default --project=aiprod-484120

# Verify Cloud NAT is working (if using private IP)
gcloud compute routers list --project=aiprod-484120
```

---

## Recovery Steps

### Step 1: Identify Bottleneck

```bash
# Check which component is slow
gcloud logging read "jsonPayload" \
  --project=aiprod-484120 \
  --limit=10 \
  --format=json | jq '.[] | .jsonPayload | {
    request_latency: .httpRequest.latency,
    db_time: .db_time_ms,
    backend_time: .backend_time_seconds,
    network_time: .network_latency_ms
  }'
```

### Step 2: Apply Targeted Fix

- If database: add index or optimize query
- If backend: scale up or enable fallback
- If network: check infrastructure or service health

### Step 3: Verify Improvement

```bash
# Monitor latency for 5 minutes
for i in {1..5}; do
  sleep 60
  P95=$(gcloud logging read "httpRequest.latency!=null" \
    --project=aiprod-484120 \
    --limit=100 \
    --format=json | jq '[.[] | .httpRequest.latency | tonumber] | sort | .[length*95/100]')
  echo "P95 at T+${i}min: ${P95}ms"
done

# P95 should be <900 seconds (under alert threshold)
```

### Step 4: Capacity Planning

- Analyze trend: is latency increasing over time?
- Plan for growth: add more resources if needed
- Set up auto-scaling

---

## Prevention

1. **Monitor continuously**: Set up latency dashboard
2. **Profile regularly**: Find bottlenecks before they impact users
3. **Test at scale**: Load test with realistic traffic patterns
4. **Maintain indexes**: Regularly review database index usage
5. **Backend diversification**: Keep fallback backends healthy

---

## Escalation

| Time     | Action                                          |
| -------- | ----------------------------------------------- |
| T+0      | Page on-call Backend Lead                       |
| T+10 min | If not improving, check external service status |
| T+20 min | If issue persists, notify Service Lead          |
| T+30 min | If not resolved, notify Engineering Manager     |
| T+60 min | If not resolved, notify VP Engineering          |

---

## Contacts

- **Backend Lead**: backend-lead@aiprod.ai
- **Database Lead**: dba@aiprod.ai
- **Infrastructure Lead**: infra-lead@aiprod.ai
- **Slack Channel**: #incidents

---

## Related Documentation

- [Database Errors Runbook](database-errors.md)
- [Performance Tips](../guides/performance.md)
- [Monitoring Setup](../monitoring/metrics.md)

---

**Last Tested**: Feb 4, 2026  
**Next Review**: Mar 4, 2026
