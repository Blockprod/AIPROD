# Runbook: High Error Rate (>1%)

**Alert ID**: `alert-error-rate`  
**Severity**: 🔴 CRITICAL  
**Oncall**: DevOps Lead  
**Last Updated**: Feb 4, 2026

---

## Quick Diagnosis

### Symptoms

- API returning 5xx errors
- Error rate > 1% (threshold alert triggered)
- Alert fired in Cloud Monitoring dashboard
- User reports seeing "Server Error" responses

### Check Current Error Rate

```bash
# View error rate in Cloud Logging
gcloud logging read "severity=ERROR AND resource.type=cloud_run_revision" \
  --project=aiprod-484120 \
  --limit=50 \
  --format=json | jq '.[] | .jsonPayload'

# Check recent error counts
gcloud logging read "httpRequest.status>=500 AND resource.service.name=aiprod-api" \
  --project=aiprod-484120 \
  --limit=100
```

### Get Error Details

```bash
# Get most recent errors
gcloud logging read "severity=ERROR" \
  --project=aiprod-484120 \
  --limit=20 \
  --format="table(timestamp, jsonPayload.message, jsonPayload.error_type)"
```

---

## Root Causes Analysis

### 1. Database Connection Error

**Indicators**:

- Logs show "connection refused" or "timeout"
- Error pattern: consistent across all requests
- Database is responding slowly

**Diagnosis**:

```bash
# Check database status
gcloud sql instances describe aiprod-postgres \
  --project=aiprod-484120 | grep -A 5 "state\|status"

# Check database availability
gcloud sql instances list --project=aiprod-484120

# If status is RUNNABLE, test connection
gcloud sql connect aiprod-postgres \
  --user=postgres \
  --project=aiprod-484120 << EOF
SELECT 1;
EOF
```

**Fix**: If database is down, failover to replica

```bash
# List available replicas
gcloud sql instances list --project=aiprod-484120 | grep replica

# Initiate failover
gcloud sql instances failover aiprod-postgres \
  --project=aiprod-484120

# Verify failover completed
gcloud sql instances describe aiprod-postgres --project=aiprod-484120
```

**Escalation**: If failover fails, contact DBA team

---

### 2. Out of Memory

**Indicators**:

- Logs show "OOM" or "memory" errors
- Error rate spikes suddenly
- Response times increase dramatically

**Diagnosis**:

```bash
# Check memory usage (Cloud Run)
gcloud monitoring time-series list \
  --filter='metric.type="run.googleapis.com/container_memory_utilizations" AND resource.service_name="aiprod-api"' \
  --project=aiprod-484120

# Check recent revisions
gcloud run revisions list \
  --service=aiprod-api \
  --region=europe-west1 \
  --project=aiprod-484120 | head -5
```

**Fix**: Restart the service with traffic drain

```bash
# Step 1: Stop sending traffic to current revision
gcloud run services update-traffic aiprod-api \
  --to-revisions LATEST=0 \
  --region=europe-west1 \
  --project=aiprod-484120

# Step 2: Wait for in-flight requests to complete
sleep 120

# Step 3: Resume traffic
gcloud run services update-traffic aiprod-api \
  --to-revisions LATEST=100 \
  --region=europe-west1 \
  --project=aiprod-484120

# Step 4: Verify error rate dropping
gcloud logging read "httpRequest.status>=500" \
  --project=aiprod-484120 \
  --limit=10
```

**Prevention**: Increase memory limit if this is recurring

```bash
gcloud run services update aiprod-api \
  --memory=4Gi \
  --region=europe-west1 \
  --project=aiprod-484120
```

---

### 3. Code Deployment Issue

**Indicators**:

- Error rate started after recent deployment
- Specific endpoint returning errors (not all)
- Error messages reference new code path

**Diagnosis**:

```bash
# Check recent deployments
gcloud run revisions list \
  --service=aiprod-api \
  --region=europe-west1 \
  --project=aiprod-484120

# Check revision details
gcloud run revisions describe SERVICE_REVISION \
  --region=europe-west1 \
  --project=aiprod-484120

# Check deployment timestamp
gcloud logging read "resource.service.name=aiprod-api" \
  --project=aiprod-484120 \
  --limit=1 \
  --format="table(timestamp, jsonPayload.message)"
```

**Fix**: Rollback to previous version

```bash
# Get previous revision name
PREVIOUS_REV=$(gcloud run revisions list \
  --service=aiprod-api \
  --region=europe-west1 \
  --project=aiprod-484120 \
  --sort-by="~revision.name" \
  --limit=2 \
  --format="value(name)" | tail -1)

# Route traffic to previous revision
gcloud run services update-traffic aiprod-api \
  --to-revisions $PREVIOUS_REV=100 \
  --region=europe-west1 \
  --project=aiprod-484120

# Verify error rate dropping
sleep 30
gcloud logging read "httpRequest.status>=500" \
  --project=aiprod-484120 \
  --limit=10
```

**Prevention**: Test before production deployment

---

### 4. Third-Party Service Failure

**Indicators**:

- Errors show "external service" or "upstream" in message
- Only specific job types failing (e.g., video rendering jobs)
- Error rate correlates with external service status

**Diagnosis**:

```bash
# Check which external services are called
gcloud logging read "jsonPayload.external_service" \
  --project=aiprod-484120 \
  --limit=20

# Check for specific service errors
gcloud logging read "severity=ERROR AND jsonPayload.service=runway" \
  --project=aiprod-484120 \
  --limit=10

# Check Runway health status
curl -s https://status.runwayml.com/api/v2/status.json | jq '.status'
```

**Fix**: Implement fallback logic or wait for service recovery

```bash
# Check if fallback is already routing jobs
gcloud logging read "jsonPayload.backend_selected" \
  --project=aiprod-484120 \
  --limit=20 | jq '.[] | .jsonPayload | {backend: .backend_selected, status: .status}'

# If fallback not working, check error logs
gcloud logging read "jsonPayload.error_type=BACKEND_UNAVAILABLE" \
  --project=aiprod-484120 \
  --limit=5
```

**Escalation**: Notify affected customers, provide ETA for recovery

---

## Recovery Steps (In Order)

### Step 1: Assess Severity

```bash
# Get error rate percentage
ERROR_COUNT=$(gcloud logging read "httpRequest.status>=500" \
  --project=aiprod-484120 \
  --limit=1000 --format=json | jq length)

TOTAL_COUNT=$(gcloud logging read "httpRequest.status!=null" \
  --project=aiprod-484120 \
  --limit=1000 --format=json | jq length)

ERROR_RATE=$(echo "scale=2; $ERROR_COUNT * 100 / $TOTAL_COUNT" | bc)
echo "Current error rate: ${ERROR_RATE}%"
```

### Step 2: Identify Root Cause

- Check database status
- Check memory usage
- Check recent deployments
- Check external service status

### Step 3: Apply Fix

- Failover database if needed
- Restart service if OOM
- Rollback deployment if code issue
- Wait for external service recovery

### Step 4: Verify Recovery

```bash
# Monitor error rate for 5 minutes
for i in {1..5}; do
  sleep 60
  ERROR_RATE=$(gcloud logging read "httpRequest.status>=500" --project=aiprod-484120 --limit=1000 --format=json | jq 'length')
  echo "Error count at T+${i}min: $ERROR_RATE"
done

# Error rate should be <0.1% (under control)
```

### Step 5: Post-Incident

- Document root cause
- Update runbook if needed
- Create incident ticket
- Schedule post-mortem if >30min downtime

---

## Prevention

1. **Monitor continuously**: Check dashboard every 15 minutes
2. **Test deployments**: Always test in staging first
3. **Maintain database**: Regular health checks and backups
4. **Capacity planning**: Monitor memory and CPU trends
5. **Circuit breakers**: Implement for external service calls

---

## Escalation

| Time     | Action                                      |
| -------- | ------------------------------------------- |
| T+0      | Page on-call DevOps                         |
| T+5 min  | If not resolved, notify Service Lead        |
| T+15 min | If not resolved, notify Engineering Manager |
| T+30 min | If not resolved, notify VP Engineering      |
| T+60 min | If not resolved, notify CEO and customers   |

---

## Contacts

- **DevOps Lead**: devops-lead@aiprod.ai
- **Service Lead**: service-lead@aiprod.ai
- **DBA**: dba@aiprod.ai
- **VP Engineering**: vp-eng@aiprod.ai
- **Slack Channel**: #incidents

---

## Related Documentation

- [Disaster Recovery Guide](disaster-recovery.md)
- [Database Runbook](database-errors.md)
- [SLA Documentation](../business/sla-details.md)

---

**Last Tested**: Feb 4, 2026  
**Next Review**: Mar 4, 2026
