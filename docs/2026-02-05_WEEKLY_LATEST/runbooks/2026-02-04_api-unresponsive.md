# Runbook: API Unresponsive / Complete Outage

**Alert ID**: `alert-api-down`  
**Severity**: 🔴 CRITICAL  
**Oncall**: DevOps Lead + Service Lead  
**Last Updated**: Feb 4, 2026

---

## Quick Diagnosis (< 2 minutes)

### Is the API Down?

```bash
# Test API health
curl -s https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/health -w "\nStatus: %{http_code}\n"

# Expected: Status: 200 with {"status": "ok"}
# If timeout or error, API is down

# Quick status check
gcloud run services describe aiprod-api \
  --region=europe-west1 \
  --project=aiprod-484120 | grep -E "state|status"

# Expected: state=READY
```

### Check Service Status

```bash
# View recent error logs
gcloud logging read "resource.service.name=aiprod-api" \
  --project=aiprod-484120 \
  --limit=10 \
  --format="table(timestamp, severity, jsonPayload.message)"

# Check if service is deployed
gcloud run services list --project=aiprod-484120 | grep aiprod-api
```

---

## Root Causes (by likelihood)

### 1. Cloud Run Service Crashed

**Indicators**:

- Service exists but showing errors
- Recent restart activity in logs
- health endpoint timeout

**Diagnosis**:

```bash
# Check if service crashed
gcloud run revisions list \
  --service=aiprod-api \
  --region=europe-west1 \
  --project=aiprod-484120 \
  --format="table(name, status, revision.annotations.client-name)"

# Check most recent revision status
gcloud run revisions describe aiprod-api \
  --region=europe-west1 \
  --project=aiprod-484120 | grep -E "status|error"

# View error details
gcloud logging read "resource.service.name=aiprod-api AND severity=ERROR" \
  --project=aiprod-484120 \
  --limit=20
```

**Fix**: Restart the service

```bash
# Force redeploy current revision
gcloud run services update aiprod-api \
  --update-env-vars RESTART_FORCE=true \
  --region=europe-west1 \
  --project=aiprod-484120

# Or deploy fresh from current source
gcloud run deploy aiprod-api \
  --source=. \
  --region=europe-west1 \
  --project=aiprod-484120 \
  --allow-unauthenticated

# Monitor recovery
watch -n 5 "gcloud logging read 'resource.service.name=aiprod-api' --project=aiprod-484120 --limit=5"
```

---

### 2. Container Build Failed

**Indicators**:

- Recent deployment error in logs
- No new revisions created
- Build logs show errors

**Diagnosis**:

```bash
# Check recent builds
gcloud builds list --project=aiprod-484120 | head -10

# View build failure details
BUILD_ID=$(gcloud builds list --project=aiprod-484120 --format="value(ID)" --limit=1)
gcloud builds log $BUILD_ID --project=aiprod-484120 | tail -100

# Check Docker image
gcloud container images list --project=aiprod-484120
gcloud container images describe gcr.io/aiprod-484120/aiprod-api:latest \
  --project=aiprod-484120
```

**Fix**: Rebuild from previous working state

```bash
# Option 1: Check out previous working commit
git log --oneline -5
git checkout KNOWN_GOOD_COMMIT

# Rebuild
gcloud run deploy aiprod-api \
  --source=. \
  --region=europe-west1 \
  --project=aiprod-484120

# Option 2: Revert to last working image tag
gcloud run deploy aiprod-api \
  --image=gcr.io/aiprod-484120/aiprod-api:v1.2.3 \
  --region=europe-west1 \
  --project=aiprod-484120

# Verify build succeeded
gcloud run revisions list \
  --service=aiprod-api \
  --region=europe-west1 \
  --project=aiprod-484120 \
  --limit=1 \
  --format="table(name, status)"
```

---

### 3. Networking / Load Balancer Issue

**Indicators**:

- Service is up but not responding
- Cloud Run metrics show traffic but no responses
- DNS or routing issues

**Diagnosis**:

```bash
# Check if Cloud Run service is healthy
gcloud run services describe aiprod-api \
  --region=europe-west1 \
  --project=aiprod-484120 | grep -E "status|traffic"

# Test direct connection to Cloud Run
# (Without load balancer)
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  https://aiprod-api-hxhx3s6eya-ew.a.run.app/health

# Check DNS resolution
nslookup aiprod-v33-api-hxhx3s6eya-ew.a.run.app
# Should resolve to Cloud Run IP

# Test network connectivity
ping -c 1 aiprod-v33-api-hxhx3s6eya-ew.a.run.app
curl -v https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/health
```

**Fix**: Update DNS and routing

```bash
# Option 1: Update Cloud Run URL in DNS
# (If using custom domain)
gcloud run services describe aiprod-api \
  --region=europe-west1 \
  --project=aiprod-484120 | grep "serviceUrl"

# Create DNS CNAME record pointing to Cloud Run URL
gcloud dns record-sets update aiprod-v33-api.example.com. \
  --rrdatas=aiprod-api-hxhx3s6eya-ew.a.run.app. \
  --ttl=300 \
  --type=CNAME \
  --zone=example-zone

# Option 2: Verify ingress settings
gcloud run services describe aiprod-api \
  --region=europe-west1 \
  --project=aiprod-484120 | grep -A 5 "ingress"

# Should be "ALLOW_ALL" or whitelisted IPs
```

---

### 4. Resource Quota Exceeded

**Indicators**:

- Errors about "quota exceeded" or "resource limit"
- Build takes longer than usual
- Many deployments in past hour

**Diagnosis**:

```bash
# Check GCP quotas
gcloud compute project-info describe --project=aiprod-484120 | grep -A 5 "quota"

# Check Cloud Run service quotas
gcloud run regions list --format="table(name, memory_concurrency, instances)"

# Check Cloud Build quotas
gcloud builds describe BUILD_ID --project=aiprod-484120 | grep "quota"

# List resource usage
gcloud compute instances list --project=aiprod-484120
gcloud sql instances list --project=aiprod-484120
gcloud compute disks list --project=aiprod-484120
```

**Fix**: Request quota increase or clean up resources

```bash
# Option 1: Request quota increase (24-48 hour wait)
# https://cloud.google.com/docs/quota/change-quota

# Option 2: Clean up unused resources
# Stop/delete test instances
gcloud compute instances delete TEST_INSTANCE --project=aiprod-484120

# Delete old Cloud Run revisions (keep last 5)
for REV in $(gcloud run revisions list \
  --service=aiprod-api \
  --region=europe-west1 \
  --project=aiprod-484120 \
  --format="value(name)" | tail -n +6); do
  gcloud run revisions delete $REV \
    --region=europe-west1 \
    --project=aiprod-484120
done
```

---

### 5. Database Dependency Chain Failure

**Indicators**:

- Service starts but exits immediately
- Error about "failed to connect to database"
- All database-dependent requests failing

**Diagnosis**:

```bash
# Check if service can start without database
gcloud logging read "resource.service.name=aiprod-api AND severity=CRITICAL" \
  --project=aiprod-484120 \
  --limit=20 | grep -i "database\|connection"

# Verify database is available
gcloud sql connect aiprod-postgres \
  --user=postgres \
  --project=aiprod-484120 << EOF
SELECT 1;
EOF

# Check connection string in Cloud Run
gcloud run services describe aiprod-api \
  --region=europe-west1 \
  --project=aiprod-484120 | grep -A 5 "env_vars\|DATABASE"
```

**Fix**: Verify database connection

```bash
# Option 1: Check environment variables
gcloud run services describe aiprod-api \
  --region=europe-west1 \
  --project=aiprod-484120 \
  --format="value(spec.containers[0].env[*].[name,value])"

# Option 2: Update connection string if needed
gcloud run services update aiprod-api \
  --set-env-vars DATABASE_URL=postgresql://... \
  --region=europe-west1 \
  --project=aiprod-484120

# Option 3: If database unavailable, failover
gcloud sql instances failover aiprod-postgres \
  --project=aiprod-484120

# Restart service after database is online
gcloud run services update aiprod-api \
  --region=europe-west1 \
  --project=aiprod-484120
```

---

### 6. Secret or Credential Issues

**Indicators**:

- Authentication errors in logs
- Service can't access secrets/credentials
- Recent secret rotation

**Diagnosis**:

```bash
# Check secret access
gcloud secrets list --project=aiprod-484120 | grep aiprod

# Check service account permissions
gcloud iam service-accounts get-iam-policy \
  aiprod-api@aiprod-484120.iam.gserviceaccount.com \
  --project=aiprod-484120

# Check if secrets are accessible
gcloud secrets versions access latest --secret=aiprod-api-key --project=aiprod-484120

# Check secret permissions
gcloud secrets get-iam-policy aiprod-api-key --project=aiprod-484120
```

**Fix**: Restore credentials or permissions

```bash
# Option 1: Update missing secret
echo "NEW_SECRET_VALUE" | gcloud secrets create aiprod-api-key \
  --replication-policy="automatic" \
  --project=aiprod-484120

# Option 2: Grant service account access to secret
gcloud secrets add-iam-policy-binding aiprod-api-key \
  --member=serviceAccount:aiprod-api@aiprod-484120.iam.gserviceaccount.com \
  --role=roles/secretmanager.secretAccessor \
  --project=aiprod-484120

# Option 3: Restart service to reload secrets
gcloud run services update aiprod-api \
  --region=europe-west1 \
  --project=aiprod-484120
```

---

## Emergency Recovery Steps (In Order)

### Step 1: Declare Incident (T+0)

```bash
# Log incident start time
INCIDENT_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "Incident started at: $INCIDENT_TIME"

# Alert team
# - Slack: #incidents
# - Page: on-call DevOps + Service Lead
# - Email: tech-leads@aiprod.ai
```

### Step 2: Assess Impact (T+1 min)

```bash
# Check customer impact
gcloud logging read "httpRequest.status>=500" \
  --project=aiprod-484120 \
  --limit=100 | jq 'length'

# Get baseline for comparison
# (If >100 errors = widespread outage)

# Estimate affected users
gcloud logging read "jsonPayload.user_id!=null AND httpRequest.status>=500" \
  --project=aiprod-484120 \
  --limit=1000 \
  --format=json | jq -r '.[] | .jsonPayload.user_id' | sort -u | wc -l
```

### Step 3: Attempt Quick Recovery (T+2 min)

```bash
# Try restart first (safest)
gcloud run services update aiprod-api \
  --region=europe-west1 \
  --project=aiprod-484120

# Wait 30 seconds
sleep 30

# Test if recovered
curl -s https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/health -w "\nStatus: %{http_code}\n"
```

### Step 4: If Still Down, Diagnose (T+5 min)

- Run diagnostics from sections above
- Identify root cause
- Apply targeted fix

### Step 5: Verify Recovery (T+10 min)

```bash
# Check error rate dropping
gcloud logging read "httpRequest.status>=500" \
  --project=aiprod-484120 \
  --limit=100 --format=json | jq 'length'

# Should be significantly lower than step 2

# Verify service responding to normal requests
curl https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/health

# Monitor for 5 minutes
for i in {1..5}; do
  sleep 60
  curl -s https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/health && echo "✓ OK" || echo "✗ FAIL"
done
```

### Step 6: Communicate Status (Ongoing)

```bash
# Every 5 minutes during outage
gcloud logging read "jsonPayload" \
  --project=aiprod-484120 \
  --limit=5 \
  --format="table(timestamp, jsonPayload.message)"

# Update status page / customer communication
# Post to Slack #incidents every 5-10 min
```

---

## Post-Outage Actions (Within 24 hours)

```bash
# 1. Gather timeline and logs
gcloud logging read "resource.service.name=aiprod-api" \
  --start-time=INCIDENT_START \
  --end-time=INCIDENT_END \
  --project=aiprod-484120 \
  > incident_logs.json

# 2. Calculate impact metrics
ERROR_COUNT=$(jq '[.[] | select(.httpRequest.status >= 500)] | length' incident_logs.json)
DOWNTIME_MINUTES=$(( $(date -d "$INCIDENT_END" +%s) - $(date -d "$INCIDENT_START" +%s) / 60 ))

# 3. Schedule post-mortem (within 48 hours)
# Include: what happened, why, how we fix it, prevention

# 4. Update runbooks with learnings
# 5. Implement preventive measures
```

---

## Prevention

1. **Monitoring**: Continuous uptime monitoring
2. **Redundancy**: Multi-region deployments
3. **Health checks**: Regular automated health tests
4. **Disaster recovery**: Test failover monthly
5. **Change control**: All changes deployed to staging first

---

## Escalation Matrix

| Time | Severity    | Action                             |
| ---- | ----------- | ---------------------------------- |
| T+0  | 🔴 CRITICAL | Page on-call DevOps + Service Lead |
| T+5  | 🔴 CRITICAL | Page VP Engineering                |
| T+10 | 🔴 CRITICAL | Page CEO                           |
| T+30 | 🔴 CRITICAL | Contact GCP Support Premium        |

---

## Critical Commands Reference

```bash
# Diagnose
gcloud run services describe aiprod-api --region=europe-west1 --project=aiprod-484120
gcloud logging read "resource.service.name=aiprod-api" --project=aiprod-484120 --limit=50

# Fix (attempt in order)
gcloud run services update aiprod-api --region=europe-west1 --project=aiprod-484120  # Restart
gcloud run deploy aiprod-api --source=. --region=europe-west1 --project=aiprod-484120  # Redeploy
gcloud sql instances failover aiprod-postgres --project=aiprod-484120  # DB failover

# Verify
curl https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/health
gcloud logging read "httpRequest.status>=500" --project=aiprod-484120 --limit=5
```

---

## Contacts

- **DevOps Lead**: devops-lead@aiprod.ai (IMMEDIATE)
- **Service Lead**: service-lead@aiprod.ai (IMMEDIATE)
- **VP Engineering**: vp-eng@aiprod.ai (T+5 min)
- **GCP Support**: https://cloud.google.com/support/premium (T+30 min)
- **Slack Channel**: #incidents (Real-time updates)

---

## Related Documentation

- [Disaster Recovery Guide](disaster-recovery.md)
- [Deployment Procedures](../guides/deployment.md)
- [Monitoring & Alerting](../monitoring/alerts.md)

---

**Last Tested**: Feb 4, 2026  
**Next Review**: Mar 4, 2026  
**Severity**: 🔴 CRITICAL — Requires monthly drill testing
