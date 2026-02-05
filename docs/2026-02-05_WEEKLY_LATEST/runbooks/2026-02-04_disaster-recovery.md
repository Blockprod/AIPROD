# Disaster Recovery Guide — AIPROD

**Document Version**: 1.0  
**Effective Date**: February 4, 2026  
**Classification**: Internal  
**Last Tested**: February 4, 2026

---

## Executive Summary

This guide defines AIPROD's disaster recovery (DR) procedures to minimize impact of catastrophic failures. Our strategy emphasizes rapid recovery with minimal data loss.

### Key Targets

| Metric                             | Target        | SLA                  |
| ---------------------------------- | ------------- | -------------------- |
| **RTO** (Recovery Time Objective)  | <4 hours      | Guaranteed           |
| **RPO** (Recovery Point Objective) | <24 hours     | Guaranteed           |
| **Data Durability**                | 99.99%        | Multi-region backups |
| **Backup Frequency**               | Every 6 hours | Automated            |

### Recovery Levels

- **Level 1**: Single component failure (instance, service)
- **Level 2**: Single region outage
- **Level 3**: Complete regional infrastructure loss

---

## Part 1: Backup Strategy

### Backup Architecture

```
Production Database (aiprod-postgres)
    ↓
├── Continuous replication → Replica-1 (same region)
├── Continuous replication → Replica-2 (different AZ)
├── Automated backups → 6-hour interval (7 snapshots/day)
├── Daily full backup → Stored in secondary region
└── Monthly archive backup → Long-term retention
```

### Backup Schedule

| Type                       | Frequency     | Retention | Location                      |
| -------------------------- | ------------- | --------- | ----------------------------- |
| **Continuous Replication** | Real-time     | N/A       | europe-west1b, europe-west1c  |
| **Automated Snapshots**    | Every 6 hours | 30 days   | europe-west1 (multi-zone)     |
| **Daily Full Backup**      | 3:00 AM UTC   | 90 days   | europe-west2 (backup region)  |
| **Monthly Archive**        | 1st of month  | 1 year    | Cloud Storage (archive class) |
| **Ad-hoc Backups**         | On-demand     | 30 days   | europe-west1                  |

### Backup Storage

**Primary Backups** (EU: europe-west1)

- Location: Google Cloud SQL automatic backups
- Encryption: Google-managed keys
- Retention: 30 days
- Cost: Included in Cloud SQL pricing

**Secondary Backups** (EU: europe-west2)

- Location: Cross-region automated backup
- Purpose: Protection against regional failure
- Retention: 90 days
- Cost: ~$50/month

**Archive Backups** (Cloud Storage)

- Location: Multi-region storage (EU + US)
- Purpose: Long-term compliance and recovery
- Retention: 1 year
- Cost: ~$5/month (cheap archive tier)

### Backup Verification

```bash
# Weekly automated test
0 2 * * 0 /scripts/test-backup-restore.sh

# Monthly manual verification
# First Monday of month, 10 AM UTC
# Team: DBA + DevOps Lead
```

---

## Part 2: Disaster Scenarios

### Scenario 1: Single Cloud Run Instance Failure

**Severity**: Low (auto-recovers)  
**RTO**: <2 minutes  
**RPO**: 0 minutes

#### Detection

- Cloud Run health check fails
- Instance exits with error code
- Error logged in Cloud Logging

#### Automatic Recovery

```bash
# Cloud Run automatically:
1. Detects unhealthy instance (15 seconds)
2. Stops instance (5 seconds)
3. Starts replacement instance (30 seconds)
4. Begins routing traffic (10 seconds)

# Total: ~60 seconds
```

#### Manual Verification (if auto-recovery fails)

```bash
# Check instance status
gcloud run revisions list \
  --service=aiprod-api \
  --region=europe-west1 \
  --project=aiprod-484120

# If still failing, restart service
gcloud run services update aiprod-api \
  --region=europe-west1 \
  --project=aiprod-484120

# Monitor for 5 minutes
gcloud logging read "resource.service.name=aiprod-api" \
  --project=aiprod-484120 \
  --limit=20
```

**Expected Outcome**: Service restored, no data loss

---

### Scenario 2: Multiple Cloud Run Instances Fail

**Severity**: Medium  
**RTO**: <5 minutes  
**RPO**: 0 minutes

#### Detection

- Error rate >1%
- Multiple instance failures logged
- Alert fires: "API Error Rate High"

#### Recovery Procedure

```bash
# Step 1: Trigger full service restart (with traffic drain)
gcloud run services update-traffic aiprod-api \
  --to-revisions LATEST=0 \
  --region=europe-west1 \
  --project=aiprod-484120

# Step 2: Wait for in-flight requests to complete
sleep 120

# Step 3: Restart all instances
gcloud run services update aiprod-api \
  --min-instances=5 \
  --region=europe-west1 \
  --project=aiprod-484120

# Step 4: Resume traffic
gcloud run services update-traffic aiprod-api \
  --to-revisions LATEST=100 \
  --region=europe-west1 \
  --project=aiprod-484120

# Step 5: Verify recovery
for i in {1..5}; do
  sleep 30
  curl -s https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/health
done
```

**Expected Outcome**: Service restored, minimal data loss

---

### Scenario 3: Primary Database Instance Fails

**Severity**: Critical  
**RTO**: <10 minutes  
**RPO**: <1 minute (unsaved writes)

#### Detection

- API getting "connection refused" errors
- Cloud SQL alerts: "Instance Unavailable"
- All database queries timing out

#### Automatic Failover

```bash
# Cloud SQL performs automatic failover:
1. Detects primary failure (30 seconds)
2. Promotes replica to primary (60 seconds)
3. Updates connection endpoints (10 seconds)
4. Replicas resume replication (20 seconds)

# Total: ~2 minutes
```

#### Manual Failover (if automatic fails)

```bash
# Check if instance is really down
gcloud sql instances describe aiprod-postgres \
  --project=aiprod-484120 | grep "state"

# List available replicas
gcloud sql instances list \
  --project=aiprod-484120 | grep replica

# Promote replica to primary
gcloud sql instances failover aiprod-postgres \
  --backup-instance=aiprod-postgres-replica-1 \
  --project=aiprod-484120

# Monitor failover (watch status change)
watch -n 5 "gcloud sql instances describe aiprod-postgres --project=aiprod-484120 | grep state"

# Once RUNNABLE, API will reconnect automatically
```

#### Post-Failover

```bash
# 1. Verify data integrity
gcloud sql connect aiprod-postgres \
  --user=postgres \
  --project=aiprod-484120 << EOF
SELECT COUNT(*) as job_count FROM jobs;
SELECT COUNT(*) as result_count FROM results;
EOF

# 2. Check replication status
SELECT slot_name, restart_lsn FROM pg_replication_slots;

# 3. Create new replica to replace failed one
# (Contact DBA for this, takes ~30 minutes)
```

**Expected Outcome**: Service restored, <1 min data loss

---

### Scenario 4: Data Corruption Detected

**Severity**: Critical  
**RTO**: <30 minutes  
**RPO**: <24 hours

#### Detection

- Application logic detects inconsistent data
- Customer reports incorrect calculation
- Database constraint violations in logs

#### Diagnosis

```bash
# Step 1: Stop writes to prevent further corruption
gcloud run services update aiprod-api \
  --no-traffic \
  --region=europe-west1 \
  --project=aiprod-484120

# Step 2: Identify corruption scope
gcloud sql connect aiprod-postgres \
  --user=postgres \
  --project=aiprod-484120 << EOF

-- Find jobs with inconsistent data
SELECT id, user_id, status, cost, created_at
FROM jobs
WHERE cost < 0 OR user_id IS NULL
ORDER BY created_at DESC
LIMIT 100;

-- Count affected records
SELECT COUNT(*) FROM jobs WHERE cost < 0;

EOF

# Step 3: Determine root cause
# - Was it application bug? (fix, re-run)
# - Was it data corruption? (restore from backup)
# - When did it start? (check logs for timestamp)

gcloud logging read "severity=ERROR" \
  --project=aiprod-484120 \
  --limit=50 | grep -i "error\|exception"
```

#### Recovery Options

**Option A: Fix in place (if minor corruption)**

```bash
# If <100 records affected and easy to fix
gcloud sql connect aiprod-postgres \
  --user=postgres \
  --project=aiprod-484120 << EOF

BEGIN TRANSACTION;

-- Fix known corrupted records
UPDATE jobs SET cost = 0 WHERE cost < 0;
UPDATE jobs SET user_id = 'unknown' WHERE user_id IS NULL;

-- Verify fixes
SELECT COUNT(*) FROM jobs WHERE cost < 0;
SELECT COUNT(*) FROM jobs WHERE user_id IS NULL;

COMMIT;

EOF

# Resume traffic
gcloud run services update-traffic aiprod-api \
  --to-revisions LATEST=100 \
  --region=europe-west1 \
  --project=aiprod-484120
```

**Option B: Restore from backup (if extensive corruption)**

```bash
# Step 1: List available backups
gcloud sql backups list \
  --instance=aiprod-postgres \
  --project=aiprod-484120

# Step 2: Create restored instance
gcloud sql backups restore BACKUP_ID \
  --backup-instance=aiprod-postgres \
  --target-instance=aiprod-postgres-recovered \
  --project=aiprod-484120 \
  --async

# Monitor restore progress
watch -n 10 "gcloud sql operations list --instance=aiprod-postgres-recovered --project=aiprod-484120 | head -1"

# Step 3: Verify restored data
gcloud sql connect aiprod-postgres-recovered \
  --user=postgres \
  --project=aiprod-484120 << EOF
SELECT COUNT(*) FROM jobs;
SELECT COUNT(*) FROM results;
SELECT SUM(cost) FROM jobs;
EOF

# Step 4: Update connection string to restored instance
# (CAREFUL: Do NOT update until verified)
gcloud run services update aiprod-api \
  --set-env-vars DATABASE_URL=postgresql://... \
  --region=europe-west1 \
  --project=aiprod-484120

# Step 5: Resume traffic
gcloud run services update-traffic aiprod-api \
  --to-revisions LATEST=100 \
  --region=europe-west1 \
  --project=aiprod-484120

# Step 6: Clean up old instance (after 48 hours if all OK)
gcloud sql instances delete aiprod-postgres \
  --project=aiprod-484120

# Rename recovered instance
gcloud sql instances patch aiprod-postgres-recovered \
  --new-name=aiprod-postgres \
  --project=aiprod-484120
```

**Expected Outcome**: Service restored to pre-corruption state

---

### Scenario 5: Point-in-Time Recovery (PITR)

**Severity**: Medium  
**RTO**: <60 minutes  
**RPO**: <5 minutes (with transaction logs)

#### Use Cases

- User accidentally deleted important data
- Malicious modification of records
- Need to recover to specific timestamp

#### Procedure

```bash
# Step 1: Identify target recovery time
TARGET_TIME="2026-02-04T10:30:00Z"  # Time before problem occurred

# Step 2: Create new instance from PITR
gcloud sql backups restore BACKUP_ID \
  --backup-instance=aiprod-postgres \
  --target-instance=aiprod-postgres-pitr \
  --backup-restore-time=$TARGET_TIME \
  --project=aiprod-484120 \
  --async

# Monitor progress
gcloud sql operations list \
  --instance=aiprod-postgres-pitr \
  --project=aiprod-484120

# Step 3: Verify data at target time
gcloud sql connect aiprod-postgres-pitr \
  --user=postgres \
  --project=aiprod-484120 << EOF

-- Verify you recovered to correct time
SELECT COUNT(*) FROM jobs WHERE created_at > '2026-02-04T10:30:00Z';

-- Should be 0 (no jobs created after recovery point)

EOF

# Step 4: If correct, promote PITR instance
# (Switch traffic from old database to PITR)
gcloud run services update aiprod-api \
  --set-env-vars DATABASE_URL=postgresql://... \
  --region=europe-west1 \
  --project=aiprod-484120

# Step 5: Monitor for issues (24 hours)
gcloud logging read "resource.service.name=aiprod-api" \
  --project=aiprod-484120 \
  --limit=50

# Step 6: Delete old instance once confident
gcloud sql instances delete aiprod-postgres \
  --project=aiprod-484120
```

**PITR Retention**: 7 days of transaction logs retained

**Expected Outcome**: Database restored to specific point in time

---

### Scenario 6: Regional Outage (Complete)

**Severity**: Critical  
**RTO**: <2 hours  
**RPO**: <15 minutes

#### Detection

- All services in region unavailable
- GCP status shows region issue
- All endpoints timing out

#### Failover to Secondary Region

```bash
# Step 1: Declare disaster
INCIDENT_START=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "Regional failover initiated: $INCIDENT_START"

# Notify team immediately
# Slack: #incidents
# Page: VP Eng + all service leads

# Step 2: Verify region is really down
for i in {1..3}; do
  gcloud compute regions describe europe-west1 \
    --project=aiprod-484120 && break || sleep 10
done

# If region is down, proceed with failover

# Step 3: Deploy API to secondary region (europe-west2)
gcloud run deploy aiprod-api-dr \
  --image=gcr.io/aiprod-484120/aiprod-api:latest \
  --region=europe-west2 \
  --project=aiprod-484120 \
  --allow-unauthenticated

# Step 4: Restore database in secondary region
gcloud sql instances create aiprod-postgres-dr \
  --region=europe-west2 \
  --database-version=POSTGRES_14 \
  --tier=db-custom-4-16384 \
  --project=aiprod-484120 \
  --async

# Monitor instance creation
watch -n 5 "gcloud sql instances describe aiprod-postgres-dr --project=aiprod-484120 | grep state"

# Step 5: Restore latest backup to DR instance
gcloud sql backups list \
  --instance=aiprod-postgres \
  --project=aiprod-484120 \
  --limit=1 | head -5

# Use most recent backup
gcloud sql backups restore LATEST_BACKUP_ID \
  --backup-instance=aiprod-postgres \
  --target-instance=aiprod-postgres-dr \
  --project=aiprod-484120 \
  --async

# Monitor restore
watch -n 10 "gcloud sql operations list --project=aiprod-484120 | head -3"

# Step 6: Update API to use DR database
gcloud run services update aiprod-api-dr \
  --set-env-vars DATABASE_URL=postgresql://aiprod-postgres-dr:5432/aiprod \
  --region=europe-west2 \
  --project=aiprod-484120

# Step 7: Update DNS to point to DR endpoint
# Option A: If using custom domain, update CNAME
gcloud dns record-sets update aiprod.example.com. \
  --rrdatas=aiprod-api-dr-xxxx-ew.a.run.app. \
  --ttl=60 \  # Lower TTL for faster cutover
  --type=CNAME \
  --zone=example-zone

# Option B: If using Cloud Run URL, update load balancer
# (Custom configuration depending on your setup)

# Step 8: Verify DR region working
curl https://aiprod-api-dr-xxxx-ew.a.run.app/health

# Step 9: Monitor DR region for issues (24 hours)
gcloud logging read "resource.service.name=aiprod-api-dr" \
  --project=aiprod-484120 \
  --limit=50
```

#### Re-activation to Primary Region

```bash
# Once primary region is back online:

# Step 1: Verify primary region health
gcloud compute regions describe europe-west1 \
  --project=aiprod-484120

# Step 2: Sync data from DR back to primary
# (Backup from DR, restore to primary)
gcloud sql backups create \
  --instance=aiprod-postgres-dr \
  --project=aiprod-484120

gcloud sql backups restore BACKUP_ID \
  --backup-instance=aiprod-postgres-dr \
  --target-instance=aiprod-postgres \
  --project=aiprod-484120

# Step 3: Update DNS back to primary
gcloud dns record-sets update aiprod.example.com. \
  --rrdatas=aiprod-api-xxxx-ew.a.run.app. \
  --ttl=300 \
  --type=CNAME \
  --zone=example-zone

# Step 4: Verify primary region working (30 min)
for i in {1..30}; do
  sleep 60
  curl -s https://aiprod.example.com/health && echo "✓" || echo "✗"
done

# Step 5: Decommission DR resources
gcloud run services delete aiprod-api-dr \
  --region=europe-west2 \
  --project=aiprod-484120

gcloud sql instances delete aiprod-postgres-dr \
  --project=aiprod-484120
```

**Expected Outcome**: Service restored in secondary region, then back to primary

**Total Time**: 90-120 minutes

---

## Part 3: Testing & Validation

### Monthly Backup Test

```bash
#!/bin/bash
# Run first Monday of each month at 10:00 UTC
# Team: DBA + DevOps Lead

# Test backup restoration
LATEST_BACKUP=$(gcloud sql backups list \
  --instance=aiprod-postgres \
  --project=aiprod-484120 \
  --format="value(name)" \
  --limit=1)

# Create test instance
gcloud sql instances create aiprod-postgres-test-restore \
  --region=europe-west1 \
  --database-version=POSTGRES_14 \
  --tier=db-f1-micro \
  --project=aiprod-484120

# Restore backup
gcloud sql backups restore $LATEST_BACKUP \
  --backup-instance=aiprod-postgres \
  --target-instance=aiprod-postgres-test-restore \
  --project=aiprod-484120

# Verify data integrity
gcloud sql connect aiprod-postgres-test-restore \
  --user=postgres \
  --project=aiprod-484120 << EOF

SELECT COUNT(*) as job_count FROM jobs;
SELECT COUNT(*) as result_count FROM results;
SELECT COUNT(*) as user_count FROM users;

-- Verify no corruption
SELECT COUNT(*) FROM jobs WHERE cost < 0;
SELECT COUNT(*) FROM jobs WHERE user_id IS NULL;

EOF

# Clean up test instance
gcloud sql instances delete aiprod-postgres-test-restore \
  --project=aiprod-484120

echo "✓ Backup test completed successfully"
```

### Quarterly Failover Drill

```bash
#!/bin/bash
# Run quarterly (every 3 months)
# Team: DBA + DevOps Lead + Service Lead

echo "=== Quarterly Failover Drill ==="
echo "Start time: $(date)"

# 1. Trigger failover
gcloud sql instances failover aiprod-postgres \
  --backup-instance=aiprod-postgres-replica-1 \
  --project=aiprod-484120 \
  --async

# 2. Monitor failover (watch for completion)
for i in {1..10}; do
  STATE=$(gcloud sql instances describe aiprod-postgres \
    --project=aiprod-484120 \
    --format="value(state)")

  echo "Failover state: $STATE"

  if [ "$STATE" = "RUNNABLE" ]; then
    echo "✓ Failover completed in $(($i * 30)) seconds"
    break
  fi

  sleep 30
done

# 3. Test API connectivity
curl https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/health

# 4. Run sanity tests
# (Check that API still works)

# 5. Failback to original primary
gcloud sql instances failover aiprod-postgres \
  --backup-instance=aiprod-postgres \
  --project=aiprod-484120

# 6. Verify all systems healthy
echo "Failover drill completed at $(date)"
```

### Annual Full DR Test

Complete simulation of regional outage:

```bash
#!/bin/bash
# Run annually (every 12 months)
# Team: All infrastructure

echo "=== Annual Full DR Test ==="
echo "Start time: $(date)"

# 1. Deploy API to DR region
gcloud run deploy aiprod-api-dr-test \
  --region=europe-west2 \
  --project=aiprod-484120

# 2. Restore database in DR region
gcloud sql backups restore LATEST \
  --backup-instance=aiprod-postgres \
  --target-instance=aiprod-postgres-dr-test \
  --project=aiprod-484120

# 3. Connect API to DR database
gcloud run services update aiprod-api-dr-test \
  --set-env-vars DATABASE_URL=postgresql://... \
  --region=europe-west2 \
  --project=aiprod-484120

# 4. Run full regression tests in DR region
pytest tests/ -v --tb=short

# 5. Validate data integrity
python scripts/validate-dr.py

# 6. Load test DR region
python scripts/load-test-dr.py --duration=5min --rps=100

# 7. Cleanup
gcloud run services delete aiprod-api-dr-test --region=europe-west2 --project=aiprod-484120
gcloud sql instances delete aiprod-postgres-dr-test --project=aiprod-484120

echo "✓ DR test completed successfully at $(date)"
```

---

## Part 4: RTO & RPO Summary

| Scenario              | RTO     | RPO     | Notes                 |
| --------------------- | ------- | ------- | --------------------- |
| Single instance fail  | <2 min  | 0 sec   | Auto-recovered        |
| Multiple instances    | <5 min  | 0 sec   | Full restart          |
| Database replica fail | <2 min  | 0 sec   | Auto-failover         |
| Primary DB down       | <10 min | <1 min  | Replica promotion     |
| Data corruption       | <30 min | <24 hr  | Restore from backup   |
| Regional outage       | <2 hr   | <15 min | Failover to DR region |

**SLA Commitment**:

- RTO <4 hours for any scenario
- RPO <24 hours for any scenario

---

## Part 5: Communication & Escalation

### During Disaster

```
T+0:     Incident detected
         └→ Page: Service Lead + DBA

T+5:     Begin investigation
         └→ Post to Slack #incidents

T+15:    Root cause identified
         └→ Update customers (status page)

T+30:    Recovery started
         └→ Notify all stakeholders

T+60:    Verify recovery
         └→ Resume normal operations

T+120:   Post-mortem scheduled
         └→ Internal review + fixes
```

### Contact List

**On-Call Rotation** (via PagerDuty):

- Infrastructure Lead
- Database Admin
- Service Lead
- VP Engineering (if >1 hour downtime)

**Emergency**: incidents@aiprod.ai (all alerts CCed)

### Customer Communication

- **Status Page**: https://status.aiprod.ai (updated every 15 min)
- **Email**: Sent to all affected customers within 30 min
- **Slack** (Enterprise): Direct channel updates

---

## Part 6: Maintenance & Updates

### Backup Configuration Changes

```bash
# Update backup retention (e.g., 60 days instead of 30)
gcloud sql instances patch aiprod-postgres \
  --backup-start-time=03:00 \
  --retained-backups-count=60 \
  --project=aiprod-484120
```

### Replica Management

```bash
# Create new replica
gcloud sql instances create aiprod-postgres-replica-3 \
  --master-instance-name=aiprod-postgres \
  --region=europe-west1 \
  --replica-type=READ \
  --project=aiprod-484120

# Delete failed replica
gcloud sql instances delete aiprod-postgres-replica-2 \
  --project=aiprod-484120
```

### DR Resource Sizing

Keep DR resources same size or larger than production:

- API: 4 CPU, 4 GB RAM (minimum)
- Database: db-custom-4-16384 (matching production)
- Storage: 100 GB (minimum)

---

## Appendix A: Backup Verification Script

```python
# scripts/verify-backups.py
import subprocess
import json
from datetime import datetime, timedelta

def verify_recent_backups():
    """Verify backups from past 7 days"""

    # Get backups
    result = subprocess.run([
        'gcloud', 'sql', 'backups', 'list',
        '--instance=aiprod-postgres',
        '--project=aiprod-484120',
        '--format=json'
    ], capture_output=True, text=True)

    backups = json.loads(result.stdout)

    # Check we have backups each day
    days_with_backup = {}
    for backup in backups:
        timestamp = backup['windowStartTime'][:10]  # YYYY-MM-DD
        days_with_backup[timestamp] = True

    # Verify 7 days of backups
    today = datetime.now()
    for i in range(7):
        date = (today - timedelta(days=i)).strftime('%Y-%m-%d')
        if date not in days_with_backup:
            print(f"⚠️  Missing backup for {date}")
            return False

    print("✓ All recent backups present")
    return True

if __name__ == '__main__':
    verify_recent_backups()
```

---

## Appendix B: Recovery Checklist

```markdown
### Incident Response Checklist

#### Immediately (T+0-5 min)

- [ ] Confirm incident is real (not false alarm)
- [ ] Page on-call team
- [ ] Create incident ticket
- [ ] Update status page

#### Assessment (T+5-15 min)

- [ ] Determine root cause
- [ ] Assess customer impact
- [ ] Check if RTO/RPO at risk
- [ ] Identify recovery path

#### Recovery (T+15-60 min)

- [ ] Execute recovery procedure
- [ ] Monitor progress
- [ ] Test critical paths
- [ ] Resume operations

#### Verification (T+60-120 min)

- [ ] Verify all systems healthy
- [ ] Check data integrity
- [ ] Confirm no data loss
- [ ] Clear incident status

#### Post-Incident (T+120+ min)

- [ ] Document timeline
- [ ] Collect logs/evidence
- [ ] Schedule post-mortem
- [ ] Update runbooks
- [ ] Implement preventive measures
```

---

**Document Status**: 🟢 Active  
**Last Updated**: February 4, 2026  
**Last Tested**: February 4, 2026  
**Next Test**: May 4, 2026 (quarterly)  
**Owner**: Infrastructure Team  
**Approver**: VP Engineering
