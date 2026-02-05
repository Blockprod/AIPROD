# Runbook: Database Errors

**Alert ID**: `alert-database-health`  
**Severity**: 🔴 CRITICAL  
**Oncall**: DBA  
**Last Updated**: Feb 4, 2026

---

## Quick Diagnosis

### Symptoms

- "Database connection refused" errors in logs
- "Connection timeout" errors
- API returns 503 Service Unavailable
- All operations failing (not just specific queries)

### Check Database Status

```bash
# Check Cloud SQL instance status
gcloud sql instances describe aiprod-postgres \
  --project=aiprod-484120 | grep -E "state|status|databaseVersion"

# Expected output: state=RUNNABLE

# If not RUNNABLE, check for maintenance
gcloud sql instances describe aiprod-postgres \
  --project=aiprod-484120 | grep -A 10 "settings:"
```

### Test Database Connectivity

```bash
# Attempt to connect to database
gcloud sql connect aiprod-postgres \
  --user=postgres \
  --project=aiprod-484120 << EOF
SELECT version();
SELECT now();
\dt  -- List tables
EOF

# If connection succeeds, database is up
# If connection fails, database is unreachable
```

---

## Root Causes Analysis

### 1. Database Instance Down

**Indicators**:

- `gcloud sql instances describe` shows state != RUNNABLE
- All connection attempts timeout
- No recent activity in logs

**Diagnosis**:

```bash
# Check instance state
gcloud sql instances describe aiprod-postgres --project=aiprod-484120

# Check for automatic maintenance
gcloud sql instances describe aiprod-postgres --project=aiprod-484120 | grep -A 5 "maintenanceWindow"

# Check recent operations
gcloud sql operations list --instance=aiprod-postgres --project=aiprod-484120 --limit=5
```

**Fix**: Wait for automatic recovery or failover manually

```bash
# If in maintenance, wait 15 minutes for completion
gcloud sql instances describe aiprod-postgres --project=aiprod-484120 --poll-interval=5 --max-wait=600

# If not recovering, failover to replica
gcloud sql instances failover aiprod-postgres \
  --backup-instance=aiprod-postgres-replica-1 \
  --project=aiprod-484120 \
  --async

# Monitor failover progress
gcloud sql operations list --instance=aiprod-postgres --project=aiprod-484120 --limit=1
```

**Escalation**: If failover fails, contact GCP Support

---

### 2. Connection Pool Exhausted

**Indicators**:

- Database responds to direct connections
- Application connections fail with "too many connections"
- Error rate increases gradually with traffic

**Diagnosis**:

```bash
# Check active connections
gcloud sql connect aiprod-postgres \
  --user=postgres \
  --project=aiprod-484120 << EOF
SELECT count(*) as total_connections FROM pg_stat_activity;
SELECT state, count(*) FROM pg_stat_activity GROUP BY state;
SELECT pid, usename, client_addr, state, query_start
FROM pg_stat_activity
ORDER BY query_start DESC LIMIT 20;
EOF

# Check max connections setting
gcloud sql connect aiprod-postgres \
  --user=postgres \
  --project=aiprod-484120 << EOF
SHOW max_connections;
EOF
```

**Fix**: Kill idle connections or increase pool size

```bash
# Option 1: Kill idle connections (quick fix)
gcloud sql connect aiprod-postgres \
  --user=postgres \
  --project=aiprod-484120 << EOF
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE state = 'idle' AND query_start < NOW() - INTERVAL '5 minutes'
AND pid != pg_backend_pid();
EOF

# Option 2: Increase max connections (permanent fix)
gcloud sql instances patch aiprod-postgres \
  --database-flags=max_connections=200 \
  --project=aiprod-484120

# Verify change
gcloud sql connect aiprod-postgres \
  --user=postgres \
  --project=aiprod-484120 << EOF
SHOW max_connections;
EOF

# Option 3: Reduce application connection pool size
# Edit .env or deployment config to use smaller pool
# Typical: pool_size=20, max_overflow=10
```

---

### 3. Disk Space Full

**Indicators**:

- Database accepting connections but returning "no space left"
- Inserts/updates failing
- Queries succeed but writes fail

**Diagnosis**:

```bash
# Check disk usage
gcloud sql instances describe aiprod-postgres \
  --project=aiprod-484120 | grep -A 5 "currentDiskSize\|settings:"

# Check database size
gcloud sql connect aiprod-postgres \
  --user=postgres \
  --project=aiprod-484120 << EOF
SELECT pg_size_pretty(pg_database_size('aiprod'));
SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
LIMIT 10;
EOF
```

**Fix**: Increase disk size or archive old data

```bash
# Option 1: Increase storage (quick fix)
gcloud sql instances patch aiprod-postgres \
  --storage-size=100GB \  # Or next size tier
  --project=aiprod-484120

# Monitor increase progress
gcloud sql operations list --instance=aiprod-postgres --project=aiprod-484120

# Option 2: Archive old data (long-term fix)
gcloud sql connect aiprod-postgres \
  --user=postgres \
  --project=aiprod-484120 << EOF
-- Archive old jobs (>90 days)
CREATE TABLE jobs_archive AS
SELECT * FROM jobs WHERE created_at < NOW() - INTERVAL '90 days';

DELETE FROM jobs WHERE created_at < NOW() - INTERVAL '90 days';

VACUUM ANALYZE jobs;
EOF
```

---

### 4. High CPU or Memory

**Indicators**:

- Database responding slowly
- Long-running queries visible in logs
- CPU/memory at 80%+ utilization

**Diagnosis**:

```bash
# Check Cloud SQL metrics
gcloud monitoring time-series list \
  --filter='metric.type="cloudsql.googleapis.com/database/cpu/utilization" AND resource.database_id="aiprod-484120:aiprod-postgres"' \
  --project=aiprod-484120

# Check memory
gcloud monitoring time-series list \
  --filter='metric.type="cloudsql.googleapis.com/database/memory/utilization"' \
  --project=aiprod-484120

# Find expensive queries
gcloud sql connect aiprod-postgres \
  --user=postgres \
  --project=aiprod-484120 << EOF
SELECT query, mean_exec_time, calls
FROM pg_stat_statements
WHERE mean_exec_time > 5000  -- > 5 seconds
ORDER BY mean_exec_time DESC
LIMIT 10;
EOF
```

**Fix**: Optimize queries or upgrade database tier

```bash
# Option 1: Kill long-running query
gcloud sql connect aiprod-postgres \
  --user=postgres \
  --project=aiprod-484120 << EOF
SELECT pid, now() - query_start as duration, query
FROM pg_stat_activity
WHERE state != 'idle'
ORDER BY query_start;

-- Kill specific query if needed
SELECT pg_terminate_backend(PID);
EOF

# Option 2: Upgrade database tier
gcloud sql instances patch aiprod-postgres \
  --tier=db-custom-4-16384 \  # 4 CPU, 16GB RAM
  --project=aiprod-484120

# Monitor upgrade
gcloud sql operations list --instance=aiprod-postgres --project=aiprod-484120
```

---

### 5. Replication Lag

**Indicators**:

- Read replicas lagging behind primary
- "Replication lag too high" alert
- Reads on replica return stale data

**Diagnosis**:

```bash
# Check replication status on primary
gcloud sql connect aiprod-postgres \
  --user=postgres \
  --project=aiprod-484120 << EOF
SELECT slot_name, restart_lsn FROM pg_replication_slots;
EOF

# Check replica lag on each replica
for REPLICA in aiprod-postgres-replica-1 aiprod-postgres-replica-2; do
  echo "=== $REPLICA ==="
  gcloud sql connect $REPLICA \
    --user=postgres \
    --project=aiprod-484120 << EOF
SELECT EXTRACT(EPOCH FROM (NOW() - pg_last_xact_replay_timestamp())) as lag_seconds;
EOF
done
```

**Fix**: Restart replication if lag is excessive

```bash
# Check if replica is still catching up
gcloud sql instances describe aiprod-postgres-replica-1 \
  --project=aiprod-484120 | grep -A 5 "replicaConfiguration"

# If lag > 60 seconds, restart replica
gcloud sql instances restart aiprod-postgres-replica-1 \
  --project=aiprod-484120

# Monitor recovery
sleep 30
gcloud sql connect aiprod-postgres-replica-1 \
  --user=postgres \
  --project=aiprod-484120 << EOF
SELECT EXTRACT(EPOCH FROM (NOW() - pg_last_xact_replay_timestamp())) as lag_seconds;
EOF
```

---

## Recovery Steps

### Step 1: Assess Severity

```bash
# Check how many operations are failing
gcloud logging read "severity=ERROR AND jsonPayload.error_type=DATABASE_ERROR" \
  --project=aiprod-484120 \
  --limit=100 | jq length

# High number (>50) indicates critical issue
```

### Step 2: Isolate Problem

- Is database instance down? → Wait or failover
- Are connections exhausted? → Kill idle or increase pool
- Is disk full? → Increase storage or archive
- Is database slow? → Optimize query or upgrade

### Step 3: Apply Fix

See specific section above for your root cause

### Step 4: Verify Recovery

```bash
# Test basic connectivity
gcloud sql connect aiprod-postgres \
  --user=postgres \
  --project=aiprod-484120 << EOF
SELECT count(*) FROM jobs;
SELECT count(*) FROM results;
EOF

# Monitor error rate dropping
gcloud logging read "severity=ERROR AND jsonPayload.error_type=DATABASE_ERROR" \
  --project=aiprod-484120 \
  --limit=10
```

---

## Prevention

1. **Monitoring**: Set up alerts for CPU, memory, disk
2. **Maintenance**: Regular VACUUM and ANALYZE
3. **Capacity**: Plan for growth, increase size proactively
4. **Backups**: Test backup/restore monthly
5. **Replicas**: Keep replicas healthy and in sync

---

## Escalation

| Time     | Action                                    |
| -------- | ----------------------------------------- |
| T+0      | Page DBA on-call                          |
| T+5 min  | If instance down, attempt failover        |
| T+10 min | If still down, notify Infrastructure Lead |
| T+30 min | If not resolved, notify VP Engineering    |
| T+60 min | If not resolved, contact GCP Support      |

---

## Commands Reference

```bash
# View database status
gcloud sql instances describe aiprod-postgres --project=aiprod-484120

# List all instances
gcloud sql instances list --project=aiprod-484120

# Failover to replica
gcloud sql instances failover aiprod-postgres --backup-instance=REPLICA_NAME --project=aiprod-484120

# Connect to database
gcloud sql connect aiprod-postgres --user=postgres --project=aiprod-484120

# Update database settings
gcloud sql instances patch aiprod-postgres --database-flags=FLAG=VALUE --project=aiprod-484120

# Check operations
gcloud sql operations list --instance=aiprod-postgres --project=aiprod-484120
```

---

## Contacts

- **DBA Lead**: dba-lead@aiprod.ai
- **Infrastructure Lead**: infra-lead@aiprod.ai
- **GCP Support**: https://cloud.google.com/support
- **Slack Channel**: #database-incidents

---

## Related Documentation

- [Disaster Recovery Guide](disaster-recovery.md)
- [Backup & Recovery](../monitoring/backup.md)
- [Performance Monitoring](../monitoring/performance.md)

---

**Last Tested**: Feb 4, 2026  
**Next Review**: Mar 4, 2026
