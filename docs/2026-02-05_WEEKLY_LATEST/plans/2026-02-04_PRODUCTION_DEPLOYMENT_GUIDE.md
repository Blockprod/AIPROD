# 📘 Production Deployment Guide - AIPROD

**Last Updated**: February 4, 2026  
**Version**: 1.0  
**Environment**: Google Cloud Platform (GCP)

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Deployment Steps](#deployment-steps)
4. [Configuration](#configuration)
5. [Monitoring](#monitoring)
6. [Troubleshooting](#troubleshooting)
7. [Operations](#operations)
8. [Rollback](#rollback)

---

## Overview

AIPROD is a complete audio-video generation pipeline deployed on Google Cloud Run with the following components:

- **API Service**: FastAPI/Uvicorn running on Cloud Run
- **Database**: Cloud SQL PostgreSQL 14
- **Storage**: Google Cloud Storage (GCS) for assets
- **Messaging**: Google Cloud Pub/Sub for async jobs
- **Monitoring**: Prometheus + Grafana + Cloud Logging

### Architecture

```
User Request
    ↓
Cloud Run (API)
    ├→ FastAPI endpoints
    ├→ StateMachine orchestration
    └→ Pub/Sub job queue
    ↓
Audio-Video Pipeline
    ├→ AudioGenerator (TTS)
    ├→ MusicComposer (Suno API)
    ├→ SoundEffectsAgent (Freesound)
    └→ PostProcessor (FFmpeg)
    ↓
GCS Storage (Output)
    └→ Final videos & assets
```

---

## Prerequisites

### Tools Required

```bash
# Google Cloud SDK
gcloud --version

# Docker (for local testing)
docker --version

# kubectl (for container management)
kubectl version

# Python 3.11+
python3 --version
```

### GCP Setup

```bash
# Set project
gcloud config set project aiprod-484120

# Verify permissions
gcloud auth list
gcloud projects get-iam-policy aiprod-484120

# Set default region
gcloud config set run/region europe-west1
```

### Required Credentials

1. **Database URL** - Cloud SQL connection string
2. **API Keys**:
   - Gemini API key
   - Suno API key
   - Freesound API key
   - ElevenLabs API key

3. **Service Account** - `aiprod-sa@aiprod-484120.iam.gserviceaccount.com`

---

## Deployment Steps

### Step 1: Build Docker Image

```bash
# Clone repository
git clone https://github.com/your-org/AIPROD.git
cd AIPROD

# Build image
docker build -t gcr.io/aiprod-484120/aiprod-v33:latest .

# Test locally
docker run -p 8000:8000 \
  -e DATABASE_URL="your_db_url" \
  -e GEMINI_API_KEY="your_key" \
  gcr.io/aiprod-484120/aiprod-v33:latest
```

### Step 2: Push to Container Registry

```bash
# Authenticate with GCR
gcloud auth configure-docker gcr.io

# Push image
docker push gcr.io/aiprod-484120/aiprod-v33:latest

# Verify
gcloud container images list-tags gcr.io/aiprod-484120/aiprod-v33
```

### Step 3: Setup Secrets in Secret Manager

```bash
# Create secrets
echo -n "your_db_url" | gcloud secrets create aiprod-db-url --data-file=-
echo -n "your_gemini_key" | gcloud secrets create gemini-api-key --data-file=-
echo -n "your_suno_key" | gcloud secrets create suno-api-key --data-file=-
echo -n "your_freesound_key" | gcloud secrets create freesound-api-key --data-file=-
echo -n "your_elevenlabs_key" | gcloud secrets create elevenlabs-api-key --data-file=-

# Grant service account access
for secret in aiprod-db-url gemini-api-key suno-api-key freesound-api-key elevenlabs-api-key; do
  gcloud secrets add-iam-policy-binding $secret \
    --member=serviceAccount:aiprod-sa@aiprod-484120.iam.gserviceaccount.com \
    --role=roles/secretmanager.secretAccessor
done
```

### Step 4: Create Pub/Sub Resources

```bash
# Create topic
gcloud pubsub topics create aiprod-jobs \
  --labels=environment=production,service=aiprod

# Create subscription
gcloud pubsub subscriptions create aiprod-jobs-sub \
  --topic=aiprod-jobs \
  --push-endpoint=https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/process-job \
  --push-auth-service-account=aiprod-sa@aiprod-484120.iam.gserviceaccount.com \
  --ack-deadline=600

# Create DLQ (Dead Letter Queue)
gcloud pubsub topics create aiprod-jobs-dlq
gcloud pubsub subscriptions create aiprod-jobs-dlq-sub \
  --topic=aiprod-jobs-dlq \
  --ack-deadline=600
```

### Step 5: Deploy to Cloud Run

```bash
# Using kubectl
kubectl apply -f deployments/cloud-run.yaml

# Or using gcloud
gcloud run deploy aiprod-v33-api \
  --image=gcr.io/aiprod-484120/aiprod-v33:latest \
  --platform=managed \
  --region=europe-west1 \
  --service-account=aiprod-sa@aiprod-484120.iam.gserviceaccount.com \
  --set-env-vars="GOOGLE_CLOUD_PROJECT=aiprod-484120,ENVIRONMENT=production" \
  --memory=2Gi \
  --cpu=2 \
  --concurrency=80 \
  --max-instances=20 \
  --min-instances=2
```

### Step 6: Verify Deployment

```bash
# Check service status
gcloud run services describe aiprod-v33-api --region europe-west1

# Test health endpoint
curl https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/health

# View logs
gcloud cloud-logging read "resource.type=cloud_run_revision AND resource.labels.service_name=aiprod-v33-api" --limit 50
```

---

## Configuration

### Environment Variables

| Variable               | Value                             | Purpose                |
| ---------------------- | --------------------------------- | ---------------------- |
| `ENVIRONMENT`          | `production`                      | Deployment environment |
| `GOOGLE_CLOUD_PROJECT` | `aiprod-484120`                   | GCP project ID         |
| `GCS_BUCKET_NAME`      | `aiprod-v33-assets`               | Storage bucket         |
| `DATABASE_URL`         | `postgresql://...`                | DB connection          |
| `LOG_LEVEL`            | `INFO`                            | Logging level          |
| `PUBSUB_TOPIC_NAME`    | `projects/.../topics/aiprod-jobs` | Job queue topic        |

### Resource Limits

```yaml
CPU: 2 vCPU
Memory: 2 GB
Timeout: 600 seconds
Concurrency: 80 requests per instance
Min Instances: 2 (always running)
Max Instances: 20 (scale under load)
```

### Database Configuration

```bash
# Cloud SQL proxy connection
export CLOUD_SQL_CONNECTION_NAME="aiprod-484120:europe-west1:aiprod-postgres"

# In DATABASE_URL
postgresql://user:password@/dbname?unix_socket_dir=/cloudsql/$CLOUD_SQL_CONNECTION_NAME
```

---

## Monitoring

### View Logs

```bash
# All logs
gcloud cloud-logging read --limit=100

# Error logs only
gcloud cloud-logging read 'severity="ERROR"' --limit=50

# Specific service
gcloud cloud-logging read 'resource.type="cloud_run_revision" AND resource.labels.service_name="aiprod-v33-api"' --limit=100

# Follow logs in real-time
gcloud cloud-logging read --limit 50 --follow
```

### Metrics & Dashboards

```bash
# View metrics
gcloud monitoring metrics-descriptors list

# Create alert
gcloud alpha monitoring policies create \
  --notification-channels=CHANNEL_ID \
  --display-name="High Error Rate" \
  --condition-display-name="Error rate > 5%"
```

### Common Metrics

- **Request Count**: `run.googleapis.com/request_count`
- **Request Latencies**: `run.googleapis.com/request_latencies`
- **Billable Time**: `run.googleapis.com/billable_time`
- **CPU Allocation**: `run.googleapis.com/cpu_allocation`
- **Memory Allocation**: `run.googleapis.com/memory_allocation`

---

## Troubleshooting

### Service Won't Start

```bash
# Check container logs
gcloud cloud-logging read 'resource.labels.service_name="aiprod-v33-api"' \
  --limit=50 \
  --sort-by=timestamp

# Common issues:
# 1. Missing secrets - verify all secrets exist
# 2. Database unreachable - check Cloud SQL proxy
# 3. Memory/CPU insufficient - increase limits
```

### High Error Rate

```bash
# Check recent errors
gcloud cloud-logging read 'severity="ERROR"' --limit=20

# Solutions:
# 1. Check external API availability (Suno, Freesound)
# 2. Verify database connection
# 3. Check request queue length
# 4. Review error logs for specific failures
```

### Slow Responses

```bash
# Check request latency
gcloud monitoring read \
  --filter='metric.type="run.googleapis.com/request_latencies"'

# Solutions:
# 1. Scale up instances (increase max_instances)
# 2. Optimize database queries
# 3. Check for long-running processes
# 4. Review audio processing times
```

### Queue Backlog

```bash
# Check Pub/Sub subscription
gcloud pubsub subscriptions describe aiprod-jobs-sub

# Check DLQ for failures
gcloud pubsub subscriptions describe aiprod-jobs-dlq-sub

# Clear backlog (if safe)
gcloud pubsub subscriptions seek aiprod-jobs-sub --time=2026-02-04T10:00:00Z
```

---

## Operations

### Scaling

```bash
# Increase capacity
gcloud run services update aiprod-v33-api \
  --max-instances=30 \
  --region=europe-west1

# Decrease capacity
gcloud run services update aiprod-v33-api \
  --max-instances=10 \
  --region=europe-west1

# Check current metrics
gcloud monitoring read \
  --filter='resource.labels.service_name="aiprod-v33-api"'
```

### Database Maintenance

```bash
# Backup database
gcloud sql backups create \
  --instance=aiprod-postgres \
  --description="Pre-deployment backup"

# Check backup status
gcloud sql backups list --instance=aiprod-postgres

# Restore from backup (if needed)
gcloud sql backups restore BACKUP_ID \
  --backup-instance=aiprod-postgres
```

### Update Configuration

```bash
# Update environment variables
gcloud run services update aiprod-v33-api \
  --set-env-vars=LOG_LEVEL=DEBUG \
  --region=europe-west1

# Update resource limits
gcloud run services update aiprod-v33-api \
  --memory=4Gi \
  --cpu=4 \
  --region=europe-west1
```

---

## Rollback

### Rollback to Previous Version

```bash
# List recent revisions
gcloud run revisions list --service=aiprod-v33-api --region=europe-west1

# Rollback to previous revision
gcloud run services update-traffic aiprod-v33-api \
  --to-revisions=aiprod-v33-api-PREVIOUS-REVISION=100 \
  --region=europe-west1

# Verify rollback
gcloud run services describe aiprod-v33-api --region=europe-west1
```

### Emergency Rollback

```bash
# If something goes wrong, immediately roll back:
gcloud run services update-traffic aiprod-v33-api \
  --to-revisions=aiprod-v33-api-LAST-STABLE=100 \
  --region=europe-west1

# Check status
gcloud run services describe aiprod-v33-api --region=europe-west1
```

---

## Performance Optimization

### Caching

```bash
# Enable CDN caching for static assets
gsutil lifecycle set - <<EOF
{
  "lifecycle": {
    "rule": [{
      "action": {"type": "SetStorageClass", "storageClass": "STANDARD"},
      "condition": {"age": 30}
    }]
  }
}
EOF
```

### Database Optimization

```bash
# Create indexes for common queries
gcloud sql connect aiprod-postgres --user=postgres

# In psql:
CREATE INDEX idx_video_user ON videos(user_id);
CREATE INDEX idx_job_status ON jobs(status);
CREATE INDEX idx_job_created ON jobs(created_at DESC);
```

### Request Optimization

```bash
# Enable compression
# Already handled by Cloud Run (gzip)

# Connection pooling
# Already configured in DATABASE_URL

# Async processing
# Jobs processed via Pub/Sub
```

---

## Security Checklist

- [x] All secrets stored in Secret Manager
- [x] Service account with minimal permissions
- [x] Cloud Run ingress set to internal
- [x] SSL/TLS enforced for all connections
- [x] Database connection encrypted
- [x] API authentication required
- [x] Rate limiting enabled
- [x] Audit logging enabled
- [x] DDoS protection via Cloud Armor
- [x] Regular security scans

---

## Support Contacts

- **On-Call Engineering**: [Slack: #aiprod-oncall]
- **Production Issues**: [Email: devops@company.com]
- **Urgent Escalation**: [Phone: +1-XXX-XXX-XXXX]

---

**Status**: Production Ready ✅  
**Last Deployment**: 2026-02-04  
**Next Review**: 2026-02-11
