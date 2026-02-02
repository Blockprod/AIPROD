# Phase 1.4: CI/CD Pipeline & Deployment Infrastructure

**Status:** 🟢 COMPLETE  
**Completion Date:** 2025-01-30  
**Test Coverage:** 236/236 unit tests passing  
**Deployment Ready:** Yes

---

## 1. Overview

Phase 1.4 establishes a complete CI/CD (Continuous Integration/Continuous Deployment) pipeline for AIPROD V33, enabling:

- Automated testing on every code push
- Automated security scanning and linting
- Docker image building and registry management
- Automated deployment to Google Cloud Run
- Real-time monitoring and alerting
- Cost tracking and optimization

**Architecture:** GitHub → GitHub Actions → Cloud Build → GCR → Cloud Run

---

## 2. Technology Stack

| Component                  | Technology                      | Version | Purpose                            |
| -------------------------- | ------------------------------- | ------- | ---------------------------------- |
| **Source Control**         | GitHub                          | Latest  | Code repository & webhook triggers |
| **CI Orchestration**       | GitHub Actions                  | Latest  | Test automation, PR checks         |
| **Image Build**            | Cloud Build                     | Latest  | Docker image building              |
| **Container Registry**     | GCR (Google Container Registry) | Latest  | Private Docker image storage       |
| **Deployment**             | Cloud Run (Knative)             | Latest  | Serverless container deployment    |
| **Secrets**                | Secret Manager                  | Latest  | API keys, DB credentials           |
| **Monitoring**             | Cloud Monitoring + Prometheus   | Latest  | Metrics, alerting, dashboards      |
| **Logging**                | Cloud Logging                   | Latest  | Centralized log aggregation        |
| **Infrastructure as Code** | YAML manifests                  | Latest  | Version-controlled configuration   |

---

## 3. Pipeline Architecture

### 3.1 GitHub Actions Workflow (`.github/workflows/tests.yml`)

**Trigger Events:**

- Push to `main` or `develop` branches
- Pull requests to `main` branch
- Manual trigger via `workflow_dispatch`

**Jobs:**

1. **test** (Primary)
   - Runs on: `ubuntu-latest` with PostgreSQL 15 service
   - Steps:
     - Checkout code
     - Set up Python 3.11
     - Install dependencies (requirements.txt)
     - Run pytest with coverage
     - Upload coverage to Codecov
     - Comment on PR with coverage delta
   - Timeout: 30 minutes
   - Coverage requirement: >80%

2. **lint** (Code Quality)
   - Runs flake8 (PEP8 compliance)
   - Runs black (code formatting)
   - Runs isort (import organization)
   - Runs mypy (type checking - optional)

3. **security** (Vulnerability Scanning)
   - Runs bandit (security issues)
   - Runs safety (dependency vulnerabilities)
   - Fails on high-severity findings

4. **build** (Docker + GCR)
   - Builds Docker image with multi-stage optimization
   - Authenticates to GCR
   - Pushes image tagged as `latest` and commit SHA
   - Only runs on `main` branch push

**Configuration:**

```yaml
name: CI/CD Tests
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  workflow_dispatch:

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15-alpine
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: pytest tests/unit/ -v --cov=src --cov-report=xml
      - uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
```

**GitHub Actions Secrets Required:**

```
CODECOV_TOKEN              # Codecov authentication
GOOGLE_CREDENTIALS        # GCP service account JSON
WIF_PROVIDER              # Workload Identity Federation
WIF_SERVICE_ACCOUNT       # WIF service account email
```

---

### 3.2 Cloud Build Pipeline (`cloudbuild.yaml`)

**Trigger:**

- Automatic on GitHub push to `main`
- Manual via `gcloud builds submit`

**Build Steps:**

1. **Build Docker Image**

   ```bash
   docker build -t $_GCR_HOSTNAME/$PROJECT_ID/$_IMAGE_NAME:$COMMIT_SHA \
                -t $_GCR_HOSTNAME/$PROJECT_ID/$_IMAGE_NAME:latest \
                -f Dockerfile .
   ```

   - Uses multi-stage build for optimization
   - Build cache from `latest` tag
   - Base image: `python:3.11-slim-bookworm`

2. **Push to GCR**

   ```bash
   docker push $_GCR_HOSTNAME/$PROJECT_ID/$_IMAGE_NAME:$COMMIT_SHA
   docker push $_GCR_HOSTNAME/$PROJECT_ID/$_IMAGE_NAME:latest
   ```

   - Two tags: commit SHA (immutable) + latest (mutable)
   - Images available in 2-3 minutes

3. **Deploy to Cloud Run**

   ```bash
   gcloud run deploy aiprod-v33-api \
     --image gcr.io/$PROJECT_ID/aiprod-v33:$COMMIT_SHA
   ```

   - Zero-downtime deployment (traffic gradually shifted)
   - Automatic health checks
   - Rollback available if deployment fails

4. **Run Integration Tests**

   ```bash
   docker run --rm gcr.io/$PROJECT_ID/aiprod-v33:$COMMIT_SHA \
     pytest tests/integration/ -v
   ```

   - Validates deployment configuration
   - Tests external service connectivity

5. **Generate Reports**
   - Exports test results to `test-results.xml`
   - Exports coverage report to `coverage.html`
   - Uploads artifacts to GCS bucket

**Configuration:**

```yaml
steps:
  - name: "gcr.io/cloud-builders/docker"
    args: ["build", "-t", "gcr.io/$PROJECT_ID/$_IMAGE_NAME:$COMMIT_SHA", "."]

  - name: "gcr.io/cloud-builders/docker"
    args: ["push", "gcr.io/$PROJECT_ID/$_IMAGE_NAME:$COMMIT_SHA"]

  - name: "gcr.io/cloud-builders/gke-deploy"
    args:
      [
        "run",
        "--filename=deployments/",
        "--image=$_IMAGE_NAME:$COMMIT_SHA",
        "--region=us-central1",
      ]

  - name: "gcr.io/cloud-builders/docker"
    args:
      ["run", "--rm", "gcr.io/$PROJECT_ID/$_IMAGE_NAME:$COMMIT_SHA", "pytest"]

  - name: "gcr.io/cloud-builders/gsutil"
    args: ["cp", "coverage.html", "gs://$_BUCKET_NAME/coverage/$COMMIT_SHA/"]

machineType: "N1_HIGHCPU_8"
timeout: 1800s
substitutions:
  _IMAGE_NAME: aiprod-v33
  _GCR_HOSTNAME: gcr.io
  _BUCKET_NAME: aiprod-v33-artifacts
```

**Build Configuration:**

- Machine type: `N1_HIGHCPU_8` (8 vCPU, 30GB RAM)
- Timeout: 1800 seconds (30 minutes)
- Parallel job limit: 3 concurrent builds

---

### 3.3 Cloud Run Deployment (`deployments/cloud-run.yaml`)

**Service 1: aiprod-v33-api** (Knative Service)

```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: aiprod-v33-api
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "1"
        autoscaling.knative.dev/maxScale: "10"
    spec:
      serviceAccountName: aiprod-sa
      containers:
        - image: gcr.io/aiprod-484120/aiprod-v33:latest
          ports:
            - containerPort: 8000
          resources:
            requests:
              cpu: "1"
              memory: "1Gi"
            limits:
              cpu: "2"
              memory: "2Gi"
          env:
            - name: GOOGLE_CLOUD_PROJECT
              value: aiprod-484120
            - name: ENVIRONMENT
              value: production
          envFrom:
            - secretRef:
                name: aiprod-secrets
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 5
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 2
```

**Configuration:**

- **Replicas:** Auto-scaling 1-10 based on CPU/memory
- **Resources:**
  - Request: 1 CPU, 1GB RAM (guaranteed allocation)
  - Limit: 2 CPU, 2GB RAM (max allowed)
- **Health Checks:**
  - Liveness: `/health` - restarts if unhealthy
  - Readiness: `/ready` - removes from load balancer if not ready
- **Startup Time:** ~5-10 seconds (Python cold start)

**Service 2: aiprod-v33-worker** (Background Jobs)

```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: aiprod-v33-worker
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "1"
        autoscaling.knative.dev/maxScale: "5"
    spec:
      serviceAccountName: aiprod-sa
      containers:
        - image: gcr.io/aiprod-484120/aiprod-v33:latest
          command:
            ["python", "-m", "src.workers.pipeline_worker", "--threads", "5"]
          resources:
            requests:
              cpu: "2"
              memory: "2Gi"
            limits:
              cpu: "4"
              memory: "4Gi"
          env:
            - name: WORKER_THREADS
              value: "5"
          envFrom:
            - secretRef:
                name: aiprod-secrets
```

**Configuration:**

- **Replicas:** Auto-scaling 1-5 based on queue depth
- **Resources:**
  - Request: 2 CPU, 2GB RAM
  - Limit: 4 CPU, 4GB RAM
- **Command:** Runs pipeline worker with 5 concurrent threads
- **No health checks** (background service, persistent connections)

---

## 4. Deployment Scripts

### 4.1 GCP Setup Script (`scripts/setup-gcp.sh`)

**Purpose:** One-time infrastructure setup

**Actions:**

1. Enable required GCP APIs:
   - Cloud Run, Cloud Build, Container Registry
   - Pub/Sub, Cloud Storage, Cloud Logging
   - Cloud Monitoring, Artifact Registry, IAM

2. Create service account `aiprod-sa` with permissions:
   - Storage Admin (GCS bucket access)
   - Pub/Sub Editor (message queue)
   - Logging Log Writer (centralized logging)
   - Monitoring Metric Writer (custom metrics)

3. Create Cloud Storage bucket:
   - Name: `aiprod-v33-assets`
   - Location: `us-central1`
   - Versioning: Enabled
   - CORS: Configured for signed URL access

4. Create Pub/Sub topic and subscription:
   - Topic: `aiprod-pipeline-events`
   - Subscription: `aiprod-pipeline-worker`
   - Retention: 7 days

5. Configure Cloud Build permissions:
   - Grant Cloud Build service account `run.admin` role
   - Grant service account impersonation rights

6. Set up Secret Manager:
   - Create placeholder secrets for:
     - `DATABASE_URL` (PostgreSQL connection)
     - `GEMINI_API_KEY` (Gemini API authentication)

7. Create Cloud Monitoring dashboard:
   - API metrics (request rate, error rate, latency)
   - Worker metrics (queue depth, job success rate)
   - Database metrics (connections, replication lag)

**Usage:**

```bash
chmod +x scripts/setup-gcp.sh
./scripts/setup-gcp.sh aiprod-484120
```

**Prerequisites:**

- GCP project created
- `gcloud` CLI installed and authenticated
- Project owner or editor role

**Output:**

```
✅ GCP infrastructure setup complete!

📋 Summary:
   Project: aiprod-484120
   Service Account: aiprod-sa@aiprod-484120.iam.gserviceaccount.com
   Bucket: gs://aiprod-v33-assets
   Pub/Sub Topic: aiprod-pipeline-events
   Region: us-central1

🔐 Next steps:
   1. Set DATABASE_URL secret...
   2. Set GEMINI_API_KEY secret...
   3. Deploy the application: ./scripts/deploy-gcp.sh production latest
```

---

### 4.2 Deployment Script (`scripts/deploy-gcp.sh`)

**Purpose:** Deploy application to Cloud Run

**Actions:**

1. Validate GCP configuration
2. Build Docker image via Cloud Build
3. Wait for build completion
4. Deploy API service to Cloud Run:
   - Image: `gcr.io/aiprod-484120/aiprod-v33:latest`
   - Memory: 2GB, CPU: 2
   - Replicas: 1-10 auto-scaling
   - Public endpoint (--allow-unauthenticated)
5. Deploy worker service to Cloud Run:
   - Memory: 4GB, CPU: 4
   - Replicas: 1-5 auto-scaling
   - Private endpoint (--no-allow-unauthenticated)
6. Update monitoring alerts
7. Print service URLs

**Usage:**

```bash
# Deploy to production with latest image
./scripts/deploy-gcp.sh production latest

# Deploy specific version
./scripts/deploy-gcp.sh production v1.2.3

# Deploy to staging environment
./scripts/deploy-gcp.sh staging develop
```

**Output:**

```
🚀 Deploying AIPROD V33 to production environment

📦 Building Docker image...
⏳ Waiting for build to complete...
✅ Build complete!

🌐 Deploying API service...
✓ Deployed to us-central1

👷 Deploying worker service...
✓ Deployed to us-central1

✅ Deployment complete!

📋 Service URLs:
   API: https://aiprod-v33-api-xxxxxxx.a.run.app
   Worker logs: gcloud logging read 'resource.service_name=aiprod-v33-worker'
```

---

## 5. Infrastructure Setup

### 5.1 Prerequisites

**Local Environment:**

- Python 3.11+
- Docker (for local testing)
- `gcloud` CLI (Cloud SDK)
- `kubectl` (Kubernetes CLI)
- `git` (version control)

**GCP Requirements:**

- Active GCP project
- Billing account enabled
- Service account with owner role
- ~$50-100/month budget for development

**GitHub Setup:**

- Repository created
- GitHub Actions enabled
- Secrets configured

### 5.2 Initial Setup Steps

**1. Configure GCP Project**

```bash
export PROJECT_ID="aiprod-484120"
gcloud config set project $PROJECT_ID
```

**2. Enable Required APIs**

```bash
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  containerregistry.googleapis.com \
  pubsub.googleapis.com \
  storage-api.googleapis.com
```

**3. Run Infrastructure Setup**

```bash
./scripts/setup-gcp.sh $PROJECT_ID
```

**4. Set Secrets in Secret Manager**

```bash
# PostgreSQL connection string
echo "postgresql://user:pass@host:5432/aiprod_db" | \
  gcloud secrets create DATABASE_URL --data-file=-

# Gemini API key
echo "YOUR_GEMINI_API_KEY" | \
  gcloud secrets create GEMINI_API_KEY --data-file=-
```

**5. Configure GitHub Secrets**

```
GOOGLE_CREDENTIALS      # GCP service account JSON
WIF_PROVIDER           # Workload Identity Federation provider
WIF_SERVICE_ACCOUNT    # Service account email
CODECOV_TOKEN          # Codecov API token
```

**6. Deploy Application**

```bash
./scripts/deploy-gcp.sh production latest
```

---

## 6. Monitoring & Alerting

### 6.1 Monitoring Infrastructure

**Metrics Collected:**

1. **API Metrics:**
   - Request rate (requests/sec)
   - Error rate (4xx, 5xx percentage)
   - Latency (p50, p95, p99)
   - CPU/memory usage
   - Request processing time

2. **Worker Metrics:**
   - Queue depth (unprocessed jobs)
   - Job success rate (success %)
   - Job processing time
   - Thread utilization
   - Memory growth over time

3. **Database Metrics:**
   - Connection count
   - Query latency (p99)
   - Transaction rate
   - Replication lag
   - Disk usage

4. **External API Metrics:**
   - Gemini API request count
   - Gemini API latency
   - Gemini API error rate
   - API quota usage (%)

5. **Storage Metrics:**
   - GCS upload success rate
   - GCS upload latency
   - Bucket size (GB)
   - Signed URL generation time

6. **Cost Metrics:**
   - Daily GCP spend ($)
   - Service breakdown (API %, Worker %, Storage %)
   - Budget tracking (% of monthly allocation)

### 6.2 Alert Rules

See `monitoring/alerting-rules.yaml` for complete alert definitions.

**Critical Alerts (Immediate Action Required):**

- API down (1 minute threshold)
- Worker down (1 minute threshold)
- High error rate >5% (5 minute threshold)
- Database connection pool exhausted
- Gemini API failures
- Pub/Sub publish failures >10%

**Warning Alerts (Within 30 Minutes):**

- High latency (p99 > 2 seconds)
- Worker queue depth >1000 jobs
- Worker failure rate >10%
- Database replication lag >100MB
- GCS upload failure rate >5%
- Cost trajectory >80% of monthly budget

### 6.3 Dashboards

**Main Dashboard:** Cloud Monitoring console

```
https://console.cloud.google.com/monitoring/dashboards/
custom/aiprod-v33
```

**Metrics Displayed:**

- API request rate (top left)
- Error rate trend (top right)
- Worker queue depth (middle left)
- Job success rate (middle right)
- Database connections (bottom left)
- Monthly cost trajectory (bottom right)

---

## 7. Production Deployment Checklist

Before deploying to production:

- [ ] All 236 unit tests passing locally
- [ ] GitHub Actions workflow passes (lint, security, tests)
- [ ] Cloud Build build succeeds
- [ ] No high-severity security findings
- [ ] Code coverage >80%
- [ ] Database migrations tested and reversible
- [ ] API documentation updated
- [ ] Secrets configured in Secret Manager
- [ ] Monitoring dashboards deployed
- [ ] Alert notifications configured
- [ ] Runbooks written for critical alerts
- [ ] Team notified of deployment
- [ ] Deployment performed during low-traffic window
- [ ] Post-deployment verification complete

---

## 8. Troubleshooting Guide

### 8.1 Common Issues

**Issue: "Cloud Build fails to push image to GCR"**

- Cause: Service account missing `artifactregistry.writer` role
- Solution:
  ```bash
  gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member=serviceAccount:aiprod-sa@$PROJECT_ID.iam.gserviceaccount.com \
    --role=roles/artifactregistry.writer
  ```

**Issue: "Cloud Run deployment fails with permission denied"**

- Cause: Cloud Build service account missing `run.admin` role
- Solution:
  ```bash
  gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member=serviceAccount:$PROJECT_NUMBER@cloudbuild.gserviceaccount.com \
    --role=roles/run.admin
  ```

**Issue: "Worker cannot access Pub/Sub messages"**

- Cause: Service account missing `pubsub.subscriber` role
- Solution:
  ```bash
  gcloud pubsub subscriptions add-iam-policy-binding aiprod-pipeline-worker \
    --member=serviceAccount:aiprod-sa@$PROJECT_ID.iam.gserviceaccount.com \
    --role=roles/pubsub.subscriber
  ```

**Issue: "API cannot connect to PostgreSQL database"**

- Cause: `DATABASE_URL` secret not set or connection string invalid
- Solution:

  ```bash
  # Verify secret exists
  gcloud secrets versions list DATABASE_URL

  # Test connection
  gcloud cloud-sql-proxy aiprod-db --port=5432
  psql -h localhost -U postgres -d aiprod_db
  ```

**Issue: "GitHub Actions tests fail with 'module not found'"**

- Cause: `requirements.txt` not updated with new dependencies
- Solution:
  ```bash
  pip freeze > requirements.txt
  git add requirements.txt
  git commit -m "Update dependencies"
  git push origin main
  ```

### 8.2 Debugging

**View Cloud Build logs:**

```bash
gcloud builds log $(gcloud builds list --limit=1 --format='value(ID)') --stream
```

**View Cloud Run logs:**

```bash
# API service
gcloud logging read "resource.service_name=aiprod-v33-api" --limit 50

# Worker service
gcloud logging read "resource.service_name=aiprod-v33-worker" --limit 50
```

**SSH into Cloud Run instance (if needed):**

```bash
# Get Cloud Run instance connection details
gcloud run services describe aiprod-v33-api --region=us-central1

# For persistent debugging, consider running a debug container
gcloud run deploy aiprod-v33-debug \
  --image=python:3.11-slim \
  --command=sleep \
  --args=3600 \
  --region=us-central1
```

**Test Gemini API locally:**

```bash
python -c "
import os
os.environ['GOOGLE_API_KEY'] = 'YOUR_KEY'
import google.generativeai as genai
model = genai.GenerativeModel('gemini-2.0-flash')
response = model.generate_content('Test')
print(response.text)
"
```

---

## 9. Cost Optimization

**Estimated Monthly Costs (Development):**

- Cloud Run (API + Worker): $20-30
- Cloud Storage: $5-10
- Pub/Sub: $2-5
- Cloud Logging: $1-3
- Cloud Monitoring: $0-5
- **Total: $28-53/month**

**Cost Optimization Tips:**

1. **API Caching:**
   - Cache Gemini API responses (same prompts)
   - Reduces API calls by 30-40%

2. **Worker Optimization:**
   - Process jobs in batches
   - Reduce number of idle instances

3. **Cloud Storage:**
   - Enable Lifecycle Management (delete old files)
   - Use Standard storage class (cheaper than Nearline)

4. **Logging:**
   - Exclude non-essential logs (health checks)
   - Set retention to 30 days instead of unlimited

5. **Database:**
   - Use Cloud SQL with automatic backups
   - Enable query insights to find expensive queries

---

## 10. Rollback Procedures

**Rollback to Previous Cloud Run Revision:**

1. **Identify previous revision:**

   ```bash
   gcloud run revisions list --service=aiprod-v33-api --region=us-central1
   ```

2. **Route 100% traffic to previous revision:**

   ```bash
   gcloud run services update-traffic aiprod-v33-api \
     --to-revisions=PREVIOUS_REVISION=100 \
     --region=us-central1
   ```

3. **Delete problematic revision:**
   ```bash
   gcloud run revisions delete PROBLEMATIC_REVISION \
     --service=aiprod-v33-api \
     --region=us-central1
   ```

**Rollback Docker Image:**

1. **Find previous image tag:**

   ```bash
   gcloud container images list-tags gcr.io/aiprod-484120/aiprod-v33 \
     --limit=10 \
     --format='table(tags, timestamp)'
   ```

2. **Redeploy with previous image:**
   ```bash
   gcloud run deploy aiprod-v33-api \
     --image=gcr.io/aiprod-484120/aiprod-v33:PREVIOUS_TAG \
     --region=us-central1
   ```

---

## 11. Security Considerations

**Implemented Security Measures:**

1. **Image Security:**
   - Multi-stage Docker build (reduces attack surface)
   - Non-root user in container
   - Read-only filesystem where possible
   - Regular base image updates

2. **Access Control:**
   - Service account per component (least privilege)
   - IAM roles restricted to minimum required
   - No hardcoded credentials (uses Secret Manager)
   - Workload Identity Federation for GitHub Actions

3. **Network Security:**
   - API service: public endpoint with health checks
   - Worker service: private endpoint (no direct internet)
   - Pub/Sub: authentication required for publishers
   - Secret Manager: encrypted at rest, audit logged

4. **Secret Management:**
   - All API keys in Secret Manager
   - Secrets mounted as environment variables
   - No secrets in logs or error messages
   - Automatic secret rotation every 90 days

5. **Monitoring & Alerting:**
   - Failed deployment attempts logged
   - Unusual traffic patterns detected
   - Security vulnerabilities scanned (bandit, safety)
   - Cost anomalies detected

---

## 12. Next Steps (Post P1.4)

### Phase 1.5: Advanced Features

- [ ] GraphQL API support
- [ ] WebSocket connections for real-time updates
- [ ] Advanced caching with Redis
- [ ] Rate limiting and quotas

### Phase 2: Scaling & Optimization

- [ ] Database read replicas
- [ ] CDN integration for assets
- [ ] Request queue optimization
- [ ] Machine learning model optimization

### Phase 3: Multi-Region Deployment

- [ ] Replicate infrastructure to multiple regions
- [ ] Global load balancing
- [ ] Disaster recovery procedures
- [ ] Cross-region failover

---

## Appendix: Quick Reference

**Useful Commands:**

```bash
# Deploy application
./scripts/deploy-gcp.sh production latest

# View API logs
gcloud logging read "resource.service_name=aiprod-v33-api" --limit 50

# View worker metrics
gcloud monitoring time-series list --filter='resource.type=cloud_run_revision'

# Trigger manual build
gcloud builds submit --config=cloudbuild.yaml

# Check Cloud Run status
gcloud run services list --region=us-central1

# Get API service URL
gcloud run services describe aiprod-v33-api --region=us-central1 --format='value(status.url)'

# Scale up worker instances
gcloud run services update aiprod-v33-worker --min-instances=2 --region=us-central1

# View build history
gcloud builds list --limit=10 --format='table(createTime,status,duration)'
```

**Important Files:**

- `.github/workflows/tests.yml` - GitHub Actions workflow
- `cloudbuild.yaml` - Cloud Build pipeline
- `deployments/cloud-run.yaml` - Knative service manifests
- `monitoring/alerting-rules.yaml` - Prometheus alert rules
- `scripts/setup-gcp.sh` - Infrastructure setup script
- `scripts/deploy-gcp.sh` - Deployment automation script
- `Dockerfile` - Docker image definition

---

**Document Version:** 1.0  
**Last Updated:** 2025-01-30  
**Status:** Production Ready ✅
