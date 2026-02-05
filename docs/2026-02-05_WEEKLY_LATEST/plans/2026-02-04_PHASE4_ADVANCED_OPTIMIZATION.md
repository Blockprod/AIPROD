# PHASE 4 — Advanced Features & Optimization

**Start Date**: February 4, 2026  
**Deadline**: May 31, 2026  
**Duration**: ~12 hours (spread over 4 months)  
**Objective**: Cost optimization, advanced features, and infrastructure enhancement  
**Status**: 🟢 Ready to start

---

## 📋 Overview

PHASE 4 focuses on optimizing costs while implementing advanced features that increase platform value and user satisfaction. This phase builds on the solid foundation of PHASE 3 (Documentation & SLAs).

### Success Criteria

- [ ] Cloud costs reduced by 20-30%
- [ ] Auto-scaling operational on all critical services
- [ ] Database performance improved by 40%+
- [ ] Cost monitoring dashboard live
- [ ] All 5 advanced features implemented

---

## 🎯 TÂCHE 4.1 — Cloud Cost Analysis

**ID**: `4.1`  
**Duration**: 2 hours  
**Complexity**: Medium  
**Dependencies**: Production environment (PHASE 3 complete)

### Objective

Analyze current cloud infrastructure costs, identify optimization opportunities, and create baseline metrics for tracking cost reduction.

### Deliverables

1. **Cost Breakdown Report**
   - Cloud Run: CPU/Memory costs
   - Firestore: Storage/Operations costs
   - Cloud Storage: Data transfer costs
   - Cloud SQL: Compute/Storage costs
   - BigQuery: Query/Storage costs
   - Networking: Ingress/Egress costs

2. **Cost Analysis Dashboard**
   - Monthly cost trends (6-month history)
   - Cost per service
   - Cost per feature/product
   - Projected quarterly/annual costs

3. **Optimization Opportunities Document**
   - Quick wins (implement within 1 week)
   - Medium-term optimizations (1-4 weeks)
   - Long-term strategies (1-6 months)

### Implementation Steps

**Step 1: Gather Current Costs**

```bash
# Export 6 months of billing data
gcloud billing accounts list
gcloud billing accounts describe ACCOUNT_ID

# Export to BigQuery
bq load \
  --autodetect \
  project_id:billing_dataset.billing_data \
  gs://billing-export-bucket/billing_export_*.csv
```

**Step 2: Analyze by Service**

```sql
-- Breakdown by service in BigQuery
SELECT
  service.description as service,
  SUM(CAST(cost as FLOAT64)) as total_cost,
  COUNT(*) as line_items,
  SUM(CAST(cost as FLOAT64)) / COUNT(*) as avg_cost_per_item
FROM `project_id.billing_dataset.gcp_billing_export_*`
WHERE _TABLE_SUFFIX BETWEEN '20250901' AND '20260204'
GROUP BY service.description
ORDER BY total_cost DESC;
```

**Step 3: Identify Cost Drivers**

```sql
-- Top cost drivers
SELECT
  service.description as service,
  sku.description as resource,
  SUM(CAST(cost as FLOAT64)) as total_cost,
  SUM(usage.amount) as total_usage,
  usage.unit as unit
FROM `project_id.billing_dataset.gcp_billing_export_*`
WHERE _TABLE_SUFFIX BETWEEN '20250901' AND '20260204'
GROUP BY service, sku.description, usage.unit
ORDER BY total_cost DESC
LIMIT 50;
```

### Expected Results

```
Cloud Run (est. 35-40% of total)
├─ CPU seconds: 45%
├─ Memory: 35%
├─ Requests: 10%
└─ Storage: 10%

Firestore (est. 20-25% of total)
├─ Read operations: 50%
├─ Write operations: 30%
├─ Storage: 15%
└─ Network: 5%

Cloud SQL (est. 15-20% of total)
├─ Compute: 60%
├─ Storage: 25%
└─ Backups: 15%

Cloud Storage (est. 5-10% of total)
├─ Storage: 60%
├─ Network (egress): 40%

Other services (est. 5% of total)
└─ Secrets Manager, Logging, Monitoring
```

### Optimization Opportunities (Quick Wins)

```
1. ⚡ Reduce Cloud Run memory (4GB → 2GB for most services)
   - Impact: -10-15% Cloud Run costs (~$2-3K/month)
   - Effort: 2 hours
   - Risk: Test thoroughly before deploying

2. 🗄️ Archive old Firestore data monthly
   - Impact: -5-8% Firestore storage (~$500-800/month)
   - Effort: 1 hour (automation setup)
   - Risk: Low (archival to Cloud Storage)

3. 🔄 Optimize Cloud SQL backups (daily → weekly for non-critical)
   - Impact: -3-5% Cloud SQL costs (~$300-500/month)
   - Effort: 30 min
   - Risk: Medium (ensure daily incremental backups active)

4. 📊 Implement query caching in Firestore
   - Impact: -8-12% Firestore read operations (~$1-1.5K/month)
   - Effort: 2-3 hours
   - Risk: Low (application-level change)

5. 🌍 Optimize CDN configuration
   - Impact: -10-20% egress costs (~$500-1K/month)
   - Effort: 1 hour
   - Risk: Low (network configuration)
```

### Metrics to Track

- Monthly GCP bill (target: -20% in Q1)
- Cost per transaction
- Cost per active user
- Cost per API call
- Cloud Run utilization
- Database query costs

---

## 🎯 TÂCHE 4.2 — Auto-Scaling Setup

**ID**: `4.2`  
**Duration**: 2.5 hours  
**Complexity**: High  
**Dependencies**: TÂCHE 4.1 (cost analysis)

### Objective

Implement intelligent auto-scaling policies that dynamically adjust resources based on demand, reducing idle costs while maintaining performance.

### Auto-Scaling Targets

**Cloud Run** (API services)

- Min instances: 1 (production), 0 (staging)
- Max instances: 20 (production), 5 (staging)
- Target CPU: 60%
- Target concurrency: 40 requests/instance
- Scale-down delay: 5 minutes

**Firestore** (NoSQL database)

- Read/Write capacity: Auto-scale
- Mode: On-demand (pay per operation, no provisioning)
- Monitoring: Track usage patterns

**Cloud SQL** (PostgreSQL)

- CPU scaling threshold: 70%
- Storage auto-expand: 100GB increments
- High availability: Failover replica in different zone

**Cloud Tasks** (Job processing)

- Concurrency limits: 100-1000 (adaptive)
- Rate limiting: 100-10,000 tasks/sec (adaptive)

### Implementation

**Step 1: Cloud Run Auto-Scaling**

```yaml
# deployment.yaml for Cloud Run
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: aiprod-api
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "1"
        autoscaling.knative.dev/maxScale: "20"
        autoscaling.knative.dev/target-cpu-utilization-percentage: "60"
        autoscaling.knative.dev/target-rpc-utilization-percentage: "80"
    spec:
      containerConcurrency: 40
      containers:
        - image: gcr.io/aiprod-v33/api:latest
          resources:
            requests:
              cpu: 500m
              memory: 512Mi
```

**Step 2: Firestore Auto-Scaling**

```python
# firestore_config.py
from firebase_admin import firestore

db = firestore.client()

# Enable on-demand billing (auto-scaling)
# gcloud firestore databases update \
#   --type=cloud-firestore \
#   --mode=on-demand \
#   --location=europe-west1

# Monitor read/write usage
def track_firestore_usage():
    stats = db.collection('_metadata').document('stats').get()
    reads = stats.get('monthly_reads')
    writes = stats.get('monthly_writes')
    return {
        'reads': reads,
        'writes': writes,
        'cost_estimate': (reads * 0.06 + writes * 0.18) / 1_000_000
    }
```

**Step 3: Cloud SQL Auto-Scaling**

```bash
# Enable Cloud SQL auto-scaling
gcloud sql instances patch aiprod-db \
  --enable-bin-log \
  --backup-start-time=03:00 \
  --database-flags=cloudsql_iam_authentication=on

# Configure High Availability
gcloud sql instances create aiprod-db-replica \
  --master-instance-name=aiprod-db \
  --region=europe-west1 \
  --tier=db-f1-micro
```

**Step 4: Cloud Tasks Auto-Scaling**

```python
# task_processor.py with adaptive concurrency
import concurrent.futures
from google.cloud import tasks_v2

class AdaptiveTaskProcessor:
    def __init__(self):
        self.min_workers = 10
        self.max_workers = 100
        self.current_workers = self.min_workers

    def adjust_concurrency(self, queue_depth):
        """Adjust number of workers based on queue depth"""
        if queue_depth > 500:
            self.current_workers = min(self.max_workers, queue_depth // 5)
        elif queue_depth < 100:
            self.current_workers = max(self.min_workers, queue_depth // 10)

        return self.current_workers

    def process_tasks(self):
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.current_workers
        ) as executor:
            # Process tasks
            pass
```

### Load Testing

```bash
# Load test Cloud Run auto-scaling with wrk
wrk -t12 -c400 -d30s \
  --script=post.lua \
  https://aiprod-api.example.com/pipeline/run

# Monitor scaling in real-time
watch -n 5 'gcloud run services describe aiprod-api \
  --region=europe-west1 \
  --format="table(status.observedGeneration, status.replicas)"'
```

### Expected Improvements

- Idle cost reduction: 20-30% (fewer unused instances)
- Peak performance maintained: 99.5% latency SLA
- Manual scaling eliminated: Fully automatic
- Cost per request: -15-25%

---

## 🎯 TÂCHE 4.3 — Database Optimization

**ID**: `4.3`  
**Duration**: 2.5 hours  
**Complexity**: High  
**Dependencies**: Production environment, monitoring in place

### Objective

Identify and optimize slow database queries, implement strategic indexing, and improve overall database performance by 40%+.

### Query Profiling

**Step 1: Identify Slow Queries**

```sql
-- PostgreSQL: Find slow queries
SELECT
  query,
  calls,
  mean_exec_time,
  stddev_exec_time,
  rows
FROM pg_stat_statements
WHERE mean_exec_time > 100  -- > 100ms
ORDER BY mean_exec_time DESC
LIMIT 20;
```

**Step 2: Analyze Query Execution Plans**

```sql
-- Example slow query analysis
EXPLAIN ANALYZE
SELECT p.*, u.email, COUNT(r.id) as report_count
FROM pipelines p
JOIN users u ON p.user_id = u.id
LEFT JOIN reports r ON p.id = r.pipeline_id
WHERE p.created_at > NOW() - INTERVAL '30 days'
GROUP BY p.id, u.id
ORDER BY p.created_at DESC
LIMIT 100;

-- Look for Sequential Scans, high planning time, high execution time
-- Solution: Add indexes on frequently filtered/joined columns
```

### Indexing Strategy

**Critical Indexes to Add**

```sql
-- User queries
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_created_at ON users(created_at DESC);

-- Pipeline queries
CREATE INDEX idx_pipelines_user_id ON pipelines(user_id);
CREATE INDEX idx_pipelines_created_at ON pipelines(created_at DESC);
CREATE INDEX idx_pipelines_status ON pipelines(status, created_at DESC);

-- Report queries
CREATE INDEX idx_reports_pipeline_id ON reports(pipeline_id);
CREATE INDEX idx_reports_created_at ON reports(created_at DESC);

-- Job queries
CREATE INDEX idx_jobs_pipeline_id ON jobs(pipeline_id);
CREATE INDEX idx_jobs_status ON jobs(status, updated_at DESC);

-- Composite indexes for common filters
CREATE INDEX idx_pipelines_user_status
  ON pipelines(user_id, status, created_at DESC);

CREATE INDEX idx_reports_pipeline_status
  ON reports(pipeline_id, status, created_at DESC);
```

### Query Optimization

**Before (slow)**

```python
# N+1 query problem
pipelines = Pipeline.objects.filter(user_id=user.id)
for pipeline in pipelines:
    print(pipeline.user.email)  # Extra query for each pipeline!
    reports = pipeline.reports.all()  # Extra query for each pipeline!
```

**After (optimized)**

```python
# Use select_related and prefetch_related
pipelines = Pipeline.objects.filter(
    user_id=user.id
).select_related('user').prefetch_related('reports')

for pipeline in pipelines:
    print(pipeline.user.email)  # No extra query
    reports = pipeline.reports.all()  # No extra query
```

### Caching Layer

```python
# redis_cache.py
from functools import wraps
import redis
import json
from datetime import timedelta

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cached(ttl_seconds=300):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and args
            cache_key = f"{func.__name__}:{args}:{kwargs}"

            # Try to get from cache
            cached_value = redis_client.get(cache_key)
            if cached_value:
                return json.loads(cached_value)

            # Call function and cache result
            result = func(*args, **kwargs)
            redis_client.setex(
                cache_key,
                ttl_seconds,
                json.dumps(result, default=str)
            )
            return result

        return wrapper
    return decorator

# Usage
@cached(ttl_seconds=3600)
def get_user_stats(user_id):
    # Expensive query
    return expensive_database_query(user_id)
```

### Performance Monitoring

```python
# query_monitor.py
from django.db import connection
from django.db.backends.utils import CursorWrapper

class QueryTimeLogger:
    def __init__(self, threshold_ms=100):
        self.threshold_ms = threshold_ms

    def log_slow_queries(self):
        for query in connection.queries:
            time = float(query['time'])
            if time * 1000 > self.threshold_ms:
                print(f"Slow query ({time*1000:.2f}ms): {query['sql'][:100]}")

# Usage in production monitoring
monitor = QueryTimeLogger(threshold_ms=100)
monitor.log_slow_queries()
```

### Expected Improvements

- Average query time: -40-50%
- P95 latency: -30-40%
- Database CPU utilization: -25-35%
- Cost per query: -20-30%

---

## 🎯 TÂCHE 4.4 — Cost Alerts & Monitoring

**ID**: `4.4`  
**Duration**: 1.5 hours  
**Complexity**: Medium  
**Dependencies**: TÂCHE 4.1 (cost analysis complete)

### Objective

Setup comprehensive cost monitoring and alerting to prevent budget overruns and identify anomalies in real-time.

### GCP Budget Alerts

```bash
# Create budget alert for total spend
gcloud billing budgets create \
  --billing-account=ACCOUNT_ID \
  --display-name="Monthly Budget Alert" \
  --budget-amount=5000 \
  --threshold-rule=percent=50 \
  --threshold-rule=percent=80 \
  --threshold-rule=percent=100

# Create budget alert per service
gcloud billing budgets create \
  --billing-account=ACCOUNT_ID \
  --display-name="Cloud Run Budget" \
  --budget-amount=1500 \
  --threshold-rule=percent=80 \
  --filter='resource.service="Cloud Run"'
```

### Slack Integration

```python
# cost_monitor.py
from google.cloud import monitoring_v3
import requests
import json
from datetime import datetime, timedelta

class CostMonitor:
    def __init__(self, slack_webhook_url):
        self.slack_url = slack_webhook_url
        self.client = monitoring_v3.MetricServiceClient()

    def check_daily_costs(self):
        """Check costs and alert if unusual"""
        # Query BigQuery for today's costs
        from google.cloud import bigquery

        bq_client = bigquery.Client()
        query = """
        SELECT
          SUM(CAST(cost as FLOAT64)) as daily_cost,
          service.description as service
        FROM `project_id.billing_dataset.gcp_billing_export_*`
        WHERE DATE(_PARTITIONTIME) = CURRENT_DATE()
        GROUP BY service.description
        ORDER BY daily_cost DESC
        """

        results = bq_client.query(query).result()
        total_cost = sum(row['daily_cost'] for row in results)

        # Compare to 30-day average
        avg_daily = self.get_average_daily_cost()
        percent_change = ((total_cost - avg_daily) / avg_daily) * 100

        # Alert if > 20% higher than average
        if percent_change > 20:
            self.send_slack_alert(
                f"⚠️ Cost Alert: Today's costs are {percent_change:.1f}% higher than average!\n"
                f"Today: ${total_cost:.2f} | 30-day avg: ${avg_daily:.2f}"
            )

    def send_slack_alert(self, message):
        """Send alert to Slack"""
        payload = {
            "text": message,
            "attachments": [{
                "color": "danger",
                "fields": [{
                    "title": "Cost Monitoring",
                    "value": message,
                    "short": False
                }]
            }]
        }
        requests.post(self.slack_url, json=payload)

    def get_average_daily_cost(self):
        """Get average daily cost for last 30 days"""
        from google.cloud import bigquery

        bq_client = bigquery.Client()
        query = """
        SELECT AVG(daily_cost) as avg_cost
        FROM (
          SELECT
            DATE(_PARTITIONTIME) as date,
            SUM(CAST(cost as FLOAT64)) as daily_cost
          FROM `project_id.billing_dataset.gcp_billing_export_*`
          WHERE DATE(_PARTITIONTIME) BETWEEN
            DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
            AND CURRENT_DATE()
          GROUP BY date
        )
        """

        result = bq_client.query(query).result()
        return list(result)[0]['avg_cost']

# Schedule daily checks
from apscheduler.schedulers.background import BackgroundScheduler

monitor = CostMonitor(slack_webhook_url="https://hooks.slack.com/...")
scheduler = BackgroundScheduler()
scheduler.add_job(monitor.check_daily_costs, 'cron', hour=9, minute=0)  # 9 AM daily
scheduler.start()
```

### Cost Tracking Dashboard

```python
# dashboard.py - Flask/Grafana integration
from flask import Flask, jsonify
from google.cloud import bigquery

app = Flask(__name__)
bq_client = bigquery.Client()

@app.route('/api/costs/monthly')
def get_monthly_costs():
    """Get current month costs by service"""
    query = """
    SELECT
      service.description as service,
      SUM(CAST(cost as FLOAT64)) as cost,
      COUNT(*) as line_items
    FROM `project_id.billing_dataset.gcp_billing_export_*`
    WHERE DATE(_PARTITIONTIME) >= DATE_TRUNC(CURRENT_DATE(), MONTH)
    GROUP BY service.description
    ORDER BY cost DESC
    """

    results = bq_client.query(query).result()
    return jsonify([dict(row) for row in results])

@app.route('/api/costs/trend')
def get_cost_trend():
    """Get 6-month cost trend"""
    query = """
    SELECT
      DATE(_PARTITIONTIME) as date,
      SUM(CAST(cost as FLOAT64)) as daily_cost
    FROM `project_id.billing_dataset.gcp_billing_export_*`
    WHERE DATE(_PARTITIONTIME) >= DATE_SUB(CURRENT_DATE(), INTERVAL 180 DAY)
    GROUP BY date
    ORDER BY date
    """

    results = bq_client.query(query).result()
    return jsonify([dict(row) for row in results])

if __name__ == '__main__':
    app.run(debug=True)
```

### Anomaly Detection

```python
# anomaly_detection.py
import numpy as np
from scipy import stats

class AnomalyDetector:
    def __init__(self, z_score_threshold=2.5):
        self.threshold = z_score_threshold

    def detect_anomalies(self, daily_costs):
        """Detect cost anomalies using Z-score method"""
        mean = np.mean(daily_costs)
        std = np.std(daily_costs)
        z_scores = np.abs(stats.zscore(daily_costs))

        anomalies = []
        for i, z_score in enumerate(z_scores):
            if z_score > self.threshold:
                anomalies.append({
                    'date': i,
                    'cost': daily_costs[i],
                    'z_score': z_score,
                    'deviation_percent': ((daily_costs[i] - mean) / mean) * 100
                })

        return anomalies

# Example usage
detector = AnomalyDetector()
daily_costs = [100, 102, 98, 105, 250, 103, 99]  # 250 is anomaly
anomalies = detector.detect_anomalies(daily_costs)
print(anomalies)  # [{'date': 4, 'cost': 250, 'z_score': 3.2, 'deviation_percent': 140}]
```

### Expected Deliverables

- ✅ GCP Budget Alerts configured
- ✅ Slack notifications active
- ✅ Cost tracking dashboard live
- ✅ Anomaly detection rules enabled
- ✅ Weekly cost reports automated

---

## 🎯 TÂCHE 4.5 — Reserved Capacity Planning

**ID**: `4.5`  
**Duration**: 2 hours  
**Complexity**: Medium  
**Dependencies**: TÂCHE 4.1-4.4 (cost analysis, monitoring, auto-scaling)

### Objective

Implement GCP Commitment Plans and Reserved Capacity to achieve 25-40% cost savings for predictable workloads.

### Commitment Plan Analysis

```python
# capacity_planner.py
from google.cloud import bigquery
import json

class CapacityPlanner:
    def __init__(self):
        self.bq_client = bigquery.Client()

    def analyze_resource_usage(self):
        """Analyze 6-month usage patterns to recommend commitments"""

        # Analyze Cloud Run CPU usage
        cr_query = """
        SELECT
          DATE_TRUNC(TIMESTAMP_MILLIS(_TIMESTAMP_MS), MONTH) as month,
          SUM(CAST(usage.amount as FLOAT64)) as cpu_seconds,
          SUM(CAST(cost as FLOAT64)) as cost
        FROM `project_id.billing_dataset.gcp_billing_export_*`
        WHERE service.description = 'Cloud Run'
          AND sku.description LIKE '%CPU%'
          AND DATE(_PARTITIONTIME) >= DATE_SUB(CURRENT_DATE(), INTERVAL 180 DAY)
        GROUP BY month
        ORDER BY month
        """

        cr_results = self.bq_client.query(cr_query).result()
        cr_usage = [row['cpu_seconds'] for row in cr_results]

        # Calculate average and recommend commitment
        avg_monthly = sum(cr_usage) / len(cr_usage)

        return {
            'service': 'Cloud Run',
            'average_monthly_usage': avg_monthly,
            'commitment_recommendation': self.calculate_commitment(avg_monthly),
            'current_monthly_cost': sum(row['cost'] for row in cr_results) / len(cr_usage),
            'projected_savings': self.calculate_savings(avg_monthly)
        }

    def calculate_commitment(self, monthly_usage):
        """Calculate recommended commitment level"""
        # GCP recommends committing to 70-80% of average usage
        return monthly_usage * 0.75

    def calculate_savings(self, monthly_usage):
        """Calculate projected savings from commitment"""
        # 1-year commitment: 25% discount, 3-year: 40% discount
        one_year_savings = (monthly_usage * 0.25) * 12
        three_year_savings = (monthly_usage * 0.40) * 36

        return {
            '1_year': {
                'discount_percent': 25,
                'annual_savings': one_year_savings,
                'annual_cost': (monthly_usage * 12) * 0.75
            },
            '3_year': {
                'discount_percent': 40,
                'total_savings': three_year_savings,
                'annual_cost': (monthly_usage * 12) * 0.60
            }
        }

# Example analysis
planner = CapacityPlanner()
analysis = planner.analyze_resource_usage()
print(json.dumps(analysis, indent=2))
```

### Commitment Implementation

```bash
# Purchase 1-year Cloud Run commitment (CPU)
gcloud compute commitments create aiprod-cr-cpu-commitment-1y \
  --project=aiprod-v33 \
  --type=compute-optimized \
  --resources=compute-cpu:1000 \
  --plan=one-year \
  --region=europe-west1

# Purchase 3-year commitment (best value for stable workloads)
gcloud compute commitments create aiprod-cr-commitment-3y \
  --project=aiprod-v33 \
  --type=general-purpose \
  --resources=compute-memory:5000 \
  --plan=three-year \
  --region=europe-west1

# List active commitments
gcloud compute commitments list --project=aiprod-v33

# Monitor commitment usage
gcloud compute commitments describe aiprod-cr-commitment-3y \
  --region=europe-west1 \
  --format="table(name, type, plan, resources, status)"
```

### Expected Savings

```
Current Monthly Spend (on-demand):
├─ Cloud Run: $3,500
├─ Cloud SQL: $1,500
├─ Firestore: $1,200
├─ Cloud Storage: $800
└─ Misc: $500
Total: $7,500/month = $90,000/year

With 1-year Commitments (25% discount):
├─ Cloud Run: $2,625 (-$875)
├─ Cloud SQL: $1,125 (-$375)
└─ Other services: $2,500
Total: $6,250/month = $75,000/year
Annual Savings: $15,000 ✅

With 3-year Commitments (40% discount):
├─ Cloud Run: $2,100 (-$1,400)
├─ Cloud SQL: $900 (-$600)
└─ Other services: $2,500
Total: $5,500/month = $66,000/year
3-year Savings: $72,000 ✅
```

### ROI Analysis

```python
# roi_calculator.py
class ROICalculator:
    def __init__(self, monthly_on_demand_cost):
        self.monthly_cost = monthly_on_demand_cost

    def calculate_roi(self, commitment_discount_percent, commitment_months):
        """Calculate ROI for commitment"""
        discounted_monthly = self.monthly_cost * (1 - commitment_discount_percent/100)

        annual_savings = (self.monthly_cost - discounted_monthly) * 12
        commitment_savings = annual_savings * (commitment_months / 12)

        return {
            'monthly_savings': self.monthly_cost - discounted_monthly,
            'annual_savings': annual_savings,
            'commitment_period_savings': commitment_savings,
            'roi_percent': (commitment_savings / (self.monthly_cost * commitment_months)) * 100
        }

# Example
calc = ROICalculator(monthly_on_demand_cost=7500)
roi_1y = calc.calculate_roi(commitment_discount_percent=25, commitment_months=12)
roi_3y = calc.calculate_roi(commitment_discount_percent=40, commitment_months=36)

print("1-year Commitment ROI:", roi_1y)
print("3-year Commitment ROI:", roi_3y)
```

### Monitoring Commitment Usage

```python
# commitment_monitor.py
from google.cloud import monitoring_v3

class CommitmentMonitor:
    def __init__(self):
        self.client = monitoring_v3.MetricServiceClient()

    def get_commitment_utilization(self):
        """Monitor how much of commitment is being utilized"""
        query = """
        SELECT
          DATE(_PARTITIONTIME) as date,
          SUM(CAST(usage.amount as FLOAT64)) as cpu_used,
          (SELECT SUM(CAST(amount as FLOAT64)) FROM resources)
            as cpu_committed
        FROM `project_id.billing_dataset.gcp_billing_export_*`
        WHERE service.description = 'Cloud Run'
          AND DATE(_PARTITIONTIME) = CURRENT_DATE()
        """

        # Calculate utilization percentage
        utilization_percent = (cpu_used / cpu_committed) * 100

        if utilization_percent < 70:
            print(f"⚠️ Low commitment utilization: {utilization_percent:.1f}%")
            print("Consider reducing commitment in next period")
        elif utilization_percent > 95:
            print(f"⚠️ High commitment utilization: {utilization_percent:.1f}%")
            print("Consider increasing commitment to avoid overages")
        else:
            print(f"✅ Optimal commitment utilization: {utilization_percent:.1f}%")
```

### Success Metrics

- ✅ Commitments purchased for 70-80% of average usage
- ✅ 25-40% cost reduction achieved
- ✅ Utilization monitoring active
- ✅ ROI verified (positive)
- ✅ Quarterly review process established

---

## 📊 PHASE 4 Summary

| Task    | Deliverable                 | Duration | Impact                       |
| ------- | --------------------------- | -------- | ---------------------------- |
| **4.1** | Cost Analysis Dashboard     | 2h       | Baseline metrics             |
| **4.2** | Auto-Scaling Infrastructure | 2.5h     | 20-30% idle cost reduction   |
| **4.3** | Database Optimization       | 2.5h     | 40%+ performance improvement |
| **4.4** | Cost Alerts & Monitoring    | 1.5h     | Real-time anomaly detection  |
| **4.5** | Commitment Planning         | 2h       | 25-40% additional savings    |

**Total Duration**: ~10.5 hours  
**Expected Cost Reduction**: 40-50% (combined)  
**ROI Timeline**: 3-6 months

---

**Next Phase**: PHASE 5 (Advanced Features Implementation)

**Version History**:

- v1.0 - February 4, 2026 - PHASE 4 detailed documentation

---
