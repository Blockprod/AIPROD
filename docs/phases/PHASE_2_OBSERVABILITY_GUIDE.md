# Phase 2: Logging & Observabilité - Guide de Démarrage

**Status:** ✅ **PRÊT POUR DÉPLOIEMENT** (Février 2, 2026)

## 📊 Vue d'ensemble Phase 2

Phase 2 ajoute l'observabilité complète au pipeline AIPROD V33:

- **Prometheus:** Collecte des métriques (P50/P95/P99, coûts, erreurs)
- **Grafana:** 3 dashboards (Performance, Coûts, SLA)
- **AlertManager:** Alertes critiques vers Slack/PagerDuty
- **Jaeger:** Distributed tracing des appels Gemini
- **Structured Logging:** Logs JSON vers Google Cloud Logging
- **Custom Metrics:** 12 métriques Prometheus critiques

---

## 🚀 Démarrage Rapide (Local)

### Étape 1: Vérifier l'environnement Python

```bash
# Vérifier .venv311 est activé
(.venv311) PS C:\Users\averr\AIPROD_V33>

# Vérifier les packages Phase 2
pip list | findstr prometheus jaeger
```

**Résultat attendu:**

```
jaeger-client 4.8.0
prometheus-client 0.24.1
prometheus-fastapi-instrumentator 7.1.0
google-cloud-logging 3.13.0
```

### Étape 2: Démarrer Prometheus + Grafana + Jaeger

```bash
# (Optionnel) Nettoyer les anciens containers
docker-compose -f docker-compose.monitoring.yml down -v

# Démarrer la stack d'observabilité
docker-compose -f docker-compose.monitoring.yml up -d

# Vérifier l'état
docker-compose -f docker-compose.monitoring.yml ps
```

**Outputs attendus:**

- `prometheus` → http://localhost:9090
- `grafana` → http://localhost:3000 (admin/admin)
- `alertmanager` → http://localhost:9093
- `jaeger` → http://localhost:16686

### Étape 3: Démarrer l'API FastAPI

```bash
# Terminal 1 - Démarrer l'API avec métriques Prometheus
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 - Vérifier les métriques
curl http://localhost:8000/metrics | head -20
```

---

## 📈 Dashboards Grafana

### Accès Grafana

```
URL: http://localhost:3000
Login: admin
Password: admin
```

### 3 Dashboards Configurés

#### 1️⃣ **Pipeline Performance**

**Clés à surveiller:**

- Execution Rate (jobs/sec)
- P95 Latency (SLA: 300s)
- Active Jobs
- Success Rate by Preset

**Thresholds d'alerte:**

- 🟡 Alerte SLA si P95 > 120s (warning)
- 🔴 Critique si P95 > 300s (SLA breach)

#### 2️⃣ **Cost Dashboard**

**Clés à surveiller:**

- 24h Total Cost ($)
- Cost per Job (moyenne)
- Hourly Trend
- % du Daily Budget ($2000)

**Thresholds d'alerte:**

- 🟡 Alerte si > 70% du budget journalier
- 🔴 Critique si > 90% du budget

#### 3️⃣ **SLA & Error Tracking**

**Clés à surveiller:**

- Success Rate (Target: 99.5%)
- Render Failures Timeline
- QA Gate Acceptance Rate
- HTTP Error Rate by Endpoint

**Thresholds d'alerte:**

- 🟡 Alerte si success < 95%
- 🔴 Critique si success < 90%
- 🔴 Critique si render failures > 5/10min

---

## 🚨 AlertManager Configuration

### Alertes Critiques (10 définies)

1. **PipelineHighErrorRate** → Slack #critical-alerts + PagerDuty
2. **PipelineCostThresholdExceeded** → Slack #sla-alerts
3. **PipelineLatencySLABreach** → Slack #critical-alerts
4. **PubSubQueueDepthHigh** → Slack #infra-alerts
5. **AIAgentTimeouts** → Slack #critical-alerts + PagerDuty
6. **QAGateHighRejectionRate** → Slack #sla-alerts
7. **VideoRenderFailures** → Slack #critical-alerts
8. **DatabaseLatencyHigh** → Slack #infra-alerts
9. **PrometheusDown** → Slack #critical-alerts + PagerDuty
10. **ActiveJobsSpike** → Slack #sla-alerts

### Configuration des Notifications

**Fichier:** `config/alertmanager.yml`

```yaml
# Slack Webhook (env var requis)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

# PagerDuty (env var requis)
PAGERDUTY_SERVICE_KEY=...
```

---

## 📊 Métriques Prometheus Exposées

### 12 Custom Metrics

| Métrique                       | Type      | Labels                     | Utilité               |
| ------------------------------ | --------- | -------------------------- | --------------------- |
| `pipeline_duration_seconds`    | Histogram | status, preset, agent_type | P50/P95/P99 latency   |
| `pipeline_cost_dollars`        | Histogram | status, backend, preset    | Coût par job          |
| `ai_agent_calls_total`         | Counter   | agent_type, status, model  | Gemini usage tracking |
| `render_failures_total`        | Counter   | reason, backend, stage     | Error categorization  |
| `user_jobs_completed_total`    | Counter   | preset, quality_tier       | Success rate          |
| `pipeline_active_jobs`         | Gauge     | status, preset             | Real-time job count   |
| `pubsub_queue_depth`           | Gauge     | topic                      | Queue latency         |
| `ai_agent_latency_seconds`     | Summary   | agent_type, model          | Quantile latency      |
| `qa_gate_acceptance_total`     | Counter   | result, stage              | QA acceptance rate    |
| `video_output_size_bytes`      | Histogram | preset, duration           | Output size dist      |
| `db_operation_latency_seconds` | Histogram | operation, table           | DB latency tracking   |
| `api_response_time_seconds`    | Histogram | method, endpoint, status   | API performance       |

---

## 🔍 Structured Logging vers Google Cloud Logging

### Activation (Configuration Requise)

**Fichier:** `src/utils/structured_logging.py`

```python
from src.utils.structured_logging import (
    logger,
    set_correlation_id,
    set_trace_id,
    set_user_id
)

# Dans chaque request handler
@app.post("/api/pipeline/execute")
async def execute_pipeline(request: PipelineRequest):
    # Générer correlation ID unique pour le job
    cid = set_correlation_id()
    tid = set_trace_id()

    logger.info(
        "Pipeline execution started",
        job_id=job_id,
        preset=request.preset,
        cost_estimate=estimated_cost
    )
```

**Variables d'environnement requises:**

```bash
GOOGLE_CLOUD_PROJECT=aiprod-v33
```

---

## 🔄 Jaeger Distributed Tracing

### Configuration des Traces

**Clients actuellement tracées:**

- `src/agents/semantic_qa.py` → Appels Gemini
- `src/agents/visual_translator.py` → Appels Gemini
- `src/api/main.py` → Endpoints HTTP

### Jaeger UI

```
URL: http://localhost:16686
Services: aiprod-api, prometheus, grafana
```

**Traces utiles:**

1. Pipeline end-to-end trace (input → agents → output)
2. Gemini API call latency distribution
3. Database operations timeline

---

## 📝 Runbooks (À Compléter en P2.4)

Créer des runbooks pour chaque alerte:

```
docs/runbooks/
├── high-error-rate.md          # Triage erreurs pipeline
├── cost-threshold.md            # Optimisation coûts
├── latency-sla.md              # Amélioration performance
├── pubsub-queue.md             # Queue troubleshooting
├── agent-timeout.md            # Agent debugging
├── qa-gate-rejection.md        # Quality troubleshooting
├── render-failures.md          # Render debugging
├── db-latency.md               # Database optimization
├── prometheus-down.md          # Monitoring recovery
└── active-jobs-spike.md        # Load investigation
```

---

## 🧪 Test de l'Observabilité (Local)

### 1. Tester les Métriques Prometheus

```bash
# Vérifier metrics endpoint
curl http://localhost:8000/metrics

# Chercher une métrique spécifique
curl -s http://localhost:8000/metrics | grep pipeline_duration

# Vérifier Prometheus scrape
curl http://localhost:9090/api/v1/query?query=pipeline_duration_seconds_bucket
```

### 2. Déclencher une Alerte Test

```bash
# Simuler une exécution pipeline (gênère des métriques)
curl -X POST http://localhost:8000/api/pipeline/execute \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Test video content",
    "preset": "quick_social",
    "duration_sec": 30
  }'
```

### 3. Vérifier Grafana Dashboards

1. Aller à http://localhost:3000
2. Chercher "AIPROD V33 - Pipeline Performance"
3. Vérifier que les métriques s'affichent (délai ~10-15 secondes)

### 4. Vérifier AlertManager

```bash
# Lister les alertes actuelles
curl http://localhost:9093/api/v1/alerts

# Vérifier la config
curl http://localhost:9093/api/v1/alerts/groups
```

---

## 🐳 Mode Conteneur (Production)

### Déployer sur Cloud Run

```bash
# 1. Build l'image avec observabilité
docker build -t aiprod-v33:latest .

# 2. Push sur GCR
docker tag aiprod-v33:latest gcr.io/aiprod-prod/aiprod-v33:latest
docker push gcr.io/aiprod-prod/aiprod-v33:latest

# 3. Deploy sur Cloud Run avec Prometheus
gcloud run deploy aiprod-v33 \
  --image=gcr.io/aiprod-prod/aiprod-v33:latest \
  --port=8000 \
  --cpu=4 \
  --memory=16Gi \
  --timeout=3600 \
  --env=GOOGLE_CLOUD_PROJECT=aiprod-prod \
  --env=SLACK_WEBHOOK_URL=... \
  --env=PAGERDUTY_SERVICE_KEY=...
```

### Prometheus Remote Write vers Google Cloud Monitoring

Ajouter à `config/prometheus.yml`:

```yaml
remote_write:
  - url: https://monitoring.googleapis.com/api/v1/projects/aiprod-prod/timeSeries
    write_relabel_configs:
      - source_labels: [__name__]
        regex: "pipeline_.*|ai_agent_.*|render_.*"
        action: keep
```

---

## 📋 Checklist Complétion Phase 2

- ✅ Prometheus + Grafana setup (local)
- ✅ 3 Dashboards Grafana créés
- ✅ 12 Custom metrics définies
- ✅ AlertManager rules (10 alertes)
- ✅ Jaeger tracing configuré
- ✅ Structured logging (JSON → Cloud Logging)
- ⏳ Tests d'intégration (P2.4)
- ⏳ Runbooks documentation (P2.4)
- ⏳ Production deployment (P2.4)
- ⏳ Slack/PagerDuty integration (P2.4)

---

## 🎯 Prochaines Étapes (P2.4)

1. **Tester localement** - Exécuter les dashboards avec données réelles
2. **Créer runbooks** - Doc pour chaque alerte critique
3. **Intégrer Slack** - Connecter webhook AlertManager
4. **Deployer production** - Cloud Run + Cloud Monitoring
5. **On-call setup** - PagerDuty escalation policies

---

## 📞 Support

**Logs détaillés:** `logs/aiprod.log`
**Grafana URL:** http://localhost:3000 (admin/admin)
**Prometheus URL:** http://localhost:9090
**Jaeger URL:** http://localhost:16686
