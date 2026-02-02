---
# ⚡ PHASE 1 - PLAN D'EXÉCUTION FONDATION

**Objectif**: Compléter Phase 1 à **100%** avant de commencer Phase 2

**Status Actuel**: 0% - À Commencer

**Timeline**: 5 Février → 19 Février 2026 (2 semaines)

---

## 🎯 PHASE 1 - OVERVIEW

### Sous-Phases

| Phase     | Task                          | Durée   | Owner          |
| --------- | ----------------------------- | ------- | -------------- |
| **P1.1**  | Persistance: RAM → PostgreSQL | 10h     | Backend        |
| **P1.2**  | Distribution: Ajouter Pub/Sub | 16h     | Backend/DevOps |
| **P1.3**  | Remplacer mocks par réels     | 11h     | Backend        |
| **P1.4**  | CI/CD Pipeline                | 4h      | DevOps         |
| **TOTAL** | **Phase 1 Complete**          | **41h** | **Team**       |

---

## 📋 P1.1 - PERSISTANCE: RAM → POSTGRESQL

**Durée**: 10 heures  
**Owner**: Backend Engineer  
**Blocage**: CRITIQUE pour P1.2

### P1.1.1: Schema PostgreSQL (2h)

**Objectif**: Créer le schema PostgreSQL pour remplacer le job manager en mémoire

**Livérables**:

- [ ] Schema SQL (jobs, job_states, job_results tables)
- [ ] Migrations Alembic setup
- [ ] Connection pooling config
- [ ] Docker-compose update avec PostgreSQL

**Files à Créer/Modifier**:

- `migrations/` - Alembic migrations directory
- `migrations/versions/001_initial_schema.py` - Initial schema
- `src/db/models.py` - SQLAlchemy models
- `docker-compose.yml` - Add postgres service

**Commands**:

```bash
# Initialize Alembic
alembic init migrations

# Create initial migration
alembic revision --autogenerate -m "Initial schema"

# Run migrations
alembic upgrade head
```

**Tests**:

- [ ] PostgreSQL starts in docker-compose
- [ ] Schema created successfully
- [ ] Connection pooling works
- [ ] Tables created with correct columns

---

### P1.1.2: Refactor JobManager (8h)

**Objectif**: Remplacer JobManager en RAM par JobManager PostgreSQL

**Livérables**:

- [ ] New JobManager class with PostgreSQL backend
- [ ] Backward compatible API (same interface)
- [ ] Connection management
- [ ] Transaction handling
- [ ] Unit tests for CRUD operations

**Files à Créer/Modifier**:

- `src/db/job_repository.py` - PostgreSQL repository
- `src/api/icc_manager.py` - Update to use DB backend
- `tests/unit/test_job_repository.py` - Repository tests
- `tests/integration/test_postgres_integration.py` - Integration tests

**Key Methods to Implement**:

```python
# Create job
create_job(job_id, content, preset, user_id)

# Get job state
get_job_state(job_id)

# Update job state
update_job_state(job_id, new_state, metadata)

# Get job results
get_job_results(job_id)

# List jobs (paginated)
list_jobs(user_id, limit, offset)

# Delete job (soft delete)
delete_job(job_id)
```

**Tests**:

- [ ] All CRUD operations work
- [ ] Concurrent access handled correctly
- [ ] Transaction rollback works
- [ ] Migration from RAM → DB doesn't lose data
- [ ] Performance acceptable (< 100ms for most queries)

---

## 📋 P1.2 - DISTRIBUTION: PUB/SUB QUEUE

**Durée**: 16 heures  
**Owner**: Backend + DevOps  
**Blocage**: CRITICAL for async scaling

### P1.2.1: Setup Pub/Sub GCP (2h)

**Objectif**: Créer infrastructure Pub/Sub dans GCP

**Livérables**:

- [ ] GCP Pub/Sub topics created
- [ ] Subscriptions configured
- [ ] Service account IAM permissions
- [ ] Dead-letter topic for failed messages
- [ ] IAM service account updated

**Topics to Create**:

```
aiprod-pipeline-jobs        - Pipeline execution requests
aiprod-pipeline-results     - Pipeline execution results
aiprod-pipeline-dlq         - Dead-letter queue
```

**GCP Commands**:

```bash
# Create topics
gcloud pubsub topics create aiprod-pipeline-jobs
gcloud pubsub topics create aiprod-pipeline-results
gcloud pubsub topics create aiprod-pipeline-dlq

# Create subscriptions
gcloud pubsub subscriptions create aiprod-pipeline-jobs-sub \
  --topic aiprod-pipeline-jobs

gcloud pubsub subscriptions create aiprod-pipeline-results-sub \
  --topic aiprod-pipeline-results

# Add IAM permissions
gcloud pubsub topics add-iam-policy-binding aiprod-pipeline-jobs \
  --member=serviceAccount:aiprod-sa@aiprod-484120.iam.gserviceaccount.com \
  --role=roles/pubsub.publisher

gcloud pubsub topics add-iam-policy-binding aiprod-pipeline-jobs \
  --member=serviceAccount:aiprod-sa@aiprod-484120.iam.gserviceaccount.com \
  --role=roles/pubsub.subscriber
```

**Tests**:

- [ ] Topics exist in Pub/Sub
- [ ] Service account has correct permissions
- [ ] Can publish test message
- [ ] Can receive test message

---

### P1.2.2: Refactor API for Pub/Sub (6h)

**Objectif**: Modifier API pour publier à Pub/Sub au lieu de traiter localement

**Livérables**:

- [ ] Pub/Sub client initialization
- [ ] Message schema definition
- [ ] Refactored `/pipeline/run` endpoint
- [ ] Request validation
- [ ] Error handling with DLQ

**Files à Créer/Modifier**:

- `src/integrations/pubsub_client.py` - Pub/Sub client
- `src/api/main.py` - Update `/pipeline/run` endpoint
- `tests/unit/test_pubsub_client.py` - Client tests

**Message Schema**:

```python
class PipelineJobMessage(BaseModel):
    job_id: str                    # Unique job ID
    user_id: str                   # User email
    content: str                   # Video description
    preset: str                    # quick_social, brand_campaign, premium_spot
    duration_sec: int              # Video duration
    timestamp: datetime            # Creation time
    metadata: Dict[str, Any]       # Extra data
```

**Tests**:

- [ ] Message published successfully
- [ ] DLQ receives failed messages
- [ ] Response to user includes job_id for tracking
- [ ] Load testing with high volume

---

### P1.2.3: Worker Pub/Sub (8h)

**Objectif**: Créer worker qui consomme messages Pub/Sub

**Livérables**:

- [ ] Worker script consuming Pub/Sub messages
- [ ] Pipeline execution
- [ ] Result publication to result topic
- [ ] Error handling and retries
- [ ] Graceful shutdown
- [ ] Metrics collection

**Files à Créer**:

- `src/workers/pipeline_worker.py` - Main worker
- `scripts/start_worker.sh` - Worker startup script
- `Dockerfile.worker` - Worker Docker image
- `kubernetes/worker-deployment.yaml` - K8s deployment

**Worker Flow**:

```
1. Listen to aiprod-pipeline-jobs topic
2. Pull message with job_id + content
3. Execute pipeline (orchestrator.execute())
4. Publish result to aiprod-pipeline-results
5. Acknowledge message to Pub/Sub
6. Handle errors with retry/DLQ
```

**Tests**:

- [ ] Worker processes message correctly
- [ ] Results published to results topic
- [ ] Errors sent to DLQ
- [ ] Retries work correctly
- [ ] Graceful shutdown without data loss
- [ ] Multiple workers don't duplicate work

---

## 📋 P1.3 - REMPLACER MOCKS PAR RÉELS

**Durée**: 11 heures  
**Owner**: Backend Engineer

### P1.3.1: SemanticQA → LLM Réel (4h)

**Objectif**: Remplacer mock SemanticQA par vrai appel Gemini

**Livérables**:

- [ ] Gemini API integration
- [ ] Prompt optimization
- [ ] Caching for common questions
- [ ] Error handling and retries
- [ ] Unit tests

**Files à Modifier**:

- `src/agents/semantic_qa.py` - Implement real Gemini calls

**Implementation**:

```python
# Replace mock response with real Gemini call
import anthropic  # or Google's genai library

class SemanticQA:
    def validate(self, content: str) -> Dict[str, Any]:
        # Call Gemini instead of returning mock
        response = self.gemini_client.generate_content(
            f"Validate this video content: {content}"
        )
        return parse_response(response)
```

**Tests**:

- [ ] Calls Gemini API correctly
- [ ] Parses response correctly
- [ ] Handles API errors gracefully
- [ ] Caching works (same prompt = cached response)

---

### P1.3.2: VisualTranslator → Gemini Real (3h)

**Objectif**: Remplacer mock VisualTranslator par vrai Gemini + image generation

**Livérables**:

- [ ] Gemini Vision API integration
- [ ] Image analysis
- [ ] Scene description generation
- [ ] Error handling

**Files à Modifier**:

- `src/agents/visual_translator.py` - Implement real Gemini Vision

---

### P1.3.3: GCP Integrator → Cloud Storage Real (4h)

**Objectif**: Remplacer mock GCP integrator par vraie Cloud Storage

**Livérables**:

- [ ] Cloud Storage integration
- [ ] Upload/download functions
- [ ] Signed URLs for public access
- [ ] Error handling

**Files à Modifier**:

- `src/agents/gcp_services_integrator.py` - Real Cloud Storage

---

## 📋 P1.4 - CI/CD PIPELINE

**Durée**: 4 heures  
**Owner**: DevOps Engineer

### P1.4.1: GitHub Actions (4h)

**Objectif**: Créer GitHub Actions pipeline pour tests et deployment

**Livérables**:

- [ ] `.github/workflows/test.yml` - Run tests on every PR
- [ ] `.github/workflows/deploy.yml` - Deploy to Cloud Run on merge
- [ ] Coverage reports
- [ ] Lint checks

**Workflows**:

```yaml
# test.yml
- Run pytest
- Run coverage
- Check code formatting
- Run security scans

# deploy.yml
- Build Docker image
- Run tests
- Deploy to Cloud Run dev
- Run smoke tests
```

**Tests**:

- [ ] Tests run on every PR
- [ ] Deployment works automatically
- [ ] Rollback capability

---

## ✅ PHASE 1 COMPLETION CHECKLIST

### Code

- [ ] PostgreSQL JobManager implementation (10h work)
- [ ] Pub/Sub client and worker (16h work)
- [ ] Real LLM implementations (11h work)
- [ ] CI/CD pipelines (4h work)

### Testing

- [ ] All 22 existing tests still passing
- [ ] 30+ new tests for P1 features
- [ ] Integration tests with real services
- [ ] Load testing (100+ jobs/minute)

### Infrastructure

- [ ] PostgreSQL deployed and running
- [ ] Pub/Sub topics/subscriptions created
- [ ] Worker container images built
- [ ] GitHub Actions workflows active

### Documentation

- [ ] Architecture update (database schema diagram)
- [ ] Worker deployment guide
- [ ] Pub/Sub message format documentation
- [ ] CI/CD workflow documentation

### Validation

- [ ] End-to-end test: submit job → process → get results
- [ ] Performance: < 5s API response time
- [ ] Reliability: 0 data loss during job processing
- [ ] Monitoring: Pub/Sub metrics visible in Prometheus

---

## 📅 PHASE 1 TIMELINE

```
Week 1 (5-12 Feb):
├─ Mon 5 Feb: P1.1 PostgreSQL schema + setup
├─ Tue-Wed 6-7 Feb: P1.1 JobManager refactor
├─ Thu-Fri 8-9 Feb: P1.1 testing + P1.2 setup
└─ Sat-Sun 10-11 Feb: Buffer/P1.2 Pub/Sub API

Week 2 (12-19 Feb):
├─ Mon 12 Feb: P1.2 Pub/Sub worker
├─ Tue-Wed 13-14 Feb: P1.3 Mock replacements
├─ Thu-Fri 15-16 Feb: P1.4 CI/CD setup
└─ Sat-Sun 17-18 Feb: Integration testing + fixes
└─ Mon 19 Feb: Phase 1 validation complete
```

---

## 🚀 PHASE 1 → PHASE 2 TRANSITION

**Phase 1 Completion Criteria**:

- ✅ All 4 sub-phases completed
- ✅ All tests passing (22 old + 30+ new)
- ✅ No production bugs introduced
- ✅ Performance acceptable
- ✅ Documentation complete

**Phase 2 Start**: 22 Février 2026

**Phase 2 Topics**:

- P2.1: Logging & Observability (Structured logging, ELK stack)
- P2.2: Tests & Coverage (Unit + Integration + E2E)
- P2.3: Monitoring & Alerting (Prometheus + Grafana + PagerDuty)
- P2.4: Operational Documentation

**Phase 2 Effort**: ~30 hours over 2-3 weeks

---

## 📝 Success Metrics

**By end of Phase 1**:

- Database latency: < 100ms (p99)
- Pub/Sub latency: < 500ms (p99)
- Job throughput: 50+ jobs/minute
- Test coverage: > 80%
- Uptime: 99.9% (during deployment window excluded)

---

✅ **PHASE 1 READY TO START**

**Start Date**: 5 Février 2026  
**Est. Completion**: 19 Février 2026  
**Next Phase**: Phase 2 (22 Février)

Prêt pour P1.1? 🚀
