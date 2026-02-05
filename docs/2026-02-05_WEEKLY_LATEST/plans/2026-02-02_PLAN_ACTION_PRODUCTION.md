# 📋 PLAN D'ACTION PRODUCTION — AIPROD_V33

**Document** : Plan d'action complet pour rendre AIPROD_V33 prêt pour la production  
**Date** : 2 février 2026  
**Scope** : Basé sur AUDIT_COMPLET + AUDIT_TECHNIQUE  
**Durée totale estimée** : 6-8 semaines  
**Équipe requise** : 2-3 backend engineers + 1 DevOps/SRE

---

## 📌 Executive Summary

AIPROD_V33 est une plateforme beta bien architecturée mais **non productible** en l'état. Les audits identifient **4 risques critiques** et **6 améliorations majeures**. Ce plan priorise les actions urgentes (24-48h), puis les blocages structurels (1-2 semaines), et enfin les optimisations (mois 1-2).

| Phase              | Durée   | Risques levés | Effort |
| ------------------ | ------- | ------------- | ------ |
| **0 - Critique**   | 24-48h  | Sécurité      | 10j    |
| **1 - Fondation**  | 1-2 sem | Scalabilité   | 15j    |
| **2 - Robustesse** | 2-3 sem | Fiabilité     | 10j    |
| **3 - Production** | 1 mois  | Opérationnel  | 15j    |

---

# 🔴 PHASE 0 — CRITIQUES (24-48 HEURES)

## P0.1 Sécurité : Secrets exposés

### Issue

- ✋ **Sévérité** : CRITIQUE
- 📍 **Fichier** : `.env`
- 🔓 **Exposition** : Clés API Gemini, Runway, Datadog, credentials GCP
- ⚠️ **Impact** : Compromission immédiate si versionné

### Actions

#### P0.1.1 Audit & Révocation (2h)

```bash
# 1. Vérifier historique git pour expositions
git log --oneline --all -- .env
git log -p --all -S "AIzaSy" | head -100  # Scanner clés Gemini
git log -p --all -S "key_" | head -100     # Scanner clés Runway

# 2. Revérifier le .env actuel
cat .env | grep -E "API_KEY|SECRET|CREDENTIALS"

# 3. Révoquer les clés compromises
# → GCP Console : Disable service account keys
# → Gemini API Console : Regenerate API keys
# → Runway ML : Regenerate API keys
# → Datadog : Regenerate API keys
```

**Checklist** :

- [ ] Toutes les clés git history supprimées (git filter-branch ou GitHub Tool)
- [ ] Clés GCP révoquées dans IAM
- [ ] Clés Gemini révoquées
- [ ] Clés Runway révoquées
- [ ] Clés Datadog révoquées
- [ ] `.env` ajouté à `.gitignore` (si pas déjà fait)

#### P0.1.2 Migration Secret Manager (3h)

```bash
# 1. Créer Secret Manager GCP
gcloud secrets create GEMINI_API_KEY --replication-policy="automatic"
gcloud secrets create RUNWAY_API_KEY --replication-policy="automatic"
gcloud secrets create DATADOG_API_KEY --replication-policy="automatic"
gcloud secrets create GCS_BUCKET_NAME --replication-policy="automatic"

# 2. Stocker les nouvelles clés
echo "AIzaSy_NEW_KEY" | gcloud secrets versions add GEMINI_API_KEY --data-file=-

# 3. Créer .env.example (sans valeurs)
cat > .env.example << 'EOF'
# Récupérés depuis GCP Secret Manager
GEMINI_API_KEY=<from GCP Secret Manager>
RUNWAY_API_KEY=<from GCP Secret Manager>
DATADOG_API_KEY=<from GCP Secret Manager>
GCS_BUCKET_NAME=<from GCP Secret Manager>
EOF
git add .env.example && git commit -m "chore: add .env.example template"
```

#### P0.1.3 Code : Charger secrets runtime (2h)

Fichier : `src/config/secrets.py` (créer)

```python
import os
from google.cloud import secretmanager

def get_secret(secret_id: str) -> str:
    """Charge un secret depuis GCP Secret Manager."""
    project_id = os.getenv("GCP_PROJECT_ID", "aiprod-484120")
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

def load_secrets():
    """Charge tous les secrets au démarrage."""
    os.environ["GEMINI_API_KEY"] = get_secret("GEMINI_API_KEY")
    os.environ["RUNWAY_API_KEY"] = get_secret("RUNWAY_API_KEY")
    os.environ["DATADOG_API_KEY"] = get_secret("DATADOG_API_KEY")
    os.environ["GCS_BUCKET_NAME"] = get_secret("GCS_BUCKET_NAME")
```

Fichier : `src/api/main.py` (modifié)

```python
# Au démarrage de l'app
from src.config.secrets import load_secrets

@app.on_event("startup")
async def startup_event():
    load_secrets()
    logger.info("Secrets chargés depuis GCP Secret Manager")
```

**Checklist** :

- [ ] Secret Manager GCP configuré
- [ ] 5 secrets créés + migrés
- [ ] Code de chargement implémenté
- [ ] Tests avec mock secrets
- [ ] `.env` supprimé du repo
- [ ] Documentation mise à jour

---

## P0.2 Sécurité : Pas d'authentification API

### Issue

- ✋ **Sévérité** : CRITIQUE
- 📍 **Endpoints affectés** : `/pipeline/run`, `/metrics`, `/alerts`, `/icc/data`
- ⚠️ **Impact** : DDOS, modification d'état, fuite de données

### Actions

#### P0.2.1 Ajouter JWT + Firebase Auth (6h)

**Installation**

```bash
pip install firebase-admin python-jose
```

Fichier : `src/auth/firebase_auth.py` (créer)

```python
import os
import firebase_admin
from firebase_admin import credentials, auth

# Initialiser Firebase (credentials via Secret Manager)
if not firebase_admin.get_app():
    cred = credentials.Certificate({
        "project_id": os.getenv("GCP_PROJECT_ID"),
        # ... charger depuis Secret Manager
    })
    firebase_admin.initialize_app(cred)

def verify_token(token: str) -> dict:
    """Vérifie un token JWT Firebase."""
    try:
        decoded = auth.verify_id_token(token)
        return decoded
    except Exception as e:
        raise ValueError(f"Token invalide: {e}")

def verify_api_key(api_key: str) -> bool:
    """Vérifie une API key (alternative JWT)."""
    # Implémenter vérification avec base de données
    # Pour MVP : liste blanche de clés
    pass
```

Fichier : `src/api/auth_middleware.py` (créer)

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthCredentials
from src.auth.firebase_auth import verify_token

security = HTTPBearer()

async def verify_request(credentials: HTTPAuthCredentials = Depends(security)):
    """Middleware pour vérifier authentification."""
    token = credentials.credentials
    try:
        user = verify_token(token)
        return user
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalide",
        )
```

Fichier : `src/api/main.py` (modifié)

```python
from src.api.auth_middleware import verify_request

# Routes protégées
@app.post("/pipeline/run")
async def run_pipeline(
    request: PipelineRequest,
    user = Depends(verify_request)  # ✅ Auth requis
):
    logger.info(f"Pipeline lancé par {user['uid']}")
    # ... reste du code

# Routes publiques
@app.get("/health")
async def health():
    return {"status": "ok"}

# Routes metrics (optionnellement protégées)
@app.get("/metrics")
async def get_metrics(user = Depends(verify_request)):
    return metrics_collector.get_internal_metrics()
```

**Checklist** :

- [ ] Firebase project créé + configuré
- [ ] Firebase Auth middleware implémenté
- [ ] JWT verification testé
- [ ] API protégée `/pipeline/run`
- [ ] Documentation auth mise à jour
- [ ] Client frontend peut obtenir token

---

## P0.3 Sécurité : Passwords/configs en dur

### Issue

- 📍 **Fichiers** : `docker-compose.yml` (Grafana password = `admin`)
- ⚠️ **Impact** : Accès non autorisé Grafana, risque données

### Actions

#### P0.3.1 Grafana (1h)

```yaml
# docker-compose.yml (modifié)
services:
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3030:3000"
    volumes:
      - ./config/grafana:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD} # ✅ Depuis env/secret
      - GF_SERVER_ROOT_URL=https://grafana.aiprod.prod # ✅ HTTPS
      - GF_SECURITY_COOKIE_SECURE=true # ✅ Secure cookies
      - GF_SECURITY_COOKIE_HTTPONLY=true
    restart: unless-stopped
```

**Checklist** :

- [ ] Password Grafana changé (> 16 chars, mixed case)
- [ ] Stocké en Secret Manager
- [ ] `docker-compose.yml` mis à jour
- [ ] HTTPS forcé Grafana
- [ ] Accès IP restreint

---

## P0.4 Sécurité : Audit log manquant

### Actions (2h)

Fichier : `src/security/audit_logger.py` (créer)

```python
import logging
import json
from datetime import datetime
from enum import Enum

class AuditAction(str, Enum):
    API_CALL = "API_CALL"
    PIPELINE_START = "PIPELINE_START"
    PIPELINE_COMPLETE = "PIPELINE_COMPLETE"
    SECRET_ACCESS = "SECRET_ACCESS"
    ERROR = "ERROR"

audit_logger = logging.getLogger("audit")

def log_audit(action: AuditAction, user_id: str, details: dict):
    """Enregistre une action pour audit trail."""
    audit_logger.info(json.dumps({
        "timestamp": datetime.utcnow().isoformat(),
        "action": action.value,
        "user_id": user_id,
        "details": details,
    }))
```

Fichier : `src/api/main.py` (modifié)

```python
from src.security.audit_logger import log_audit, AuditAction

@app.post("/pipeline/run")
async def run_pipeline(request: PipelineRequest, user = Depends(verify_request)):
    log_audit(
        AuditAction.PIPELINE_START,
        user["uid"],
        {"content": request.content[:50], "preset": request.preset}
    )
    # ... reste du code
```

**Checklist** :

- [ ] Audit logger implémenté
- [ ] Tous les endpoints critiques loggent
- [ ] Logs exportés vers Cloud Logging

---

## 📋 P0 Summary

| Action                  | Durée   | Owner      | Status |
| ----------------------- | ------- | ---------- | ------ |
| P0.1 - Secrets exposés  | 7h      | Backend    | [ ]    |
| P0.2 - Auth API         | 6h      | Backend    | [ ]    |
| P0.3 - Grafana password | 1h      | DevOps     | [ ]    |
| P0.4 - Audit logging    | 2h      | Backend    | [ ]    |
| **Total P0**            | **16h** | **24-48h** | [ ]    |

**Validation** :

```bash
# Vérifier tous les secrets sont hors du code
git grep "AIzaSy" -- ':!.env.example'  # Doit être vide
git grep "key_" -- ':!.env.example'

# Vérifier API requiert auth
curl http://localhost:8000/pipeline/run  # Doit retourner 401
curl -H "Authorization: Bearer TOKEN" http://localhost:8000/pipeline/run  # Doit passer auth

# Vérifier Grafana password changé
curl http://localhost:3030 -u admin:admin  # Doit échouer
```

---

# 🟠 PHASE 1 — FONDATION (1-2 SEMAINES)

## P1.1 Persistance : Remplacer JobManager RAM par PostgreSQL

### Issue

- 📍 **Actuel** : `src/api/icc_manager.py:JobManager._jobs` en Dict RAM
- ⚠️ **Impact** : Perte d'état au redémarrage, pas de multi-instance

### Actions

#### P1.1.1 Schema PostgreSQL (2h)

Fichier : `migrations/001_create_jobs_table.sql` (créer)

```sql
CREATE TABLE IF NOT EXISTS jobs (
    id VARCHAR(36) PRIMARY KEY,
    content TEXT NOT NULL,
    preset VARCHAR(50),
    state VARCHAR(50) NOT NULL,
    priority VARCHAR(20) DEFAULT 'low',
    lang VARCHAR(10) DEFAULT 'en',
    brand_id VARCHAR(255),

    production_manifest JSONB,
    consistency_markers JSONB,
    cost_estimate JSONB,
    render_result JSONB,
    qa_report JSONB,

    approved BOOLEAN DEFAULT FALSE,
    approval_timestamp TIMESTAMP,
    edits_history JSONB DEFAULT '[]'::jsonb,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_state (state),
    INDEX idx_created_at (created_at),
    INDEX idx_brand_id (brand_id)
);

CREATE TABLE IF NOT EXISTS job_events (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR(36) REFERENCES jobs(id),
    event_type VARCHAR(50),
    event_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_job_id (job_id),
    INDEX idx_created_at (created_at)
);
```

#### P1.1.2 Refactor JobManager (8h)

Fichier : `src/persistence/db.py` (créer)

```python
import os
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://aiprod:password@localhost:5432/aiprod_v33"
)

engine = create_engine(
    DATABASE_URL,
    echo=False,
    poolclass=NullPool,  # Cloud Run serverless
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

Fichier : `src/persistence/models.py` (créer)

```python
from sqlalchemy import Column, String, Boolean, DateTime, JSON, func
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class JobModel(Base):
    __tablename__ = "jobs"

    id = Column(String(36), primary_key=True)
    content = Column(String, nullable=False)
    preset = Column(String(50))
    state = Column(String(50), nullable=False)
    priority = Column(String(20), default="low")
    lang = Column(String(10), default="en")
    brand_id = Column(String(255))

    production_manifest = Column(JSON)
    consistency_markers = Column(JSON)
    cost_estimate = Column(JSON)
    render_result = Column(JSON)
    qa_report = Column(JSON)

    approved = Column(Boolean, default=False)
    approval_timestamp = Column(DateTime)
    edits_history = Column(JSON, default=[])

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
```

Fichier : `src/api/icc_manager.py` (refactorisé)

```python
from sqlalchemy.orm import Session
from src.persistence.models import JobModel

class JobManager:
    def __init__(self, db: Session):
        self.db = db

    async def create_job(self, content: str, **kwargs) -> JobModel:
        """Crée un job en base de données."""
        job = JobModel(
            id=str(uuid.uuid4())[:8],
            content=content,
            state=JobState.CREATED,
            **kwargs
        )
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        return job

    async def get_job(self, job_id: str) -> JobModel:
        return self.db.query(JobModel).filter(JobModel.id == job_id).first()

    async def update_job_state(self, job_id: str, new_state: JobState):
        job = self.db.query(JobModel).filter(JobModel.id == job_id).first()
        if job:
            job.state = new_state
            job.updated_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(job)
        return job
```

Fichier : `src/api/main.py` (modifié)

```python
from sqlalchemy.orm import Session
from src.persistence.db import get_db
from src.api.icc_manager import JobManager

@app.post("/pipeline/run")
async def run_pipeline(
    request: PipelineRequest,
    user = Depends(verify_request),
    db: Session = Depends(get_db)
):
    job_manager = JobManager(db)
    job = await job_manager.create_job(
        content=request.content,
        priority=request.priority,
        lang=request.lang
    )
    # ... exécute pipeline avec db persistence
```

**Checklist** :

- [ ] PostgreSQL deployed (Cloud SQL recommandé)
- [ ] Migrations exécutées
- [ ] Models SQLAlchemy créés
- [ ] JobManager refactorisé + testé
- [ ] Connection pooling configuré
- [ ] Backups automatiques configurés

---

## P1.2 Distribution : Ajouter queue Pub/Sub

### Issue

- 📍 **Actuel** : Rendu synchrone bloque l'API
- ⚠️ **Impact** : Pas de scalabilité, timeout sur requêtes longues

### Actions

#### P1.2.1 Setup Pub/Sub GCP (2h)

```bash
# Créer topics
gcloud pubsub topics create aiprod-pipeline-requests
gcloud pubsub topics create aiprod-pipeline-results

# Créer subscriptions
gcloud pubsub subscriptions create aiprod-render-worker \
  --topic aiprod-pipeline-requests \
  --push-endpoint=https://render-worker.aiprod.prod/process

gcloud pubsub subscriptions create aiprod-results-processor \
  --topic aiprod-pipeline-results
```

#### P1.2.2 Refactor API pour Pub/Sub (6h)

Fichier : `src/queue/publisher.py` (créer)

```python
from google.cloud import pubsub_v1
import json

publisher = pubsub_v1.PublisherClient()
PROJECT_ID = "aiprod-484120"
TOPIC_ID = "aiprod-pipeline-requests"

async def publish_job(job_id: str, pipeline_request: dict):
    """Publie un job dans la queue."""
    topic_path = publisher.topic_path(PROJECT_ID, TOPIC_ID)
    message_json = json.dumps({
        "job_id": job_id,
        "request": pipeline_request,
    })
    future = publisher.publish(topic_path, message_json.encode('utf-8'))
    message_id = future.result()
    logger.info(f"Job {job_id} publié: {message_id}")
    return message_id
```

Fichier : `src/api/main.py` (modifié)

```python
from src.queue.publisher import publish_job

@app.post("/pipeline/run")
async def run_pipeline(
    request: PipelineRequest,
    user = Depends(verify_request),
    db: Session = Depends(get_db)
):
    # Créer job en DB
    job_manager = JobManager(db)
    job = await job_manager.create_job(content=request.content, ...)

    # Publier dans queue
    await publish_job(job.id, request.model_dump())

    # Retourner immédiatement au client
    return {
        "job_id": job.id,
        "status": "queued",
        "message": "Pipeline lancé, vous recevrez une notification à la completion"
    }
```

#### P1.2.3 Worker Pub/Sub (8h)

Fichier : `src/workers/render_worker.py` (créer)

```python
from google.cloud import pubsub_v1
from src.orchestrator.state_machine import StateMachine
from src.persistence.db import SessionLocal

subscriber = pubsub_v1.SubscriberClient()
PROJECT_ID = "aiprod-484120"
SUBSCRIPTION_ID = "aiprod-render-worker"

subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)

def process_job(message):
    """Traite un job de la queue."""
    try:
        data = json.loads(message.data.decode('utf-8'))
        job_id = data["job_id"]
        request = data["request"]

        # Charger job depuis DB
        db = SessionLocal()
        job = db.query(JobModel).filter(JobModel.id == job_id).first()

        if not job:
            logger.error(f"Job {job_id} not found")
            message.ack()
            return

        # Exécuter pipeline
        state_machine = StateMachine()
        result = asyncio.run(state_machine.run(request))

        # Mettre à jour job
        job.state = PipelineState.DELIVERED
        job.render_result = result
        db.commit()

        logger.info(f"Job {job_id} completed")
        message.ack()  # Acknowledge après succès

    except Exception as e:
        logger.error(f"Job processing failed: {e}")
        message.nack()  # Requeue si erreur

def listen():
    """Écoute la queue indéfiniment."""
    streaming_pull_future = subscriber.subscribe(subscription_path, process_job)
    logger.info(f"Listening on {subscription_path}")

    try:
        streaming_pull_future.result()
    except KeyboardInterrupt:
        streaming_pull_future.cancel()
```

Fichier : `scripts/run_render_worker.py` (créer)

```python
#!/usr/bin/env python
import asyncio
from src.workers.render_worker import listen

if __name__ == "__main__":
    asyncio.run(listen())
```

**Checklist** :

- [ ] Pub/Sub topics créés
- [ ] API refactorisée (async return)
- [ ] Worker implémenté + testé
- [ ] Worker déployé en Cloud Run (job)
- [ ] Queue monitoring en place
- [ ] Dead letter queue configurée

---

## P1.3 Remplacer mocks par implémentations réelles

### Issue

- 📍 **Mocks** : `SemanticQA`, `VisualTranslator`, `GCP Integrator`
- ⚠️ **Impact** : Résultats non fiables

### Actions

#### P1.3.1 SemanticQA → LLM réel (4h)

Fichier : `src/agents/semantic_qa.py` (refactorisé)

```python
import asyncio
import google.generativeai as genai
from src.utils.monitoring import logger

class SemanticQA:
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-1.5-pro-vision")

    async def run(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Valide sémantiquement les outputs avec LLM réel."""
        try:
            render_output = outputs.get("render", {})
            video_url = render_output.get("video_url", "")

            prompt = f"""Analysez cette vidéo générée par IA:
            - URL: {video_url}
            - Prompt original: {outputs.get('prompt', '')}

            Évaluez:
            1. Qualité visuelle (0-1)
            2. Pertinence au prompt (0-1)
            3. Artefacts/erreurs?
            4. Note globale (0-1)

            Format JSON: {{"quality": X, "relevance": X, "artifacts": [], "score": X}}
            """

            response = await self.model.generate_content_async(prompt)
            result = json.loads(response.text)

            logger.info(f"SemanticQA: quality={result['quality']}, relevance={result['relevance']}")

            return {
                "semantic_valid": result["score"] >= 0.7,
                "quality_score": result["quality"],
                "relevance_score": result["relevance"],
                "artifacts": result.get("artifacts", []),
                "overall_score": result["score"],
            }
        except Exception as e:
            logger.error(f"SemanticQA error: {e}")
            return {"semantic_valid": False, "error": str(e)}
```

#### P1.3.2 VisualTranslator → Gemini real (3h)

Similar to SemanticQA, mais avec prompts de traduction visuelle.

#### P1.3.3 GCP Integrator → Cloud Storage real (4h)

Fichier : `src/agents/gcp_services_integrator.py` (refactorisé)

```python
from google.cloud import storage
import asyncio

class GoogleCloudServicesIntegrator:
    def __init__(self):
        self.storage_client = storage.Client()
        self.bucket_name = os.getenv("GCS_BUCKET_NAME")

    async def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Upload réel vers GCS."""
        try:
            bucket = self.storage_client.bucket(self.bucket_name)

            # Upload video
            video_blob = bucket.blob(f"videos/{inputs['job_id']}/output.mp4")
            video_blob.upload_from_filename(inputs["video_path"])

            # Generate signed URL (7 jours)
            video_url = video_blob.generate_signed_url(
                version="v4",
                expiration=datetime.timedelta(days=7),
                method="GET"
            )

            return {
                "status": "uploaded",
                "video_url": video_url,
                "storage_path": f"gs://{self.bucket_name}/videos/{inputs['job_id']}/output.mp4",
            }
        except Exception as e:
            logger.error(f"GCP upload error: {e}")
            return {"status": "error", "error": str(e)}
```

**Checklist** :

- [ ] SemanticQA implémenté + testé
- [ ] VisualTranslator implémenté + testé
- [ ] GCP Integrator implémenté + testé
- [ ] Error handling pour API failures
- [ ] Fallback stratégies en place

---

## P1.4 CI/CD Pipeline

### Issue

- 📍 **Actuellement** : Pas de CI/CD
- ⚠️ **Impact** : Déploiements manuels, risqué

### Actions

#### P1.4.1 GitHub Actions (4h)

Fichier : `.github/workflows/test.yml` (créer)

```yaml
name: Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov black ruff

      - name: Lint with Ruff
        run: ruff check src/

      - name: Format check with Black
        run: black --check src/

      - name: Run tests
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost/aiprod_test
        run: pytest tests/ -v --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

Fichier : `.github/workflows/deploy.yml` (créer)

```yaml
name: Deploy to Production

on:
  push:
    branches: [main]
    tags: [v*]

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v1
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}

      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v1

      - name: Build and push Docker image
        run: |
          gcloud builds submit \
            --tag gcr.io/${{ secrets.GCP_PROJECT }}/aiprod-api:${{ github.sha }} \
            --substitutions _IMAGE_NAME=aiprod-api

      - name: Deploy to Cloud Run
        run: |
          gcloud run deploy aiprod-api \
            --image gcr.io/${{ secrets.GCP_PROJECT }}/aiprod-api:${{ github.sha }} \
            --platform managed \
            --region us-central1 \
            --allow-unauthenticated=false
```

**Checklist** :

- [ ] GitHub Actions configuré
- [ ] Tests exécutés à chaque PR
- [ ] Coverage rapporté
- [ ] Linting en place (Ruff, Black)
- [ ] Déploiement automatique main → prod

---

## 📋 P1 Summary

| Action               | Durée   | Owner          | Status |
| -------------------- | ------- | -------------- | ------ |
| P1.1 - PostgreSQL    | 10h     | Backend        | [ ]    |
| P1.2 - Pub/Sub queue | 16h     | Backend/DevOps | [ ]    |
| P1.3 - Mocks → réels | 11h     | Backend        | [ ]    |
| P1.4 - CI/CD         | 4h      | DevOps         | [ ]    |
| **Total P1**         | **41h** | **1-2 sem**    | [ ]    |

---

# 🟡 PHASE 2 — ROBUSTESSE (2-3 SEMAINES)

## P2.1 Logging & Observabilité

### Actions (8h)

#### P2.1.1 Structured JSON Logging

Fichier : `src/config/logging_config.py` (créer)

```python
import logging
import json
import sys
from pythonjsonlogger import jsonlogger

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Stdout (pour Cloud Logging)
    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
```

#### P2.1.2 OpenTelemetry Tracing

```bash
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-gcp-trace
```

Fichier : `src/config/tracing.py` (créer)

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.gcp_trace import CloudTraceExporter

exporter = CloudTraceExporter()
trace.set_tracer_provider(TracerProvider(resource_attributes={"service.name": "aiprod"}))
trace.get_tracer_provider().add_span_processor(
    opentelemetry.sdk.trace.export.SimpleSpanProcessor(exporter)
)

tracer = trace.get_tracer(__name__)
```

**Checklist** :

- [ ] JSON logs configurés
- [ ] Cloud Logging reçoit les logs
- [ ] Tracing exporté en Cloud Trace
- [ ] Latency monitored

---

## P2.2 Tests & Couverture

### Actions (10h)

- [ ] Ajouter tests security (injection, auth bypass)
- [ ] Tests de charge (k6 ou locust)
- [ ] Tests de concurrence (multi-jobs)
- [ ] Coverage > 80% pour core logic
- [ ] Mutation testing (pit)

---

## P2.3 Monitoring & Alerting

### Actions (6h)

Fichier : `deployments/monitoring.yaml` (mise à jour)

```yaml
alertPolicy:
  displayName: "AIPROD High Latency"
  conditions:
    - displayName: "Pipeline latency > 60s"
      conditionThreshold:
        filter: |
          resource.type="cloud_run_revision"
          metric.type="custom.googleapis.com/pipeline_latency_ms"
        comparison: COMPARISON_GT
        thresholdValue: 60000
        duration: 300s
  notificationChannels:
    - projects/aiprod-484120/notificationChannels/123456 # Slack/Email
```

**Checklist** :

- [ ] Alertes PagerDuty/Slack
- [ ] Seuils SLO définis
- [ ] Dashboard Grafana créé
- [ ] On-call rotation en place

---

## P2.4 Documentation Opérationnel

### Actions (6h)

Créer fichiers :

- `docs/RUNBOOK.md` - Incident response
- `docs/DEPLOYMENT.md` - Deploy procedure
- `docs/TROUBLESHOOTING.md` - Common issues
- `docs/SECURITY.md` - Security practices

**Checklist** :

- [ ] Runbook complet
- [ ] Deployment checklist
- [ ] Incident templates
- [ ] On-call guide

---

## 📋 P2 Summary

| Action          | Durée   | Owner       | Status |
| --------------- | ------- | ----------- | ------ |
| P2.1 - Logging  | 8h      | Backend     | [ ]    |
| P2.2 - Tests    | 10h     | Backend     | [ ]    |
| P2.3 - Alerting | 6h      | DevOps      | [ ]    |
| P2.4 - Docs     | 6h      | Tech Lead   | [ ]    |
| **Total P2**    | **30h** | **2-3 sem** | [ ]    |

---

# 🟢 PHASE 3 — PRODUCTION (1 MOIS)

## P3.1 Infrastructure as Code (Terraform)

### Actions (12h)

Fichier : `terraform/main.tf` (créer)

```hcl
terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
}

# Cloud SQL (PostgreSQL)
resource "google_sql_database_instance" "aiprod" {
  name             = "aiprod-db"
  database_version = "POSTGRES_15"
  region           = var.gcp_region

  settings {
    tier              = "db-custom-2-8192"
    availability_type = "REGIONAL"
    backup_configuration {
      enabled = true
      backup_retention_settings {
        retained_backups = 30
      }
    }
  }
}

# Cloud Run
resource "google_cloud_run_service" "aiprod_api" {
  name     = "aiprod-api"
  location = var.gcp_region

  template {
    spec {
      containers {
        image = var.docker_image
        env {
          name  = "DATABASE_URL"
          value = google_sql_database_instance.aiprod.connection_name
        }
      }
      service_account_name = google_service_account.aiprod.email
    }
  }
}

# Cloud Monitoring Uptime Check
resource "google_monitoring_uptime_check_config" "aiprod" {
  display_name = "AIPROD API Uptime"
  timeout      = "10s"
  period       = "60s"

  http_check {
    path = "/health"
    port = 443
  }

  selected_regions = ["USA", "EUROPE", "ASIA_PACIFIC"]
}
```

**Checklist** :

- [ ] Terraform code écrit + documenté
- [ ] État Terraform distant (GCS)
- [ ] Déploiement via Terraform approuvé
- [ ] Staging environnement en Terraform

---

## P3.2 Scalabilité & Performance

### Actions (10h)

- [ ] Horizontal scaling tested (5+ instances)
- [ ] Database connection pooling (PgBouncer)
- [ ] Redis cache layer
- [ ] CDN pour assets vidéo (Cloud CDN)
- [ ] Load testing reproductible (k6 + CI)

---

## P3.3 Disaster Recovery

### Actions (8h)

- [ ] Backup strategy documentée
- [ ] RTO/RPO défini
- [ ] Failover procedure tested
- [ ] Secrets rotation policy
- [ ] Data retention policy

---

## P3.4 Cost Optimization

### Actions (6h)

- [ ] Cost analysis (Recommender)
- [ ] Reserved instances pour Cloud SQL
- [ ] Spot VMs pour workers
- [ ] Budget alerts >150% budget

---

## 📋 P3 Summary

| Action             | Durée   | Owner           | Status |
| ------------------ | ------- | --------------- | ------ |
| P3.1 - Terraform   | 12h     | DevOps          | [ ]    |
| P3.2 - Scalabilité | 10h     | Backend/DevOps  | [ ]    |
| P3.3 - DR          | 8h      | DevOps/Security | [ ]    |
| P3.4 - Costs       | 6h      | DevOps/PM       | [ ]    |
| **Total P3**       | **36h** | **1 mois**      | [ ]    |

---

# 📊 TIMELINE GLOBALE

```
JOUR 1-2        | P0 - Sécurité (16h)       [████████████████]
SEMAINE 1-2     | P1 - Fondation (41h)      [████████████████████████████████████████]
SEMAINE 3-4     | P2 - Robustesse (30h)     [██████████████████████████████]
SEMAINE 5-8     | P3 - Production (36h)     [████████████████████████████████████]

TOTAL: 16h + 41h + 30h + 36h = 123 jours-homme (~6-8 semaines, 2-3 personnes)
```

---

# ✅ VALIDATION & SIGN-OFF

## Pre-Production Checklist

### Sécurité

- [ ] Secrets hors du code (Secret Manager)
- [ ] Auth API (Firebase/JWT)
- [ ] HTTPS forcé
- [ ] Audit trail complet
- [ ] Vulnérabilité scan passé (Trivy)

### Scalabilité

- [ ] PostgreSQL avec replicas
- [ ] Pub/Sub queue en place
- [ ] Horizontal scaling testé
- [ ] Load test passé (1000 req/s)
- [ ] Database queries optimisées

### Fiabilité

- [ ] Tests coverage > 80%
- [ ] Mocks remplacés
- [ ] Error handling normalisé
- [ ] Retry logic testé
- [ ] Health checks en place

### Opérationnel

- [ ] CI/CD pipeline automatisé
- [ ] Logs JSON en Cloud Logging
- [ ] Monitoring + alerting actifs
- [ ] Runbook écrit + testé
- [ ] On-call rotation définie

### Coût

- [ ] Budget mensuel < $500
- [ ] Cost alerts configurées
- [ ] Reserved instances réservées

## Sign-off

| Role          | Responsable     | Sign-off |
| ------------- | --------------- | -------- |
| **Architect** | Backend Lead    | [ ]      |
| **DevOps**    | Cloud Engineer  | [ ]      |
| **Security**  | Security Lead   | [ ]      |
| **PM**        | Product Manager | [ ]      |

---

# 🎯 POST-PROD (SEMAINES 9-12)

## Monitoring & Optimization

- Analyser métriques réelles
- Tuning performance database
- Coûts réels vs prévisions
- User feedback collection
- Roadmap Phase 4

---

**Document signé le** : 2 février 2026  
**Version** : 1.0  
**Prochain review** : Post-P0 (48h)
