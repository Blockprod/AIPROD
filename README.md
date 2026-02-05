# 🎬 AIPROD v3.3 - Pipeline de Génération Vidéo IA

**Une plateforme complète et orchestrée pour la génération, composition et traitement de contenu vidéo avec IA**

![Version](https://img.shields.io/badge/version-3.3.0-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![License](https://img.shields.io/badge/license-MIT-black)
![Status](https://img.shields.io/badge/status-Production%20Ready-brightgreen)

---

## 📋 Table des Matières

- [Vue d'ensemble](#vue-densemble)
- [Fonctionnalités](#fonctionnalités)
- [Architecture](#architecture)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Configuration](#configuration)
- [Utilisation](#utilisation)
- [API REST](#api-rest)
- [Tests](#tests)
- [Déploiement](#déploiement)
- [Documentation](#documentation)
- [Support](#support)

---

## 🎯 Vue d'ensemble

**AIPROD** est une plateforme enterprise pour la génération et le traitement de contenu vidéo haut de gamme utilisant l'IA. Elle orchestrate un ensemble complexe d'agents spécialisés (directeur créatif, compositeur musical, traitement vidéo, etc.) avec un système de validation QA intégré.

### Points clés

✨ **Orchestration Multi-Agents** - Coordination automatique de 10+ agents spécialisés  
🎥 **Pipeline Vidéo Complet** - Du concept à la vidéo finale en qualité 4K  
💰 **Gestion Financière Intégrée** - Suivi des coûts et estimations en temps réel  
🔐 **Authentification d'Entreprise** - Firebase Auth + API Keys + JWT Token  
📊 **Observabilité Complète** - Prometheus, Jaeger, Google Cloud Logging  
🚀 **Déploiement Cloud** - Cloud Run, Cloud Functions, Docker  
♻️ **Haute Disponibilité** - Redis Cache, Disaster Recovery, Load Balancing

---

## ✨ Fonctionnalités

### 🎨 Générateurs de Contenu

- **Creative Director** - Direction créative et concept visual
- **Visual Translator** - Conversion de prompts en visual assets
- **Audio Generator** - Génération de voix et dialogues
- **Music Composer** - Composition de musique de fond
- **Sound Effects Agent** - Effets sonores spécialisés

### 🎬 Traitement Vidéo

- **Post Processor** - Édition, montage, couleur grading
- **Render Executor** - Rendu haute qualité (4K)
- **Fast Track Agent** - Mode accéléré pour prototypes

### ✅ Assurance Qualité

- **Technical QA Gate** - Validation technique du pipeline
- **Semantic QA** - Validation sémantique du contenu
- **Consistency Cache** - Vérification de cohérence

### 📈 Gestion & Monitoring

- **Financial Orchestrator** - Gestion des budgets et coûts
- **State Machine** - Gestion d'état du pipeline
- **Metrics Collector** - Collecte de métriques (Prometheus)
- **Monitoring System** - Dashboards Grafana

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI REST API                         │
│                    (Authentication & Endpoints)                 │
└────────────┬────────────────────────────────┬───────────────────┘
             │                                │
    ┌────────▼──────────────┐        ┌────────▼──────────────┐
    │  Input Sanitizer      │        │  Financial            │
    │  (Validation)         │        │  Orchestrator         │
    └────────┬──────────────┘        └────────┬──────────────┘
             │                                │
    ┌────────▼───────────────────────────────▼──────────────┐
    │              State Machine Orchestrator                │
    │         (Gestion d'état et transitions)               │
    └────────┬───────────────────────────────┬──────────────┘
             │                               │
    ┌────────▼──────────────┐       ┌────────▼──────────────┐
    │   Agent Pool (10+)    │       │   Technical QA Gate   │
    │  - Creative Director  │       │  - Validation Tech    │
    │  - Visual Translator  │       │  - Semantic QA        │
    │  - Audio Generator    │       │  - Cache Manager      │
    │  - Music Composer     │       └────────┬──────────────┘
    │  - Post Processor     │                │
    │  - Render Executor    │                │
    │  - etc...             │                │
    └────────┬──────────────┘                │
             │                               │
    ┌────────▼───────────────────────────────▼──────────────┐
    │     External Integrations                             │
    │  - Runway ML (Video Generation)                       │
    │  - Google Cloud (Storage, Vision, Logging)            │
    │  - Replicate (Model Inference)                        │
    │  - Datadog (Observability)                            │
    │  - Firebase (Authentication)                          │
    └───────────────────────────────────────────────────────┘
             │
    ┌────────▼───────────────────────────┐
    │   Data Layer                        │
    │  - PostgreSQL (Persistence)         │
    │  - Redis (Cache & Sessions)         │
    │  - Google Cloud Storage (Assets)    │
    └─────────────────────────────────────┘
```

### Composants Principaux

| Composant           | Description                               | Technologie                   |
| ------------------- | ----------------------------------------- | ----------------------------- |
| **API Server**      | Endpoints REST et WebSocket               | FastAPI + Uvicorn             |
| **Orchestrator**    | Coordination des agents via state machine | Python asyncio                |
| **Memory Manager**  | Gestion de l'état partagé et cache        | Redis + Python                |
| **Pipeline Worker** | Exécution asynchrone des tâches           | Asyncio + Pub/Sub             |
| **Auth System**     | Authentification multi-layer              | Firebase + JWT + API Keys     |
| **Monitoring**      | Observabilité complète                    | Prometheus + Grafana + Jaeger |
| **Database**        | Persistence des données                   | PostgreSQL + SQLAlchemy       |

---

## 📋 Prérequis

### Système

- **OS**: Linux, macOS, ou Windows (avec WSL2)
- **Python**: 3.10 ou supérieur
- **Docker**: 20.10+ (pour déploiement containerisé)
- **RAM**: 8GB minimum (16GB recommended)
- **Disk**: 50GB+ libre

### Services Externes

- **Google Cloud Project** (avec APIs activées)
  - Cloud Storage
  - Cloud Monitoring
  - Cloud Logging
  - Cloud AI Platform
  - Secret Manager
- **Firebase Project** (pour authentification)
- **Redis Server** (3.11+ ou Redis Cloud)
- **PostgreSQL Database** (12+ ou Cloud SQL)

---

## 🚀 Installation

### 1. Cloner le Repository

```bash
git clone https://github.com/Blockprod/AIPROD.git
cd AIPROD
```

### 2. Créer un Environnement Virtual Python

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux/macOS
python -m venv .venv
source .venv/bin/activate
```

### 3. Installer les Dépendances

```bash
pip install -r requirements.txt
```

### 4. Installation Optionnelle pour GCP

```bash
pip install -r requirements-ci.txt  # Pour CI/CD
pip install google-cloud-aiplatform google-cloud-storage google-cloud-logging
```

### 5. Vérifier l'Installation

```bash
python -c "import fastapi; print(f'FastAPI {fastapi.__version__} ✓')"
python -c "import pydantic; print(f'Pydantic {pydantic.__version__} ✓')"
```

---

## ⚙️ Configuration

### 1. Variables d'Environnement

Créer un fichier `.env` à la racine du projet:

```bash
# Copier le template
cp .env.example .env

# Éditer avec vos valeurs
# voir .env.example pour la liste complète
```

### 2. Configuration Core (`.env`)

```env
# ========== API ==========
API_HOST=0.0.0.0
API_PORT=8000
API_ENV=development  # ou 'production'
API_WORKERS=4

# ========== Database ==========
DATABASE_URL=postgresql://user:password@localhost:5432/aiprod
REDIS_URL=redis://localhost:6379/0

# ========== Google Cloud ==========
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json

# ========== Firebase ==========
FIREBASE_CONFIG_JSON={...}

# ========== External APIs ==========
RUNWAY_API_KEY=your-runway-key
REPLICATE_API_TOKEN=your-replicate-token
DATADOG_API_KEY=your-datadog-key

# ========== Features ==========
ENABLE_MONITORING=true
ENABLE_QA_VALIDATION=true
ENABLE_COST_TRACKING=true
DEBUG_MODE=false
```

### 3. Initialiser la Base de Données

```bash
# Créer les migrations
alembic upgrade head

# Ou manuellement
python -c "from src.db import init_db; init_db()"
```

### 4. Configuration GCP (Production)

```bash
# Authentifier avec GCP
gcloud auth application-default login

# Ou utiliser une clé de service
export GOOGLE_APPLICATION_CREDENTIALS="./credentials/terraform-key.json"
```

---

## 📖 Utilisation

### Démarrage du Serveur API

```bash
# Mode développement (avec reload)
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Ou utiliser la tâche VS Code
# Cmd+Shift+P → "Tasks: Run Task" → "Run API Server"
```

### Vérifier la Santé de l'API

```bash
curl http://localhost:8000/health

# Réponse attendue:
# {"status":"healthy","timestamp":"2026-02-05T...","version":"3.3.0"}
```

### Accéder à la Documentation Interactive

```
http://localhost:8000/docs        # Swagger UI
http://localhost:8000/redoc       # ReDoc
```

---

## 🔌 API REST

### Endpoints Principaux

#### 1️⃣ Création de Projet Vidéo

```bash
POST /api/v1/projects
Content-Type: application/json

{
  "name": "Mon Premier Projet",
  "description": "Description du contenu vidéo",
  "script": "Dialogue et scènes du script",
  "budget_limit": 500.00,
  "settings": {
    "quality": "4K",
    "duration": 60,
    "style": "cinematic"
  }
}
```

#### 2️⃣ Lancer le Pipeline

```bash
POST /api/v1/projects/{project_id}/execute
Authorization: Bearer {token}

{
  "mode": "full",  # ou "fast_track"
  "agent_selection": ["audio_generator", "music_composer", "post_processor"]
}
```

#### 3️⃣ Récupérer le Statut

```bash
GET /api/v1/projects/{project_id}/status
Authorization: Bearer {token}

# Réponse:
{
  "project_id": "uuid",
  "state": "executing",
  "progress": 45,
  "current_stage": "audio_generation",
  "estimated_completion": "2026-02-05T14:30:00Z",
  "cost_estimate": 125.50,
  "cost_actual": 87.30
}
```

#### 4️⃣ Exporter le Résultat

```bash
GET /api/v1/projects/{project_id}/export?format=mp4
Authorization: Bearer {token}

# Retourne le fichier vidéo
```

### Authentification

```bash
# Via API Key (header)
curl -H "X-API-Key: your-api-key" http://localhost:8000/api/v1/projects

# Via JWT Token (header)
curl -H "Authorization: Bearer {your-jwt-token}" http://localhost:8000/api/v1/projects

# Obtenir un token via Firebase
POST /auth/token
{
  "id_token": "firebase-id-token"
}
```

---

## 🧪 Tests

### Lancer Tous les Tests

```bash
# Mode verbeux
python -m pytest tests -v

# Ou utiliser la tâche
# Cmd+Shift+P → "Tasks: Run Task" → "Run Tests"
```

### Tests avec Couverture

```bash
python -m pytest tests -v --cov=src --cov-report=html

# Voir le rapport dans htmlcov/index.html
```

### Tests Spécifiques

```bash
# Unit tests
pytest tests/unit -v

# Load tests
pytest tests/load -v

# Tests d'un module
pytest tests/unit/test_cost_estimator.py -v
```

### Structure des Tests

```
tests/
├── unit/                    # Tests unitaires
│   ├── test_cost_estimator.py
│   ├── test_presets.py
│   ├── test_consistency_cache.py
│   └── ...
├── load/                    # Tests de charge
│   ├── test_concurrent_jobs.py
│   └── test_cost_limits.py
└── integration/             # Tests d'intégration (si applicable)
```

---

## 🐳 Déploiement

### Option 1: Docker Local

```bash
# Build l'image
docker build -t aiprod-v33:latest .

# Run le container
docker run -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/logs:/app/logs \
  aiprod-v33:latest
```

### Option 2: Docker Compose (Avec Redis & PostgreSQL)

```bash
# Démarrer tout le stack
docker-compose up -d

# Arrêter
docker-compose down

# Voir les logs
docker-compose logs -f api
```

### Option 3: Google Cloud Run (Recommandé)

```bash
# Authentication
gcloud auth login
gcloud config set project YOUR_GCP_PROJECT_ID

# Deploy
gcloud run deploy aiprod-v33 \
  --source . \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 3600 \
  --set-env-vars-file .env.cloud.yaml
```

### Option 4: Kubernetes (Production)

```bash
# Appliquer les configurations
kubectl apply -f deployments/

# Vérifier le déploiement
kubectl get pods -l app=aiprod

# Port forward pour accéder localement
kubectl port-forward svc/aiprod-service 8000:8000
```

---

## 📚 Documentation

### Documentation Principale

- [Quick Start Guide](docs/guides/QUICK_START.md) - Démarrage rapide
- [API Reference](docs/guides/2026-02-04_api-integration.md) - Documentation complète des endpoints
- [Architecture Design](docs/guides/2026-02-04_INTEGRATION_FULL_PIPELINE.md) - Architecture détaillée

### Guides Avancés

- [GCP Setup](docs/guides/2026-02-03_ETAPE_1_GCP_SETUP_STATUS.md) - Configuration GCP
- [Troubleshooting](docs/guides/2026-02-04_COMPREHENSIVE_TROUBLESHOOTING.md) - Diagnostic et résolution
- [Security Audit](docs/reports/2026-02-04_SECURITY_AUDIT_PHASE1.md) - Audit de sécurité
- [SLA Details](docs/business/2026-02-04_sla-details.md) - SLAs et disponibilité

### Rapports Techniques

- [Phase 2.1 Monitoring](docs/2026-02-05_WEEKLY_LATEST/PHASE_2.1_MONITORING_COMPLETE.md)
- [Phase 4 Completion](docs/archive/phases/phase_4/PHASE_4_COMPLETION.md)
- [Audit Complet](docs/2026-02-05_WEEKLY_LATEST/2026-02-05_AUDIT_COMPLET_PRECIS_FINAL.md)

### Plans d'Action

- [Production Deployment Plan](docs/2026-02-05_WEEKLY_LATEST/plans/2026-02-04_PHASE6_PRODUCTION_DEPLOYMENT.md)
- [Disaster Recovery](docs/2026-02-05_WEEKLY_LATEST/runbooks/2026-02-04_disaster-recovery.md)

---

## 🔒 Sécurité

### Authentification & Autorisation

- ✅ Firebase Authentication
- ✅ JWT Tokens
- ✅ API Key Management
- ✅ Role-Based Access Control (RBAC)

### Chiffrement

- ✅ HTTPS/TLS en production
- ✅ Secrets encryptés (Google Secret Manager)
- ✅ Database SSL/TLS

### Monitoring & Audit

- ✅ Audit logging complet
- ✅ Alertes en temps réel (Datadog)
- ✅ Tracing distribué (Jaeger)

### Best Practices

```python
# Charger les secrets de manière sécurisée
from src.config.secrets import get_secret

api_key = get_secret("RUNWAY_API_KEY")  # Jamais en dur!

# Masquer les secrets dans les logs
from src.config.secrets import mask_secret
log_entry = mask_secret(sensitive_data)
```

---

## 📊 Monitoring & Observabilité

### Métriques Prometheus

Disponible sur: `http://localhost:8000/metrics`

```
# Exemples de métriques
http_request_duration_seconds_bucket
pipeline_execution_time_seconds
cost_tracking_total
memory_usage_bytes
```

### Dashboards Grafana

Accès: `http://localhost:3000` (si Docker Compose)

- API Performance Overview
- Pipeline Execution Status
- Cost Analysis
- Resource Utilization

### Logging

```python
from src.utils.monitoring import logger

logger.info("Project created", extra={"project_id": project_id})
logger.error("Pipeline failed", exc_info=True)
```

---

## 🛠️ Développement

### Structure du Projet

```
src/
├── api/                     # Couche API REST
│   ├── main.py             # Point d'entrée FastAPI
│   ├── cost_estimator.py   # Estimation des coûts
│   ├── icc_manager.py      # Gestion ICC
│   ├── presets.py          # Presets prédéfinis
│   ├── openapi_docs.py     # Documentation OpenAPI
│   ├── auth_middleware.py  # Middleware d'auth
│   └── functions/          # Fonctions métier
├── agents/                  # Agents IA et orchestration
│   ├── creative_director.py
│   ├── audio_generator.py
│   ├── music_composer.py
│   ├── post_processor.py
│   ├── render_executor.py
│   └── ...
├── orchestrator/           # Orchestration des agents
│   ├── state_machine.py    # State machine
│   └── transitions.py      # Transitions d'état
├── auth/                   # Système d'authentification
│   ├── firebase_auth.py
│   ├── token_manager.py
│   ├── api_key_manager.py
│   └── ...
├── memory/                 # Gestion de l'état partagé
│   ├── memory_manager.py
│   ├── consistency_cache.py
│   ├── exposed_memory.py
│   └── schema_validator.py
├── infra/                  # Infrastructure & DevOps
│   ├── cdn_config.py
│   ├── dr_manager.py
│   ├── rbac.py
│   ├── security_audit.py
│   └── ...
├── pubsub/                 # Pub/Sub (messages)
│   └── client.py
├── utils/                  # Utilitaires
│   ├── monitoring.py
│   ├── metrics_collector.py
│   ├── gcp_client.py
│   ├── llm_wrappers.py
│   └── ...
├── db/                     # Couche persistence
│   └── __init__.py
├── workers/                # Workers asynchrones
│   └── pipeline_worker.py
└── config/                 # Configurations
    ├── secrets.py
    └── ...
```

### Ajouter un Nouvel Endpoint API

```python
# Dans src/api/main.py

from fastapi import Router

router = Router(prefix="/api/v1", tags=["custom"])

@router.post("/custom-endpoint")
async def my_custom_endpoint(request: MyRequestModel) -> MyResponseModel:
    """
    Description de l'endpoint
    """
    # Votre logique ici
    return result
```

### Code Style & Linting

```bash
# Format le code
black src/

# Check style issues
ruff check src/

# Type checking
mypy src/
```

---

## 🐛 Troubleshooting

### Problèmes Courants

#### 1. "ModuleNotFoundError: No module named 'src'"

```bash
# Solution: Ajouter le chemin au PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Ou via .env
PYTHONPATH=/app
```

#### 2. "Connection refused: Cannot connect to Redis"

```bash
# Vérifier que Redis est running
redis-cli ping
# Ou avec Docker
docker run -d -p 6379:6379 redis:latest
```

#### 3. "Database connection error"

```bash
# Vérifier la connexion PostgreSQL
psql -U user -h localhost -d aiprod

# Ou exécuter les migrations
alembic upgrade head
```

#### 4. "GCP credentials not found"

```bash
# Vérifier la variable d'environnement
echo $GOOGLE_APPLICATION_CREDENTIALS

# Ou s'authentifier
gcloud auth application-default login
```

Voir [Comprehensive Troubleshooting Guide](docs/guides/2026-02-04_COMPREHENSIVE_TROUBLESHOOTING.md) pour plus de détails.

---

## 📞 Support

### Ressources

- 📖 [Documentation Complète](docs/)
- 🐛 [Issues GitHub](https://github.com/Blockprod/AIPROD/issues)
- 💬 [Discussions](https://github.com/Blockprod/AIPROD/discussions)
- 📧 Email: team@aiprod.ai

### Rapporter un Bug

1. Vérifier que le bug n'existe pas déjà
2. Incluir les étapes de reproduction
3. Inclure les logs d'erreur
4. Indiquer la version de AIPROD

```bash
# Générer les infos système
python -c "import platform; import sys; print(f'Python {sys.version}'); print(f'Platform {platform.platform()}')"
```

### Contribuer

Les contributions sont bienvenues! Voir [CONTRIBUTING.md](CONTRIBUTING.md) pour les guidelines.

---

## 📜 License

AIPROD est sous license MIT. Voir [LICENSE](LICENSE) pour les détails.

---

## 👥 Auteurs & Remerciements

**AIPROD Team** - [team@aiprod.ai](mailto:team@aiprod.ai)

---

## 📈 Roadmap

### Phase 2.5 (Février 2026)

- [ ] Enhanced Monitoring Dashboards
- [ ] Multi-language Support
- [ ] Advanced Cost Predictions

### Phase 3 (Mars 2026)

- [ ] Real-time Collaboration
- [ ] Custom Model Training
- [ ] API v2 Release

### Phase 4+ (Avril+)

- [ ] Mobile App
- [ ] Marketplace Integration
- [ ] Enterprise Features

---

**Version:** 3.3.0  
**Last Updated:** 5 Février 2026  
**Status:** Production Ready ✅

---

<div align="center">

**[⬆ Retour au début](#-aiprod-v33---pipeline-de-génération-vidéo-ia)**

Made with ❤️ by AIPROD Team

</div>
