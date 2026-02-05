# Architecture AIPROD

## Vue d'ensemble

AIPROD est un pipeline de génération vidéo IA cloud-native construit sur une architecture asynchrone modulaire. Le système orche
stre plusieurs agents spécialisés pour transformer un texte en vidéo optimisée, avec double validation technique et sémantique.

## Composants principaux

### 1. Orchestrator (State Machine)

- **Fichier** : `src/orchestrator/state_machine.py`
- **Rôle** : Gère les états du pipeline et les transitions
- **États** : INIT → INPUT_SANITIZED → AGENTS_EXECUTED → QA_SEMANTIC → FINAL_APPROVAL → DELIVERED
- **Fonctionnalités** :
  - Gestion des transitions conditionnelles (fast vs full pipeline)
  - Retry policy avec backoff (max 3 tentatives)
  - Intégration des agents

### 2. Memory Manager

- **Fichier** : `src/memory/memory_manager.py`
- **Rôle** : Gère la mémoire partagée et le cache de cohérence
- **Fonctionnalités** :
  - Validation de schéma avec Pydantic
  - Mémoire exposée pour ICC (Interface Client Collaboratif)
  - Cache TTL 168h pour cohérence
  - Logging structuré

### 3. Agents

#### Creative Director

- **Fichier** : `src/agents/creative_director.py`
- **Rôle** : Fusion des outputs, gestion du cache et fallback Gemini
- **Comportement** : Utilise les résultats en cache si disponibles, sinon exécute la fusion

#### Fast Track Agent

- **Fichier** : `src/agents/fast_track_agent.py`
- **Rôle** : Pipeline simplifié pour les requêtes de faible complexité
- **Contraintes** : maxDurationSec: 30, maxScenes: 3, priority-based
- **Target latence** : < 20 secondes

#### Render Executor

- **Fichier** : `src/agents/render_executor.py`
- **Rôle** : Exécution du rendu des assets
- **Comportement** : Simule le temps de rendu (mock en développement)

#### Semantic QA

- **Fichier** : `src/agents/semantic_qa.py`
- **Rôle** : Validation sémantique des outputs
- **Validation** : Basée sur vision LLM (Gemini 1.5 Pro Vision)

#### Visual Translator

- **Fichier** : `src/agents/visual_translator.py`
- **Rôle** : Traduction et adaptation des assets visuels
- **Support** : Multi-langue (en, fr, etc.)

### 4. Fonctions métier

#### Financial Orchestrator

- **Fichier** : `src/api/functions/financial_orchestrator.py`
- **Rôle** : Optimisation coût/qualité sans LLM
- **Fonctionnalités** :
  - Dynamic pricing avec updateIntervalHours: 24
  - Audit trail pour chaque opération
  - Certification des coûts

#### Technical QA Gate

- **Fichier** : `src/api/functions/technical_qa_gate.py`
- **Rôle** : Vérifications binaires déterministes
- **Checks** : asset_count, manifest_complete, cost_valid, quality_acceptable

#### Input Sanitizer

- **Fichier** : `src/api/functions/input_sanitizer.py`
- **Rôle** : Nettoyage et validation des entrées utilisateur
- **Validations** : Pydantic + nettoyage (trim, lowercase, etc.)

### 5. API REST

- **Fichier** : `src/api/main.py`
- **Framework** : FastAPI avec documentation Swagger
- **Endpoints** : Pipeline, métriques, alertes, ICC, optimisation financière, QA technique

### 6. Monitoring & Métriques

- **Fichier** : `src/utils/monitoring.py` et `src/utils/metrics_collector.py`
- **Métriques** : Latence, coût, qualité, taux d'erreur
- **Alertes** : high_latency (>5s), high_cost (>$1), low_quality (<60%), high_error_rate

## Flux d'exécution global

```
User Request
    ↓
Input Sanitizer (validation + nettoyage)
    ↓
Orchestrator State Machine
    ├─→ Fast Track (priority=high) → Latence < 20s
    └─→ Full Pipeline (priority=low)
          ├─→ Creative Director (fusion + cache)
          ├─→ Render Executor (rendu)
          ├─→ Semantic QA (validation)
          └─→ Visual Translator (adaptation)
    ↓
Financial Orchestrator (optimisation coût/qualité)
    ↓
Technical QA Gate (vérifications binaires)
    ↓
Metrics Collector (tracking performance)
    ↓
API Response + ICC Data
```

## Configuration

Configuration externalisée depuis `config/v33.json` avec les paramètres :

- retry: { maxRetries: 3, backoffSec: 15 }
- cache: { ttl: 168 (heures) }
- fastTrack: { maxDurationSec: 30, maxScenes: 3, costCeiling: 0.3 }
- financial: { updateIntervalHours: 24 }

## Logging & Monitoring

- **Logging** : Format structuré avec rotation des fichiers (5MB max, 5 backups)
- **Répertoire logs** : `logs/AIPROD.log`
- **Niveaux** : INFO, ERROR, WARNING avec contexte détaillé

## Tests

- **Tests unitaires** : `tests/unit/` (14+ tests)
- **Tests d'intégration** : `tests/integration/` (3+ tests)
- **Tests de performance** : `tests/performance/` (2+ tests)
- **Couverture** : Memory Manager, Orchestrator, Agents, API, Metrics

## Déploiement

- **Plateforme** : Google Cloud Platform
- **Container** : Docker avec uvicorn
- **Infrastructure** : Cloud Run, Cloud Functions, Cloud Storage
- **Monitoring** : Cloud Monitoring, Cloud Logging

## Extensions futures

- Intégration Sora pour la génération vidéo native
- Backends multi-LLM (Claude, GPT-4, etc.)
- Webhooks asynchrones pour les notifications
- Persistence en Firestore/BigQuery pour les historiques
