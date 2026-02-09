# 📊 AUDIT - État d'Implémentation vs Plan "Cost Intelligence First"

**Date Audit**: Février 6, 2026  
**Projet**: AIPROD  
**Plan Référence**: `IMPROVEMENT_PLAN_COST_FIRST.md`

---

## 🎯 Synthèse Exécutive

| Métrique                   | État                                              |
| -------------------------- | ------------------------------------------------- |
| **Couverture Plan P0**     | 🟡 **35%** (Partiellement existant)               |
| **Couverture Plan P1**     | 🔴 **0%** (À implémenter)                         |
| **Couverture Plan P2**     | 🟡 **20%** (Fragments existants)                  |
| **Prêt pour P0 démarrage** | ✅ **OUI** (peut commencer immédiatement)         |
| **Effort estimation**      | ~40h (moins que prévu, car infrastructure existe) |

---

## 📋 P0 - COST INTELLIGENCE CORE (Semaine 1)

### 1.1 CostEstimator Engine

#### ✅ EXISTE DÉJÀ:

**Fichier**: `src/api/cost_estimator.py` (217 lignes)

```python
# Ce qui existe:
✅ estimate_gemini_cost(complexity)
✅ estimate_runway_cost(duration_sec, mode)
✅ estimate_gcs_cost(duration_sec)
✅ estimate_cloud_run_cost(job_duration_sec)
✅ get_full_cost_estimate()  # Retourne dict avec breakdown
✅ Pricing dict avec RW, Gemini, GCS, CloudRun
```

**Problème**: C'est une ancienne version qui:

- ❌ Ne retourne pas les 3 GenerationPlan (premium/balanced/economy)
- ❌ N'a pas de classe GenerationTier Enum
- ❌ N'a pas de recommendation logic
- ❌ Ne prend pas en compte "user_preferences"
- ❌ Pas intégré avec Runway credit checking

#### 🔴 MANQUE:

- [ ] Classe `GenerationTier` (Enum: PREMIUM, BALANCED, ECONOMY)
- [ ] Classe `GenerationPlan` (@dataclass)
- [ ] Méthode `estimate_plans()` retournant List[GenerationPlan]
- [ ] Logic de recommendation (scoring par user_preference)
- [ ] Integration avec `_check_runway_credits()` (existe dans RenderExecutor!)
- [ ] Contrainte filtering (max_cost, min_quality, max_time_sec)

**Effort Refactor**: 4h (remplacer/étendre l'existant)

---

### 1.2 API Endpoints /video/plan et /video/generate

#### ✅ EXISTE PARTIELLEMENT:

**Fichier**: `src/api/main.py` (2222 lignes - très complet!)

**Endpoints proches**:

```
✅ POST /pipeline/run               # Job creation (générique)
✅ GET  /pipeline/job/{job_id}      # Job status
✅ POST /cost-estimate              # Cost breakdown (ligne 948)
✅ GET  /job/{job_id}/costs         # Actual costs (ligne 980)
```

**Problème**: Ces endpoints sont génériques (pour tout le pipeline AIPROD), pas spécifiques à `/video`:

- ❌ Pas d'endpoint `/video/plan` dédié
- ❌ Pas d'endpoint `/video/generate` dédié
- ❌ Pas de format response avec GenerationPlan options

#### 🔴 MANQUE:

- [ ] `POST /video/plan` endpoint
  - Input: `prompt`, `user_id` (optional), `constraints` (optional)
  - Output: 3 GenerationPlan options + ai_wisdom
- [ ] `POST /video/generate` endpoint
  - Input: `prompt`, `tier` (premium/balanced/economy), `user_id`
  - Output: video_url + cost_receipt (estimé vs réel)

**Effort Nouvelle Route**: 2h (wrapper autour RenderExecutor)

---

### 1.3 Frontend VideoPlanner UI

#### ✅ EXISTE:

**Aucun dashboard frontend trouvé**

```
❌ Pas de dossier dashboard/
❌ Pas de VideoPlanner.jsx
❌ Pas de React component
❌ Pas d'UI pour plan selection
```

Le projet semble être backend-focused (pas de frontend React visible).

#### 🔴 MANQUE:

- [ ] Dossier `dashboard/` avec React setup
- [ ] `dashboard/src/components/VideoPlanner.jsx`
- [ ] `dashboard/src/components/VideoPlanner.css`
- [ ] `dashboard/package.json` et `next.config.js`
- [ ] Connexion au backend API

**Effort création UI**: 6h (React + CSS + API wiring)

---

## 📍 P1 - QUALITÉ & PROFILS (Semaine 2)

### 2.1 Veo 3.0 Testing + Quality Validator

#### ✅ EXISTE:

**Fichier**: `scripts/generate_veo_video.py` (existe)

```python
# Utilise déjà Gemini API pour Veo
# Modèle actuellement: veo-2.0-generate-001
```

**Fichier**: `src/agents/render_executor.py` (709 lignes)

```python
✅ class VideoBackend(Enum): RUNWAY, VEO3, REPLICATE, AUTO
✅ class BackendConfig: configurations des backends
✅ _check_runway_credits()  (ligne ~92-126)
   # Vérifie solde Runway via SDK
   # Retourne nombre de credits disponibles
```

#### 🔴 MANQUE:

- [ ] Classe `VideoQualityValidator` pour ffprobe checks
- [ ] Classe `QualitySpec` (@dataclass) par tier
- [ ] Méthode `validate(video_path, tier)` retournant dict
- [ ] Conversion Veo 2.0 → Veo 3.0 dans generate_veo_video.py
- [ ] Upscaling Real-ESRGAN (dépendance + classe VideoUpscaler)

**Dépendances à ajouter** (pas dans requirements.txt):

```
realesrgan  # Pour upscaling
ffmpeg-python  # Déjà présent ✅
av  # Déjà présent ✅
```

**Effort**: 5h

---

### 2.2 Resolution Profile System

#### ✅ EXISTE:

**Fichier**: `src/api/presets.py` (existe!)

```python
# Semble avoir un système de presets existant
# Mais pas spécifique aux résolutions
```

#### 🔴 MANQUE:

- [ ] Classe `ResolutionProfile` (Enum: SOCIAL, WEB, BROADCAST)
- [ ] Classe `ProfileSpec` (@dataclass) avec specs par profile
- [ ] Classe `ResolutionProfileSelector` avec `select(use_case)`
- [ ] Integration dans `/video/generate` endpoint

**Effort**: 3h

---

## 🔔 P2 - OBSERVABILITÉ (Semaine 3)

### 2.1 Real-Time Metrics Dashboard

#### ✅ EXISTE PARTIELLEMENT:

**Fichier**: `src/api/websocket_manager.py` (278 lignes)

```python
✅ class WebSocketConnectionManager
✅ Gère subscriptions job updates
✅ Broadcasting d'events
✅ Connection auth tracking
```

**Fichier**: `src/webhooks.py` (387 lignes)

```python
✅ class WebhookEventType (enums d'événements)
✅ class WebhookEvent
✅ class WebhookDelivery
✅ Retry logic avec exponential backoff
```

**Monitoring**:

```python
✅ src/monitoring/metrics_collector.py
✅ src/monitoring/monitoring_middleware.py
✅ Prometheus integration (prometheus-fastapi-instrumentator)
✅ Datadog integration (requirements.txt)
```

#### 🔴 MANQUE:

- [ ] Dashboard frontend temps réel
- [ ] WebSocket `/ws/metrics` endpoint
- [ ] Classe `MetricsCollector` pour agg coûts
- [ ] Cost metrics collection (vs seulement perf metrics)
- [ ] React dashboard avec graphs + cards

**Effort**: 8h (frontend + backend metrics)

---

### 2.2 Webhook System pour notifications

#### ✅ EXISTE:

**Complet** - voir fichier `src/webhooks.py`

```python
✅ WebhookManager avec retry logic
✅ HMAC signing (sécurité)
✅ Event types: job.created, job.completed, etc
✅ Delivery tracking
```

#### 🔴 MANQUE:

- [ ] Integration avec cost accuracy tracking
- [ ] Événement "video.cost_receipt" (réel vs estimé)
- [ ] Callback dans `/video/generate` après exécution

**Effort**: 1h (juste wiring existant)

---

## 🏗️ Infrastructure Existante Utilisable

### ✅ Très Bon (à exploiter):

1. **RenderExecutor** déjà multi-backend avec credit checking
2. **WebSocket + Webhooks** déjà implémentés
3. **Cost estimation** existe (à moderniser)
4. **Main.py** infrastructure robuste pour ajouter routes
5. **Auth/Security** déjà en place (Firebase + JWT)

### ⚠️ À Améliorer:

1. **Cost Estimator** ancien format (pas GenerationPlan)
2. **UI/Frontend** inexistant ou minimal
3. **Quality validation** manquant completement
4. **Resolution profiles** pas spécifique
5. **Cost metrics** pas agrégés pour dashboard

---

## 📊 Tableau Détaillé par Tâche Plan P0

| Tâche                               | Existe?    | État     | Effort | Blockers    |
| ----------------------------------- | ---------- | -------- | ------ | ----------- |
| **1.1a** CostEstimator class        | 🟡 Partiel | Refactor | 4h     | Aucun       |
| **1.1b** GenerationPlan @dataclass  | 🔴 Non     | À créer  | 1h     | Aucun       |
| **1.1c** `/video/plan` endpoint     | 🔴 Non     | New      | 2h     | Aucun       |
| **1.2a** `/video/generate` endpoint | 🔴 Non     | New      | 2h     | Aucun       |
| **1.2b** VideoPlanner.jsx           | 🔴 Non     | New      | 6h     | Aucun       |
| **1.2c** VideoPlanner.css           | 🔴 Non     | New      | 2h     | Aucun       |
| **Dashboard setup**                 | 🔴 Non     | New      | 3h     | Node.js/npm |

**Total P0**: ~20h (vs 10h plan, car UI = nouveau)

---

## 📊 Tableau Détaillé par Tâche Plan P1

| Tâche                          | Existe?          | État         | Effort | Blockers       |
| ------------------------------ | ---------------- | ------------ | ------ | -------------- |
| **2.1a** Veo 3.0 test          | 🟡 Script existe | Change model | 0.5h   | Aucun          |
| **2.1b** QualityValidator      | 🔴 Non           | New          | 3h     | ffprobe OK     |
| **2.1c** Real-ESRGAN upscaling | 🔴 Non           | New          | 4h     | realesrgan pip |
| **2.2** ResolutionProfiles     | 🟡 Presets exist | Adapt        | 3h     | Aucun          |

**Total P1**: ~10.5h

---

## 🎯 Quick Start Immédiat

### Semaine 1 (P0) - Order de Priorité:

```
JOUR 1-2 (4h):
  1. Refactor src/api/cost_estimator.py
     - Ajouter GenerationTier Enum
     - Ajouter GenerationPlan @dataclass
     - Refactor estimate_plans() method

JOUR 3-4 (4h):
  2. Ajouter endpoints /video/plan et /video/generate dans main.py
     - Wrapper CostEstimator.estimate_plans()
     - Wrapper RenderExecutor.run()
     - Retourner format générationPlan + receipt

JOUR 5-7 (6h + setup):
  3. Créer dashboard React
     - dashboard/package.json + next.js ou vite
     - VideoPlanner.jsx component
     - API wiring avec fetch

JOUR 8 (Validation):
  4. E2E test:
     - Appeler /video/plan → 3 options reçues
     - Appeler /video/generate → Vidéo + receipt retourné
     - UI affiche les 3 options, UI de sélection marche
```

---

## 🚨 Dépendances Manquantes

Ajouter à `requirements.txt`:

```
realesrgan>=0.3.0  # Pour upscaling vidéo
```

Frontend (new):

```
node >= 18
npm ou yarn
next.js ou vite
react
```

---

## 💡 Opportunités Supplémentaires

### Réutiliser existant:

1. **Statistics de jobs** → Afficher sur dashboard (data déjà collectée!)
2. **Cloud Monitoring Prometheus** → Déjà configuré, juste besoin de cost metrics
3. **Auth Firebase** → Réutiliser pour dashboard authentication
4. **Job history DB** → Afficher dans cost breakdown par user

---

## 🎬 Commencer Immédiatement?

### ✅ OUI, tu peux commencer P0 demain:

1. **Zero blockers** - tout le backend infrastructure existe
2. **RenderExecutor** fonctionne déjà (vérifié earlier)
3. **Runway integration** OK (credit checking existe)
4. **Seule complexité**: UI (mais standard React)

### 📝 Ordre Recommandé:

1. **Jour 1-2**: Refactor cost_estimator.py (core logic)
2. **Jour 3**: Endpoints /video/plan et /video/generate
3. **Jour 4-6**: Frontend dashboard
4. **Jour 7**: End-to-end test et validation
5. **Semaine 2**: P1 (quality + profiles)

---

## 🔍 Fichiers Clés À Connaître

```
Core Business Logic:
  src/api/cost_estimator.py     ← À refactor
  src/agents/render_executor.py ← À integrer
  src/api/main.py               ← Ajouter endpoints

Existing Infrastructure:
  src/api/websocket_manager.py  ← Réutiliser
  src/webhooks.py               ← Réutiliser
  src/api/presets.py            ← Inspiration pour profiles

À Créer:
  src/agents/video_quality_validator.py  (NEW)
  src/agents/resolution_profiles.py      (NEW)
  dashboard/                              (NEW - React)
```

---

## 📞 Prochaines Questions

1. **UI Déploiement**: Veux-tu dashboard sur même serveur (port 3000) que FastAPI (port 8000)?
2. **Database**: user_preferences stockés où? Redis? PostgreSQL?
3. **Frontend Framework**: Préférence React + Vite ou Next.js?
4. **Authentication**: Même Firebase token pour dashboard + backend API?

---

**Audit Complété**: ✅  
**Prêt pour P0**: ✅  
**Recommendation**: Commence par cost_estimator.py refactor → endpoints → then frontend
