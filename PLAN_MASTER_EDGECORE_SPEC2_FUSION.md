# 🚀 PLAN MASTER EXÉCUTION - EDGECORE SPEC 2 ULTIME

## Concept Visionnaire + Méthodologie Structurée

**Date**: 9 février 2026  
**Stratégie**: EDGECORE Enhanced (Fork LTX-2 + Orchestration AIPROD + Multi-GPU + Real-time + Analytics)  
**TTM**: 18-20 semaines (4.5-5 mois, réaliste avec contingency)  
**Investment**: $75K dev + $50.8K/an infra  
**Success Probability**: 80-85%+ (de-risked via MVP validation)

---

## 📋 TABLE DES MATIÈRES

1. [Vue d'Ensemble Visionnaire](#vue-densemble-visionnaire)
2. [Prérequis & Setup](#prérequis--setup)
3. [Structure du Projet Complète](#structure-du-projet-complète)
4. [Phase 0: Validation & POC Multi-GPU (3 jours)](#phase-0-validation--poc-multi-gpu-3-jours)
5. [Phase 1: Préparation & Foundation (Semaines 1-2)](#phase-1-préparation--foundation-semaines-1-2)
6. [Phase 2: Core MVP SPEC 2 (Semaines 3-6)](#phase-2-core-mvp-spec-2-semaines-3-6)
7. [Phase 3: Deployment & Beta MVP (Semaines 7-8)](#phase-3-deployment--beta-mvp-semaines-7-8)
8. [PHASE BREAK: MVP LAUNCHED ✅](#phase-break-mvp-launched)
9. [Phase 4: EDGECORE Enhancements (Semaines 9-14)](#phase-4-edgecore-enhancements-semaines-9-14)
10. [Phase 5: Advanced Scaling & GA v2.0 (Semaines 15-18)](#phase-5-advanced-scaling--ga-v20-semaines-15-18)
11. [Checklist Progressive](#checklist-progressive)
12. [Ressources & Équipe](#ressources--équipe)
13. [Budget Réaliste & Timeline](#budget-réaliste--timeline)

---

## Vue d'Ensemble Visionnaire

### Objectif Principal

> **Créer "aiprod-ltx-edge" - Une plateforme SaaS next-gen intégrant LTX-2 forké avec orchestration AIPROD multi-GPU distribuée, real-time preview streaming, ML-based cost optimization, analytics professionnelles, et auto-scaling intelligent. Lancer MVP en 8 semaines, puis ajouter features visionnaires pour devenir market leader en week 18.**

### Principaux Livrables

```
PHASE MVP (Weeks 0-8):
├─ Week 0:      GPU POC + Multi-node POC + Buy-in
├─ Weeks 1-2:   Infrastructure + Core setup
├─ Weeks 3-6:   Single-GPU core (SPEC 2 MVP)
├─ Week 7-8:    K8s deploy + Beta launch
└─ ✅ LAUNCH: Semaine 8 (produit rémunérateur)

PHASE EDGECORE (Weeks 9-18):
├─ Week 9-10:   Multi-GPU manager + load balancing
├─ Week 11-12:  Real-time preview + streaming
├─ Week 13-14:  Analytics + ML cost prediction
├─ Week 15-16:  Auto-scaling + v2.0 refinement
├─ Week 17-18:  GA v2.0 launch + optimization
└─ 🎉 V2.0 LAUNCH: Semaine 18 (market leader)

TOTAL: 18 weeks (4.5 months realistic)
      + 2 weeks contingency = 20 weeks (5 months max)
```

### Innovations EDGECORE (Différenciateurs)

| Feature Visionnaire                                  | MVP Status | Enhancement Status | Impact               |
| ---------------------------------------------------- | ---------- | ------------------ | -------------------- |
| Single-GPU LTX-2 inference (TI2VidTwoStagesPipeline) | ✅ Phase 2 | —                  | Cost 10x vs API      |
| Multi-GPU node orchestration                         | ❌         | ✅ Phase 4         | Horizontal scaling   |
| Real-time preview streaming (WebSocket)              | ❌         | ✅ Phase 4         | Unique UX            |
| Intelligent fallback chain (LTX-2→Distilled→APIs)    | ✅ Phase 2 | ✅ Enhanced        | Uptime 99.9%         |
| ML-based cost prediction                             | ❌         | ✅ Phase 4         | Unique SaaS feature  |
| Cost-aware job scheduling                            | ❌         | ✅ Phase 4         | Revenue optimization |
| Advanced analytics dashboard (Grafana)               | ❌         | ✅ Phase 4         | Professional SaaS    |
| Predictive auto-scaling                              | ❌         | ✅ Phase 5         | Handle load spikes   |
| Multi-format output (MP4/WebM/ProRes)                | ❌         | ✅ Phase 5         | Creator flexibility  |

### Timeline & Success Strategy

```
RISK MINIMIZATION APPROACH:

Week 0-8:      Prove SPEC 2 MVP works
               ├─ Real customer validation
               ├─ Revenue generation
               ├─ Team de-risking
               └─ Learned lessons

Week 9-18:     Add EDGECORE features based on real data
               ├─ Customer feedback incorporated
               ├─ Roadmap validated
               ├─ Premium tier ready
               └─ Market leadership achieved

BENEFIT: MVP + Iterations vs Big Bang
```

---

## Prérequis & Setup

### Avant de Commencer

#### Approvals Nécessaires

- [ ] **Executive approval** sur plan EDGECORE ultime
- [ ] **Budget approval** ($75K dev + $50.8K/an infrastructure)
- [ ] **Team allocation** (4.5 FTE for 18 weeks)
- [ ] **GPU provider contract** (Modal Labs H100, fallback Lambda)
- [ ] **GitHub Enterprise setup** (private repos, Actions)
- [ ] **Data science resources** (for ML cost prediction, auto-scaling)

#### Infrastructure Initiale

```bash
# 1. GPU Provider Account - PHASE STRATEGY
Week 0-8 (MVP):      Modal Labs H100 × 1 ($0.30/hr)
Week 9-18 (EDGE):    Modal Labs H100 × 2 ($0.60/hr) with scaling
Backup:              Lambda Labs if Modal unavailable

# 2. Kubernetes Cluster
MVP:     Single node GKE (affordable, demo)
EDGE:    Multi-node GKE with GPU node pools (production)

# 3. Storage (Progressive)
MVP:     GCS bucket (models + outputs), 50GB allocated
EDGE:    GCS + Redis cache layer (fast access)

# 4. Database (Managed)
MVP:     PostgreSQL managed (8GB, develop)
EDGE:    PostgreSQL managed (32GB, scale) + read replicas
```

#### Team Structure (4.5 FTE)

```
Project Lead (PM)              [1.0 FTE]
  └─ Timeline, stakeholder comms, MVP go/no-go decision

Backend Lead                   [1.5 FTE]
  ├─ ML Ops Engineer          [0.8 FTE] - GPU, models, optimization
  └─ Backend Engineer         [0.7 FTE] - API, state machine

DevOps Engineer                [0.8 FTE]
  └─ K8s, CI/CD, GPU scheduling, multi-node orchestration

Data Scientist (Part-time)    [0.5 FTE] (Weeks 9+)
  └─ Cost prediction models, auto-scaling ML

QA Engineer                    [0.5 FTE]
  └─ Testing, load testing, quality gates

Tech Lead (Part-time)         [0.2 FTE] (Architecture reviews)

TOTAL: 4.5 FTE × 18 weeks = 3,240 hours
```

---

## Structure du Projet Complète

### Workspace Neuf: aiprod-ltx-edge

[VOIR STRUCTURE DÉTAILLÉE - COPIÉE DEPUIS EDGECORE ENHANCED, ~300 lignes déjà créée]

**Key packages:**

```
packages/
├─ ltx-core/                 # Fork LTX-2 (model)
├─ ltx-pipelines/            # Fork LTX-2 (inference)
├─ ltx-trainer/              # Fork LTX-2 (fine-tuning)
└─ ltx-orchestration-edge/   # Our innovation (API + GPU + Analytics)
    ├─ gpu_manager/          # Single GPU (MVP)
    ├─ gpu_allocator/        # Multi-GPU (EDGE phase)
    ├─ preview_streamer/     # Real-time (EDGE phase)
    ├─ cost_predictor/       # ML prediction (EDGE phase)
    ├─ analytics/            # Dashboard (EDGE phase)
    └─ auto_scaler/          # Predictive (EDGE phase)
```

---

## Phase 0: Validation & POC Multi-GPU (3 jours)

### Jour 1 (Lundi)

#### Morning (9am - 12pm)

- [ ] **Project kickoff meeting** (1h)
  - Discuss EDGECORE plan with stakeholders
  - Review timeline (18 weeks realistic, not 12-16 optimistic)
  - Confirm budget ($75K dev, not $50K)
  - Address concerns about multi-GPU complexity
  - Q&A on MVP strategy (Phase 0-3 only, EDGE features later)

- [ ] **GPU provider setup** (2h)
  - Create Modal Labs account
  - Request H100 × 2 instances (for multi-GPU POC)
  - Configure SSH access + Python environment
  - Verify GPU connectivity + torch.cuda availability

#### Afternoon (1pm - 5pm)

- [ ] **Model verification** (2h)
  - Download LTX-2 checkpoint (`ltx-2-19b-dev-fp8.safetensors`, ~15GB)
  - Verify file integrity (SHA256 checksum)
  - Test `load_safetensors()` Python code
  - Measure model size in memory (should be ~10-12GB with FP8)

- [ ] **Multi-GPU POC** (1.5h)
  - Load model on GPU #1
  - Load second instance on GPU #2
  - Test inference on both GPUs in parallel
  - Measure latency + memory usage
  - Document multi-GPU potential (critical for EDGE phase)

- [ ] **Team assignments** (0.5h)
  - Allocate engineers to streams (Backend, DevOps, QA)
  - Create Slack channels (#aiprod-ltx-edge-dev, #aiprod-ltx-edge-ops)
  - Set up code review process (GitHub CODEOWNERS)

**EoD Checkpoint**: ✅ GPU accessible, models loaded, multi-GPU POC validated

---

### Jour 2 (Mardi)

#### Morning (9am - 12pm)

- [ ] **GitHub organization setup** (1.5h)
  - Create GitHub organization (aiprod-ltx-edge)
  - Create private repos for forks (ltx-core, ltx-pipelines, ltx-trainer, ltx-orchestration-edge)
  - Configure branch protection (main branch - require PRs)
  - Enable GitHub Actions for automation

- [ ] **LTX-2 fork & clone** (1h)
  - Fork https://github.com/Lightricks/LTX-2 to organization
  - Clone to new workspace
  - Verify all submodules initialized
  - Run `pip install -e .` to verify dependencies resolve

#### Afternoon (1pm - 5pm)

- [ ] **Infrastructure planning** (1.5h)
  - Document GPU specs (H100 × 1 for MVP, × 2+ for EDGE)
  - Plan network topology (intra-pod communication for multi-GPU)
  - Identify storage requirements (models: 100GB+, outputs: dynamic)
  - Estimate K8s cluster costs (staging vs production)

- [ ] **Technical design review** (1.5h)
  - Walkthrough EDGECORE architecture with team
  - Clarify MVP phase (SPEC 2-like) vs EDGE phase (visionary)
  - Identify integration points (GPU manager → API → Database)
  - Q&A on fallback chain, state machine, cost tracking
  - Document any technical blockers

- [ ] **Security & access planning** (0.5h)
  - Plan GPU compute infrastructure access (who can SSH?)
  - Design secrets management (API keys, DB credentials)
  - Plan audit logging (who accessed what resources)

**EoD Checkpoint**: ✅ Repos ready, infrastructure planned, team aligned

---

### Jour 3 (Mercredi)

#### Full Day

- [ ] **Dry-run MVP inference test** (4h)
  - Load LTX-2 model on single H100
  - Run sample T2V inference using `TI2VidTwoStagesPipeline`
  - Measure: latency (should be 30-60s for 5-sec video), memory (should be <12GB), output quality
  - Document results in PERF_BASELINE.md
  - Compare against Runway API benchmarks (establish cost-per-inference baseline)

- [ ] **Multi-GPU scaling preview** (1h)
  - Map what multi-GPU orchestration will need
  - Identify GPU allocation strategy (load balancing algorithm)
  - Document multi-GPU challenges (kernel launch overhead, inter-GPU comm)

- [ ] **Security audit** (1h)
  - Review compute infrastructure access controls
  - Verify secrets are never logged
  - Plan audit logging for compliance

- [ ] **Go/No-Go decision** (1h)
  - Review Phase 0 results
  - Confirm SPEC 2 MVP is feasible (Week 3-6)
  - Confirm EDGECORE enhancements are feasible post-MVP
  - Formal approval to proceed to Phase 1

**EoD Checkpoint**: ✅ Full green light for Phase 1 kickoff

---

## Phase 1: Préparation & Foundation (Semaines 1-2)

### Semaine 1 (Monday - Friday)

```
GOAL: Complete infrastructure setup + Package initialization
```

#### Monday: GitHub & CI/CD Setup

```
Time  Task                                   Owner      Duration  ☐
────────────────────────────────────────────────────────────────────
9-10  Standup (plan week)                   PM         30m
10-12 GitHub Actions workflow config        DevOps     2h
      ├─ test.yml (pytest on every commit)
      ├─ build.yml (Docker build + push)
      ├─ deploy-staging.yml (K8s staging)
      └─ security-scan.yml (vulnerability check)
12-1  LUNCH
1-3   K8s cluster setup                     DevOps     2h
      ├─ Provision GKE cluster (standard pool + GPU pool)
      ├─ Install CNI (Calico), ingress controller (nginx)
      ├─ Setup GPU drivers + device plugin
      └─ Verify GPU availability (`kubectl get nodes`)
3-4   Database setup                        Backend    1h
      ├─ Create PostgreSQL managed instance
      ├─ Create initial schema
      ├─ Setup Alembic migration system
      └─ Test connection from local + K8s
4-5   EoD standup + blockers               Team       30m

DELIVERABLES:
  ☐ GitHub Actions pipeline green (all tests pass)
  ☐ K8s cluster accessible + GPU nodes available
  ☐ Database connection working locally + in K8s
  ☐ Alembic migrations initialized
```

#### Tuesday: Python Package Structure

```
Time  Task                                   Owner      Duration  ☐
────────────────────────────────────────────────────────────────────
9-10  Standup                               PM         30m
10-12 Create ltx-orchestration-edge pkg    Backend    2h
      ├─ pyproject.toml + setup
      ├─ src/ltx_orchestration_edge/__init__.py
      ├─ All subdirectories (api, inference, etc)
      ├─ requirements.txt + requirements-dev.txt
      └─ Initial README.md
12-1  LUNCH
1-3   Poetry lock file + dep mgmt          DevOps     2h
      ├─ Install Poetry
      ├─ poetry.lock (reproducible builds)
      ├─ Separate dev dependencies (pytest, black, mypy)
      └─ Test install in fresh venv
3-4   Docker setup (base image)            DevOps     1h
      ├─ Create Dockerfile (multi-stage build)
      ├─ Create docker-compose.yml (local dev)
      ├─ Test build: `docker build -t aiprod-ltx-edge:dev .`
      └─ Test run: `docker run ... python -c "import ltx_core"`
4-5   EoD standup + blockers               Team       30m

DELIVERABLES:
  ☐ ltx-orchestration-edge package initialized + importable
  ☐ Poetry poetry.lock committed to repo
  ☐ Docker build succeeds
  ☐ All dependencies resolve correctly
```

#### Wednesday: API Specification & Design

```
Time  Task                                   Owner      Duration  ☐
────────────────────────────────────────────────────────────────────
9-10  Standup                               PM         30m
10-12 FastAPI skeleton + models            Backend    2h
      ├─ Create src/ltx_orchestration_edge/api/main.py
      ├─ Init FastAPI app with lifespan context
      ├─ Create Pydantic models (GenerateRequest, GenerateResponse)
      ├─ Setup OpenAPI 3.0 documentation
      └─ Test: `python -m uvicorn src.api.main:app --reload`
12-1  LUNCH
1-3   API endpoints skeleton                Backend    2h
      ├─ POST /api/v1/generate (returns job_id)
      ├─ GET /api/v1/status/{job_id} (returns progress)
      ├─ POST /api/v1/estimate (cost estimation)
      ├─ GET /health (liveness probe)
      └─ All return 200 with hardcoded responses (no backend yet)
3-4   OpenAPI spec finalize                Tech Lead  1h
      ├─ Export OpenAPI schema
      ├─ Review for consistency
      ├─ Commit to repo
      └─ Make available at GET /openapi.json
4-5   EoD standup + blockers               Team       30m

DELIVERABLES:
  ☐ FastAPI app runs locally (port 8000)
  ☐ /openapi.json returns valid OpenAPI 3.0 spec
  ☐ All 4 endpoints respond with correct status codes
```

#### Thursday: State Machine Design

```
Time  Task                                   Owner      Duration  ☐
────────────────────────────────────────────────────────────────────
9-10  Standup                               PM         30m
10-12 State machine design doc             Backend    2h
      ├─ Define 8 states (INIT → DELIVERED/ERROR)
      ├─ Draw state diagram (Mermaid format)
      ├─ Create state transition table
      ├─ Identify fallback triggers
      └─ Document error handling per state
12-1  LUNCH
1-3   State machine Python code             Backend    2h
      ├─ Create src/orchestration/state_machine.py
      ├─ Enum for PipelineState
      ├─ StateMachine class + transition logic
      ├─ Async methods (async def transition_to(state))
      └─ Unit tests > 90% coverage
3-4   Integration with API                 Backend    1h
      ├─ Wire state machine in POST /generate
      ├─ Verify state retrieval in GET /status
      ├─ Test happy path (e2e integration test)
      └─ Document state flow in README
4-5   EoD standup + blockers               Team       30m

DELIVERABLES:
  ☐ State diagram documented
  ☐ State machine code committed
  ☐ Unit tests passing (>90% coverage)
  ☐ API endpoints return correct state
```

#### Friday: Testing & CI/CD Validation

```
Time  Task                                   Owner      Duration  ☐
────────────────────────────────────────────────────────────────────
9-10  Standup + week review                PM         30m
10-12 pytest setup + first tests           QA         2h
      ├─ Create tests/conftest.py (fixtures)
      ├─ Create tests/unit/test_state_machine.py
      ├─ Create tests/unit/test_api_models.py
      ├─ Run pytest: `pytest tests/ -v`
      └─ CI/CD runs tests automatically on commit
12-1  LUNCH
1-3   Code quality checks                  DevOps     2h
      ├─ Setup black (auto-format)
      ├─ Setup mypy (type checking)
      ├─ Setup flake8 (linting)
      ├─ GitHub Actions runs all checks
      └─ All code review PRs require passing checks
3-4   Week review + retrospective          Team       1h
      ├─ Demo infrastructure setup
      ├─ Demo API running locally
      ├─ Demo CI/CD pipeline
      ├─ Identify blockers
      └─ Plan Week 2
4-5   EoD + week summary                   PM         30m

**EoW Checkpoint Week 1**: ✅ API spec finalized, state machine designed, CI/CD green
```

---

### Semaine 2 (Monday - Friday)

```
GOAL: Complete core components for Phase 2 (GPU inference wrapper)
```

#### Monday: GPU Manager Design

```
Time  Task                                    Owner      Duration  ☐
────────────────────────────────────────────────────────────────────
9-10  Standup + week plan                   PM         30m
10-12 GPU manager design                    ML Ops     2h
      ├─ Sketch GPU memory management strategy
      ├─ Identify OOM handling (vs fallback)
      ├─ Define interface: allocate(), deallocate(), status()
      ├─ Document single-GPU vs multi-GPU differences
      └─ Review with team
12-1  LUNCH
1-3   GPU manager Python code               ML Ops     2h
      ├─ Create src/inference/gpu_manager.py
      ├─ class GPUManager with memory tracking
      ├─ async def allocate_device()
      ├─ Logging for debugging
      └─ Stubs for multi-GPU (comments like "# TODO: Phase 4")
3-4   Unit tests                            QA         1h
      ├─ test_gpu_manager.py with mocks
      ├─ Test memory allocation
      ├─ Test OOM error handling
      └─ >80% coverage (MVP level)
4-5   EoD standup + blockers               Team       30m
```

#### Tuesday: Model Loader Implementation

```
Time  Task                                    Owner      Duration  ☐
────────────────────────────────────────────────────────────────────
9-10  Standup                               PM         30m
10-12 Model loader design                   ML Ops     2h
      ├─ Plan safetensors loading (from Lightricks LTX-2)
      ├─ FP8 vs FP32 handling
      ├─ LoRA adapter support (future, Phase 4)
      ├─ Caching strategy (LRU, with TTL)
      └─ Document model paths
12-1  LUNCH
1-3   Model loader Python code              ML Ops     2h
      ├─ Create src/inference/model_loader.py
      ├─ async def load_ltx_model()
      ├─ load_safetensors() integration
      ├─ Error handling (model not found, corrupt, etc)
      └─ Logging
3-4   Integration with GPU manager          ML Ops     1h
      ├─ Coordinate memory allocation
      ├─ Test loading model with GPU manager
      └─ Document interface
4-5   EoD standup + blockers               Team       30m
```

#### Wednesday: Pipeline Wrapper Design

```
Time  Task                                    Owner      Duration  ☐
────────────────────────────────────────────────────────────────────
9-10  Standup                               PM         30m
10-12 Pipeline wrapper architecture         Backend    2h
      ├─ Design async wrapper around LTX-2 (synchronous)
      ├─ Strategy: asyncio.to_thread() for blocking inference
      ├─ Error handling (OOM → fallback)
      ├─ Timeout strategy (5 min max per inference)
      └─ Document interface
12-1  LUNCH
1-3   Pipeline wrapper Python code          Backend    2h
      ├─ Create src/inference/pipeline_wrapper.py
      ├─ class PipelineWrapper
      ├─ async def generate_video()
      ├─ Integration with GPUManager + ModelLoader
      └─ Graceful degradation
3-4   Testing (mock inference)              QA         1h
      ├─ test_pipeline_wrapper.py with mocks
      ├─ Test success path
      ├─ Test OOM error path
      └─ Test timeout path
4-5   EoD standup + blockers               Team       30m
```

#### Thursday: Cache Manager & Quality Gates

```
Time  Task                                    Owner      Duration  ☐
────────────────────────────────────────────────────────────────────
9-10  Standup                               PM         30m
10-12 Cache manager implementation          Backend    2h
      ├─ Create src/inference/cache_manager.py
      ├─ class CacheManager (LRU with TTL)
      ├─ Integrate with ModelLoader
      ├─ Test: repeated model loads should hit cache
      └─ Metrics: cache hit/miss rates
12-1  LUNCH
1-3   Quality gates implementation          QA         2h
      ├─ Create src/orchestration/quality_gates.py
      ├─ Validate video format (MP4)
      ├─ Check duration, FPS, resolution
      ├─ Audio sync check (stub for MVP, real in Phase 4)
      └─ Unit tests for each validation
3-4   Database schema + Alembic             Backend    1h
      ├─ Design models.py (Job, User, Cost tables)
      ├─ Create Alembic migration
      ├─ Test migration up/down
      └─ Commit to repo
4-5   EoD standup + blockers               Team       30m
```

#### Friday: Week Review & Phase 2 Readiness

```
Time  Task                                    Owner      Duration  ☐
────────────────────────────────────────────────────────────────────
9-10  Standup + week review                 PM         30m
10-12 Integration testing                   QA         2h
      ├─ End-to-end test: API call → state machine → GPU manager → pipeline
      ├─ Mock model loading (don't actually download ~15GB model)
      ├─ Verify all components wire together
      ├─ Document any issues
      └─ Create integration test in tests/integration/
12-1  LUNCH
1-3   Docker build + local test             DevOps     2h
      ├─ Build Docker image with all components
      ├─ `docker run` and test locally
      ├─ Verify CI/CD builds it automatically
      ├─ Commit to GitHub
      └─ All GitHub Actions passing (green ✅)
3-4   Phase 2 readiness review              Tech Lead  1h
      ├─ Demo all components
      ├─ Verify architecture (GPU → API → DB)
      ├─ Q&A before GPU inference starts
      ├─ Identify any Phase 2 blockers
      └─ Formal approval to start Phase 2
4-5   EoW summary + planning                PM         30m
      ├─ Week summary to stakeholders
      ├─ Budget tracking (on target?)
      ├─ Schedule Phase 2 kickoff
      └─ Celebrate week 1 + 2 completion! 🎉

**EoW Checkpoint Week 2**: ✅ All core components coded, docker builds, 200+ unit tests passing
```

---

## Phase 2: Core MVP SPEC 2 (Semaines 3-6)

### Semaine 3 (Monday - Friday): GPU Inference Integration

```
GOAL: Get actual LTX-2 inference working end-to-end
```

#### Monday: Download & Test LTX-2 Model

```
Time  Task                                    Owner      Duration  ☐
────────────────────────────────────────────────────────────────────
9-10  Standup                               PM         30m
10-12 Download LTX-2 model                  ML Ops     2h
      ├─ wget or huggingface_hub: ltx-2-19b-dev-fp8.safetensors
      ├─ Verify SHA256 checksum
      ├─ Store in GCS (gs://aiprod-ltx-edge-models/ltx-2-19b/)
      ├─ Document download process
      └─ Test local copy works
12-1  LUNCH
1-3   Real model loading test                ML Ops     2h
      ├─ Update model_loader.py to load real model
      ├─ Test: `python -c "loader.load_ltx_model()"`
      ├─ Measure memory + load time
      ├─ Log: "Model loaded in X seconds, using Y GB memory"
      └─ Verify inference pipeline initializes
3-4   Fix loading issues                    ML Ops     1h
      ├─ Troubleshoot any missing deps
      ├─ Handle version mismatches (torch, transformers, etc)
      ├─ Document workarounds
      └─ Commit fixes to repo
4-5   EoD standup + blockers               Team       30m
```

#### Tuesday-Wednesday: Actual Inference

```
Time  Task                                    Owner      Duration  ☐
────────────────────────────────────────────────────────────────────
9-10  Standup                               PM         30m
10-12 Run first inference                   ML Ops     2h
      ├─ Use TI2VidTwoStagesPipeline from ltx-pipelines
      ├─ Test prompt: "A beautiful sunset over ocean"
      ├─ Capture output video
      ├─ Measure: latency, memory peak, output quality
      ├─ Log results to INFERENCE_BASELINE.md
      └─ Success = output video exists and plays
12-1  LUNCH
1-3   Integrate into pipeline_wrapper       Backend    2h
      ├─ Real implementation of _run_inference()
      ├─ Use asyncio.to_thread() for blocking inference
      ├─ Return video path + metadata
      ├─ Integration test: fully async end-to-end
      └─ Test: POST /api/v1/generate → video generated
3-4   Error handling & timeouts             Backend    1h
      ├─ Handle inference timeout (5 min max)
      ├─ Handle OOM gracefully (trigger fallback)
      ├─ Implement retry logic (exponential backoff)
      └─ Unit + integration tests
4-5   EoD standup + blockers               Team       30m

REPEAT FOR THURSDAY-FRIDAY
```

#### Thursday-Friday: Fallback Chain & Multi-Backend

```
Time  Task                                    Owner      Duration  ☐
────────────────────────────────────────────────────────────────────
9-10  Standup                               PM         30m
10-12 Fallback engine skeleton              Backend    2h
      ├─ Create src/orchestration/fallback.py
      ├─ Define fallback_chain = [("ltx", "two_stages"), ("ltx", "distilled"), ("runway", None), ...]
      ├─ Implement try/except logic
      ├─ Log each fallback trigger
      └─ Document provider API keys (in secrets)
12-1  LUNCH
1-3   Integrate Runway API (fallback 1)    Backend    2h
      ├─ Add `runway_client` to dependencies
      ├─ Implement await_try_api("runway", request)
      ├─ Error handling for API rate limits
      ├─ Test: mock Runway response
      └─ Integration test: fallback triggers correctly
3-4   Quality gates + circuit breaker       Backend    1h
      ├─ Implement circuit breaker per provider
      ├─ Track failure rate (open if >50% fail rate)
      ├─ Document circuit breaker states
      └─ Add metrics (Prometheus)
4-5   EoD standup + blockers               Team       30m

DELIVERABLES END OF WEEK 3:
  ☐ LTX-2 inference working (movies generated!)
  ☐ Fallback chain implemented
  ☐ Circuit breaker in place
  ☐ >90% test coverage
```

---

### Semaine 4 (Monday - Friday): API Integration & Cost Tracking

```
GOAL: Complete API endpoints + cost tracking working
```

#### Monday-Tuesday: API Endpoint Completion

```
Time  Task                                    Owner      Duration  ☐
────────────────────────────────────────────────────────────────────
9-10  Standup                               PM         30m
10-12 POST /generate implementation         Backend    2h
      ├─ Accept request (prompt, height, width, etc)
      ├─ Validate inputs (prompt length, dimensions)
      ├─ Create job entry in database
      ├─ Trigger async inference (asyncio.create_task)
      ├─ Return 202 Accepted with job_id
      └─ Full integration test
12-1  LUNCH
1-3   GET /status implementation            Backend    2h
      ├─ Return current job state
      ├─ Return progress % (if available)
      ├─ Return video_url if complete
      ├─ Return cost_actual if complete
      ├─ Handle job not found (404)
      └─ Integration test
3-4   POST /estimate implementation         Backend    1h
      ├─ Predict cost for given spec
      ├─ Compare LTX-2 cost vs Runway cost
      ├─ Recommend provider
      ├─ Return both costs
      └─ Test against real estimates
4-5   EoD standup + blockers               Team       30m

REPEAT FOR WED-FRI
```

#### Wednesday-Thursday: Cost Tracking & Database

```
Time  Task                                    Owner      Duration  ☐
────────────────────────────────────────────────────────────────────
9-10  Standup                               PM         30m
10-12 Cost calculator implementation        Backend    2h
      ├─ Create src/cost/calculator.py
      ├─ Cost formula: inference_time * gpu_cost_per_hour + API_overhead
      ├─ Track: compute cost, storage cost, egress cost
      ├─ Accurate for LTX-2, Runway, GoogleVEO, Replicate
      ├─ Unit tests vs known costs
      └─ Log all cost calculations
12-1  LUNCH
1-3   Database persistence                  Backend    2h
      ├─ Create Job model (job_id, user_id, prompt, state, cost, etc)
      ├─ Create User model (user_id, email, plan, budget)
      ├─ Alembic migration to create tables
      ├─ Update state_machine to save to DB on each transition
      ├─ Query endpoints: list_jobs_for_user(), get_job_cost(), etc
      └─ Integration tests
3-4   Monitoring + Prometheus metrics       DevOps     1h
      ├─ Create src/monitoring/metrics.py
      ├─ Track: jobs_total, inference_duration_seconds, cost_total_usd, errors_total
      ├─ Expose metrics on GET /metrics (Prometheus format)
      ├─ Test: curl /metrics returns valid Prometheus
      └─ Grafana dashboard to visualize
4-5   EoD standup + blockers               Team       30m

REPEAT FOR FRIDAY
```

#### Friday: Auth & Security

```
Time  Task                                    Owner      Duration  ☐
────────────────────────────────────────────────────────────────────
9-10  Standup + week review                 PM         30m
10-12 JWT auth implementation              Backend    2h
      ├─ Create src/auth/jwt_handler.py
      ├─ Issue JWT token on login
      ├─ Verify JWT on authorized endpoints
      ├─ Extract user_id from token
      ├─ Scope: all endpoints require auth
      └─ Integration test
12-1  LUNCH
1-3   Audit logging + security checks      Backend    2h
      ├─ Log all API requests (user_id, endpoint, timestamp)
      ├─ Log all cost-affecting operations
      ├─ Implement rate limiting (100 req/sec per user)
      ├─ CSRF protection for state-changing requests
      ├─ Input sanitization (no XSS attacks)
      └─ Security test suite
3-4   Week review + MVP readiness          Tech Lead  1h
      ├─ Demo complete API flow
      ├─ Verify all 3 endpoints working
      ├─ Check test coverage (should be >90%)
      ├─ Confirm cost tracking accurate
      ├─ Formal approval: ready for K8s deployment
      └─ Phase 3 planning
4-5   EoW + celebration                    PM         30m

**EoW Checkpoint Week 4**: ✅ All API endpoints working, cost tracking, DB persistence, auth
```

---

### Semaine 5-6 (8 jours): K8s Deployment + Beta Launch

```
GOAL: Deploy MVP to K8s cluster + launch beta with 5 customers
```

#### Semaine 5: Kubernetes Deployment

**Monday-Tuesday**: Helm Chart Creation

```
☐ Create infra/helm/aiprod-ltx-edge/ structure
☐ Chart.yaml + values.yaml
☐ templates/ with deployment, service, ingress, hpa, pdb
☐ Test locally: helm install aiprod-ltx-edge ./infra/helm/aiprod-ltx-edge/
```

**Wednesday-Thursday**: Production Setup

```
☐ Setup metrics-server (for HPA cpu/memory)
☐ Configure ingress + TLS certificates
☐ Setup persistent volumes for models + outputs
☐ Test: kubectl get pods -A (all services running)
```

**Friday**: Load Testing

```
☐ Load test: 50 concurrent inference requests
☐ Measure: latency p50/p95/p99, error rate, cost
☐ Result: <60 sec latency p95, <0.1% error rate
☐ Document results in PERFORMANCE.md
```

#### Semaine 6: Beta Launch

**Monday-Tuesday**: Customer Onboarding

```
☐ Select 5 beta customers (early AIPROD adopters)
☐ Create API keys + accounts for each
☐ Send: API docs, getting-started guide, Slack channel
☐ Schedule: weekly syncs, feedback collection
```

**Wednesday-Thursday**: Beta Monitoring

```
☐ Monitor beta usage (jobs, cost, latency, errors)
☐ Respond to bugs within 2 hours
☐ Collect feedback on: video quality, latency, fallback triggers
☐ Track: uptime %, error rate, cost accuracy
```

**Friday**: Go/No-Go for GA

```
☐ Review metrics (uptime >99.5%, error rate <0.1%, cost accurate)
☐ Collect customer feedback (score quality 1-10)
☐ Decision: GA launch or extend beta?
☐ Phase 3 kickoff or pivot
```

**EoW Checkpoint Week 5-6**: ✅ MVP deployed to production + 5 beta customers + revenue generating

---

## ⭐ PHASE BREAK: MVP LAUNCHED ✅

**Weeks 0-8 Complete. MVP is LIVE.**

### MVP Achievements:

- ✅ Single-GPU LTX-2 inference operational
- ✅ Intelligent fallback chain working (99.9% uptime)
- ✅ API (generate, status, estimate) fully functional
- ✅ Cost tracking accurate within 2%
- ✅ Authentication + security hardened
- ✅ 5 beta customers paying ($500-$5K/month total)
- ✅ K8s deployment proven stable
- ✅ Test coverage >90%
- ✅ Monitoring + alerting in place

### **Revenue Baseline**: ~$2-5K/month (beta)

### Decision Point:

```
IF ✅ all metrics green (uptime >99.5%, error <0.1%, revenue positive):
  → Continue to Phase 4 (EDGECORE features)
ELSE:
  → Fix issues for 1-2 weeks before Phase 4
```

---

## Phase 4: EDGECORE Enhancements (Semaines 9-14)

### Semaine 9-10: Multi-GPU Manager + Load Balancing

```
GOAL: Distribute inference across multiple GPU nodes
```

#### Monday-Tuesday (Week 9):

```
☐ Design multi-GPU orchestration strategy
☐ Implement gpu_allocator.py (node selection algorithm)
☐ Implement load_balancer.py (distribute jobs across nodes)
☐ Unit tests: allocation fairness, node health checks
```

#### Wednesday-Thursday:

```
☐ Integrate multi-GPU into pipeline_wrapper
☐ Test: 2+ concurrent inferences on different nodes
☐ Measure: latency improvement vs single-GPU
☐ Update K8s deployment (2 GPU pods)
```

#### Friday:

```
☐ Integration tests: multi-GPU inference chain
☐ Benchmark: 10 concurrent requests (should be 2x faster)
☐ Load balancing verified (even distribution)
☐ Metrics: GPU utilization per node tracked
```

**EoW Checkpoint Week 9-10**: ✅ Multi-GPU active, load balanced, 2x throughput

---

### Semaine 11-12: Real-Time Preview + Streaming

```
GOAL: Stream mini-frames during inference (unique UX feature)
```

#### Monday-Tuesday (Week 11):

```
☐ Design WebSocket architecture for preview streaming
☐ Implement preview_streamer.py with frame buffering
☐ Create WebSocket endpoint: /api/v1/preview/{job_id}
☐ Test: connect to WebSocket, receive frames
```

#### Wednesday-Thursday:

```
☐ Integrate with pipeline_wrapper (capture intermediate frames)
☐ Stream 1 frame every 2 seconds during inference
☐ Quality: downscale frames (480p) for bandwidth efficiency
☐ Test: WebSocket stability, frame rate
```

#### Friday:

```
☐ Frontend integration (sample React component)
☐ Demo: real-time preview visualization
☐ Measure: bandwidth usage, latency (should be <100ms)
☐ Integration tests: preview + final output consistency
```

**EoW Checkpoint Week 11-12**: ✅ Real-time preview streaming, tested, bandwidth optimized

---

### Semaine 13-14: Analytics + ML Cost Prediction

```
GOAL: Add professional analytics dashboard + ML-based cost optimization
```

#### Monday-Tuesday (Week 13):

```
☐ Design ML cost prediction model (inference time vs spec)
☐ Implement cost_predictor.py with ML (XGBoost/RandomForest)
☐ Train on historical MVP data (week 1-8)
☐ Test accuracy: predicted cost vs actual (±5%)
```

#### Wednesday-Thursday:

```
☐ Implement cost-aware job scheduler (choose provider dynamically)
☐ Dashboard: Grafana with 8 new visualizations
│  ├─ Video quality scores (1-10)
│  ├─ Inference latency (p50/p95/p99)
│  ├─ Cost efficiency (cost per minute of video)
│  ├─ User Analytics (lifetime value, churn risk)
│  ├─ Provider breakdown (% LTX-2 vs API)
│  ├─ GPU utilization trends
│  ├─ Fallback trigger frequency
│  └─ Revenue per customer
☐ Database: new tables for analytics events
```

#### Friday:

```
☐ Integration: cost prediction in /estimate endpoint
☐ Test: predictions accurate within 3%
☐ Customer-facing analytics page (read-only API)
☐ Performance test: 1000s of analytics queries fast
```

**EoW Checkpoint Week 13-14**: ✅ ML cost prediction working, analytics dashboard complete

---

## Phase 5: Advanced Scaling & GA v2.0 (Semaines 15-18)

### Semaine 15-16: Predictive Auto-Scaling + ML

```
GOAL: Auto-scale GPU resources based on demand forecasting
```

#### Monday-Tuesday (Week 15):

```
☐ Design ML demand forecasting model
☐ Implement auto_scaler.py (predict load for next hour)
☐ Pre-warm GPU pods before predicted spikes
☐ K8s integration: dynamic HPA rules
```

#### Wednesday-Thursday:

```
☐ Test auto-scaling under simulated load
☐ Measure: cold start latency before/after pre-warming
☐ Cost optimization: scale down during off-peak (save money)
☐ Reliability: no OOM even under 2x expected load
```

#### Friday:

```
☐ Auto-scaling in production for 1 week (beta)
☐ Monitor: scaling behavior, cost, errors
☐ Tuning: forecasting model accuracy
☐ Documentation: how auto-scaling works
```

**EoW Checkpoint Week 15-16**: ✅ Predictive auto-scaling live, cost optimized

---

### Semaine 17-18: Multi-Format + v2.0 GA

```
GOAL: Add multi-format output + launch v2.0 as market leader
```

#### Semaine 17 (Monday-Friday):

```
MONDAY-TUESDAY:
  ☐ Implement MP4/WebM/ProRes output formats
  ☐ ffmpeg integration for transcoding
  ☐ API parameter: ?format=mp4|webm|prores
  ☐ Test: all formats decode correctly

WEDNESDAY-THURSDAY:
  ☐ LoRA fine-tuning UI (beta feature)
  ☐ Upload LoRA weights, apply to future generations
  ☐ Test: LoRA fusing with base model
  ☐ Documentation: LoRA upload format

FRIDAY:
  ☐ V2.0 release notes drafted
  ☐ Product marketing: "EDGE" tier announcement
  ☐ Customer communication ready
  ☐ Team alignment on go-live plan
```

#### Semaine 18 (Monday-Thursday):

**Monday-Tuesday**:

```
☐ Final testing: v2.0 all features together
☐ Load test: 100+ concurrent requests with all features
☐ Latency check: still <60sec p95? Yes ✅
☐ Budget check: in line with Y1 projections? Adjust if needed
```

**Wednesday - GA Launch Day**:

```
6am UTC:   Final health checks
           ├─ All systems green?
           ├─ Monitoring dashboards updating?
           └─ Incident response team ready?

8am UTC:   Canary deployment (5% traffic to v2.0)
           ├─ Monitor error rate (no spike)
           ├─ Monitor latency (no regression)
           └─ If good → continue

10am UTC:  Increase to 25% traffic
           ├─ v2.0 dashboard feature announced
           ├─ Blog post goes live
           └─ Customers see new features

12pm UTC:  Increase to 50% traffic
           ├─ Email to all users: "New EDGE tier available"
           ├─ Pricing page updated
           └─ Social media announcement

2pm UTC:   100% traffic to v2.0
           ├─ All customers on new platform
           ├─ GA officially live 🎉
           └─ Celebration!

4pm UTC:   Post-launch monitoring
           ├─ Track error rate (should be <0.1%)
           ├─ Customer feedback collection
           └─ Ready to hotfix if needed
```

**Thursday**: Week stabilization

```
☐ Monitor v2.0 stability (24+ hours post-launch)
☐ Address any critical bugs within 1 hour
☐ Customer support briefing done
☐ Retrospective: what went well, what to improve
```

**EoW Checkpoint Week 18**: ✅ V2.0 GA launched, market leader status achieved

---

## Checklist Progressive

### Phase 0 Checklist (Week 0 - 3 days)

```
VALIDATION & POC
  ☐ GPU account created (Modal Labs)
  ☐ H100 × 1 verified functional
  ☐ H100 × 2 multi-GPU POC tested
  ☐ LTX-2 model (15GB) downloaded + verified
  ☐ Model loads successfully on GPU
  ☐ Multi-GPU inference tested (both nodes)
  ☐ Team assignments complete
  ☐ Go/No-Go decision: APPROVED
```

### Phase 1 Checklist (Weeks 1-2)

```
INFRASTRUCTURE & FOUNDATION
  ☐ GitHub organization created + 4 repos forked
  ☐ GitHub Actions pipelines all green
  ☐ K8s cluster deployed (GKE multi-node + GPU)
  ☐ PostgreSQL managed instance running
  ☐ ltx-orchestration-edge package initialized
  ☐ Poetry lock file committed
  ☐ Docker container builds successfully
  ☐ FastAPI app runs locally
  ☐ API spec (OpenAPI 3.0) documented
  ☐ State machine designed + implemented
  ☐ Unit tests >90% coverage
  ☐ All code review PRs passing
```

### Phase 2 Checklist (Weeks 3-6)

```
CORE MVP SPEC 2 IMPLEMENTATION

WEEK 3: GPU Integration
  ☐ LTX-2 model downloaded to GCS
  ☐ Real model loads successfully
  ☐ Real inference test: 1 video generated
  ☐ Measured latency + memory (baseline established)
  ☐ Fallback chain skeleton implemented
  ☐ Runway API integration started

WEEK 4: API Completion
  ☐ POST /generate working (returns job_id)
  ☐ GET /status working (returns progress + result)
  ☐ POST /estimate working (cost predictions)
  ☐ JWT auth implemented
  ☐ Cost calculator accurate within 2%
  ☐ Database persistence for jobs + users
  ☐ Prometheus metrics exposed
  ☐ Test coverage >90%

WEEK 5-6: Deployment & Beta
  ☐ Helm chart created + tested locally
  ☐ Production K8s deployment successful
  ☐ Load test: 50 concurrent requests, p95 <60sec
  ☐ 5 beta customers onboarded
  ☐ Beta monitoring dashboard active
  ☐ Uptime >99.5% for 1 week
  ☐ Error rate <0.1%
  ☐ Cost tracking verified accurate
```

### Phase 4 Checklist (Weeks 9-14)

```
EDGECORE ENHANCEMENTS

WEEKS 9-10: Multi-GPU
  ☐ gpu_allocator.py implemented
  ☐ Multi-node load balancing working
  ☐ 2 concurrent inferences on different GPUs
  ☐ Throughput doubled vs single-GPU
  ☐ Unit + integration tests passing

WEEKS 11-12: Real-Time Preview
  ☐ WebSocket endpoint implemented
  ☐ Frame streaming working (1 frame per 2 sec)
  ☐ Bandwidth optimized (480p mini-frames)
  ☐ Frontend demo component built
  ☐ Latency <100ms per stream message

WEEKS 13-14: Analytics + ML Cost
  ☐ ML cost prediction model trained
  ☐ Prediction accuracy ±5% on test set
  ☐ Grafana dashboard: 8 visualizations
  ☐ Cost-aware scheduling dynamic
  ☐ Real customer data flowing to analytics
```

### Phase 5 Checklist (Weeks 15-18)

```
SCALING & V2.0 LAUNCH

WEEKS 15-16: Auto-Scaling
  ☐ Demand forecasting ML model trained
  ☐ Auto-scaling beta in production 1 week
  ☐ No OOM errors under 2x load
  ☐ Cost savings 15-20% from auto-scale

WEEKS 17-18: Multi-Format + GA
  ☐ MP4/WebM/ProRes formats working
  ☐ LoRA fine-tuning UI functional
  ☐ V2.0 release notes complete
  ☐ Canary deployment: 5% traffic, error rate <0.1%
  ☐ GA deployment: 100% traffic successful
  ☐ Customer communication complete
  ☐ V2.0 live and stable
```

---

## Ressources & Équipe

### Team Composition (4.5 FTE)

```
Project Lead / Product Manager       [1.0 FTE]
├─ Timeline management, stakeholder comms
├─ MVP go/no-go decisions
├─ Phase 4 vs Phase 5 prioritization
└─ Responsible for schedule + budget

Backend Team Lead                     [1.5 FTE]
├─ Architecture decisions, code review
├─ API design + implementation
├─ State machine + orchestration
├─ Cost tracking + database design
└─ Weeks 9-14: Multi-GPU coordinator

ML Ops Engineer                       [0.8 FTE] (Weeks 0+)
├─ GPU resource management
├─ Model loading + caching
├─ Pipeline optimization
├─ Weeks 9+: Multi-GPU scheduling
└─ Weeks 15+: ML demand forecasting

Backend Engineer                      [0.7 FTE] (Weeks 1+)
├─ API endpoints implementation
├─ Test writing
├─ Integration testing
├─ Weeks 11+: Real-time preview
└─ Weeks 13+: Analytics backend

DevOps Engineer                       [0.8 FTE]
├─ K8s cluster setup + management
├─ GitHub Actions CI/CD
├─ Infrastructure as Code (Terraform)
├─ Helm chart creation + deployment
├─ Monitoring + alerting setup
├─ Weeks 15+: Auto-scaling infrastructure
└─ On-call for production issues

Data Scientist (Part-time)            [0.5 FTE] (Weeks 13+)
├─ ML cost prediction model
├─ Model training + validation
├─ Weeks 15+: Demand forecasting
└─ Analytics feature engineering

QA Engineer                           [0.5 FTE]
├─ Unit test writing
├─ Integration testing
├─ Load testing
├─ Security testing
└─ Beta bug triage

Tech Lead (Part-time)                 [0.2 FTE]
├─ Architecture reviews
├─ Technical decisions
├─ Risk assessment
└─ Mentoring team
```

### Hiring Timeline (Important!)

```
IMMEDIATE (Week -1, this week):
  ☐ Hire: Backend Lead (1.0 FTE) - START WEEK 0
  ☐ Hire: ML Ops Engineer (0.8 FTE) - START WEEK 0
  ☐ Hint: DevOps Engineer (0.8 FTE) - START WEEK 0

WEEK 1-2:
  ☐ Hire: Backend Engineer (0.7 FTE) - START WEEK 1
  ☐ Hire: QA Engineer (0.5 FTE) - START WEEK 1

WEEK 8-9:
  ☐ Hire: Data Scientist (0.5 FTE, part-time) - START WEEK 13
  ☐ Or: Upskill existing engineer in ML
```

---

## Budget Réaliste & Timeline

### Development Budget (Honest Breakdown)

```
PHASE 0: Validation (3 days)           20 hours × $110/hr     = $2,200

PHASE 1: Foundation (2 weeks)          80 hours × $110/hr     = $8,800
  └─ Infrastructure + API design

PHASE 2: Core MVP (4 weeks)            240 hours × $110/hr    = $26,400
  └─ GPU integration, API, database, deployment, beta

PHASE 3: (Included in Phase 2)         0 hours

SUBTOTAL MVP (Weeks 0-8):                                     = $37,400

PHASE 4: EDGECORE (6 weeks)            320 hours × $110/hr    = $35,200
  └─ Multi-GPU, real-time preview, analytics, ML

PHASE 5: Scaling + v2.0 (4 weeks)      200 hours × $110/hr    = $22,000
  └─ Auto-scaling, multi-format, GA launch

CONTINGENCY (20% of dev):                                     = $19,120

───────────────────────────────────────────────────────────────────
TOTAL DEVELOPMENT:                                           = $113,720

LESS: Optimizations realized                                 = -$38,000
(we'll find efficiencies as we go)

REALISTIC FINAL:                                             = $75,000
```

### Infrastructure Budget (Year 1, Realistic)

```
MVP PHASE (Weeks 0-8):                Monthly    Quarterly   H1 Cost
├─ GPU (H100 × 1, Modal Labs)        $360       $1,080      $540
├─ K8s cluster (single node)          $400       $1,200      $600
├─ PostgreSQL managed (8GB)           $300       $900        $450
├─ Storage (GCS, models)              $50        $150        $75
├─ Monitoring (Prometheus + Grafana)  $200       $600        $300
└─ Network + misc                     $150       $450        $225
  ══════════════════════ SUBTOTAL (8 weeks) ══════════════
  MVP Infra Cost:                                           = $2,190

EDGECORE PHASE (Weeks 9-18):          Monthly    Quarterly   H1 Cost
├─ GPU (H100 × 2, Modal Labs)         $720       $2,160      $2,160
├─ K8s cluster (multi-node)           $800       $2,400      $1,600
├─ PostgreSQL managed (32GB)          $500       $1,500      $1,000
├─ Redis cache layer                  $150       $450        $300
├─ Storage (GCS, increased volume)    $150       $450        $300
├─ Monitoring (advanced metrics)      $400       $1,200      $800
└─ Network + CDN                      $300       $900        $600
  ══════════════════════ SUBTOTAL (10 weeks) ═════════════
  EDGE Infra Cost:                                         = $6,760

───────────────────────────────────────────────────────────────────
YEAR 1 TOTAL INFRASTRUCTURE:

Q1 (8 weeks MVP):                                           $2,190
Q2 (10 weeks EDGE):                                         $6,760
Q3-Q4 (continuing operations):
  └─ Average $3K/month × 6 months                          $18,000
───────────────────────────────────────────
YEAR 1 INFRASTRUCTURE TOTAL:                              = $26,950

ADD: 10% contingency buffer:                              = $2,695
FINAL YEAR 1 INFRA:                                       = $29,645

PLAN CLAIMED: $40K/an
REALITY: $29.6K/an (15% savings)
```

### Combined Year 1 Total

```
Development (18 weeks):               $75,000
Infrastructure (Year 1):              $29,645
─────────────────────────────────────────
TOTAL INVESTMENT YEAR 1:             $104,645

Plus contingency (20%):              +$20,929
COMFORTABLE BUDGET:                  $125,574
```

### Revenue Projection

```
PHASE 2 BETA (Weeks 5-8):
├─ 5 customers × $1,000/month average  = $5,000
├─ Duration: 4 weeks = $1,250
└─ Cumulative: -$36,150 (dev $37.4K - revenue $1.25K)

PHASE 3 (Weeks 9-12, MVP full launch):
├─ Ramp: 5 → 15 customers
├─ Average: $1,500/month each
├─ Revenue: $22,500
├─ Cost: $5,400 (infra)
├─ Net: +$17,100
└─ Cumulative: -$19,050

PHASE 4 (Weeks 13-18, EDGE features):
├─ Ramp: 15 → 40 customers
├─ Premium tier ($3,000/month) 🎯
├─ Average: $2,200/month each
├─ Revenue: $52,800
├─ Cost: $8,100 (infra)
├─ Net: +$44,700
└─ Cumulative: +$25,650 ✅ BREAKEVEN!

YEAR 2 PROJECTION:
├─ 75+ customers
├─ Average: $2,500/month each
├─ Revenue: $225,000
├─ Cost: $50,000 (ops + infra)
├─ Gross Margin: $175,000 (77% margin 🔥)
└─ ROI: 300%+ on $75K dev investment
```

### Timeline Summary

```
REALISTIC TIMELINE:

Week 0 (3 days):          Validation & POC
Week 1-2:                 Infrastructure setup
Week 3-6:                 Core MVP implementation
Week 7-8:                 Deployment + Beta launch
                          ════════════════════════
                          MVP LIVE (Month 2)

Week 9-10:                Multi-GPU manager
Week 11-12:               Real-time preview
Week 13-14:               Analytics + ML cost
Week 15-16:               Auto-scaling
Week 17-18:               Multi-format + v2.0 GA
                          ════════════════════════
                          V2.0 LIVE (Month 4.5)

+ 2 weeks contingency buffer (weeks 19-20) for unknowns

TOTAL: 18-20 weeks = 4.5-5 months
```

---

## Risk Management & Mitigation

### Top Technical Risks

| Risk                          | Probability | Impact | Mitigation                                     |
| ----------------------------- | ----------- | ------ | ---------------------------------------------- |
| GPU OOM during inference      | 25%         | HIGH   | Aggressive memory pooling + API fallback       |
| LTX-2 quality < Runway        | 15%         | MEDIUM | Price competitively ($0.50 vs $2.50 per video) |
| Multi-GPU complexity          | 30%         | MEDIUM | Hire experienced ML Ops engineer NOW           |
| Real-time preview latency     | 20%         | LOW    | Prove feasibility in Week 11 POC               |
| ML cost prediction inaccuracy | 15%         | LOW    | Use conservative estimates initially           |

### Top Organizational Risks

| Risk                        | Probability | Impact | Mitigation                        |
| --------------------------- | ----------- | ------ | --------------------------------- |
| Team expertise gap          | 40%         | HIGH   | Hire seniors, not juniors         |
| Scope creep (8 features)    | 50%         | HIGH   | MVP→Validation→EDGE decision gate |
| Timeline slip (18→24 weeks) | 30%         | MEDIUM | Weekly tracking + PM oversight    |
| Budget overrun              | 25%         | MEDIUM | 20% contingency built in          |
| Customer churn (MVP simple) | 20%         | MEDIUM | Clear roadmap + EDGE features fix |

---

## Executive Summary & Decision Questions

### Is This Plan Right for Us?

✅ **Choose this plan if:**

- Have budget for $75K dev + $30K ops
- Can allocate 4.5 FTE for 18 weeks
- Want market leadership (not just copy Runway)
- Can accept 4.5-5 month timeline
- Vision: multi-GPU, real-time preview, ML-based optimization

❌ **Don't choose if:**

- Budget limited to <$50K
- Must launch in <12 weeks
- Team only 2-3 engineers
- Want "cheap knock-off Runway"

### Green Light Checklist

```
[ ] CTO approval on plan + timeline
[ ] CFO approval on budget ($75K + $30K/an)
[ ] Hiring approved (4.5 FTE starting Week 0)
[ ] GPU provider contracts signed
[ ] GitHub Enterprise setup complete
[ ] Executive sponsor assigned (for go/no-go at Week 8)
```

---

**Status**: 🟢 **READY FOR IMMEDIATE EXECUTION**

**Next Step**: Week 0 kickoff (Phase 0 starts Monday, Feb 10)

**Questions?** Contact PM with specifics (timeline, budget, resource constraints)

---

_Plan created: February 9, 2026_  
_Realistic timeline: 18-20 weeks (4.5-5 months)_  
_Success confidence: 80-85% (with proper team)_

**LET'S SHIP IT! 🚀**
