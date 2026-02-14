# AIPROD - Proprietary Model Creation Plan

**Status** : 🟡 Phase 0: Research (Not Started Yet)
**Decision Date** : 2026-02-10
**Owner** : Averroes
**Visibility** : Private (Propriétaire)

---

## ⚠️ CRITICAL CLARIFICATION

**🔴 THIS PLAN IS AN ADDENDUM - NOT A PROJECT REFACTORING**

### What STAYS UNCHANGED ✅
- ✅ **Code Architecture** : 3 packages (core, pipelines, trainer) - UNCHANGED
- ✅ **Pipelines** : distilled, one_stage, two_stages, ic_lora, keyframe_interpolation - UNCHANGED
- ✅ **Infrastructure** : GTX 1070, PyTorch 2.5.1+cu121 - UNCHANGED
- ✅ **Project Structure** : All folders, configs, scripts - UNCHANGED
- ✅ **Concept** : AIPROD framework design - UNCHANGED

### What CHANGES ⚡
- ⚡ **ONLY**: Adding proprietary model weights (Phase 0-4)
- ⚡ **ONLY**: Research to inform model architecture
- ⚡ **ONLY**: Training code for new models

### Bottom Line
This plan = **"How to create proprietary models for AIPROD"**
NOT = "How to refactor/redesign AIPROD"

---

## Executive Summary

**PROJECT STATUS** : AIPROD 90% complete
- ✅ Code source complet (3 packages: core, pipelines, trainer)
- ✅ Infrastructure GPU configurée (GTX 1070, PyTorch 2.5.1+cu121)
- ✅ Pipelines opérationnels (distilled, one_stage, two_stages)
- ✅ Environment prêt à l'emploi

**WHAT'S MISSING** : Proprietary model weights (Phase 0-4)

**STRATEGY** : 
1. **Phase 0** : Research LTX-2 to understand patterns (NOT copy)
2. **Phase 1** : Design novel architecture based on learnings
3. **Phase 2** : Train proprietary models using AIPROD's existing code
4. **Phase 3** : Validate models with AIPROD's pipelines
5. **Phase 4** : Release to HuggingFace

**SCOPE OF THIS PLAN** :
- ✅ Research phase (2-3 weeks)
- ✅ Architecture design phase (1 week)
- ✅ Model training (1-3 months on GTX 1070)
- ✅ Validation & release (2-4 weeks)

**SCOPE NOT INCLUDED** (STAYS AS-IS) :
- ❌ Modifying existing code architecture
- ❌ Changing pipeline implementations
- ❌ Refactoring infrastructure
- ❌ Altering project structure

**Résultat** : 100% propriétaire models, zéro modification du code AIPROD existant

**Timeline** : 2-6 mois total (phase 0-4)
**Budget** : Variable (GTX 1070 = lent, Cloud H100 = 1-5K€)

---

## ✅ CONFIRMATION: What This Plan Does and Doesn't Do

### This Plan DIRECTLY USES Your Existing AIPROD Code
```python
# Phase 2 Training will use:
from aiprod_trainer import APIPRODTrainer  # ← Existing
from aiprod_pipelines import DistilledPipeline  # ← Existing
from aiprod_core import schedulers, guiders  # ← Existing

# Your new models will run through existing pipelines:
pipeline = DistilledPipeline(model_path="./models/aiprod_proprietary.safetensors")
video = pipeline.infer(prompt="Your prompt")  # ← Uses existing code
```

### This Plan DOES NOT Modify
- ❌ `packages/aiprod-core/` source code
- ❌ `packages/aiprod-pipelines/` implementations
- ❌ `packages/aiprod-trainer/` architecture
- ❌ Any configuration files
- ❌ Project structure or organization

### This Plan ONLY Adds
- ✅ `models/ltx2_research/` (reference materials)
- ✅ `models/aiprod_proprietary/` (your new models)
- ✅ `docs/AIPROD_V2_RESEARCH_NOTES.md` (research documentation)
- ✅ Training data pipeline (external, not in code)

---

## 🎯 RECOMMENDED MODEL CONFIGURATION FOR YOUR PROJECT

### Your Project Analysis ✅
| Component | Specification |
|-----------|---------------|
| **GPU** | GTX 1070 (8GB VRAM) |
| **PyTorch** | 2.5.1+cu121 (CUDA enabled) |
| **Pipelines** | 5 optimized pipelines ready |
| **Architecture** | 3 packages (core, pipelines, trainer) |

### Scenario 1: BEST BALANCE (Recommended to Start) ⭐

**What to Download Now:**
```
1. ltx-2-19b-dev-fp8.safetensors (18GB)
   ├─ Optimal pipeline: TI2VidTwoStagesPipeline
   ├─ Quality: HIGH (production-ready)
   ├─ Speed: ~2-3 min per video
   └─ Recommended for: Production, quality optimization

2. ltx-2-spatial-upscaler-x2-1.0.safetensors (6GB)
   ├─ Purpose: 2x spatial upsampling
   ├─ Required for: two_stages pipeline (BEST QUALITY)
   └─ Adds: Professional-grade visuals

Total Size: ~24GB on disk
VRAM Used: 6-7GB on GTX 1070 (comfortable, leaves room for OS)
Total Time to Download: 30-60 minutes
```

**Use Case**: Production, maximum quality output

---

### Scenario 2: MAXIMUM PERFORMANCE (If Space Limited)

**What to Download:**
```
1. ltx-2-19b-distilled-fp8.safetensors (5GB)
   ├─ Optimal pipeline: DistilledPipeline
   ├─ Quality: GOOD (acceptable degradation)
   ├─ Speed: ~30-60 sec per video
   └─ Recommended for: Prototyping, quick tests

Total Size: 5GB only
VRAM Used: 3-4GB on GTX 1070 (very comfortable)
Total Time to Download: 5-10 minutes
```

**Use Case**: Rapid prototyping, testing workflows

---

## 🚀 BEST RECOMMENDATION FOR YOU

### ✅ START WITH SCENARIO 1 (Best Balance)

| Component | Size | Reason | Download Link |
|-----------|------|--------|---------------|
| **ltx-2-19b-dev-fp8.safetensors** | 18GB | Optimal quality/performance for GTX 1070 | https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-dev-fp8.safetensors |
| **ltx-2-spatial-upscaler-x2-1.0.safetensors** | 6GB | Enables two_stages pipeline (best output) | https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors |

**Total: ~24GB** (comfortable for GTX 1070 8GB VRAM)

### Why This Configuration ✨

| Aspect | Detail |
|--------|--------|
| **Quality** | FP8 provides 95% quality of full precision |
| **VRAM** | 6-7GB used on GTX 1070 (leaves ~1-2GB for OS/others) |
| **Production-Ready** | Spatial upscaler included = professional quality |
| **Flexibility** | Can test both DistilledPipeline AND TI2VidTwoStagesPipeline |
| **Officially Tested** | Recommended config by LTX-2 for GTX 1070 |

---

## 🔄 WHICH AIPROD PIPELINE TO USE

### For Phase 1 (Development & Testing)
```python
from aiprod_pipelines import DistilledPipeline

# Fast testing - verify workflows work
pipeline = DistilledPipeline(model_path="./models/ltx2_research/ltx-2-19b-distilled-fp8.safetensors")
video = pipeline.infer(prompt="Test prompt")  # ~30-60 sec
```

### For Phase 2+ (Production Training & Inference)
```python
from aiprod_pipelines import TI2VidTwoStagesPipeline

# Best quality - full production pipeline
pipeline = TI2VidTwoStagesPipeline(
    model_path="./models/ltx2_research/ltx-2-19b-dev-fp8.safetensors",
    upsampler_path="./models/ltx2_research/ltx-2-spatial-upscaler-x2-1.0.safetensors"
)
video = pipeline.infer(prompt="Production prompt")  # ~2-3 min but BEST quality
```

---

## 📥 DOWNLOAD COMMAND (Optimized)

### One-Click Download
```powershell
cd C:\Users\averr\AIPROD
.\scripts\download_ltx2_research.ps1

# When prompted, choose: Option 1 (RECOMMENDED)
# FP8 + Spatial Upscaler (~24GB total)
```

### Or Manual Download
```powershell
# Create directory
mkdir models/ltx2_research
cd models/ltx2_research

# Download main model
huggingface-cli download Lightricks/LTX-2 \
  --repo-type model \
  --local-dir . \
  --include "ltx-2-19b-dev-fp8.safetensors"

# Download spatial upscaler
huggingface-cli download Lightricks/LTX-2 \
  --repo-type model \
  --local-dir . \
  --include "ltx-2-spatial-upscaler-x2-1.0.safetensors"
```

### Expected Timeline
- **Download time**: 30-60 minutes (depends on internet speed)
- **Extracted space**: ~24GB on disk
- **Ready to use**: Phase 0 research immediately after

---

## 📋 BEFORE YOU START PHASE 0

### Checklist
- [ ] Read this plan completely
- [ ] Understand: This adds models, doesn't refactor AIPROD
- [ ] Confirm: You want Option 1 (FP8 + Upscaler)
- [ ] Check: You have ~25GB free disk space
- [ ] Ready: HuggingFace account (for authentication)

### What Happens Next
1. **Now** → Download models to `models/ltx2_research/`
2. **Week 1** → Analyze LTX-2 architecture (Phase 0)
3. **Week 2-3** → Design novel AIPROD architecture (Phase 0)
4. **Month 1+** → Train proprietary models (Phase 1-2)

---

## 🏗️ MASTER PROJECT TIMELINE (Model + Deployment Roadmap)

### Overview: Parallel Tracks

```
Your project has TWO parallel tracks:
════════════════════════════════════

TRACK 1 (ML): Model Training & Architecture
├─ Goal: Create proprietary AI models
├─ Phases: 0 (Research), 1-4 (Training, Validation, Release)
└─ Timeline: Feb 2026 → Aug 2026 (6 months)

TRACK 2 (Ops): Deployment Infrastructure
├─ Goal: Make models accessible & professional
├─ Phases: Phase 0 Ops (nothing), Phase 1 Ops (API+DB), Phase 2 Ops (Docker+Monitor)
└─ Timeline: May 2026 → Sept 2026 (parallel with track 1)

RESULT: By Sept 2026 = Models trained + Infrastructure production-ready
```

### Week-by-Week Timeline

```
FEBRUARY 2026 (PHASE 0: Model Research)
════════════════════════════════════════

Week 1 (Feb 10-16):
├─ ML: Download LTX-2 models, begin architecture analysis
├─ Ops: ❌ SKIP (focus 100% on ML research)
└─ Status: "Infrastructure decisions are frozen, focus is pure research"

Week 2-4 (Feb 17 - Mar 10):
├─ ML: Complete Phase 0 research document
├─ ML: Define 5 Innovation Domains (backbone, VAE, text, temporal, training)
├─ Ops: ❌ SKIP (still in research phase)
└─ Status: "Architecture decisions locked in before Phase 1"


MAY 2026 (PHASE 1: MVP Production - Model Training Begins + API Kickoff)
═════════════════════════════════════════════════════════════════════════

Week 1-2 (May 1-15): Parallel Start
├─ ML: Begin Phase 1 training setup + Stage 1 training starts
├─ OPS: 🟡 START: REST API implementation (Week 1-2 effort: 1 week)
│       └─ POST /api/v1/generate (basic)
│       └─ GET /api/v1/jobs/{id}
│       └─ Database schema design (conceptual)
├─ Result: API skeleton ✅
└─ Status: "Both tracks moving in parallel"

Week 3-4 (May 16-31):
├─ ML: Stage 1 training continues (on GPU, will take weeks)
├─ Ops: 🟡 CONTINUE: Database implementation (Week 2-3 effort: 2 weeks)
│       └─ Create PostgreSQL schema (jobs, cost_log)
│       └─ Integrate DB with API
│       └─ Test end-to-end
├─ Result: API + DB functional ✅
└─ Status: "Core infrastructure ready"


JUNE 2026 (PHASE 1: MVP Production Completing)
═════════════════════════════════════════════════

Week 1-2 (Jun 1-15):
├─ ML: Stage 1 training nearing completion
├─ Ops: 🟡 CONTINUE: Dead-simple auth (1 week effort)
│       └─ API key validation (3 days)
│       └─ Rate limiting basic (3 days)
├─ Result: API + Database + Basic Auth ✅
└─ Status: "MVP infrastructure complete"

Week 3-4 (Jun 16-30):
├─ ML: Stage 1 training complete + Start Stage 2 (fine-tuning)
├─ Ops: 🟡 START: Docker containerization (2 weeks effort)
│       └─ Create Dockerfile
│       └─ docker-compose.yml for local testing
│       └─ Test deployment locally
├─ Result: Code containerized & testable ✅
└─ Status: "Ready to onboard first beta clients (July)"


JULY 2026 (PHASE 1.5: Beta Launch + PHASE 2 Ops Begins)
═════════════════════════════════════════════════════════

Week 1-2 (Jul 1-15):
├─ ML: Stage 2 training + Initial validation
├─ Ops: 🟡 MILESTONE: First beta clients onboarded!
│       └─ Deploy to production (GPU server)
│       └─ Support 3-5 clients with APIs
├─ Revenue: ✅ FIRST REVENUE (licensing model starts)
└─ Status: "Operational with paying customers"

Week 3-4 (Jul 16-31):
├─ ML: Validation & quality testing
├─ Ops: 🟡 START: Monitoring + Logging (3 weeks effort total)
│       └─ Health checks (API up?)
│       └─ GPU health monitoring
│       └─ Error tracking & logging
├─ Result: Observability in place ✅
└─ Status: "Professional monitoring active"


AUGUST 2026 (PHASE 2: Professional Operations)
═════════════════════════════════════════════════

Week 1-4 (Aug 1-31):
├─ ML: Phase 3 validation + Optimization
├─ Ops: 🟡 CONTINUE: All monitoring + Add Cost tracking
│       └─ Cost calculation per video
│       └─ Billing system (manual invoices or Stripe)
│       └─ CI/CD pipeline (automated testing + Docker build)
├─ Result: Production-grade infrastructure ✅
└─ Status: "Scaling to 10-20 paying customers"


SEPTEMBER 2026 (PHASE 2 Complete)
═══════════════════════════════════

├─ ML: Model optimization + Finalization
├─ Ops: Infrastructure mature
│      └─ 10-20 clients running
│      └─ Automated deployments
│      └─ Professional operations
└─ Status: "Ready for Phase 3 enterprise features IF needed"


OCTOBER 2026+ (PHASE 3: Enterprise IF Requested)
══════════════════════════════════════════════════

├─ ML: Start Phase 4 (Release) or continue fine-tuning
├─ Ops: Only IF customer demands
│      ├─ JWT + Firebase → Add (6-7 weeks)
│      ├─ RBAC → Add (1-2 weeks)
│      ├─ Prometheus metrics → Add (2-3 weeks)
│      ├─ Audit logging → Add (1-2 weeks)
│      └─ Only allocate effort when client signs contract
└─ Status: "Enterprise-ready by Q4 2026 IF needed, else focus on models"
```

---

## 📋 DEPLOYMENT TODO LIST (Phases Parallèles)

### PHASE 0 OPS: February-April 2026 (Nothing)

```
❌ DO NOT START operational work yet
✅ Focus: Pure ML research only

Rationale:
├─ Every minute on Ops = less time on model research
├─ Model quality >> infrastructure polish (at this stage)
├─ Premature optimization wastes 2-3 weeks
└─ Infrastructure tasks wait, research doesn't

Effort Saved: 2-3 weeks = valuable research time
```

### PHASE 1 OPS: May-June 2026 (MVP Production Layer)

```
🎯 Goal: Build minimal API so external clients can use your models

TASK 1: REST API (Minimal, 10 endpoints only)
─────────────────────────────────────────────
Status: ⏳ START in May (Week 1-2)
├─ Implement:
│  ├─ POST   /api/v1/generate              (main endpoint)
│  ├─ GET    /api/v1/jobs/{id}             (check status)
│  ├─ GET    /api/v1/jobs/{id}/download    (get video)
│  ├─ POST   /api/v1/jobs/{id}/cancel      (stop job)
│  ├─ GET    /api/v1/models                (list available)
│  ├─ POST   /api/v1/estimate-cost         (pricing)
│  ├─ GET    /api/v1/admin/stats           (your dashboard)
│  └─ 3+ internal endpoints
│
├─ Framework: FastAPI + Uvicorn (Python)
├─ Effort: 2 weeks
├─ Checklist:
│  - [ ] Install FastAPI, Pydantic, Uvicorn
│  - [ ] Create main.py with all 10 endpoints
│  - [ ] Add request validation
│  - [ ] Add error handling
│  - [ ] Test with curl + Python client
│  - [ ] Document API (auto-generated by FastAPI)
│
└─ Priority: 🔴 CRITICAL (without API, clients can't use you)

TASK 2: Database (Simple schema, 2 tables)
──────────────────────────────────────────
Status: ⏳ START in May (Week 3-4)
├─ Setup:
│  ├─ PostgreSQL local (for dev) + RDS ready (for prod)
│  ├─ SQLAlchemy ORM models
│  ├─ Alembic migrations (database versioning)
│
├─ Schema (only 2 tables):
│  │
│  ├─ Table 1: jobs
│  │  ├─ job_id (UUID)
│  │  ├─ api_key (who requested)
│  │  ├─ prompt (what they asked)
│  │  ├─ model_version (which model)
│  │  ├─ status (pending/running/completed/error)
│  │  ├─ output_path (where video saved)
│  │  ├─ cost_usd (how much charged)
│  │  ├─ created_at, completed_at, error_message
│  │  └─ metadata (JSON)
│  │
│  └─ Table 2: cost_log
│     ├─ date (YYYY-MM-DD)
│     ├─ total_cost_usd (your daily costs)
│     ├─ total_videos_generated (volume)
│     ├─ profit_margin
│     └─ notes
│
├─ Effort: 3 weeks
├─ Checklist:
│  - [ ] Install PostgreSQL + SQLAlchemy + Alembic
│  - [ ] Define SQLAlchemy models (jobs, cost_log, api_keys)
│  - [ ] Create Alembic migration script
│  - [ ] Test CRUD operations
│  - [ ] Integrate with API (save jobs to DB)
│  - [ ] Test multi-client isolation
│
└─ Priority: 🔴 CRITICAL (without DB, can't track anything professionally)

TASK 3: Dead-Simple Auth (API keys)
───────────────────────────────────
Status: ⏳ START in June (Week 1-2)
├─ Implementation:
│  ├─ Table: api_keys (key_hash, client_name, active, created_at)
│  ├─ Generate: Random string as API key
│  ├─ Validate: Check key on every request
│  ├─ Log: Track usage (who did what, when)
│
├─ Code:
│  @app.post("/api/v1/generate")
│  def generate_video(request: GenerateRequest, api_key: str = Header(...)):
│      key = db.session.query(APIKey).filter_by(key=hash(api_key)).first()
│      if not key:
│          return {"error": "Invalid API key"}, 401
│      # Log usage
│      db.session.add(JobLog(api_key_id=key.id, ...))
│      # Run pipeline
│      return {"job_id": "..."}, 202
│
├─ Effort: 3 days
├─ Checklist:
│  - [ ] Add api_keys table to DB
│  - [ ] Create key generation function
│  - [ ] Add auth middleware to all endpoints
│  - [ ] Add usage logging
│  - [ ] Test with real API calls
│
└─ Priority: 🟡 HIGH (security of base, prevents random abuse)

TASK 4: Docker Containerization
────────────────────────────────
Status: ⏳ START in June (Week 3-4)
├─ Files:
│  ├─ Dockerfile (production image)
│  │  ├─ Base: nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04
│  │  ├─ Install: Python 3.11, PyTorch, AIPROD packages
│  │  ├─ Copy: API code + model paths
│  │  ├─ Expose: Port 8000
│  │  └─ CMD: ["python", "-m", "uvicorn", "api.main:app"]
│  │
│  └─ docker-compose.yml (local dev)
│     ├─ Service 1: api (FastAPI)
│     ├─ Service 2: postgres (database)
│     ├─ Volumes: models, data, logs
│     └─ Network: api ↔ postgres
│
├─ Effort: 2 weeks
├─ Checklist:
│  - [ ] Create Dockerfile (optimize for size)
│  - [ ] Test Docker build locally
│  - [ ] Create docker-compose.yml
│  - [ ] Test with docker-compose up
│  - [ ] Verify API works in container
│  - [ ] Verify GPU access in container
│  - [ ] Test volume mounts
│
└─ Priority: 🟡 HIGH (enables any deployment)

SUMMARY PHASE 1 OPS:
───────────────────
Duration: May-June 2026 (7-8 weeks)
Parallel: During model training (Stage 1)
Effort: ~2 weeks + 3 weeks + 3 days + 2 weeks = ~7.5 weeks
Result: ✅ REST API + Database + Auth + Docker Container
Status: "MVP infrastructure production-ready"
```

### PHASE 2 OPS: July-September 2026 (Professional Operations)

```
🎯 Goal: Make infrastructure professional & reliable

TASK 1: Monitoring + Health Checks
──────────────────────────────────
Status: ⏳ START in July (Week 1-2)
├─ Implement:
│  ├─ Health Check Endpoint
│  │  ├─ GET /health → returns {"status": "ok"}
│  │  └─ Used by load balancers + monitoring
│  │
│  ├─ GPU Health Monitoring
│  │  ├─ VRAM usage (alert if > 90%)
│  │  ├─ Temperature (alert if > 80°C)
│  │  ├─ No critical memory errors
│  │  └─ Exposed as /metrics/gpu
│  │
│  ├─ API Monitoring
│  │  ├─ Request count per endpoint
│  │  ├─ Error rate (2xx vs 4xx vs 5xx)
│  │  ├─ Latency per endpoint (p50, p95, p99)
│  │  └─ Exposed as /metrics/api
│  │
│  └─ Error Tracking
│     ├─ Every error to error_log table
│     ├─ Stack trace captured
│     ├─ User impact scored
│     └─ Dashboard: "Top 10 errors today"
│
├─ Effort: 2 weeks
├─ Checklist:
│  - [ ] Add /health endpoint
│  - [ ] Monitor GPU with nvidia-ml-py
│  - [ ] Track metrics with Python counters
│  - [ ] Add error logging table
│  - [ ] Create simple dashboard (HTML + JS)
│  - [ ] Set up alerts (email when GPU temp > 80°C)
│  - [ ] Test failure scenarios
│
└─ Priority: 🟡 MEDIUM (prevents surprises)

TASK 2: Cost Tracking + Billing
───────────────────────────────
Status: ⏳ START in August (Week 1-2)
├─ Implement:
│  ├─ Cost Calculation
│  │  ├─ Per-video cost = (GPU time * hourly_cost) + storage
│  │  ├─ Example: 100 sec video on GTX 1070 = ~0.50€
│  │  ├─ Store in jobs table (cost_usd)
│  │  └─ Aggregate to cost_log daily
│  │
│  ├─ Billing System
│  │  ├─ Option 1: Manual invoices (Excel → Send email)
│  │  ├─ Option 2: Stripe integration (automatic)
│  │  ├─ Choose based on client sophistication
│  │  └─ Track: revenue per customer
│  │
│  └─ Dashboard
│     ├─ Your daily profit = revenue - costs
│     ├─ Per-customer margin
│     ├─ Volume (videos/day)
│     └─ Projection (monthly run rate)
│
├─ Effort: 1-2 weeks
├─ Checklist:
│  - [ ] Define cost formula (your decision)
│  - [ ] Test calculations
│  - [ ] Create billing dashboard (Excel or simple UI)
│  - [ ] Integrate with DB
│  - [ ] Test invoicing workflow
│  - [ ] Send first invoice to beta client
│
└─ Priority: 🟠 MEDIUM (need to know profitability)

TASK 3: CI/CD Pipeline
──────────────────────
Status: ⏳ START in August (Week 2-3)
├─ Implement:
│  ├─ GitHub Actions workflow (.github/workflows/deploy.yml)
│  │  ├─ Trigger: on git push to main
│  │  ├─ Step 1: Run tests (pytest)
│  │  ├─ Step 2: Build Docker image
│  │  ├─ Step 3: Push to registry (Docker Hub or private)
│  │  └─ Step 4: Deploy to production
│  │
│  ├─ Tests to run:
│  │  ├─ Fast: Unit tests (1 min)
│  │  ├─ Medium: Integration tests (5 min)
│  │  └─ Only fast tests on every push (full tests on schedule)
│  │
│  └─ Deployment:
│     ├─ ssh to prod server
│     ├─ Pull new Docker image
│     ├─ Stop old container
│     ├─ Start new container
│     └─ Health check (verify /health endpoint)
│
├─ Effort: 1-2 weeks
├─ Checklist:
│  - [ ] Create .github/workflows/deploy.yml
│  - [ ] Configure GitHub secrets (SSH key, registry credentials)
│  - [ ] Test CI locally (act --reuse-containers)
│  - [ ] Push test commit, verify deployment
│  - [ ] Rollback procedure documented
│  - [ ] Monitoring alert on deploy failure
│
└─ Priority: 🟠 LOW-MEDIUM (nice to have, manual deploy fine initially)

SUMMARY PHASE 2 OPS:
───────────────────
Duration: July-September 2026 (5-7 weeks)
Parallel: During model validation (Phase 3 ML)
Effort: ~2 weeks + ~2 weeks + ~1.5 weeks = ~5.5 weeks
Result: ✅ Professional monitoring + cost tracking + automated deployment
Status: "Enterprise-quality operations (minus fancy dashboards)"
```

### PHASE 3 OPS: October 2026+ (Enterprise IF Needed)

```
🎯 Goal: Only implement IF enterprise customers demand

ONLY DO IF:
├─ Customer XYZ says: "We need JWT + RBAC"
├─ Customer ABC says: "We need audit logging"
├─ Your compliance person says: "We need SOC2"
└─ Budget = customer contract value

IF customer demands JWT + RBAC:
├─ Implement: 6-7 weeks
├─ Impact: ✅ Production
└─ Timeline: October 2026+

IF customer demands Prometheus metrics:
├─ Implement: 2-3 weeks (add to existing health checks)
├─ Impact: ✅ Beautiful dashboards
└─ Timeline: October 2026+

ELSE (no customer demands):
├─ Skip: All Phase 3 ops features
├─ Focus: Improve model quality instead
├─ Timeline: N/A (not needed)
└─ Mantra: "Ship models, not infrastructure"

Decision Rule:
└─ "Revenue > Cost" → Do it
└─ "Cost > Revenue" → Postpone
```

---

## Phase 0 : Research & Strategy (Semaine 1-4)

**⚠️ PARALLEL OPS STATUS: FROZEN**
```
During Phase 0:
├─ 🎯 ALL effort → Model research
├─ ❌ NO operational work yet
├─ ⏸️  Infrastructure tasks paused
└─ Rationale: Model research cannot wait; ops can
```

### Tâche 0.1 : Analyser LTX-2 Architecture (RESEARCH ONLY - Pour apprendre)
- [ ] Télécharger modèles LTX-2 dans `models/ltx2_research/` (référence)
- [ ] Étudier backbone Transformer LTX-2 (inspirations seulement)
- [ ] Analyser VAE design LTX-2 (concepts, pas code)
- [ ] Documenter text encoder integration (comprendre patterns)
- [ ] Identifier pain points et limitations
- [ ] **IMPORTANT**: Prendre notes, PAS copier code ou poids
**Owner**: Averroes | **Due**: Week 1

### Tâche 0.2 : Définir Innovation Domains
Documenter pour chaque domaine:

#### Domain 1: Backbone Architecture
**Question** : Garder Transformer ou innover?
- [ ] Option A: Mamba/SSM instead of Attention
- [ ] Option B: Hybrid Attention + Local Conv
- [ ] Option C: Reformer/Performer sparse patterns
- [ ] Option D: Hybrid Vision+Language backbone
**Decision** : _________

#### Domain 2: Video Codec (VAE)
**Question** : VAE structure et compression?
- [ ] Custom VAE from scratch
- [ ] Improve temporal compression
- [ ] Multi-scale latent space
- [ ] Quantization strategy
**Decision** : _________

#### Domain 3: Text Understanding
**Question** : Intégration language model?
- [ ] Keep Gemma (fast path)
- [ ] Add multilingual support
- [ ] Custom embeddings
- [ ] Vision-language fusion
**Decision** : _________

#### Domain 4: Temporal Modeling
**Question** : How to track motion over time?
- [ ] Cross-frame attention
- [ ] Optical flow guidance
- [ ] Predictive latents
- [ ] Novel frame interpolation
**Decision** : _________

#### Domain 5: Training Methodology
**Question** : New training approaches?
- [ ] Custom loss functions
- [ ] Curriculum learning strategy
- [ ] Multi-stage training (stage1=base, stage2=quality)
- [ ] Reinforcement learning rewards
**Decision** : _________

### Tâche 0.3 : Resource Planning
```
GPU Budget         : [  ] €/month
Timeline Estimate  : [  ] months
Data Size Target   : [  ] hours video
Infrastructure     : [  ] (GTX1070 | Cloud | Colab | Hybrid)
```

**Owner**: Averroes | **Due**: TBD

---

## Phase 1 : Model Creation & Training (Month 1-3) - UTILISER CODE AIPROD EXISTANT

**⚠️ PARALLEL OPS MILESTONE: START REST API + DATABASE**
```
During Phase 1 (May-June):
├─ ML Track: Stage 1 training in progress
├─ Ops Track: API + Database layer implementation (7-8 weeks)
├─ Timing: API ready by end of June (before beta clients)
├─ Result: By June 30: Models training + Infrastructure ready
└─ See timeline above for detailed schedule
```

### Tâche 1.1 : Concevoir Architecture AIPROD (basée sur Phase 0 research)
- [ ] Documenter architecture decisions (backbone, VAE, text encoder, temporal, training)
- [ ] Créer `AIPROD_architecture_spec.md` (design document)
- [ ] Valider avec Phase 0 research notes
- [ ] **NOTE**: Code infrastructure AIPROD déjà existant et prêt

**Architecture to Design** (utilisant framework AIPROD existant):
```
packages/aiprod-pipelines/src/aiprod_pipelines/models/
├── AIPROD/
│   ├── __init__.py
│   ├── backbone.py       (Your novel architecture)
│   ├── vae.py            (Custom codec)
│   ├── text_encoder.py   (Language integration)
│   ├── inference.py      (Unified pipeline) ← Utilise code existant
│   └── config.py         (Hyperparameters)
```

**Owner**: Averroes | **Due**: Week 3 (after Phase 0)

### Tâche 1.2 : Prepare Training Data (utiliser outils AIPROD existants)
- [ ] Source/collect video data (target: 100-500 hours)
- [ ] Use AIPROD preprocessing pipeline (déjà existant)
- [ ] Encode to latent space (scripts AIPROD trainer ready)
- [ ] Create caption annotations (aiprod-trainer tools)
- [ ] Split train/val/test (AIPROD utilities)

**Data Structure** (AIPROD-compliant):
```
data/aiprod_training/
├── raw_videos/          (MP4, MKV, etc)
├── preprocessed/        (Encoded latents)
├── captions.json        (Text descriptions)
└── splits/
    ├── train.json
    ├── val.json
    └── test.json
```

**Owner**: Averroes | **Due**: Month 1-2

### Tâche 1.3 : Infrastructure Setup (✅ DÉJÀ EXISTANT)
- [x] GTX 1070 configured and tested (ALREADY DONE)
- [x] PyTorch 2.5.1+cu121 installed (ALREADY DONE)
- [x] AIPROD environment ready (ALREADY DONE)
- [ ] Decide additional: Cloud H100 for Phase 2? (Optional)
- [ ] AIPROD logging/monitoring already configured
- [ ] Checkpointing strategy in AIPROD trainer

**Owner**: Averroes | **Due**: N/A (COMPLETE)

---

## Phase 2 : Model Training (Month 2-6) - UTILISER AIPROD TRAINER EXISTANT

**⚠️ PARALLEL OPS MILESTONE: BETA LAUNCH + PROFESSIONAL OPS**
```
During Phase 2 (July-September):
├─ ML Track: Stage 2 training + Validation in progress
├─ Ops Track: Docker deployment + Monitoring/Cost tracking (5-7 weeks)
├─ Timing: First 3-5 beta clients onboarded by July 1
├─ Revenue: ✅ First licensing revenue begins
└─ Result: By September: Professional operations + 10-20 paying customers
```

### Tâche 2.1 : Implement Training Loop (utiliser AIPROD trainer existant)
- [ ] Utiliser `packages/aiprod-trainer/scripts/train.py` (déjà existant)
- [ ] Configurer pour architecture AIPROD novel
- [ ] Optimizer loss functions (AIPROD has framework ready)
- [ ] Utiliser learning rate scheduling AIPROD

**Optimization for GTX 1070**:
```python
# Mixed precision + checkpointing
model = model.to(torch.bfloat16)
model = checkpoint_sequential(model, segments=4)
batch_size = 1  # Very small
gradient_accumulation = 16
```

**Owner**: Averroes | **Due**: TBD

### Tâche 2.2 : Stage 1 Training (Base model)
- [ ] Train backbone + VAE
- [ ] Target: 50 hours video data
- [ ] Monitor: Loss curves, VRAM usage
- [ ] Save checkpoints every 1000 steps

**Owner**: Averroes | **Timeline**: 4-8 weeks

### Tâche 2.3 : Stage 2 Training (Quality refinement)
- [ ] Fine-tune on curated high-quality data
- [ ] Focus on prompt adherence
- [ ] Optimize inference speed
- [ ] Save production checkpoint

**Owner**: Averroes | **Timeline**: 2-4 weeks

---

## Phase 3 : Validation (Month 5-7)

**⚠️ PARALLEL OPS STATUS: ENTERPRISE FEATURES ONLY IF CUSTOMER DEMANDS**
```
During Phase 3 (Sep-Oct):
├─ ML Track: Quality validation + Optimization
├─ Ops Track: PAUSE (only add if customer contract justifies)
│  ├─ IF customer demands JWT → Allocate 6-7 weeks
│  ├─ IF customer demands Prometheus → Allocate 2-3 weeks
│  └─ ELSE → Focus on model improvements
└─ Mantra: "Revenue-driven ops, not feature-creep"
```

### Tâche 3.1 : Qualitative Testing
- [ ] Generate samples from various prompts
- [ ] Compare AIPROD v2 vs LTX-2 baselines (benchmarking seulement)
- [ ] Document AIPROD strengths/weaknesses
- [ ] Iterate AIPROD architecture if needed

**Owner**: Averroes | **Due**: TBD

### Tâche 3.2 : Performance Optimization
- [ ] Profile model (where is time spent?)
- [ ] Implement optimizations (kernel fusion, pruning)
- [ ] Benchmark on GTX 1070: inference FPS
- [ ] Create optimization guide

**Owner**: Averroes | **Due**: TBD

### Tâche 3.3 : Documentation
- [ ] Write model card (architecture, training data, license)
- [ ] Create usage examples
- [ ] Document all design decisions
- [ ] License: © Averroes (100% proprietary)

**Owner**: Averroes | **Due**: TBD

---

## Phase 4 : Release (Month 7-8)

**⚠️ PARALLEL OPS STATUS: INFRASTRUCTURE STABLE + SCALING**
```
During Phase 4 (Oct-Nov):
├─ ML Track: Final models available for public release
├─ Ops Track: Everything running smoothly
│  ├─ 20+ paying customers
│  ├─ Automated CI/CD deployments
│  ├─ Professional monitoring active
│  └─ Ready to scale
└─ Status: "Models complete + Ops infrastructure production-grade"
```

### Tâche 4.1 : Upload to Averroes10/AIPROD
- [ ] Create model weights release
- [ ] Upload to HuggingFace (private or public)
- [ ] Version: `AIPROD_base_final.safetensors`
- [ ] Update README with v2 info

**License Header**:
```
AIPROD v2 Model Weights
© 2026 Averroes. All rights reserved.
Proprietary Model - Restricted Use
Architecture: Fully original, not derivative of LTX-2
Training: Custom data, custom methodology
```

### Tâche 4.2 : Update Inference Pipeline
- [ ] Modify `examples/quickstart.py` to use v2
- [ ] Add v2-specific optimizations
- [ ] Test end-to-end pipeline
- [ ] Benchmark latency

### Tâche 4.3 : Public Communication
- [ ] Blog post: "AIPROD v2 Released"
- [ ] Technical report: Architecture details
- [ ] Model card on HuggingFace

---

## Budget & Resources

### Compute Options

| Option | Cost | Duration | Quality |
|--------|------|----------|---------|
| **GTX 1070 Solo** | 0€ | 6-12 mo | Good (slow) |
| **H100 Rental (40h)** | 1200€ | 5-10 days | Excellent (fast) |
| **Modal/Lambda Cloud** | 2-5K€ | 2-4 weeks | Excellent |
| **On-prem Colab** | 0€ | 3-5 mo | Fair (interrupts) |

**Recommendation** : Hybrid
- Phase 1: GTX 1070 (setup, testing)
- Phase 2: H100 rental for stage 1 (1200€ one-time)
- Phase 3: GTX 1070 (validation, optimization)

### Data Costs
- Collection/annotation: 5K-20K€
- Licensing (if using commercial): Variable
- Storage (50GB models): 100€/year HF

### Total Budget Estimate
- **Low** : 0€ (self-hosted, takes 12 months)
- **Medium** : 2K€ (1-2 H100 sessions, 6 months)
- **High** : 20K€ (full commercial data + compute)

---

## Decision Checkpoints

**Before Phase 1 START** ✅
- [ ] Which innovations to pursue? (5 domains documented)
- [ ] Compute budget approved?
- [ ] Data plan finalized?

**Before Phase 2 START**
- [ ] Prototype architecture validated?
- [ ] Training data ready?
- [ ] Compute provisioned?

**Before Phase 3 START**
- [ ] Stage 1 training complete?
- [ ] Quality baseline established?
- [ ] Performance acceptable?

**Before Phase 4 START**
- [ ] Stage 2 training complete?
- [ ] Optimization done?
- [ ] All tests passing?

---

## 🎯 QUICK REFERENCE: What's Happening When?

### One-Page Overview (Copy-Paste into Calendar)

```
FÉVRIER 2026
════════════
W1 (10-16):  Research start              (Ops: 0%)
W2-4:        Complete Phase 0 research   (Ops: 0%)

MAI 2026
════════
W1-2:        Stage 1 training begins     (Ops: Start REST API 20%)
W3-4:        Stage 1 continues           (Ops: Build database 50%)

JUIN 2026
═════════
W1-2:        Stage 1 finishing           (Ops: Add auth + Docker 50%)
W3-4:        Stage 2 training starts     (Ops: Docker ready 100%)

JUILLET 2026
═════════════
W1-2:        Stage 2 training            (Ops: Deploy to prod, 1st clients! 📊)
W3-4:        Validation starting         (Ops: Add monitoring 50%)

AOÛT 2026
═════════
W1-4:        Validation + optimization   (Ops: Cost tracking + CI/CD 100%)

SEPTEMBRE 2026
═══════════════
W1-4:        Final tuning                (Ops: Mature infrastructure ✅)

OCTOBRE 2026+
══════════════
After:       Release + Scale             (Ops: Enterprise features IF needed)
```

### Key Decision Points

```
BEFORE MAY 1 (Phase 1 Start):
├─ [ ] Phase 0 research complete?
├─ [ ] Innovation domains decided (backbone, VAE, etc)?
├─ [ ] Training data prepared?
└─ → Go/No-go decision for Phase 1

BEFORE JULY 1 (Op's Launch):
├─ [ ] REST API code complete?
├─ [ ] Database schema tested?
├─ [ ] Docker container working?
├─ [ ] Stage 1 training on schedule?
└─ → Ready to take first clients?

BEFORE OCTOBER 1 (Enterprise Phase):
├─ [ ] 10+ paying customers happy?
├─ [ ] Stage 2 training complete?
├─ [ ] Professional monitoring active?
├─ [ ] Any customer demanding advanced auth?
└─ → Decide: enterprise features yes/no?
```

---

## Historical Log

| Date | Event | Owner |
|------|-------|-------|
| 2026-02-10 | Decision: Option A (Analyze LTX-2, build 100% novel AIPROD) | Averroes |
| 2026-02-10 | CONFIRMED: AIPROD project 90% complete, only models missing | Averroes |
| 2026-02-10 | Download LTX-2 models to models/ltx2_research/ (reference study) | Averroes |
| TBD | Phase 0 strategy doc complete | - |
| TBD | Phase 1 prototype ready | - |
| TBD | Phase 2 training starts | - |
| TBD | Phase 3 validation complete | - |
| TBD | Phase 4 v2 released | - |

---

## Next Immediate Actions

1. **TODAY** : Read this document, understand scope
2. **This week** : Answer all "Domain Decision" questions above
3. **Next week** : Setup Phase 1 prototype environment
4. **Month 1** : Complete Phase 0 research and decisions

---

**Questions? Ambiguities?** 
Document them in `AIPROD_FAQ.md`
