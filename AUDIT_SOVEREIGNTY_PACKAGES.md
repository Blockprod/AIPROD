# SOVEREIGNTY AUDIT — `aiprod-pipelines` & `aiprod-trainer`

**Date:** 2026-02-14  
**Scope:** ALL Python source files in `packages/aiprod-pipelines/` and `packages/aiprod-trainer/`  
**Method:** File-by-file read + grep scan for the `inference/` mega-module  

---

## TABLE OF CONTENTS

1. [ALL Import Statements](#1-all-import-statements)
2. [HTTP / API Calls](#2-http--api-calls)
3. [External API References](#3-external-api-references)
4. [API Key References](#4-api-key-references)
5. [Model Loading Code](#5-model-loading-code)
6. [Network Calls & URLs](#6-network-calls--urls)
7. [Real vs Stub Assessment](#7-real-vs-stub-assessment)
8. [from_pretrained() Calls](#8-from_pretrained-calls)
9. [wandb / mlflow Tracking](#9-wandb--mlflow-tracking)
10. [Cloud SDK Usage](#10-cloud-sdk-usage)

---

## 1. ALL Import Statements

### 1.1 `aiprod-trainer` — External (Non-stdlib, Non-internal) Imports

| Module | Import | File(s) |
|--------|--------|---------|
| `torch` | `import torch`, `torch.nn`, `torch.utils.data`, `torch.distributed`, `torch.cuda.amp` | trainer.py, vae_trainer.py, datasets.py, validation_sampler.py, gemma_8bit.py, timestep_samplers.py, curriculum_training.py, captioning.py |
| `wandb` | `import wandb` | trainer.py, vae_trainer.py |
| `peft` | `LoraConfig`, `get_peft_model` | trainer.py |
| `accelerate` | `Accelerator` | trainer.py |
| `safetensors.torch` | `load_file`, `save_file` | trainer.py |
| `bitsandbytes.optim` | `AdamW8bit` (lazy) | trainer.py |
| `optimum.quanto` | `freeze`, `qfloat8`, `quantize` | quantization.py |
| `huggingface_hub` | `HfApi`, `create_repo`, `upload_folder` | hf_hub_utils.py |
| `google.generativeai` | `import google.generativeai as genai` | captioning.py |
| `transformers` | `AutoModel`, `AutoTokenizer`, `BitsAndBytesConfig` | gemma_8bit.py, captioning.py |
| `torchvision.models` | `vgg16` | vae_trainer.py |
| `boto3` | `boto3.client('s3')` | streaming/sources.py |
| `google.cloud.storage` | `storage.Client()` | streaming/sources.py |
| `huggingface_hub` | `hf_hub_download`, `list_files_in_repo` | streaming/sources.py |
| `zstandard` | `ZstdCompressor`, `ZstdDecompressor` | streaming/cache.py |
| `imageio` | `mimwrite` | hf_hub_utils.py |
| `rich` | `Console`, `Table`, `Progress`, `Panel` | config_display.py, progress.py, __init__.py |
| `pydantic` | `BaseModel`, `Field`, `model_validator` | config.py |
| `numpy` | `import numpy as np` | utils.py, captioning.py |
| `PIL` | `Image` | utils.py, captioning.py |
| `av` | `av.open` | video_utils.py |

### 1.2 `aiprod-trainer` — Internal (aiprod_core) Imports

| Import | File(s) |
|--------|---------|
| `aiprod_core.loader.SingleGPUModelBuilder` | model_loader.py |
| `aiprod_core.components.*` | model_loader.py, validation_sampler.py |
| `aiprod_core.text_encoders.LLMBridge` | model_loader.py |
| `aiprod_core.vae.*` | model_loader.py, validation_sampler.py |

### 1.3 `aiprod-pipelines` — External (Non-stdlib, Non-internal) Imports

| Module | Import | File(s) |
|--------|--------|---------|
| `torch` | `import torch`, `torch.nn`, `torch.nn.functional` | all pipelines, color/, editing/, inference/ |
| `safetensors.torch` | `load_file` | distilled.py, ic_lora.py, keyframe_interpolation.py |
| `fastapi` | `FastAPI`, `WebSocket`, `HTTPException`, `Depends` | api/gateway.py, api/collaboration/websocket.py, inference/video_editing/api_gateway.py |
| `pydantic` | `BaseModel`, `Field` | api/gateway.py, api/schema/*.py |
| `google.generativeai` | `import google.generativeai as genai` | api/integrations/gemini_client.py |
| `google.cloud.storage` | `storage.Client()` | api/adapters/gcp_services.py |
| `google.cloud.logging` | `cloud_logging.Client()` | api/adapters/gcp_services.py |
| `google.cloud.monitoring_v3` | `MetricServiceClient()` | api/adapters/gcp_services.py |
| `google.api_core.exceptions` | exceptions | api/adapters/gcp_services.py |
| `stripe` | `stripe.Customer`, `stripe.Subscription` (lazy) | api/billing_service.py |
| `asyncpg` | `asyncpg.create_pool` (lazy) | api/tenant_store.py |
| `transformers` | `AutoModelForCausalLM`, `AutoTokenizer` (lazy) | inference/scenarist/scenarist.py |
| `auto_gptq` | `AutoGPTQForCausalLM` (lazy) | inference/optimization.py |
| `awq` | `AutoAWQForCausalLM` (lazy) | inference/optimization.py |
| `cv2` / `opencv` | lazy import | inference/video_editing/backend.py, inference/validation/ |
| `einops` | `rearrange` | inference/tiling/ |
| `numpy` | `import numpy as np` | color/, export/, inference/ (various) |
| `av` | `av.open` | utils/media_io.py |
| `rich` | `Console`, `Progress` | various |

### 1.4 `aiprod-pipelines` — Internal (aiprod_core / aiprod_pipelines) Imports

| Import | File(s) |
|--------|---------|
| `aiprod_core.loader.SingleGPUModelBuilder` | utils/model_ledger.py |
| `aiprod_core.components.*` | all pipelines, utils/helpers.py |
| `aiprod_core.text_encoders.LLMBridge` | utils/model_ledger.py |
| `aiprod_core.vae.*` | all pipelines |
| `aiprod_pipelines.utils.*` | all pipelines |

---

## 2. HTTP / API Calls

| Location | Mechanism | Target | Direction | Real/Stub |
|----------|-----------|--------|-----------|-----------|
| **trainer/captioning.py** — `GeminiFlashCaptioner` | `google.generativeai` SDK | Gemini Flash API (`gemini-flash-lite-latest`) | **OUTBOUND** | **REAL** |
| **trainer/hf_hub_utils.py** | `huggingface_hub.HfApi` | HuggingFace Hub | **OUTBOUND** (upload) | **REAL** |
| **trainer/streaming/sources.py** — `HuggingFaceDataSource` | `huggingface_hub.hf_hub_download` | HuggingFace Hub | **OUTBOUND** (download) | **REAL** |
| **trainer/streaming/sources.py** — `S3DataSource` | `boto3.client('s3').download_file` | AWS S3 | **OUTBOUND** | **REAL** |
| **trainer/streaming/sources.py** — `GCSDataSource` | `google.cloud.storage.Client` | Google Cloud Storage | **OUTBOUND** | **REAL** |
| **pipelines/api/sdk.py** — `AIPRODClient` | `urllib.request.urlopen` | `https://api.aiprod.ai` (configurable) | **OUTBOUND** | **REAL** |
| **pipelines/api/gateway.py** | `urllib.request` | Health-check endpoints | Internal | **REAL** |
| **pipelines/api/webhooks.py** — `WebhookManager` | `urllib.request.urlopen` | Tenant-configured callback URLs | **OUTBOUND** | **REAL** |
| **pipelines/api/integrations/gemini_client.py** | `google.generativeai` SDK | Gemini 1.5 Pro API | **OUTBOUND** | **REAL** (fallback to mock) |
| **pipelines/api/adapters/gcp_services.py** | `google.cloud.storage/logging/monitoring_v3` | GCP services | **OUTBOUND** | **REAL** |
| **pipelines/api/billing_service.py** | `stripe` SDK (lazy) | Stripe API | **OUTBOUND** | **REAL** (optional) |
| **pipelines/api/tenant_store.py** | `asyncpg` (lazy) | PostgreSQL database | **OUTBOUND** (DB) | **REAL** (optional) |
| **pipelines/api/adapters/render.py** | — | "runway_gen3" / "replicate_wan25" backends | — | **STUB** (mock) |
| **pipelines/api/adapters/qa_semantic.py** | — | Vision LLM API | — | **STUB** (heuristic fallback) |
| **pipelines/inference/desktop_plugins.py** | HTTP | `http://localhost:9100/v1` | **LOCAL** | Config default |

---

## 3. External API References

| External Service | Where Referenced | How Used | Active Call? |
|------------------|-----------------|----------|--------------|
| **Google Gemini API** | trainer/captioning.py, pipelines/api/integrations/gemini_client.py | Text/video generation, content analysis | **YES** |
| **HuggingFace Hub** | trainer/hf_hub_utils.py, trainer/streaming/sources.py | Model push, data download | **YES** |
| **AWS S3** | trainer/streaming/sources.py | Data source streaming | **YES** |
| **Google Cloud Storage** | trainer/streaming/sources.py, pipelines/api/adapters/gcp_services.py | Data source + asset storage | **YES** |
| **Google Cloud Logging** | pipelines/api/adapters/gcp_services.py | Structured logging | **YES** |
| **Google Cloud Monitoring** | pipelines/api/adapters/gcp_services.py | Custom metrics | **YES** |
| **Stripe** | pipelines/api/billing_service.py | Billing (optional, lazy import) | **YES** (if installed) |
| **PostgreSQL** (asyncpg) | pipelines/api/tenant_store.py | Tenant data storage | **YES** (if installed) |
| **Runway Gen3** | pipelines/api/adapters/render.py, financial_orchestrator.py | Referenced as render backend | **NO — STUB** |
| **Replicate / Wan2.5** | pipelines/api/adapters/render.py, financial_orchestrator.py | Referenced as render backend | **NO — STUB** |
| **Veo3** | pipelines/api/adapters/financial_orchestrator.py | Referenced as cost model backend | **NO — STUB** |
| **W&B (Weights & Biases)** | trainer/trainer.py, trainer/vae_trainer.py | Experiment tracking | **YES** |
| **HuggingFace Transformers** | trainer/captioning.py, gemma_8bit.py, inference/scenarist/scenarist.py, inference/optimization.py | Model loading via from_pretrained | **YES** (lazy) |
| **PyTorch Hub / torchvision** | trainer/vae_trainer.py | `vgg16(pretrained=True)` | **YES** |

---

## 4. API Key References

| Key / Secret | Env Var / Config | File | Usage |
|--------------|-----------------|------|-------|
| **Gemini API Key** | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | trainer/captioning.py | `genai.configure(api_key=...)` |
| **Gemini API Key** | Constructor param `api_key` | pipelines/api/integrations/gemini_client.py | `genai.configure(api_key=self.api_key)` |
| **AIPROD API Secret** | `AIPROD_API_SECRET` | pipelines/api/gateway.py | JWT signing, HMAC-SHA256 auth |
| **Stripe API Key** | Constructor param `api_key` → `stripe.api_key` | pipelines/api/billing_service.py | Stripe SDK auth |
| **HuggingFace Token** | Via `huggingface_hub` auth (implicit) | trainer/hf_hub_utils.py, streaming/sources.py | HF Hub API auth |
| **AWS Credentials** | Implicit via boto3 credential chain | trainer/streaming/sources.py | S3 access |
| **GCP Credentials** | Implicit via ADC (Application Default Credentials) | trainer/streaming/sources.py, pipelines/api/adapters/gcp_services.py | GCS / Logging / Monitoring |
| **Webhook Secret** | `secret` param in webhook config | pipelines/api/webhooks.py | HMAC-SHA256 payload signing |
| **Distributed env vars** | `WORLD_SIZE`, `RANK`, `LOCAL_RANK`, `MASTER_ADDR`, `MASTER_PORT` | pipelines/inference/tensor_parallelism/distributed_config.py | PyTorch distributed (not secrets) |
| **multi_tenant secret_key** | Internal auth field | pipelines/inference/multi_tenant_saas/ | Internal tenant auth token |

---

## 5. Model Loading Code

### 5.1 LOCAL-ONLY Loading (Sovereign ✅)

| File | Mechanism | Notes |
|------|-----------|-------|
| **trainer/model_loader.py** | `aiprod_core.loader.SingleGPUModelBuilder` with local path | All components (transformer, VAE, text encoder) from local safetensors |
| **pipelines/utils/model_ledger.py** | `aiprod_core.loader.SingleGPUModelBuilder` with local path | Central model coordinator, enforces local paths |
| **pipelines/distilled.py** | `safetensors.torch.load_file(ckpt_path)` | LoRA weights from local path |
| **pipelines/ic_lora.py** | `safetensors.torch.load_file(lora_path)` | IC-LoRA from local path |
| **pipelines/keyframe_interpolation.py** | `safetensors.torch.load_file(...)` | Keyframe model from local path |
| **pipelines/ti2vid_one_stage.py** | `safetensors.torch.load_file(...)` | Single-stage model from local path |
| **pipelines/ti2vid_two_stages.py** | `safetensors.torch.load_file(...)` | Two-stage model from local path |
| **trainer/datasets.py** | `torch.load(path)` | Training data (.pt files) from local disk |
| **trainer/gemma_8bit.py** | `AutoModel.from_pretrained(..., local_files_only=True)` | **Forces local-only** ✅ |

### 5.2 REMOTE / DOWNLOAD Loading (Sovereignty Risk ⚠️)

| File | Mechanism | Default Model | Risk |
|------|-----------|---------------|------|
| **trainer/captioning.py** — `QwenOmniCaptioner` | `AutoModel.from_pretrained()` | `Qwen/Qwen2.5-Omni-7B` | Downloads 7B model from HF Hub |
| **trainer/captioning.py** — `QwenOmniCaptioner` | `AutoTokenizer.from_pretrained()` | `Qwen/Qwen2.5-Omni-7B` | Downloads tokenizer from HF Hub |
| **trainer/vae_trainer.py** | `vgg16(pretrained=True)` | torchvision VGG16 (ImageNet) | Downloads VGG16 weights from PyTorch Hub |
| **inference/scenarist/scenarist.py** — `LLMScenarist` | `AutoTokenizer.from_pretrained()` + `AutoModelForCausalLM.from_pretrained()` | `mistralai/Mistral-7B-Instruct-v0.3` | Downloads 7B model from HF Hub |
| **inference/optimization.py** | `AutoGPTQForCausalLM.from_pretrained()` | User-supplied model | Downloads GPTQ-quantized model from HF Hub |
| **inference/optimization.py** | `AutoAWQForCausalLM.from_pretrained()` | User-supplied model | Downloads AWQ-quantized model from HF Hub |

---

## 6. Network Calls & URLs

### 6.1 Hardcoded URLs

| URL | File | Purpose |
|-----|------|---------|
| `https://api.aiprod.ai` | pipelines/api/sdk.py (`DEFAULT_BASE_URL`) | SDK client default endpoint |
| `http://localhost:9100/v1` | pipelines/inference/desktop_plugins.py | Local desktop plugin API |
| `postgresql://aiprod:aiprod@localhost:5432/aiprod` | pipelines/api/tenant_store.py | Default PostgreSQL DSN |

### 6.2 Reference-only URLs (comments / docs)

| URL | File | Context |
|-----|------|---------|
| `https://arxiv.org/abs/2004.09602` | inference/quantization/__init__.py | Paper reference |
| `https://arxiv.org/abs/1905.12322` | inference/quantization/__init__.py | Paper reference |
| `https://arxiv.org/abs/1806.08342` | inference/quantization/__init__.py | Paper reference |
| `https://en.wikipedia.org/wiki/Cache_replacement_policies#LRU` | inference/caching/__init__.py | Doc reference |
| `https://arxiv.org/abs/1802.04742` | inference/caching/__init__.py | Paper reference |

### 6.3 Dynamic Network Calls

| File | Call Type | Target |
|------|-----------|--------|
| pipelines/api/sdk.py | `urllib.request.urlopen` | Configurable API base URL |
| pipelines/api/webhooks.py | `urllib.request.urlopen` | Tenant webhook callback URLs (dynamic) |
| pipelines/api/gateway.py | `urllib.request.urlopen` | Internal health checks |
| pipelines/export/multi_format.py | `subprocess` → `ffmpeg` | Local FFmpeg binary (no network) |
| pipelines/api/integrations/video_probe.py | `subprocess` → `ffprobe` | Local FFprobe binary (no network) |
| trainer/gpu_utils.py | `subprocess` → `nvidia-smi` | Local GPU query (no network) |

---

## 7. Real vs Stub Assessment

### 7.1 REAL Production-Ready Implementations ✅

| Component | File(s) | Notes |
|-----------|---------|-------|
| All 5 video pipelines | distilled.py, ic_lora.py, keyframe_interpolation.py, ti2vid_*.py | Fully functional, local model loading |
| Color grading pipeline | color/color_pipeline.py | LUT application, HDR, auto-grading |
| Editing / timeline engine | editing/timeline.py | Pacing, transitions, timeline generation |
| Multi-format export | export/multi_format.py | FFmpeg-based, real encoding |
| Model ledger / loader | utils/model_ledger.py, trainer/model_loader.py | Full model lifecycle |
| Trainer | trainer/trainer.py | LoRA + full fine-tuning, DDP, mixed precision |
| VAE Trainer | trainer/vae_trainer.py | Full VAE training loop |
| Streaming data sources | trainer/streaming/ | S3, GCS, HuggingFace, Local sources with caching |
| API Gateway | api/gateway.py | JWT + API key auth, rate limiting |
| SDK Client | api/sdk.py | Full HTTP client with retry |
| Webhook delivery | api/webhooks.py | HMAC-signed, exponential backoff |
| GCP services adapter | api/adapters/gcp_services.py | GCS, Cloud Logging, Monitoring |
| Stripe billing | api/billing_service.py | Customer, subscription, usage reporting |
| Gemini client | api/integrations/gemini_client.py | Text + video analysis |
| Checkpoint manager | api/checkpoint/manager.py | JSON-based state persistence |
| WebSocket collaboration | api/collaboration/websocket.py | Real-time rooms |
| Schema transformer | api/schema/transformer.py | Bidirectional conversion |
| Inference (majority) | inference/ (tiling, quantization, kernel fusion, etc.) | ~40+ modules, pure torch/numpy |
| LLM Scenarist | inference/scenarist/scenarist.py | Full LLM storyboard generator + rule-based fallback |
| GPTQ / AWQ quantization | inference/optimization.py | INT4 quantization |

### 7.2 STUB / MOCK / PLACEHOLDER Implementations ⚠️

| Component | File | What's Stubbed | Fallback |
|-----------|------|----------------|----------|
| **Render executor backends** | api/adapters/render.py, render_new.py | `_render_with_backend()` generates **mock assets** (returns dict with placeholder URLs). No actual calls to Runway Gen3 / Replicate / Wan2.5 | Returns mock video asset dicts |
| **Semantic QA vision LLM** | api/adapters/qa_semantic.py | `_call_vision_llm()` comment says "In production, would call actual vision LLM API". Returns **heuristic score** | Heuristic analysis |
| **Technical QA validation** | api/adapters/qa_technical.py | Checks metadata dicts, not actual video files. Binary pass/fail on dict fields | Still runs 10 checks but on metadata |
| **GCS download in video_probe** | api/integrations/video_probe.py | `_download_from_gcs()` for `gs://` URLs is **not implemented** | Only works with local files |
| **Gateway rate limiter** | api/gateway.py | In-memory dict (comment: "Redis-backed in production") | Functional but non-distributed |
| **Tenant store** | api/tenant_store.py | `InMemoryBackend` as fallback when asyncpg unavailable | Functional but non-persistent |

### 7.3 OPTIONAL (Graceful Degradation)

| Component | File | When Missing |
|-----------|------|-------------|
| bitsandbytes | trainer/trainer.py | Falls back to standard AdamW |
| stripe | api/billing_service.py | Falls back to local-only billing |
| asyncpg | api/tenant_store.py | Falls back to InMemoryBackend |
| google.generativeai | api/integrations/gemini_client.py | Falls back to mock mode |
| transformers | inference/scenarist/scenarist.py | Falls back to RuleBasedDecomposer |
| auto_gptq / awq | inference/optimization.py | Feature disabled |
| cv2 / opencv | inference/video_editing/ | Feature disabled |
| wandb | trainer/trainer.py | Tracking disabled |

---

## 8. `from_pretrained()` Calls

| # | File | Class/Function | Call | Default Model | local_files_only | Risk |
|---|------|---------------|------|---------------|-----------------|------|
| 1 | **trainer/captioning.py** | `QwenOmniCaptioner.__init__` | `AutoModel.from_pretrained(model_name)` | `Qwen/Qwen2.5-Omni-7B` | ❌ No | **HIGH** — downloads 7B parameters |
| 2 | **trainer/captioning.py** | `QwenOmniCaptioner.__init__` | `AutoTokenizer.from_pretrained(model_name)` | `Qwen/Qwen2.5-Omni-7B` | ❌ No | MEDIUM — tokenizer download |
| 3 | **trainer/gemma_8bit.py** | `GemmaBridge.__init__` | `AutoModel.from_pretrained(model_path, local_files_only=True, ...)` | User-supplied local path | ✅ Yes | **NONE** — local-only enforced |
| 4 | **trainer/gemma_8bit.py** | `GemmaBridge.__init__` | `AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)` | User-supplied local path | ❌ No (but typically local) | LOW — usually local path |
| 5 | **inference/scenarist/scenarist.py** | `LLMScenarist.load_model` | `AutoTokenizer.from_pretrained(model_name)` | `mistralai/Mistral-7B-Instruct-v0.3` | ❌ No | **HIGH** — downloads tokenizer |
| 6 | **inference/scenarist/scenarist.py** | `LLMScenarist.load_model` | `AutoModelForCausalLM.from_pretrained(model_name, ...)` | `mistralai/Mistral-7B-Instruct-v0.3` | ❌ No | **HIGH** — downloads 7B parameters |
| 7 | **inference/optimization.py** | GPTQ quantization | `AutoGPTQForCausalLM.from_pretrained(model, quant_config)` | User-supplied | ❌ No | MEDIUM — depends on input |
| 8 | **inference/optimization.py** | AWQ quantization | `AutoAWQForCausalLM.from_pretrained(model)` | User-supplied | ❌ No | MEDIUM — depends on input |
| 9 | **trainer/vae_trainer.py** | `PerceptualLoss.__init__` | `vgg16(pretrained=True)` | torchvision VGG16 | ❌ No | **MEDIUM** — downloads ~550MB |

**Total: 9 from_pretrained() call sites (6 with remote download risk)**

---

## 9. wandb / mlflow Tracking

### wandb

| File | Import | Usage | Conditional |
|------|--------|-------|-------------|
| **trainer/trainer.py** | `import wandb` (top-level) | `wandb.init(project=..., config=..., name=...)`, `wandb.log({metrics})`, `wandb.Video(...)`, `wandb.Image(...)` | Gated by `config.wandb.enabled` |
| **trainer/vae_trainer.py** | `import wandb` (top-level) | `wandb.init(...)`, `wandb.log(...)` | Gated by config |

### mlflow

**Not found** — zero mlflow imports or references in either package.

### Summary

- **wandb is used** for experiment tracking in 2 files
- Both usages are **conditional** — training works without wandb enabled
- wandb sends training metrics/media to **Weights & Biases cloud** when enabled
- **No mlflow** integration exists

---

## 10. Cloud SDK Usage

### 10.1 Google Cloud Platform (GCP)

| SDK | File | Classes/Methods Used | Purpose |
|-----|------|---------------------|---------|
| `google.generativeai` | trainer/captioning.py | `genai.configure()`, `genai.GenerativeModel()`, `model.generate_content()`, `genai.upload_file()` | Gemini Flash captioning |
| `google.generativeai` | pipelines/api/integrations/gemini_client.py | `genai.configure()`, `genai.GenerativeModel()`, `model.generate_content_async()`, `model.count_tokens()` | Gemini 1.5 Pro text/video analysis |
| `google.cloud.storage` | trainer/streaming/sources.py | `storage.Client()`, `client.bucket()`, `bucket.blob()`, `blob.download_to_filename()` | Streaming data from GCS |
| `google.cloud.storage` | pipelines/api/adapters/gcp_services.py | `storage.Client()`, `client.create_bucket()`, `bucket.blob()`, `blob.upload_from_filename()`, CORS config | Asset storage and delivery |
| `google.cloud.logging` | pipelines/api/adapters/gcp_services.py | `cloud_logging.Client()`, `client.logger()`, `logger.log_struct()` | Structured logging |
| `google.cloud.monitoring_v3` | pipelines/api/adapters/gcp_services.py | `monitoring_v3.MetricServiceClient()`, `client.create_time_series()` | Custom metric writing |
| `google.api_core.exceptions` | pipelines/api/adapters/gcp_services.py | `exceptions.NotFound`, `exceptions.Conflict` | GCP error handling |

### 10.2 Amazon Web Services (AWS)

| SDK | File | Classes/Methods Used | Purpose |
|-----|------|---------------------|---------|
| `boto3` | trainer/streaming/sources.py | `boto3.client('s3')`, `client.download_file()`, `client.list_objects_v2()` | Streaming training data from S3 |

### 10.3 Third-Party Cloud Services

| SDK | File | Classes/Methods Used | Purpose |
|-----|------|---------------------|---------|
| `stripe` | pipelines/api/billing_service.py | `stripe.Customer.create()`, `stripe.Subscription.create()`, `stripe.SubscriptionItem.create_usage_record()` | SaaS billing (optional, lazy) |
| `huggingface_hub` | trainer/hf_hub_utils.py | `HfApi()`, `api.create_repo()`, `api.upload_folder()` | Push trained models to HF Hub |
| `huggingface_hub` | trainer/streaming/sources.py | `hf_hub_download()`, `list_files_in_repo()` | Download training data from HF |

### 10.4 No Azure / Other Cloud SDK

**Confirmed: Zero Azure, DigitalOcean, or other cloud SDK imports.**

---

## SOVEREIGNTY RISK MATRIX

| Risk Level | Finding | Remediation |
|------------|---------|-------------|
| 🔴 **CRITICAL** | Gemini API calls in 2 files (captioning.py, gemini_client.py) send data to Google servers | Replace with local LLM or add `local_files_only` enforcement |
| 🔴 **CRITICAL** | 4 `from_pretrained()` sites download multi-GB models from HuggingFace Hub at runtime | Add `local_files_only=True` or pre-download models to local cache |
| 🔴 **CRITICAL** | GCP services adapter makes real calls to Cloud Storage / Logging / Monitoring | Abstract behind interface; allow local-only backends |
| 🟠 **HIGH** | `vgg16(pretrained=True)` downloads ~550MB from PyTorch Hub | Pre-download or use local weights path |
| 🟠 **HIGH** | W&B tracking sends experiment data to cloud when enabled | Already conditional — ensure default is `enabled=False` |
| 🟠 **HIGH** | SDK client defaults to `https://api.aiprod.ai` — assumes cloud deployment | Document; make configurable (already is) |
| 🟡 **MEDIUM** | boto3 S3 data source is a real AWS external dependency | Already optional; LocalDataSource exists as alternative |
| 🟡 **MEDIUM** | Stripe billing active when package installed | Already lazy+optional |
| 🟡 **MEDIUM** | HuggingFace Hub uploads in `hf_hub_utils.py` | Only triggered by explicit `push_to_hub` flag |
| 🟡 **MEDIUM** | Webhook manager sends HTTP POST to arbitrary external URLs | By design; tenant-controlled |
| 🟢 **LOW** | asyncpg PostgreSQL — standard database dependency | Already has InMemoryBackend fallback |
| 🟢 **LOW** | FFmpeg / FFprobe subprocess calls | Local binary, no network |
| 🟢 **LOW** | `gemma_8bit.py` uses `local_files_only=True` | **Already sovereign** ✅ |
| 🟢 **LOW** | All 5 main pipeline files load from local safetensors only | **Already sovereign** ✅ |

---

## SUMMARY STATISTICS

| Metric | Count |
|--------|-------|
| Total external (non-stdlib) package dependencies | ~25 |
| `from_pretrained()` call sites | 9 (6 with remote download risk) |
| External API integrations (active) | 6 (Gemini ×2, HF Hub ×2, S3, GCS) |
| External API integrations (stub/mock) | 3 (Runway, Replicate, Veo3) |
| API key / secret references | 6 distinct keys |
| Hardcoded URLs | 3 (1 production, 2 localhost) |
| wandb tracking locations | 2 files |
| mlflow tracking locations | 0 |
| Cloud SDKs used | 4 (GCP ×4 services, AWS S3, Stripe, HuggingFace Hub) |
| Stub/mock implementations | 6 components |
| Files with zero sovereignty risk | ~85% of all files |
| Files with critical sovereignty risk | ~8 files across both packages |

---

*End of Sovereignty Audit*
