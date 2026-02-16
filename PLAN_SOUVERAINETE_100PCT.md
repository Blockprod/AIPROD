# PLAN SOUVERAINETÉ 100% PROPRIÉTAIRE — AIPROD
## De « Dépendant masqué » à « 100% Propriétaire réel »

**Date :** 2026-02-15  
**Objectif :** Fusionner le meilleur de AIPROD_V33 (orchestration SaaS) et LTX-2 (diffusion vidéo) dans un système **entièrement propriétaire**, entraîné sur ses propres modèles, opérable en air-gapped.  
**Horizon :** 12 semaines (3 mois)  
**Référence :** `AUDIT_AIPROD_FUSION_100PCT_PROPRIETAIRE.md`

---

## SYNTHÈSE EXÉCUTIVE

```
État actuel  →  3/10 souveraineté  →  Pipeline API mock, modèles téléchargés dynamiquement
Objectif     →  9/10 souveraineté  →  Pipeline réel, modèles internes, zéro API externe
```

### 4 Phases — 12 semaines

| Phase | Nom | Durée | Objectif |
|---|---|---|---|
| **Phase 1** | Couper les fils | Semaines 1-2 | Éliminer toutes les dépendances externes |
| **Phase 2** | Connecter le moteur | Semaines 3-5 | Brancher le vrai GPU pipeline sur l'API |
| **Phase 3** | Entraîner ses modèles | Semaines 6-10 | Fine-tuner et posséder chaque modèle |
| **Phase 4** | Verrouiller et certifier | Semaines 11-12 | Reproductibilité, tests, documentation souveraine |

---

## PHASE 1 — COUPER LES FILS (Semaines 1-2)

**Objectif :** Zéro appel réseau sortant vers un service tiers. Tout fonctionne offline.

---

### 1.1 Pré-provisionner TOUS les modèles localement

**Problème :** 6 appels `from_pretrained()` téléchargent ~40 GB depuis HuggingFace Hub.  
**Solution :** Script de provisionnement unique + flag `local_files_only=True` partout.

#### Action 1.1.1 — Créer `scripts/provision_models.py`

```python
"""
Télécharge et stocke TOUS les modèles nécessaires en local.
À exécuter UNE FOIS sur machine connectée, puis le projet est 100% offline.
"""
from huggingface_hub import snapshot_download
from pathlib import Path
import hashlib, json

MODELS = {
    "models/text-encoder/gemma-3-1b": {
        "repo": "google/gemma-3-1b-pt",
        "revision": "main",  # Figer au commit SHA exact après téléchargement
    },
    "models/scenarist/mistral-7b": {
        "repo": "mistralai/Mistral-7B-Instruct-v0.3",
        "revision": "main",
    },
    "models/captioning/qwen-omni-7b": {
        "repo": "Qwen/Qwen2.5-Omni-7B",
        "revision": "main",
    },
}

def download_all():
    manifest = {}
    for local_path, spec in MODELS.items():
        path = Path(local_path)
        path.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {spec['repo']} → {local_path}")
        snapshot_download(
            repo_id=spec["repo"],
            local_dir=str(path),
            revision=spec["revision"],
        )
        # Générer hash de vérification
        manifest[local_path] = {
            "repo": spec["repo"],
            "revision": spec["revision"],
            "files": [str(f) for f in path.rglob("*.safetensors")],
        }
    
    # Sauvegarder le manifeste
    with open("models/MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("✅ Tous les modèles provisionnés. MANIFEST.json créé.")

if __name__ == "__main__":
    download_all()
```

#### Action 1.1.2 — Forcer `local_files_only=True` partout

| Fichier | Modification |
|---|---|
| `packages/aiprod-core/src/aiprod_core/model/text_encoder/bridge.py` | `AutoModel.from_pretrained(..., local_files_only=True)` + `AutoTokenizer.from_pretrained(..., local_files_only=True)` |
| `packages/aiprod-pipelines/src/aiprod_pipelines/inference/scenarist/scenarist.py` | `AutoModelForCausalLM.from_pretrained(..., local_files_only=True)` + `AutoTokenizer.from_pretrained(..., local_files_only=True)` |
| `packages/aiprod-trainer/src/aiprod_trainer/captioning.py` | `from_pretrained(..., local_files_only=True)` dans `QwenOmniCaptioner` |
| `packages/aiprod-trainer/src/aiprod_trainer/gemma_8bit.py` | ✅ Déjà fait |

#### Action 1.1.3 — Mettre à jour les chemins par défaut

| Fichier | Ancien default | Nouveau default |
|---|---|---|
| `bridge.py` → `LLMBridgeConfig.model_name` | `"meta-llama/Llama-3.2-1B"` | `"models/text-encoder/gemma-3-1b"` |
| `scenarist.py` → `LLMScenarist` | `"mistralai/Mistral-7B-Instruct-v0.3"` | `"models/scenarist/mistral-7b"` |
| `captioning.py` → `QwenOmniCaptioner.MODEL_ID` | `"Qwen/Qwen2.5-Omni-7B"` | `"models/captioning/qwen-omni-7b"` |

**Critère de validation :** `python -c "from aiprod_core.model.text_encoder import LLMBridge; b = LLMBridge(); b.encode_text('test')"` fonctionne **sans connexion réseau**.

---

### 1.2 Supprimer la dépendance Google Gemini

**Problème :** `google.generativeai` importé en dur dans 2 fichiers. Appels API vers Google.

#### Action 1.2.1 — Rendre `gemini_client.py` optionnel

```
Fichier : packages/aiprod-pipelines/src/aiprod_pipelines/api/integrations/gemini_client.py
```

- Transformer `import google.generativeai as genai` en `try/except ImportError`
- Le mode mock (déjà implémenté) devient le **mode par défaut**
- Si le SDK Google est installé ET une clé API fournie → mode live (opt-in explicite)

#### Action 1.2.2 — Remplacer `GeminiFlashCaptioner` comme captioner par défaut

```
Fichier : packages/aiprod-trainer/src/aiprod_trainer/captioning.py
```

- `GeminiFlashCaptioner` → déplacé dans un module optionnel `captioning_external.py`
- Le captioner par défaut devient `QwenOmniCaptioner` (local) ou `CachedCaptioner`
- Google Gemini = import optionnel, jamais chargé par défaut

#### Action 1.2.3 — Nettoyer le Dockerfile principal

```
Fichier : deploy/docker/Dockerfile
```

- Retirer `google-generativeai==0.3.0` de la ligne pip install
- Retirer `google-cloud-logging==3.8.0` et `google-cloud-monitoring==2.16.0`
- Garder uniquement les dépendances souveraines

**Critère de validation :** `pip install` du projet fonctionne sans aucun package `google-*`.

---

### 1.3 Isoler les backends cloud en modules optionnels

**Problème :** `boto3`, `google-cloud-storage`, `stripe` sont dans les requirements globaux.

#### Action 1.3.1 — Créer des extras optionnels dans `pyproject.toml`

```toml
[project.optional-dependencies]
cloud-gcs = ["google-cloud-storage>=2.10"]
cloud-s3 = ["boto3>=1.35"]
billing-stripe = ["stripe>=7.0"]
tracking-wandb = ["wandb>=0.16"]
tracking-gemini = ["google-generativeai>=0.3"]
```

#### Action 1.3.2 — Guard imports dans le code

| Fichier | Import actuel | Import sécurisé |
|---|---|---|
| `streaming/sources.py` → `S3DataSource` | `import boto3` en dur | `try: import boto3 except ImportError: raise ...` |
| `streaming/sources.py` → `GCSDataSource` | `from google.cloud import storage` en dur | `try/except ImportError` |
| `billing_service.py` → `StripeIntegration` | ✅ Déjà en try/except | OK |
| `vae_trainer.py` → `wandb` | `import wandb` en dur | `try/except ImportError` avec fallback console |

#### Action 1.3.3 — Mettre à jour `requirements.txt`

```
# CORE (100% souverain — aucune dépendance SaaS)
torch==2.5.1
torchvision==0.20.1
torchaudio==2.5.1
transformers==4.47.0
safetensors==0.4.5
accelerate==1.2.0
peft==0.14.0
einops==0.8.0
numpy==1.26.4
scipy==1.14.0
pillow==10.4.0
opencv-python==4.10.0.84
imageio==2.36.0
av==13.1.0
librosa==0.10.2
matplotlib==3.9.3
pydantic==2.10.0
fastapi==0.115.0
uvicorn==0.32.0
aiohttp==3.11.0
rich==13.9.0
typer==0.14.0
tqdm==4.67.0
pyyaml==6.0.2
cachetools==5.5.0
scenedetect==0.6.4
lpips==0.1.4
zstandard==0.23.0
xformers==0.0.28
bitsandbytes==0.45.0
optimum-quanto==0.2.6

# OBSERVABILITÉ (auto-hébergeable)
prometheus-client==0.21.0
opentelemetry-api==1.29.0
opentelemetry-sdk==1.29.0
opentelemetry-exporter-otlp==1.29.0
structlog==24.4.0
mlflow==2.19.0

# OPTIONNEL (installer séparément si besoin)
# pip install aiprod[cloud-gcs]    → google-cloud-storage
# pip install aiprod[cloud-s3]     → boto3
# pip install aiprod[billing]      → stripe
# pip install aiprod[tracking]     → wandb
# pip install aiprod[gemini]       → google-generativeai
```

**Critère de validation :** `pip install -r requirements.txt` n'installe AUCUN package Google, AWS, Stripe ou Wandb.

---

### 1.4 Figer les versions et créer un lockfile

#### Action 1.4.1

```bash
# Générer le lockfile complet
pip freeze > requirements.lock

# Vérifier l'intégrité
pip install --no-deps -r requirements.lock  # doit être identique
```

#### Action 1.4.2 — Hasher les poids de modèles

Créer `scripts/verify_model_integrity.py` :
- SHA-256 de chaque fichier `.safetensors` et `.pt`
- Stocker dans `models/CHECKSUMS.sha256`
- Vérification automatique au démarrage du pipeline

**Critère de validation :** `python scripts/verify_model_integrity.py` retourne ✅ pour chaque fichier.

---

## PHASE 2 — CONNECTER LE MOTEUR (Semaines 3-5)

**Objectif :** Le pipeline API génère de VRAIES vidéos sur GPU local. Zéro mock.

---

### 2.1 Implémenter le GPU Worker

**Problème :** Le gateway enqueue des jobs mais rien ne les traite. Le vrai moteur (`ti2vid_one_stage.py`) est un CLI déconnecté.

#### Action 2.1.1 — Créer `packages/aiprod-pipelines/src/aiprod_pipelines/api/gpu_worker.py`

**Architecture :**

```
Gateway (FastAPI, CPU)
    │
    ▼ (Redis queue ou in-memory queue)
    │
GPU Worker (consomme les jobs)
    │
    ├── Charge TI2VidOneStagePipeline (une fois au boot)
    │   ├── Transformer SHDT (25 GB, FP8)
    │   ├── Video VAE Encoder/Decoder
    │   ├── Audio VAE + Vocoder
    │   └── Text Encoder (Gemma local)
    │
    ├── Pour chaque job :
    │   ├── Valide le prompt (InputSanitizer)
    │   ├── Décompose en scenes (Scenarist — local LLM ou rule-based)
    │   ├── Génère chaque clip (pipeline.generate())
    │   ├── Assemble les clips (ffmpeg)
    │   ├── QA technique (resolution, codec, durée)
    │   └── Retourne le résultat (fichier vidéo + métadonnées)
    │
    └── Sauvegarde le résultat
        ├── Système de fichiers local
        └── Webhook notification au client
```

**Spécification du worker :**

```python
class GPUWorker:
    """
    Worker GPU souverain — consomme les jobs de la queue et génère de vraies vidéos.
    """
    
    def __init__(self, config: WorkerConfig):
        # Charger le pipeline UNE FOIS au démarrage
        self.pipeline = TI2VidOneStagePipeline(
            checkpoint_path=config.checkpoint_path,        # models/ltx2_research/
            gemma_root=config.text_encoder_path,           # models/text-encoder/gemma-3-1b/
            loras=[],
            fp8transformer=True,
        )
        self.scenarist = RuleBasedDecomposer()             # 100% local, pas de LLM
        self.output_dir = Path(config.output_dir)
    
    async def process_job(self, job: Job) -> JobResult:
        """Traite un job de génération vidéo."""
        # 1. Décomposer le prompt en shots
        storyboard = self.scenarist.decompose(job.prompt)
        
        # 2. Générer chaque clip
        clips = []
        for shot in storyboard.shots:
            video, audio = self.pipeline(
                prompt=shot.prompt,
                negative_prompt=job.negative_prompt or "",
                seed=job.seed + shot.index,
                height=job.height or 720,
                width=job.width or 1280,
                num_frames=shot.num_frames,
                frame_rate=job.frame_rate or 24.0,
                num_inference_steps=job.steps or 30,
                video_guider_params=MultiModalGuiderParams(cfg_scale=3.0),
                audio_guider_params=MultiModalGuiderParams(cfg_scale=3.0),
                images=[],
            )
            clips.append((video, audio))
        
        # 3. Encoder et sauvegarder
        output_path = self.output_dir / f"{job.job_id}.mp4"
        encode_video(video=clips[0][0], audio=clips[0][1], ...)
        
        # 4. QA technique
        qa_result = self.technical_qa(output_path)
        
        return JobResult(
            job_id=job.job_id,
            status="completed",
            output_path=str(output_path),
            qa_score=qa_result.score,
        )
```

#### Action 2.1.2 — Remplacer le RenderExecutorAdapter mock

```
Fichier : packages/aiprod-pipelines/src/aiprod_pipelines/api/adapters/render.py
```

Réécrire `_render_with_backend()` pour appeler le vrai `GPUWorker` au lieu de retourner des assets fabriqués.

**Supprimer :**
- Tous les `random.random()` qui simulent des échecs
- Les URLs `gs://aiprod-assets/` hardcodées
- Le commentaire `"In production, would call actual backend API"`

**Remplacer par :**
- Appel réel à `self.gpu_worker.process_job(job)`
- Le backend unique = `"aiprod_sovereign"` (pas runway, pas replicate, pas veo3)

#### Action 2.1.3 — Implémenter la queue de jobs persistante

```
Nouveau fichier : packages/aiprod-pipelines/src/aiprod_pipelines/api/job_store.py
```

**Option A (minimaliste) :** SQLite local  
**Option B (production) :** Redis (déjà prévu dans K8s config)

```python
class JobStore:
    """Stockage persistant des jobs — survit aux redémarrages."""
    
    def enqueue(self, job: Job) -> str: ...
    def dequeue(self) -> Optional[Job]: ...
    def update_status(self, job_id: str, status: str, result: dict): ...
    def get_job(self, job_id: str) -> Optional[Job]: ...
    def list_pending(self) -> list[Job]: ...
```

**Critère de validation :**
```bash
# Soumettre un job via l'API
curl -X POST http://localhost:8080/v1/generate \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"prompt": "A cat walking in a garden", "seed": 42}'

# Le résultat est un VRAI fichier vidéo .mp4, pas un mock
ffprobe output/job_xxx.mp4  # → codec h264, durée > 0, résolution réelle
```

---

### 2.2 Réécrire la config architecturale

**Problème :** `AIPROD_V33.json` décrit un système Google-centric qui ne reflète plus la réalité.

#### Action 2.2.1 — Créer `config/AIPROD_V34_SOVEREIGN.json`

Changements clés par rapport à V33 :

| Bloc V33 | Bloc V34 Souverain | Changement |
|---|---|---|
| `creativeDirector` → `llmProvider: "google"`, `llmModel: "gemini-1.5-pro"` | `llmProvider: "local"`, `llmModel: "models/scenarist/mistral-7b"` | LLM local |
| `visualTranslator` → `llmProvider: "google"` | `llmProvider: "local"`, `llmModel: "models/scenarist/mistral-7b"` | LLM local |
| `inputSanitizer` → `googleOptimized: true` | `googleOptimized: false`, `geminiPromptFormat: null` | Suppression Google |
| `renderExecutor` → `veo3Configuration`, `runwayGen3`, `replicate_wan25` | `backend: "aiprod_sovereign"`, GPU local via `TI2VidOneStagePipeline` | Rendu local |
| `financialOrchestrator` → `backendPriority: ["veo3", "runwayGen3", ...]` | `backendPriority: ["aiprod_sovereign"]` | Backend unique |
| `semanticQA` → `llmProvider: "google"`, `visionModel: "gemini-1.5-pro-vision"` | `llmProvider: "local"`, `visionModel: "models/vision-qa/clip-local"` | QA local |
| `supervisor` → `llmProvider: "google"` | Score-based rules (pas de LLM) | Suppression LLM pour approbation |
| `googleCloudServices` | **SUPPRIMÉ** | Aucun service cloud |
| `googleStackConfiguration` | **SUPPRIMÉ** | Aucune clé API |
| `dynamicPricing.sources` → `["google_cloud_billing", ...]` | `sources: ["internal_gpu_meter"]` | Coûts internes |

#### Action 2.2.2 — Mettre à jour l'orchestrateur

```
Fichier : packages/aiprod-pipelines/src/aiprod_pipelines/api/orchestrator.py
```

Charger `AIPROD_V34_SOVEREIGN.json` par défaut. Ajouter une variable d'environnement :

```python
CONFIG_PATH = os.environ.get("AIPROD_CONFIG", "config/AIPROD_V34_SOVEREIGN.json")
```

---

### 2.3 Implémenter le QA sémantique souverain

**Problème :** Le QA sémantique dans V33 repose sur `gemini-1.5-pro-vision` (API Google).

#### Action 2.3.1 — Créer un QA sémantique basé sur CLIP local

```
Nouveau fichier : packages/aiprod-pipelines/src/aiprod_pipelines/api/adapters/qa_semantic_local.py
```

**Architecture :**

```python
class LocalSemanticQA:
    """
    QA sémantique 100% local — évalue la cohérence prompt ↔ vidéo générée.
    Utilise CLIP (ou SigLIP) en local pour le score de similarité.
    """
    
    def __init__(self, model_path: str = "models/qa/siglip-base"):
        self.model = load_clip_model(model_path)  # local_files_only=True
    
    def evaluate(self, video_path: str, prompt: str) -> SemanticQAResult:
        # 1. Extraire des frames clés de la vidéo
        frames = extract_keyframes(video_path, n=8)
        
        # 2. Calculer la similarité CLIP prompt ↔ frames
        text_embedding = self.model.encode_text(prompt)
        frame_embeddings = [self.model.encode_image(f) for f in frames]
        similarity_scores = [cosine_sim(text_embedding, fe) for fe in frame_embeddings]
        
        # 3. Évaluer la cohérence temporelle (inter-frames)
        temporal_coherence = compute_temporal_coherence(frame_embeddings)
        
        return SemanticQAResult(
            prompt_adherence=mean(similarity_scores),
            temporal_coherence=temporal_coherence,
            overall_score=weighted_mean(...),
        )
```

**Modèle à préprovisionner :** `google/siglip-base-patch16-224` (350 MB, Apache 2.0, local inference)

---

### 2.4 Pipeline complet end-to-end

Une fois Phase 2 terminée, le flux est :

```
Client → API Gateway → Job Queue (SQLite/Redis)
                              │
                              ▼
                        GPU Worker
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
             Scenarist   Text Encode  Noise Init
             (local      (Gemma 3,   (Gaussian,
              Mistral     local)      seeded)
              ou rules)
                    │         │         │
                    └─────────┼─────────┘
                              ▼
                     Diffusion Loop
                     (SHDT Transformer, 
                      Euler steps, CFG,
                      FP8, local GPU)
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
              VAE Decode  Audio Decode  Upsampler
              (HW-VAE)   (NAC Codec)   (à implémenter)
                    │         │         │
                    └─────────┼─────────┘
                              ▼
                      Video Encoding
                      (ffmpeg, H.264)
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
              QA Technique  QA Sémantique  Billing
              (deterministic) (CLIP local)  (interne)
                              │
                              ▼
                      Résultat → Client
                      (fichier .mp4 + métadonnées)
```

**Critère de validation Phase 2 :**  
Générer une vidéo de 5 secondes, 720p, via l'API, en moins de 5 minutes, sur un GPU A100, **sans aucune connexion internet**.

---

## PHASE 3 — ENTRAÎNER SES MODÈLES (Semaines 6-10)

**Objectif :** Posséder chaque modèle. Ne plus dépendre des poids LTX-2/Lightricks d'origine.

---

### 3.1 Stratégie de fine-tuning — Transformer SHDT

Le transformer est le cœur du système. Le plan suit une progression curriculum (déjà implémenté dans `curriculum_training.py`).

#### Phase 3.1.1 — LoRA Fine-tuning (Semaine 6-7)

**Objectif :** Adapter le transformer LTX-2 aux cas d'usage AIPROD avec un coût GPU minimal.

```yaml
Stratégie: LoRA
Rang LoRA: 32
Alpha: 32
Target modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
Modèle de base: models/ltx2_research/ltx-2-19b-dev-fp8.safetensors
Données: 500-1000 vidéos courtes (5-10s) avec captions
Batch size: 1 (gradient accumulation 8)
Learning rate: 1e-5
GPU: 1× A100 80GB (ou 2× A100 40GB avec accelerate)
Epochs: 3-5
Durée estimée: 48-72h
```

**Commande :**

```bash
python -m aiprod_trainer.trainer \
    --config configs/train/lora_phase1.yaml \
    --checkpoint models/ltx2_research/ltx-2-19b-dev-fp8.safetensors \
    --gemma-path models/text-encoder/gemma-3-1b \
    --output checkpoints/aiprod_lora_v1/ \
    --use-wandb false
```

**Résultat :** `checkpoints/aiprod_lora_v1/adapter_model.safetensors` (~50-200 MB)

#### Phase 3.1.2 — Full Fine-tuning (Semaine 8-9)

**Objectif :** Entraîner le modèle complet pour devenir indépendant des poids LTX-2 originaux.

```yaml
Stratégie: Full fine-tuning
Modèle de base: LTX-2 + LoRA v1 fusionné
Données: 5000+ vidéos avec curriculum (résolution/durée croissantes)
Curriculum:
  Phase 1: 256×256, 16 frames, lr=5e-6  (jours 1-3)
  Phase 2: 512×512, 32 frames, lr=3e-6  (jours 4-6)
  Phase 3: 768×768, 64 frames, lr=1e-6  (jours 7-9)
  Phase 4: 1024×576, 97 frames, lr=5e-7 (jours 10-12)
GPU: 4× A100 80GB (DDP via accelerate)
Gradient checkpointing: True
Mixed precision: bf16
Durée estimée: 10-14 jours
```

**Résultat :** `checkpoints/aiprod_shdt_v1.safetensors` (~25 GB en bf16, ~12 GB en FP8)

#### Phase 3.1.3 — Quantization propriétaire (Semaine 10)

```bash
# Quantizer le modèle full en FP8 propriétaire
python -m aiprod_trainer.quantization \
    --input checkpoints/aiprod_shdt_v1.safetensors \
    --output models/aiprod-sovereign/aiprod-shdt-v1-fp8.safetensors \
    --format fp8_e4m3
```

**Résultat final :** `models/aiprod-sovereign/aiprod-shdt-v1-fp8.safetensors` — **modèle 100% propriétaire**

---

### 3.2 Fine-tuning du Video VAE

**Déjà implémenté** dans `vae_trainer.py`. Le Haar Wavelet VAE (`HWVAEEncoder`/`HWVAEDecoder`) est propriétaire.

```yaml
Données: Mêmes vidéos que le transformer
Loss: Reconstruction L1 + Perceptual (VGG) + Spectral
GPU: 1× A100 80GB
Epochs: 50-100
Durée: 3-5 jours
```

**Résultat :** `models/aiprod-sovereign/aiprod-hwvae-v1.safetensors`

---

### 3.3 Fine-tuning Audio VAE + Vocoder

L'audio codec NAC est implémenté dans `aiprod_core/model/audio_vae/codec.py`.

```yaml
Données: 1000+ clips audio (musique d'ambiance, voix, effets sonores)
Loss: Reconstruction + RVQ commitment loss + Spectral
GPU: 1× A100
Durée: 2-3 jours
```

**Résultat :** `models/aiprod-sovereign/aiprod-audio-vae-v1.safetensors`

---

### 3.4 Entraînement du TTS

L'architecture est complète dans `aiprod_core/model/tts/` (Tacotron + HiFi-GAN + Prosody).

```yaml
Données: LJSpeech (13k clips, 24h, domaine public) + LibriTTS (585h, CC BY 4.0)
Phase 1: TextEncoder + MelDecoder sur LJSpeech (5 jours)
Phase 2: VocoderTTS (HiFi-GAN) sur mel spectrograms (3 jours)
Phase 3: ProsodyModeler fine-tuning (2 jours)
GPU: 1× A100
Total: 10 jours
```

**Résultat :** `models/aiprod-sovereign/aiprod-tts-v1.safetensors`

---

### 3.5 Entraînement / Provisionnement du Text Encoder

**Option A (rapide — Semaine 6) :** Utiliser Gemma-3-1B pré-entraîné, stocké localement.  
Les poids Gemma sont Apache 2.0 — pas de restriction de licence.

**Option B (souverain total — Semaine 8+) :** Fine-tuner un text encoder plus petit sur des données de prompts vidéo.

```yaml
Modèle de base: Gemma-3-1B (Apache 2.0)
Données: 100k paires (prompt, video caption)
Méthode: LoRA sur les couches d'embedding → projection vers l'espace latent SHDT
Résultat: models/aiprod-sovereign/aiprod-text-encoder-v1.safetensors
```

---

### 3.6 Structure finale du répertoire modèles

```
models/
├── aiprod-sovereign/                      ← MODÈLES PROPRIÉTAIRES
│   ├── aiprod-shdt-v1-fp8.safetensors     ← Transformer diffusion (~12 GB)
│   ├── aiprod-hwvae-v1.safetensors        ← Video VAE (~500 MB)
│   ├── aiprod-audio-vae-v1.safetensors    ← Audio codec (~200 MB)
│   ├── aiprod-tts-v1.safetensors          ← TTS complet (~300 MB)
│   ├── aiprod-text-encoder-v1.safetensors ← Text encoder (~1 GB)
│   ├── aiprod-upsampler-v1.safetensors    ← Spatial upsampler (~500 MB)
│   ├── MANIFEST.json                      ← Versions + SHA-256 checksums
│   └── MODEL_CARD.md                      ← Documentation complète
│
├── ltx2_research/                         ← MODÈLES DE BASE (Lightricks)
│   ├── ltx-2-19b-dev-fp8.safetensors      ← Base pour fine-tuning
│   └── ltx-2-spatial-upscaler-x2-1.0.safetensors
│
├── text-encoder/                          ← Pré-provisionné Phase 1
│   └── gemma-3-1b/
│
├── scenarist/                             ← Pré-provisionné Phase 1
│   └── mistral-7b/
│
└── qa/                                    ← QA sémantique
    └── siglip-base/
```

**Critère de validation Phase 3 :**
- Générer une vidéo de 10s, 1080p, **uniquement avec les modèles `aiprod-sovereign/`**
- Aucun fichier du dossier `ltx2_research/` n'est nécessaire
- FID score ≤ 1.5× celui du modèle LTX-2 de base
- Pas de régression de qualité subjective (évaluation humaine A/B)

---

## PHASE 4 — VERROUILLER ET CERTIFIER (Semaines 11-12)

**Objectif :** Le système passe une due diligence technique. Reproductible. Documenté. Certifiable.

---

### 4.1 Reproductibilité complète

#### Action 4.1.1 — Fixer le determinism PyTorch

```python
# À ajouter dans aiprod_core/utils.py → seed_everything()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)
```

#### Action 4.1.2 — Lockfile complet

```
requirements.lock          ← pip freeze exact
cuda_version.txt           ← nvidia-smi output
docker_image_hash.txt      ← sha256 de l'image Docker
models/CHECKSUMS.sha256    ← hash de chaque fichier de poids
```

#### Action 4.1.3 — Test de reproductibilité

```bash
# Même prompt, même seed → même vidéo (bit-exact ou perceptuellement identique)
python -m aiprod_pipelines.ti2vid_one_stage \
    --prompt "A cat walking" --seed 42 --output test_repro_1.mp4

python -m aiprod_pipelines.ti2vid_one_stage \
    --prompt "A cat walking" --seed 42 --output test_repro_2.mp4

# Vérifier : PSNR(test_repro_1, test_repro_2) == inf (ou > 60 dB)
python scripts/compare_videos.py test_repro_1.mp4 test_repro_2.mp4
```

---

### 4.2 Tests automatisés complets

#### Action 4.2.1 — Tests de souveraineté automatisés

```
Nouveau fichier : tests/test_sovereignty.py
```

```python
"""Tests automatisés de souveraineté — exécutés en CI sans réseau."""

import socket

class TestSovereignty:
    def test_no_network_calls_during_import(self):
        """Importer tout le projet ne déclenche aucun appel réseau."""
        original = socket.socket
        socket.socket = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Network blocked"))
        try:
            import aiprod_core
            import aiprod_pipelines
            import aiprod_trainer
        finally:
            socket.socket = original
    
    def test_all_models_present_locally(self):
        """Tous les modèles requis existent sur le filesystem."""
        assert Path("models/aiprod-sovereign/aiprod-shdt-v1-fp8.safetensors").exists()
        assert Path("models/aiprod-sovereign/aiprod-hwvae-v1.safetensors").exists()
        # ...
    
    def test_no_google_imports_in_core(self):
        """Aucun import Google dans aiprod_core."""
        for py_file in Path("packages/aiprod-core/src").rglob("*.py"):
            content = py_file.read_text()
            assert "google.generativeai" not in content
            assert "google.cloud" not in content
    
    def test_no_from_pretrained_without_local_only(self):
        """Chaque from_pretrained() a local_files_only=True."""
        for py_file in Path("packages").rglob("*.py"):
            content = py_file.read_text()
            if "from_pretrained(" in content:
                assert "local_files_only=True" in content or "local_files_only" in content
    
    def test_inference_offline(self):
        """Générer une vidéo sans connexion réseau."""
        # Bloque le réseau, charge le pipeline, génère 1 frame
        ...
```

#### Action 4.2.2 — CI/CD pipeline

```yaml
# .github/workflows/sovereignty-check.yml
name: Sovereignty Check
on: [push, pull_request]

jobs:
  sovereignty:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install dependencies
        run: pip install -r requirements.txt  # Pas de packages cloud
      - name: Block network
        run: |
          # Iptables block all outbound except localhost
          sudo iptables -A OUTPUT -d 127.0.0.1 -j ACCEPT
          sudo iptables -A OUTPUT -j DROP
      - name: Run sovereignty tests
        run: pytest tests/test_sovereignty.py -v
```

---

### 4.3 Dockerfile souverain

#### Action 4.3.1 — Réécrire `deploy/docker/Dockerfile.gpu`

```dockerfile
# === AIPROD SOVEREIGN GPU IMAGE ===
# Zéro dépendance SaaS. Zéro appel réseau en runtime.

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder

# ... (build steps identiques)

# CHANGEMENT CRITIQUE : ne PAS installer google-*, boto3, stripe, wandb
RUN pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir \
    -r /app/requirements.txt  # requirements.txt souverain (sans cloud deps)

# Copier les modèles dans l'image (build-time, pas runtime download)
COPY models/aiprod-sovereign/ /app/models/aiprod-sovereign/
COPY models/text-encoder/ /app/models/text-encoder/
COPY models/scenarist/ /app/models/scenarist/

# Variable d'environnement : config souveraine
ENV AIPROD_CONFIG=/app/config/AIPROD_V34_SOVEREIGN.json
ENV AIPROD_MODELS_DIR=/app/models/aiprod-sovereign
```

---

### 4.4 Documentation due diligence

#### Action 4.4.1 — Créer `docs/SOVEREIGNTY_CERTIFICATE.md`

Contenu :

```
1. Inventaire complet des modèles avec licences
   - SHDT Transformer : Propriétaire (entraîné par AIPROD)
   - HW-VAE : Propriétaire
   - NAC Audio Codec : Propriétaire
   - TTS : Propriétaire (entraîné sur LJSpeech CC0 + LibriTTS CC BY 4.0)
   - Text Encoder : Basé sur Gemma (Apache 2.0) + fine-tuning propriétaire
   - QA CLIP : SigLIP (Apache 2.0, local inference)

2. Licences des dépendances open-source
   - PyTorch : BSD-3
   - transformers : Apache 2.0
   - safetensors : Apache 2.0
   - FastAPI : MIT
   - ffmpeg : LGPL 2.1

3. Inventaire des appels réseau
   - Runtime : ZÉRO
   - Build-time : PyPI (pip install), PyTorch wheel index
   - Provisionnement unique : HuggingFace Hub (pour modèles de base)

4. Test air-gapped réalisé le [DATE]
   - Environnement : [description]
   - Résultat : [PASS/FAIL]
   - Vidéos générées : [échantillons]
```

---

## PLANNING CONSOLIDÉ

```
SEMAINE  1  ████████  Phase 1.1 — Pré-provisionner les modèles
SEMAINE  2  ████████  Phase 1.2-1.4 — Couper Google, isoler cloud, figer deps
SEMAINE  3  ████████  Phase 2.1 — GPU Worker (architecture + queue)
SEMAINE  4  ████████  Phase 2.1 — GPU Worker (intégration pipeline)  
SEMAINE  5  ████████  Phase 2.2-2.4 — Config V34, QA local, test end-to-end
SEMAINE  6  ████████  Phase 3.1.1 — LoRA fine-tuning transformer
SEMAINE  7  ████████  Phase 3.1.1 — LoRA validation + Phase 3.2 VAE
SEMAINE  8  ████████  Phase 3.1.2 — Full fine-tuning (début)
SEMAINE  9  ████████  Phase 3.1.2 — Full fine-tuning (suite) + 3.3 Audio
SEMAINE 10  ████████  Phase 3.1.3 — Quantization + 3.4 TTS + 3.5 Text Encoder
SEMAINE 11  ████████  Phase 4.1-4.2 — Reproductibilité + tests souveraineté
SEMAINE 12  ████████  Phase 4.3-4.4 — Docker souverain + documentation certifiée
```

---

## BUDGET GPU ESTIMÉ

| Phase | GPU nécessaire | Durée | Coût cloud estimé |
|---|---|---|---|
| Phase 1 (couper les fils) | 0 GPU | 2 semaines | $0 |
| Phase 2 (connecter moteur) | 1× A100 40GB (tests) | 3 semaines | ~$500 |
| Phase 3 — LoRA | 1× A100 80GB | 3 jours | ~$200 |
| Phase 3 — Full fine-tune | 4× A100 80GB | 14 jours | ~$5,000 |
| Phase 3 — VAE | 1× A100 80GB | 5 jours | ~$400 |
| Phase 3 — Audio + TTS | 1× A100 40GB | 15 jours | ~$1,100 |
| Phase 3 — Quantization | 1× A100 40GB | 1 jour | ~$70 |
| Phase 4 (tests) | 1× A100 40GB | 2 jours | ~$140 |
| **TOTAL** | | | **~$7,400** |

Avec GPU propre (achat A100 80GB ~$15,000) : amortissement en 2 cycles de fine-tuning.

---

## CRITÈRES DE SUCCÈS FINAUX

| # | Critère | Mesure | Seuil |
|---|---|---|---|
| 1 | **Air-gapped** | Générer une vidéo 10s sans réseau | ✅ Fonctionne |
| 2 | **Zéro API externe** | `grep -r "google\|openai\|anthropic\|stripe" --include="*.py"` dans les imports actifs | 0 résultat |
| 3 | **Modèles propriétaires** | Tous les poids dans `models/aiprod-sovereign/` | 100% fichiers présents |
| 4 | **Pipeline réel** | POST `/v1/generate` retourne un `.mp4` valide | ffprobe OK |
| 5 | **Reproductible** | Même seed → même vidéo | PSNR > 55 dB |
| 6 | **Requirements figés** | `requirements.lock` existe et est valide | pip install OK |
| 7 | **Docker souverain** | Image Docker fonctionne air-gapped | ✅ Fonctionne |
| 8 | **Tests CI** | `tests/test_sovereignty.py` passe à 100% | 0 échecs |
| 9 | **Qualité vidéo** | FVD vs LTX-2 base | ≤ 1.3× baseline |
| 10 | **Documentation** | `SOVEREIGNTY_CERTIFICATE.md` complet | Auditable |

---

## SCORE CIBLE

| Critère | Actuel | Cible Phase 4 |
|---|---|---|
| Souveraineté réelle | 3/10 | **9/10** |
| Robustesse technique | 5/10 | **8/10** |
| Scalabilité GPU | 6/10 | **8/10** |
| Reproductibilité | 4/10 | **9/10** |
| Viabilité économique | 3/10 | **7/10** |

**Verdict cible : 👉 100% propriétaire réel**

---

*Plan établi le 2026-02-15.*  
*Première milestone : Phase 1 complétée → score souveraineté 6/10.*  
*Milestone finale : Phase 4 complétée → score souveraineté 9/10.*
