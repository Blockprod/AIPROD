<h1 align="center">COMPARAISON CONCEPT : AIPROD vs TAPNOW</h1>

<table align="center">
  <tr>
    <td><strong>Date</strong></td><td>20 février 2026</td>
    <td><strong>Version</strong></td><td>1.0</td>
  </tr>
  <tr>
    <td><strong>Type</strong></td><td>Audit comparatif structuré</td>
    <td><strong>Méthodologie</strong></td><td>Exploration exhaustive du code AIPROD + analyse publique de TapNow.ai</td>
  </tr>
</table>

---

## TABLE DES MATIÈRES

1. [Étape 1 — Exploration complète AIPROD](#1-exploration-complète-aiprod)
2. [Étape 2 — Cartographie technique AIPROD](#2-cartographie-technique-aiprod)
3. [Étape 3 — Évaluation technique interne AIPROD](#3-évaluation-technique-interne-aiprod)
4. [Étape 4 — Analyse comparative avec TapNow](#4-analyse-comparative-avec-tapnow)
5. [Étape 5 — Synthèse stratégique](#5-synthèse-stratégique)

---

## 1. EXPLORATION COMPLÈTE AIPROD

### 1.1 Arborescence des modules principaux

```
AIPROD/
├── packages/
│   ├── aiprod-core/        → Moteurs IA propriétaires (SHDT, HWVAE, NAC, TTS, LipSync)
│   ├── aiprod-pipelines/   → Orchestration, inférence, API, SaaS multi-tenant
│   ├── aiprod-trainer/     → Entraînement LoRA, curriculum, VAE, TTS
│   └── aiprod-cloud/       → Intégrations cloud optionnelles (GCP, Stripe, HF Hub)
├── config/                 → AIPROD_V34_SOVEREIGN.json (config pipeline souveraine)
├── configs/train/          → YAML d'entraînement (audio_vae, full_finetune, etc.)
├── models/                 → Checkpoints, modèles pré-entraînés, text-encoder
├── sovereign/              → Modèles souverains (.safetensors)
├── scripts/                → Utilitaires (download, quantize, test, deploy)
├── tests/                  → Tests unitaires et d'intégration
├── deploy/                 → Docker, Kubernetes, scripts de déploiement
├── docs/                   → Documentation, certificat de souveraineté
└── notebooks/              → Colab training notebook
```

### 1.2 Modules principaux identifiés

#### `aiprod-core` — Moteurs IA propriétaires

<table>
  <thead>
    <tr>
      <th width="200">Module</th>
      <th width="360">Fichiers source</th>
      <th width="340">Rôle</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>model/transformer/</code></td>
      <td>model.py · block.py · attention.py · norm.py · position.py</td>
      <td>SHDT — Backbone vidéo diffusion</td>
    </tr>
    <tr>
      <td><code>model/video_vae/</code></td>
      <td>encoder.py · decoder.py · config.py</td>
      <td>HWVAE — Encodeur/décodeur latent vidéo</td>
    </tr>
    <tr>
      <td><code>model/audio_vae/</code></td>
      <td>codec.py</td>
      <td>NAC — Codec audio neural (RVQ)</td>
    </tr>
    <tr>
      <td><code>model/tts/</code></td>
      <td>model.py · prosody.py · speaker_embedding.py · text_frontend.py · vocoder_tts.py</td>
      <td>TTS — Synthèse vocale multi-langue</td>
    </tr>
    <tr>
      <td><code>model/lip_sync/</code></td>
      <td>model.py</td>
      <td>Synchronisation labiale audio→vidéo</td>
    </tr>
    <tr>
      <td><code>model/text_encoder/</code></td>
      <td>bridge.py</td>
      <td>Pont vers encodeur texte (LLM Bridge)</td>
    </tr>
    <tr>
      <td><code>monitoring/</code></td>
      <td>logging.py · metrics.py · tracing.py</td>
      <td>Observabilité (Prometheus, OpenTelemetry)</td>
    </tr>
    <tr>
      <td><code>resilience/</code></td>
      <td>resilience.py</td>
      <td>GPU Health · OOM Fallback · Circuit Breaker</td>
    </tr>
  </tbody>
</table>

#### `aiprod-pipelines` — Orchestration & Inférence

<table>
  <thead>
    <tr>
      <th width="260">Module</th>
      <th width="100" align="right">Volume</th>
      <th width="340">Rôle</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>api/orchestrator.py</code></td>
      <td align="right">326 loc</td>
      <td>Machine à états 11 étapes avec checkpoints</td>
    </tr>
    <tr>
      <td><code>api/gateway.py</code></td>
      <td align="right">670 loc</td>
      <td>API FastAPI (JWT, rate limiting, tiers)</td>
    </tr>
    <tr>
      <td><code>api/gpu_worker.py</code></td>
      <td align="right">485 loc</td>
      <td>Worker GPU souverain (consomme la queue)</td>
    </tr>
    <tr>
      <td><code>inference/graph.py</code></td>
      <td align="right">385 loc</td>
      <td>DAG d'inférence composable</td>
    </tr>
    <tr>
      <td><code>inference/nodes.py</code></td>
      <td align="right">508 loc</td>
      <td>Nodes concrets (TextEncode, Denoise, Decode)</td>
    </tr>
    <tr>
      <td><code>inference/scenarist/</code></td>
      <td align="right">547 loc</td>
      <td>Scénariste IA local (Mistral / Llama)</td>
    </tr>
    <tr>
      <td><code>export/multi_format.py</code></td>
      <td align="right">403 loc</td>
      <td>Export multi-codec via FFmpeg</td>
    </tr>
    <tr>
      <td><code>api/billing_service.py</code></td>
      <td align="right">421 loc</td>
      <td>Facturation usage-based (Stripe optionnel)</td>
    </tr>
    <tr>
      <td><code>inference/distributed_lora/</code></td>
      <td align="right">6 fichiers</td>
      <td>LoRA distribué · fédéré · merge engine</td>
    </tr>
    <tr>
      <td><code>inference/tensor_parallelism/</code></td>
      <td align="right">7 fichiers</td>
      <td>Sharding · load balancer · gradient accumulation</td>
    </tr>
    <tr>
      <td><code>inference/multi_tenant_saas/</code></td>
      <td align="right">10+ fichiers</td>
      <td>Auth · billing · tenant isolation · job manager</td>
    </tr>
    <tr>
      <td><code>inference/edge_deployment/</code></td>
      <td align="right">7 fichiers</td>
      <td>Mobile runtime · pruning · quantization edge</td>
    </tr>
  </tbody>
</table>

#### `aiprod-trainer` — Entraînement & `aiprod-cloud` — Intégrations

<table>
  <thead>
    <tr>
      <th width="120">Package</th>
      <th width="230">Module</th>
      <th width="100" align="right">Volume</th>
      <th width="310">Rôle</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>trainer</code></td>
      <td><code>trainer.py</code></td>
      <td align="right">1 028 loc</td>
      <td>Entraîneur principal (Accelerate + PEFT + LoRA)</td>
    </tr>
    <tr>
      <td><code>trainer</code></td>
      <td><code>curriculum_training.py</code></td>
      <td align="right">—</td>
      <td>Entraînement progressif par curriculum</td>
    </tr>
    <tr>
      <td><code>trainer</code></td>
      <td><code>vae_trainer.py</code></td>
      <td align="right">—</td>
      <td>Entraîneur VAE dédié</td>
    </tr>
    <tr>
      <td><code>trainer</code></td>
      <td><code>tts_trainer.py</code></td>
      <td align="right">—</td>
      <td>Entraîneur TTS dédié</td>
    </tr>
    <tr>
      <td><code>cloud</code></td>
      <td><em>6 fichiers</em></td>
      <td align="right">—</td>
      <td>Optionnel : GCP · Gemini · Stripe · HF Hub</td>
    </tr>
  </tbody>
</table>

### 1.3 Dépendances identifiées

#### Dépendances core (obligatoires, souveraines)

<table>
  <thead>
    <tr>
      <th width="140">Catégorie</th>
      <th width="480">Dépendances</th>
      <th width="200" align="center">Souveraineté</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>GPU / ML</strong></td>
      <td>torch 2.5+ · torchaudio · torchvision · xformers · accelerate · peft</td>
      <td align="center">✅ Open-source, exécution locale</td>
    </tr>
    <tr>
      <td><strong>Diffusion</strong></td>
      <td>transformers 4.47+ · safetensors · einops · diffusers</td>
      <td align="center">✅ HuggingFace open-source</td>
    </tr>
    <tr>
      <td><strong>Vidéo</strong></td>
      <td>opencv-python · imageio · av (PyAV) · scenedetect</td>
      <td align="center">✅ Local</td>
    </tr>
    <tr>
      <td><strong>Audio</strong></td>
      <td>librosa · torchaudio</td>
      <td align="center">✅ Local</td>
    </tr>
    <tr>
      <td><strong>API</strong></td>
      <td>fastapi · uvicorn · pydantic</td>
      <td align="center">✅ Local</td>
    </tr>
    <tr>
      <td><strong>Observabilité</strong></td>
      <td>prometheus-client · opentelemetry · structlog · mlflow</td>
      <td align="center">✅ Auto-hébergeable</td>
    </tr>
    <tr>
      <td><strong>Quantisation</strong></td>
      <td>bitsandbytes · optimum-quanto</td>
      <td align="center">✅ Local</td>
    </tr>
    <tr>
      <td><strong>Data</strong></td>
      <td>numpy · pandas · scipy · scikit-learn</td>
      <td align="center">✅ Open-source</td>
    </tr>
  </tbody>
</table>

#### Dépendances optionnelles (cloud, non requises)

<table>
  <thead>
    <tr>
      <th width="160">Service</th>
      <th width="300">Dépendance</th>
      <th width="340">Impact souveraineté</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Stockage cloud</td>
      <td>google-cloud-storage · boto3</td>
      <td>⚠️ Optionnel uniquement</td>
    </tr>
    <tr>
      <td>Paiement</td>
      <td>stripe</td>
      <td>⚠️ SaaS nécessaire pour monétisation</td>
    </tr>
    <tr>
      <td>Tracking expé</td>
      <td>wandb</td>
      <td>⚠️ Remplaçable par MLflow local</td>
    </tr>
    <tr>
      <td>Captioning</td>
      <td>google-generativeai (Gemini)</td>
      <td>⚠️ Optionnel, fallback local existe</td>
    </tr>
    <tr>
      <td>Model hub</td>
      <td>huggingface-hub</td>
      <td>⚠️ Téléchargement initial uniquement</td>
    </tr>
  </tbody>
</table>

### 1.4 Moteurs IA réellement utilisés

<table>
  <thead>
    <tr>
      <th width="180">Moteur</th>
      <th width="130">Type</th>
      <th width="280">Implémentation</th>
      <th width="280">Poids entraînés</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>SHDT</strong><br/><sub>Scalable Hybrid Diffusion Transformer</sub></td>
      <td>Backbone vidéo</td>
      <td>✅ Code complet — GQA, dual-stream spatial/temporal<br/><sub>model.py · block.py · attention.py</sub></td>
      <td>⚠️ Checkpoint <code>models/ltx2_research</code> — poids probablement basés sur LTX-2</td>
    </tr>
    <tr>
      <td><strong>HWVAE</strong><br/><sub>Hierarchical Wavelet VAE</sub></td>
      <td>Encodeur/décodeur latent</td>
      <td>✅ Code complet — Haar wavelet, separable conv</td>
      <td>⚠️ Entraînement VAE en cours<br/><sub>configs/train/audio_vae.yaml</sub></td>
    </tr>
    <tr>
      <td><strong>NAC</strong><br/><sub>Neural Audio Codec</sub></td>
      <td>Codec audio neural</td>
      <td>✅ Code complet — RVQ multi-bande, Snake activation</td>
      <td>⚠️ Pas de checkpoint entraîné visible</td>
    </tr>
    <tr>
      <td><strong>TTS</strong><br/><sub>FastSpeech 2 + HiFi-GAN</sub></td>
      <td>Synthèse vocale</td>
      <td>✅ Code complet — text_frontend, prosody, vocoder</td>
      <td>⚠️ Pas de checkpoint entraîné visible</td>
    </tr>
    <tr>
      <td><strong>LipSync</strong></td>
      <td>Audio → animation faciale</td>
      <td>✅ Code complet — Conv1D + BiLSTM → FLAME 52 params</td>
      <td>⚠️ Pas de checkpoint entraîné visible</td>
    </tr>
    <tr>
      <td><strong>Text Encoder Bridge</strong></td>
      <td>Encodage texte</td>
      <td>✅ Code pont flexible</td>
      <td>✅ <code>aiprod-text-encoder-v1</code> présent</td>
    </tr>
    <tr>
      <td><strong>Scenarist</strong><br/><sub>Mistral-7B</sub></td>
      <td>Planification scènes</td>
      <td>✅ Code complet + fallback rule-based</td>
      <td>⚠️ <code>models/scenarist/mistral-7b</code> référencé</td>
    </tr>
    <tr>
      <td><strong>QA Sémantique</strong><br/><sub>CLIP / SigLIP</sub></td>
      <td>Évaluation qualité</td>
      <td>✅ Intégré dans pipeline</td>
      <td>⚠️ <code>models/clip</code> référencé</td>
    </tr>
  </tbody>
</table>

### 1.5 Modules simulés ou incomplets

<table>
  <thead>
    <tr>
      <th width="80" align="center">Statut</th>
      <th width="260">Module</th>
      <th width="480">Détail</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">🟡</td>
      <td>Entraînement SHDT from scratch</td>
      <td>Le trainer utilise LoRA fine-tuning sur checkpoints existants (LTX-2) — pas d'entraînement from scratch du backbone 2B+ params</td>
    </tr>
    <tr>
      <td align="center">🔴</td>
      <td>Poids NAC (Audio Codec)</td>
      <td>Architecture codée mais aucun <code>.safetensors</code> entraîné visible</td>
    </tr>
    <tr>
      <td align="center">🔴</td>
      <td>Poids TTS</td>
      <td>Architecture complète mais pas de modèle entraîné dans <code>models/</code></td>
    </tr>
    <tr>
      <td align="center">🔴</td>
      <td>Poids LipSync</td>
      <td>Architecture complète mais pas de modèle entraîné</td>
    </tr>
    <tr>
      <td align="center">🟡</td>
      <td>Edge deployment</td>
      <td>7 fichiers d'architecture mais pas de déploiement mobile validé</td>
    </tr>
    <tr>
      <td align="center">🟡</td>
      <td>Tensor parallelism</td>
      <td>Code d'infrastructure mais nécessite cluster multi-GPU pour validation</td>
    </tr>
    <tr>
      <td align="center">🟡</td>
      <td>Multi-tenant SaaS</td>
      <td>Architecture complète (auth, billing, tenant) mais pas déployé en prod</td>
    </tr>
    <tr>
      <td align="center">🟡</td>
      <td>Reward modeling / RLHF</td>
      <td>Code présent mais nécessite données humaines et infrastructure</td>
    </tr>
    <tr>
      <td align="center">🟡</td>
      <td>Federated LoRA training</td>
      <td>Architecture codée, non validée à l'échelle</td>
    </tr>
  </tbody>
</table>

---

## 2. CARTOGRAPHIE TECHNIQUE AIPROD

### 2.1 Orchestrateur

```
┌─────────────────────────────────────────────────────────┐
│              AIPROD Orchestrator v34                      │
│         Machine à États (11 états + transitions)         │
├─────────────────────────────────────────────────────────┤
│ INIT → ANALYSIS → CREATIVE_DIRECTION → VISUAL_TRANSLATION│
│      → RENDER_EXECUTION → QA_TECHNICAL → QA_SEMANTIC     │
│      → FINALIZE                                          │
│                                                          │
│ Fast Track: INIT → FAST_TRACK → RENDER_EXECUTION         │
│ Error:      * → ERROR → INIT (retry max 3×, backoff 5s)  │
│                                                          │
│ Checkpoint: Sauvegarde avant chaque état                 │
│ Recovery:   Resume depuis dernier checkpoint réussi      │
│ Config:     AIPROD_V34_SOVEREIGN.json                    │
└─────────────────────────────────────────────────────────┘
```

**Forces :** Machine à états formelle, checkpoint/resume, fast-track pour prompts simples, retry policy configurable.  
**Risque :** Complexité élevée pour un produit pre-prod.

### 2.2 Pipeline Vidéo

```
Prompt utilisateur
    │
    ▼
[InputSanitizer] ─── Normalisation pure function
    │
    ▼
[CreativeDirector] ─── Mistral-7B local → script + shot_list + consistency_markers
    │
    ▼
[VisualTranslator] ─── Text Encoder local → prompts LTX-2 optimisés
    │
    ▼
[InferenceGraph DAG]
    ├── TextEncodeNode (text → embeddings)
    ├── DenoiseNode (SHDT diffusion loop, guidance, LoRA)
    ├── UpsampleNode (2× spatial)
    └── DecodeVideoNode (HWVAE latent → frames, tiled decoding)
    │
    ▼
[ExportEngine] ─── FFmpeg subprocess → H.264/H.265/ProRes/DNxHR/VP9/AV1
    │
    ▼
[TechnicalQAGate] ─── Checks déterministes (format, durée, codec, taille)
    │
    ▼
[SemanticQA] ─── CLIP/SigLIP local → similarité prompt/vidéo
    │
    ▼
[Supervisor] ─── Approbation finale (règles déterministes)
```

**Résolutions supportées :** 512×768, 768×512, 512×512, extensible à 1920×1080, 3840×2160  
**FPS :** 24, 25, 30, 48, 60  
**Durée max :** 120s (config gateway), clips de 10s max par segment  
**Seed déterministe :** Oui, hash-based

### 2.3 Pipeline Audio

```
Texte (narration)
    │
    ▼
[TextFrontend] ─── Tokenisation phonémique (vocab 150)
    │
    ▼
[TextEncoder] ─── Transformer 4 couches (384 dim)
    │
    ▼
[ProsodyModeler] ─── Durée, F0, énergie
    │
    ▼
[MelDecoder] ─── Transformer 6 couches → mel-spectrogram (80 bandes)
    │
    ▼
[VocoderTTS] ─── HiFi-GAN → waveform 24kHz
    │
    ▼
[NeuralAudioCodec] ─── RVQ 8 codebooks, multi-bande, compression 480×
    │
    ▼
[LipSync] ─── mel → Conv1D → BiLSTM → FLAME 52 params (30fps)
```

**Statut :** Architecture complète mais **poids non entraînés** pour TTS, NAC et LipSync.

### 2.4 Rendering Engine

<table>
  <thead>
    <tr>
      <th width="200">Composant</th>
      <th width="380">Technologie</th>
      <th width="140" align="center">Statut</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Backbone diffusion</td>
      <td>SHDT (dual-stream, GQA, adaptive RMSNorm)</td>
      <td align="center">✅ Code + checkpoint</td>
    </tr>
    <tr>
      <td>Latent space</td>
      <td>HWVAE (Haar wavelet, separable conv 2D+1D)</td>
      <td align="center">✅ Code ok · poids ⚠️</td>
    </tr>
    <tr>
      <td>Text conditioning</td>
      <td>LLM Bridge → cross-attention à chaque layer</td>
      <td align="center">✅ Fonctionnel</td>
    </tr>
    <tr>
      <td>Scheduler</td>
      <td>Adaptive Flow Scheduler (configurable)</td>
      <td align="center">✅ Code</td>
    </tr>
    <tr>
      <td>Quantisation</td>
      <td>FP8 transformer, 8-bit text encoder</td>
      <td align="center">✅ Fonctionnel</td>
    </tr>
    <tr>
      <td>Tiled decoding</td>
      <td>Auto-tiler avec blending</td>
      <td align="center">✅ Code complet</td>
    </tr>
    <tr>
      <td>Kernel fusion</td>
      <td>Adaptive fusion engine</td>
      <td align="center">🟡 Structure</td>
    </tr>
  </tbody>
</table>

### 2.5 Infrastructure GPU

<table>
  <thead>
    <tr>
      <th width="200">Élément</th>
      <th width="540">Implémentation</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Device</td><td>CUDA (GPU NVIDIA)</td></tr>
    <tr><td>Precision</td><td>bf16 / fp16 / fp8 (configurable)</td></tr>
    <tr><td>VRAM management</td><td>OOM Fallback automatique (résolution/batch downgrade)</td></tr>
    <tr><td>Health monitoring</td><td>GPUHealthMonitor (VRAM, température, ECC, utilisation)</td></tr>
    <tr><td>Multi-GPU</td><td>Tensor parallelism (sharding, gradient accumulation)</td></tr>
    <tr><td>Batch scheduling</td><td>Dynamic batch sizing (adaptive, memory profiler)</td></tr>
    <tr><td>Thermal</td><td><code>optimize_gpu_thermal.bat</code> · <code>configure_gpu_thermal.ps1</code></td></tr>
    <tr><td>Worker</td><td>GPUWorker async (poll job queue, exécute InferenceGraph)</td></tr>
  </tbody>
</table>

### 2.6 Gestion des Seeds

```python
# reproducibility.py — Garantie bit-exact
set_deterministic_mode(seed=42)
├── random.seed(seed)
├── os.environ["PYTHONHASHSEED"] = str(seed)
├── numpy.random.seed(seed)
├── torch.manual_seed(seed) + cuda.manual_seed_all(seed)
├── cudnn.deterministic = True, benchmark = False
├── CUBLAS_WORKSPACE_CONFIG = ":4096:8"
└── torch.use_deterministic_algorithms(True, warn_only=True)
```

**Seed source :** hash-based (config : `"seedSource": "hash"`)  
**Reproductibilité :** Visée bit-exact (avec warn_only pour ops non-déterministes)

### 2.7 Gestion des Erreurs

<table>
  <thead>
    <tr>
      <th width="200">Mécanisme</th>
      <th width="540">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Orchestrator retry</strong></td>
      <td>Max 3 retries, backoff 5s, escalation vers ERROR</td>
    </tr>
    <tr>
      <td><strong>Checkpoint / Resume</strong></td>
      <td>Sauvegarde état avant chaque étape, restore depuis dernier succès</td>
    </tr>
    <tr>
      <td><strong>RecoveryManager</strong></td>
      <td>Actions : RETRY (avec context restauré) · ROLLBACK (retour INIT) · SKIP · ERROR</td>
    </tr>
    <tr>
      <td><strong>CircuitBreaker</strong></td>
      <td>Ouverture après N erreurs consécutives, half-open probing</td>
    </tr>
    <tr>
      <td><strong>OOMFallback</strong></td>
      <td>Downgrade automatique résolution/batch sur CUDA OOM</td>
    </tr>
    <tr>
      <td><strong>DeadlineManager</strong></td>
      <td>Timeout par étape avec annulation gracieuse</td>
    </tr>
    <tr>
      <td><strong>DataIntegrity</strong></td>
      <td>Vérification SHA-256 des checkpoints et datasets</td>
    </tr>
    <tr>
      <td><strong>DriftDetector</strong></td>
      <td>Monitoring FID / CLIP-Score vs distribution de référence</td>
    </tr>
  </tbody>
</table>

### 2.8 Export / Format

<table>
  <thead>
    <tr>
      <th width="180">Usage</th>
      <th width="260">Codec</th>
      <th width="120" align="center">Container</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Web standard</td><td>H.264</td><td align="center"><code>.mp4</code></td></tr>
    <tr><td>Haute qualité</td><td>H.265 (HEVC)</td><td align="center"><code>.mp4</code> · <code>.mkv</code></td></tr>
    <tr><td>Post-production</td><td>ProRes 422 / 4444</td><td align="center"><code>.mov</code></td></tr>
    <tr><td>Broadcast</td><td>DNxHD / DNxHR</td><td align="center"><code>.mxf</code></td></tr>
    <tr><td>Web moderne</td><td>VP9</td><td align="center"><code>.webm</code></td></tr>
    <tr><td>Next-gen</td><td>AV1</td><td align="center"><code>.webm</code></td></tr>
    <tr><td>Image sequence</td><td>EXR / DPX</td><td align="center">dossier</td></tr>
    <tr><td>Audio</td><td>AAC · OPUS · FLAC · PCM · Dolby Digital / Atmos</td><td align="center">muxé</td></tr>
  </tbody>
</table>

**Pipeline :** tensor frames → raw pipe → FFmpeg subprocess → container

---

## 3. ÉVALUATION TECHNIQUE INTERNE AIPROD

### 3.1 Cohérence architecturale — **8/10**

<table>
  <thead>
    <tr>
      <th width="60" align="center">⬤</th>
      <th width="320">Aspect</th>
      <th width="400">Évaluation</th>
    </tr>
  </thead>
  <tbody>
    <tr><td align="center">✅</td><td>Séparation core / pipelines / trainer / cloud</td><td>Excellente modularité monorepo</td></tr>
    <tr><td align="center">✅</td><td>Config centralisée (V34_SOVEREIGN.json)</td><td>Single source of truth</td></tr>
    <tr><td align="center">✅</td><td>Pipeline DAG composable</td><td>Design élégant et extensible</td></tr>
    <tr><td align="center">✅</td><td>Machine à états + checkpoint</td><td>Pattern robuste pour production</td></tr>
    <tr><td align="center">✅</td><td>Cohérence des conventions</td><td>Docstrings, typing, dataclasses systématiques</td></tr>
    <tr><td align="center">⚠️</td><td><strong>Point faible</strong></td><td>Architecture très ambitieuse par rapport au stade de développement. Risque de sur-ingénierie prématurée.</td></tr>
  </tbody>
</table>

### 3.2 Modularité réelle — **8.5/10**

- **4 packages indépendants** installables séparément (pyproject.toml workspace UV)
- **Dépendances cloud isolées** dans `aiprod-cloud` (optionnel)
- **Nodes d'inférence pluggables** via InferenceGraph DAG
- **Adapters pattern** pour chaque étape du pipeline
- **Training strategies** (text-to-video, video-to-video) comme plugins
- **Point faible :** Certains modules ont des dépendances implicites sur la structure de fichiers locale

### 3.3 Robustesse technique — **6/10**

<table>
  <thead>
    <tr>
      <th width="230">Aspect</th>
      <th width="90" align="center">Score</th>
      <th width="440">Justification</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Gestion d'erreurs</td>
      <td align="center"><strong>8</strong>/10</td>
      <td>Circuit breaker, OOM fallback, checkpoint/resume</td>
    </tr>
    <tr>
      <td>Tests</td>
      <td align="center"><strong>5</strong>/10</td>
      <td>~120+ fichiers de test mais couverture qualité incertaine, tests principalement structurels</td>
    </tr>
    <tr>
      <td>Poids entraînés</td>
      <td align="center"><strong>4</strong>/10</td>
      <td>Seul le text encoder et LTX-2 base semblent disponibles. TTS, NAC, LipSync sans poids</td>
    </tr>
    <tr>
      <td>Validation end-to-end</td>
      <td align="center"><strong>3</strong>/10</td>
      <td>Pas de preuve de génération vidéo complète fonctionnelle avec poids entraînés</td>
    </tr>
    <tr>
      <td>CI/CD</td>
      <td align="center"><strong>6</strong>/10</td>
      <td>GitHub Actions (sovereignty-check.yml) mais pas de pipeline de test GPU</td>
    </tr>
  </tbody>
</table>

### 3.4 Reproductibilité — **7.5/10**

- ✅ Module dédié `reproducibility.py` avec contrôle complet des seeds
- ✅ `torch.use_deterministic_algorithms(True, warn_only=True)`
- ✅ Seed hash-based dans la config pipeline
- ⚠️ `warn_only=True` signifie que certaines opérations restent non-déterministes
- ⚠️ Pas de tests de reproductibilité bit-exact documentés

### 3.5 Scalabilité SaaS — **7/10**

<table>
  <thead>
    <tr>
      <th width="60" align="center">⬤</th>
      <th width="340">Composant SaaS</th>
      <th width="360">Détail</th>
    </tr>
  </thead>
  <tbody>
    <tr><td align="center">✅</td><td>Multi-tenant isolation</td><td>Architecturé (tenant_context, access_control)</td></tr>
    <tr><td align="center">✅</td><td>API Gateway (rate limiting, tiers)</td><td>Code complet (Free / Pro / Enterprise)</td></tr>
    <tr><td align="center">✅</td><td>Billing (usage-based, Stripe)</td><td>Code complet avec fallback local</td></tr>
    <tr><td align="center">✅</td><td>Job queue (async GPU)</td><td>GPUWorker + JobStore (SQLite)</td></tr>
    <tr><td align="center">🟡</td><td>Collaboration (WebSocket)</td><td>Structure présente</td></tr>
    <tr><td align="center">🟡</td><td>Dashboard analytics</td><td>Structure présente</td></tr>
    <tr><td align="center">🔴</td><td><strong>Déploiement réel</strong></td><td><strong>Non déployé en production</strong></td></tr>
  </tbody>
</table>

### 3.6 Souveraineté technologique — **8.5/10**

**Score autoproclamé : 9/10 — Score audité : 8.5/10**

<table>
  <thead>
    <tr>
      <th width="60" align="center">⬤</th>
      <th width="340">Critère</th>
      <th width="380">Résultat</th>
    </tr>
  </thead>
  <tbody>
    <tr><td align="center">✅</td><td>Exécution 100% locale possible</td><td>Oui (GPU local, pas de cloud requis)</td></tr>
    <tr><td align="center">✅</td><td>Zéro API externe obligatoire</td><td>Confirmé (cloud = optionnel)</td></tr>
    <tr><td align="center">✅</td><td>Modèles locaux</td><td>Text encoder souverain + LTX-2 en local</td></tr>
    <tr><td align="center">✅</td><td>Données ne quittent pas la machine</td><td>Architecture locale-first</td></tr>
    <tr><td align="center">✅</td><td>Dépendance open-source uniquement</td><td>PyTorch, HuggingFace, FastAPI</td></tr>
    <tr><td align="center">✅</td><td>Monitoring auto-hébergeable</td><td>Prometheus, OpenTelemetry, MLflow</td></tr>
    <tr><td align="center">⚠️</td><td><strong>Points de dépendance résiduels</strong></td><td>Stripe (monétisation) · HF Hub (download initial) · FFmpeg (binaire externe)</td></tr>
  </tbody>
</table>

### 3.7 Viabilité économique supposée — **5/10**

<table>
  <thead>
    <tr>
      <th width="200">Facteur</th>
      <th width="540">Analyse</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Coûts GPU</td><td>Très élevés (A100/H100 requis pour inférence haute qualité)</td></tr>
    <tr><td>Pricing planifié</td><td>Free (5 req/j) · Pro (100 req/j) · Enterprise (100K req/j)</td></tr>
    <tr><td>Coût par vidéo</td><td>Estimé $0.01–$0.10 / seconde (résolution-dépendant)</td></tr>
    <tr><td>Compétition pricing</td><td>TapNow : ~$0.009 / vidéo via crédits agrégés multi-modèles</td></tr>
    <tr><td>Revenus</td><td>Aucun actuellement (pre-product)</td></tr>
    <tr><td>Burn rate</td><td>Infrastructure GPU significative avant revenus</td></tr>
    <tr><td><strong>⚠️ Risque majeur</strong></td><td><strong>Investissement R&D massif sans preuve de marché, coûts d'entraînement des modèles propriétaires non chiffrés</strong></td></tr>
  </tbody>
</table>

---

## 4. ANALYSE COMPARATIVE AVEC TAPNOW

### 4.1 Positionnement produit

<table>
  <thead>
    <tr>
      <th width="180">Dimension</th>
      <th width="330">AIPROD</th>
      <th width="330">TapNow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Positionnement</strong></td>
      <td>Moteur IA vidéo souverain, propriétaire, full-stack</td>
      <td>Plateforme d'orchestration multi-modèles (agrégateur)</td>
    </tr>
    <tr>
      <td><strong>Cible</strong></td>
      <td>Studios pro, entreprises souveraines, marché B2B</td>
      <td>Créateurs, agences, PME, e-commerce (B2C + B2B)</td>
    </tr>
    <tr>
      <td><strong>Proposition de valeur</strong></td>
      <td>« Possédez votre moteur IA vidéo »</td>
      <td>« Tous les meilleurs modèles en un seul workflow »</td>
    </tr>
    <tr>
      <td><strong>Stade</strong></td>
      <td>Pre-product (architecture avancée, pas de produit live)</td>
      <td>Produit live, 1M+ utilisateurs, revenus actifs</td>
    </tr>
    <tr>
      <td><strong>Marché géographique</strong></td>
      <td>Europe (souveraineté)</td>
      <td>Global (forte présence Asie : ByteDance, Tencent, JD.com, Alibaba)</td>
    </tr>
  </tbody>
</table>

### 4.2 Niveau d'automatisation apparent

<table>
  <thead>
    <tr>
      <th width="240">Fonctionnalité</th>
      <th width="100" align="center">AIPROD</th>
      <th width="100" align="center">TapNow</th>
      <th width="340">Commentaire</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Prompt → vidéo</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td>AIPROD : 11 étapes automatisées · TapNow : multi-modèles</td>
    </tr>
    <tr>
      <td>Script → storyboard</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td>AIPROD : Scenarist local (Mistral-7B) · TapNow : one-click</td>
    </tr>
    <tr>
      <td>QA automatique</td>
      <td align="center">✅</td>
      <td align="center">⚠️</td>
      <td>AIPROD : double gate (technique + sémantique) · TapNow : non documenté</td>
    </tr>
    <tr>
      <td>Multi-shot narrative</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td>TapNow : « 16s multi-shot narrative in one click » (Vidu Q3)</td>
    </tr>
    <tr>
      <td>Draw to video</td>
      <td align="center">🔴</td>
      <td align="center">✅</td>
      <td>—</td>
    </tr>
    <tr>
      <td>Pose control</td>
      <td align="center">🔴</td>
      <td align="center">✅</td>
      <td>—</td>
    </tr>
    <tr>
      <td>Motion transfer</td>
      <td align="center">🔴</td>
      <td align="center">✅</td>
      <td>TapNow : « Motion Asset Library »</td>
    </tr>
    <tr>
      <td>Camera control</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td>AIPROD : <code>camera_control.py</code> · TapNow : live</td>
    </tr>
    <tr>
      <td>Lip sync</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td>AIPROD : architecture complète · TapNow : « AI lip-sync avatar »</td>
    </tr>
    <tr>
      <td>Image enhancement</td>
      <td align="center">🔴</td>
      <td align="center">✅</td>
      <td>TapNow : « 4K video / 1080p image enhancement »</td>
    </tr>
    <tr>
      <td>E-commerce templates</td>
      <td align="center">🔴</td>
      <td align="center">✅</td>
      <td>TapNow : « One-click e-commerce photo sets »</td>
    </tr>
    <tr>
      <td colspan="4"></td>
    </tr>
    <tr>
      <td><strong>Score automatisation</strong></td>
      <td align="center"><strong>6/10</strong></td>
      <td align="center"><strong>9/10</strong></td>
      <td><em>AIPROD : architecturé mais non validé · TapNow : live et fonctionnel</em></td>
    </tr>
  </tbody>
</table>

### 4.3 Complexité technologique supposée

<table>
  <thead>
    <tr>
      <th width="220">Aspect</th>
      <th width="310">AIPROD</th>
      <th width="310">TapNow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Modèles propriétaires</strong></td>
      <td>✅ SHDT · HWVAE · NAC · TTS · LipSync — code original</td>
      <td>🔴 Aucun — agrège des modèles tiers</td>
    </tr>
    <tr>
      <td><strong>Architecture IA custom</strong></td>
      <td>✅ GQA · dual-stream · wavelet VAE · RVQ audio</td>
      <td>🔴 Utilise Kling, Veo, Runway, Hailuo, Seedance…</td>
    </tr>
    <tr>
      <td><strong>Entraînement</strong></td>
      <td>✅ Pipeline complet (LoRA, curriculum, VAE, TTS)</td>
      <td>⚠️ Probablement fine-tuning minimal ou aucun</td>
    </tr>
    <tr>
      <td><strong>Orchestration</strong></td>
      <td>✅ Machine à états formelle avec checkpoints</td>
      <td>✅ Tapflow (workflow canvas infini)</td>
    </tr>
    <tr>
      <td><strong>Infra GPU</strong></td>
      <td>✅ Gestion complète (OOM, thermal, parallelism)</td>
      <td>☁️ Délégué aux providers (Google, Kuaishou…)</td>
    </tr>
    <tr>
      <td><strong>Frontend</strong></td>
      <td>🔴 Pas d'interface utilisateur</td>
      <td>✅ Web app riche (canvas, templates, community)</td>
    </tr>
    <tr>
      <td colspan="3"></td>
    </tr>
    <tr>
      <td><strong>Score complexité</strong></td>
      <td><strong>9/10</strong></td>
      <td><strong>6/10</strong> <em>(complexité dans l'intégration, pas dans le moteur)</em></td>
    </tr>
  </tbody>
</table>

### 4.4 Différenciation stratégique

<table>
  <thead>
    <tr>
      <th width="200">Facteur</th>
      <th width="320">AIPROD</th>
      <th width="320">TapNow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>IP propriétaire</strong></td>
      <td>✅ Fort — architectures IA originales, code source complet</td>
      <td>🔴 Faible — dépend de modèles tiers</td>
    </tr>
    <tr>
      <td><strong>Souveraineté</strong></td>
      <td>✅ Maximal — exécution 100% locale possible</td>
      <td>🔴 Aucune — 100% cloud, dépend de Google, ByteDance…</td>
    </tr>
    <tr>
      <td><strong>Flexibilité modèles</strong></td>
      <td>✅ Un seul moteur maîtrisé de bout en bout</td>
      <td>✅ Accès à 15+ modèles simultanément</td>
    </tr>
    <tr>
      <td><strong>Time-to-market</strong></td>
      <td>🔴 Très lent (pre-product)</td>
      <td>✅ Déjà live avec traction</td>
    </tr>
    <tr>
      <td><strong>Effet réseau</strong></td>
      <td>🔴 Inexistant</td>
      <td>✅ TapTV community · « Selected Recipes » · 1M+ users</td>
    </tr>
    <tr>
      <td><strong>Lock-in client</strong></td>
      <td>✅ Déploiement on-premise possible</td>
      <td>🔴 Dépendance plateforme + crédits non transférables</td>
    </tr>
  </tbody>
</table>

### 4.5 Défensabilité

<table>
  <thead>
    <tr>
      <th width="170">Barrière</th>
      <th width="330">AIPROD</th>
      <th width="330">TapNow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Moat technologique</strong></td>
      <td>✅ Fort si modèles entraînés — architecture unique difficile à reproduire</td>
      <td>🔴 Faible — n'importe qui peut agréger les mêmes APIs</td>
    </tr>
    <tr>
      <td><strong>Moat réseau</strong></td>
      <td>🔴 Aucun</td>
      <td>✅ Communauté · templates · « Recipes » partagées</td>
    </tr>
    <tr>
      <td><strong>Moat données</strong></td>
      <td>🔴 Pas encore de données utilisateur</td>
      <td>✅ Données d'usage de 1M+ utilisateurs</td>
    </tr>
    <tr>
      <td><strong>Moat marque</strong></td>
      <td>🔴 Inconnue du marché</td>
      <td>✅ Partenaire officiel Google · clients enterprise (Honda, ByteDance, WPP)</td>
    </tr>
    <tr>
      <td><strong>Switching cost</strong></td>
      <td>✅ Fort pour on-premise (intégration profonde)</td>
      <td>🟡 Moyen (crédits non transférables mais workflow reproductible ailleurs)</td>
    </tr>
  </tbody>
</table>

### 4.6 Dépendance externe probable

<table>
  <thead>
    <tr>
      <th width="400">AIPROD — Dépendances</th>
      <th width="400">TapNow — Dépendances</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>PyTorch / CUDA (NVIDIA)</td><td>Google Veo API</td></tr>
    <tr><td>HuggingFace (modèles de base)</td><td>Kuaishou Kling API</td></tr>
    <tr><td>FFmpeg (encodage)</td><td>MiniMax Hailuo API</td></tr>
    <tr><td>NVIDIA GPU hardware</td><td>Shengshu Vidu API</td></tr>
    <tr><td>Stripe (monétisation optionnelle)</td><td>Alibaba Wan API</td></tr>
    <tr><td></td><td>ByteDance Seedance API</td></tr>
    <tr><td></td><td>Runway API</td></tr>
    <tr><td></td><td>Midjourney API</td></tr>
    <tr><td></td><td>TopazLabs API</td></tr>
    <tr><td></td><td>GPT (OpenAI) pour assistance</td></tr>
    <tr><td></td><td>Flux / Imagen APIs</td></tr>
    <tr>
      <td><strong>Score indépendance : 8/10</strong></td>
      <td><strong>Score indépendance : 2/10</strong></td>
    </tr>
  </tbody>
</table>

> ⚠️ **Zone d'incertitude TapNow :** Les termes exacts des accords API avec ces fournisseurs ne sont pas publics. TapNow pourrait avoir des accords privilégiés ou être revendeur officiel, ce qui réduirait le risque de coupure.

### 4.7 Barrières à l'entrée

<table>
  <thead>
    <tr>
      <th width="200">Barrière</th>
      <th width="310">AIPROD <sub>(pour un concurrent)</sub></th>
      <th width="310">TapNow <sub>(pour un concurrent)</sub></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Capital requis</strong></td>
      <td>$$$$$ (GPU, recherche, entraînement)</td>
      <td>$$ (intégration API, frontend)</td>
    </tr>
    <tr>
      <td><strong>Temps de développement</strong></td>
      <td>18–36 mois minimum</td>
      <td>3–6 mois pour un MVP comparable</td>
    </tr>
    <tr>
      <td><strong>Talent requis</strong></td>
      <td>PhD-level ML researchers</td>
      <td>Senior full-stack + UX designers</td>
    </tr>
    <tr>
      <td><strong>Risque technique</strong></td>
      <td>Très élevé (modèles peuvent ne pas converger)</td>
      <td>Faible (APIs stables testées)</td>
    </tr>
    <tr>
      <td><strong>Risque marché</strong></td>
      <td>Élevé (produit non validé)</td>
      <td>Moyen (marché validé mais concurrence forte)</td>
    </tr>
  </tbody>
</table>

---

## 5. SYNTHÈSE STRATÉGIQUE

### 5.1 Quel concept possède l'avantage long terme ?

**→ AIPROD (si les modèles sont entraînés avec succès)**

<table>
  <thead>
    <tr>
      <th width="200">Argument</th>
      <th width="560">Détail</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Propriété intellectuelle</td><td>AIPROD possède son moteur. TapNow loue ceux des autres.</td></tr>
    <tr><td>Souveraineté</td><td>Marché croissant (réglementations EU, RGPD, AI Act). AIPROD est positionné. TapNow est vulnérable.</td></tr>
    <tr><td>Marges</td><td>AIPROD peut atteindre des marges élevées à terme (pas de coût API tiers). TapNow paie chaque génération aux providers.</td></tr>
    <tr><td>Différenciation</td><td>Les agrégateurs sont commoditisés rapidement. Les moteurs propriétaires créent de la valeur durable.</td></tr>
    <tr><td><strong>⚠️ Condition sine qua non</strong></td><td><strong>Les modèles SHDT, HWVAE, NAC, TTS doivent être entraînés et atteindre la qualité state-of-the-art.</strong> Sans cela, AIPROD reste une architecture sans produit.</td></tr>
  </tbody>
</table>

### 5.2 Quel concept est techniquement plus ambitieux ?

**→ AIPROD, de très loin**

<table>
  <thead>
    <tr>
      <th width="230">Critère</th>
      <th width="310">AIPROD</th>
      <th width="310">TapNow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Architectures IA originales</td>
      <td><strong>5</strong> (SHDT · HWVAE · NAC · TTS · LipSync)</td>
      <td><strong>0</strong></td>
    </tr>
    <tr>
      <td>Code de recherche ML</td>
      <td>~15 000+ lignes</td>
      <td>~0</td>
    </tr>
    <tr>
      <td>Innovation algorithmique</td>
      <td>GQA · dual-stream · wavelet VAE · RVQ multi-bande</td>
      <td>Intégration d'API existantes</td>
    </tr>
    <tr>
      <td>Complexité d'infrastructure</td>
      <td>GPU management · tensor parallelism · edge deployment</td>
      <td>Cloud functions + API routing</td>
    </tr>
    <tr>
      <td>Risque d'exécution</td>
      <td>Extrême (modèles à entraîner, qualité à prouver)</td>
      <td>Faible (modèles déjà prouvés par les providers)</td>
    </tr>
  </tbody>
</table>

### 5.3 Quel concept est plus viable économiquement ?

**→ TapNow, à court et moyen terme**

<table>
  <thead>
    <tr>
      <th width="220">Facteur</th>
      <th width="300">AIPROD</th>
      <th width="300">TapNow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Revenus actuels</td>
      <td>$0</td>
      <td>Actifs (1M+ users, plans $12.75–$360/mois)</td>
    </tr>
    <tr>
      <td>Time-to-revenue</td>
      <td>12–24 mois (optimiste)</td>
      <td>Immédiat</td>
    </tr>
    <tr>
      <td>Coût de développement</td>
      <td>Très élevé (GPU training, salaires PhD)</td>
      <td>Modéré (intégration, UX)</td>
    </tr>
    <tr>
      <td>Product-market fit</td>
      <td>Non prouvé</td>
      <td>Prouvé (clients enterprise : Honda, WPP, TikTok)</td>
    </tr>
    <tr>
      <td>Scalabilité économique</td>
      <td>Excellente à long terme (pas de coût API marginal)</td>
      <td>Comprimée (marge = prix client − coût API provider)</td>
    </tr>
    <tr>
      <td><strong>Verdict</strong></td>
      <td><strong>Viable à long terme si financement suffisant</strong></td>
      <td><strong>Viable immédiatement, mais marges sous pression</strong></td>
    </tr>
  </tbody>
</table>

### 5.4 Risques majeurs pour AIPROD

<table>
  <thead>
    <tr>
      <th width="30" align="center">#</th>
      <th width="260">Risque</th>
      <th width="100" align="center">Probabilité</th>
      <th width="80" align="center">Impact</th>
      <th width="320">Mitigation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">1</td>
      <td><strong>Modèles n'atteignent pas la qualité SotA</strong></td>
      <td align="center">Élevée</td>
      <td align="center">Critique</td>
      <td>Utiliser LTX-2/open-source comme base et fine-tuner</td>
    </tr>
    <tr>
      <td align="center">2</td>
      <td><strong>Coût d'entraînement prohibitif</strong></td>
      <td align="center">Élevée</td>
      <td align="center">Critique</td>
      <td>Stratégie LoRA + curriculum training (déjà en place)</td>
    </tr>
    <tr>
      <td align="center">3</td>
      <td><strong>Time-to-market trop long</strong></td>
      <td align="center">Élevée</td>
      <td align="center">Élevé</td>
      <td>Lancer un MVP avec LTX-2 base avant modèles propriétaires complets</td>
    </tr>
    <tr>
      <td align="center">4</td>
      <td><strong>Concurrence des modèles open-source</strong></td>
      <td align="center">Moyenne</td>
      <td align="center">Élevé</td>
      <td>Se différencier par la pipeline complète (pas juste le modèle)</td>
    </tr>
    <tr>
      <td align="center">5</td>
      <td><strong>Manque de financement</strong></td>
      <td align="center">Élevée</td>
      <td align="center">Critique</td>
      <td>Démontrer un proto fonctionnel pour lever des fonds</td>
    </tr>
    <tr>
      <td align="center">6</td>
      <td><strong>Sur-ingénierie</strong></td>
      <td align="center">Moyenne</td>
      <td align="center">Moyen</td>
      <td>Prioriser le chemin critique : prompt → vidéo fonctionnel</td>
    </tr>
    <tr>
      <td align="center">7</td>
      <td><strong>Pas d'interface utilisateur</strong></td>
      <td align="center">Élevée</td>
      <td align="center">Élevé</td>
      <td>API-first est viable pour B2B mais freine l'adoption B2C</td>
    </tr>
    <tr>
      <td align="center">8</td>
      <td><strong>Talent et équipe</strong></td>
      <td align="center">Incertaine</td>
      <td align="center">Critique</td>
      <td>Entraîner SHDT/HWVAE/NAC nécessite une équipe ML senior</td>
    </tr>
  </tbody>
</table>

---

## MATRICE DE COMPARAISON FINALE

<table>
  <thead>
    <tr>
      <th width="230">Critère</th>
      <th width="140" align="center">AIPROD</th>
      <th width="140" align="center">TapNow</th>
      <th width="150" align="center">Avantage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Innovation technique</td>
      <td align="center">★★★★★</td>
      <td align="center">★★☆☆☆</td>
      <td align="center"><strong>AIPROD</strong></td>
    </tr>
    <tr>
      <td>Souveraineté</td>
      <td align="center">★★★★★</td>
      <td align="center">★☆☆☆☆</td>
      <td align="center"><strong>AIPROD</strong></td>
    </tr>
    <tr>
      <td>Produit fonctionnel</td>
      <td align="center">★★☆☆☆</td>
      <td align="center">★★★★★</td>
      <td align="center"><strong>TapNow</strong></td>
    </tr>
    <tr>
      <td>Traction marché</td>
      <td align="center">★☆☆☆☆</td>
      <td align="center">★★★★★</td>
      <td align="center"><strong>TapNow</strong></td>
    </tr>
    <tr>
      <td>Défensabilité long terme</td>
      <td align="center">★★★★☆</td>
      <td align="center">★★☆☆☆</td>
      <td align="center"><strong>AIPROD</strong></td>
    </tr>
    <tr>
      <td>Viabilité économique CT</td>
      <td align="center">★★☆☆☆</td>
      <td align="center">★★★★★</td>
      <td align="center"><strong>TapNow</strong></td>
    </tr>
    <tr>
      <td>Viabilité économique LT</td>
      <td align="center">★★★★☆</td>
      <td align="center">★★★☆☆</td>
      <td align="center"><strong>AIPROD</strong></td>
    </tr>
    <tr>
      <td>Scalabilité technique</td>
      <td align="center">★★★★☆</td>
      <td align="center">★★★★☆</td>
      <td align="center">Égalité</td>
    </tr>
    <tr>
      <td>UX / Accessibilité</td>
      <td align="center">★☆☆☆☆</td>
      <td align="center">★★★★★</td>
      <td align="center"><strong>TapNow</strong></td>
    </tr>
    <tr>
      <td>Robustesse production</td>
      <td align="center">★★★☆☆</td>
      <td align="center">★★★★★</td>
      <td align="center"><strong>TapNow</strong></td>
    </tr>
    <tr>
      <td>Indépendance fournisseurs</td>
      <td align="center">★★★★★</td>
      <td align="center">★★☆☆☆</td>
      <td align="center"><strong>AIPROD</strong></td>
    </tr>
    <tr>
      <td>Profondeur R&D</td>
      <td align="center">★★★★★</td>
      <td align="center">★☆☆☆☆</td>
      <td align="center"><strong>AIPROD</strong></td>
    </tr>
  </tbody>
</table>

---

## VERDICT

**AIPROD et TapNow ne sont pas en compétition directe — ce sont deux stratégies fondamentalement différentes :**

- **TapNow** = plateforme d'agrégation (« la meilleure interface sur les modèles des autres »)
- **AIPROD** = moteur propriétaire (« possédez votre IA de bout en bout »)

**TapNow gagne aujourd'hui** : produit live, revenus, communauté, clients enterprise.

**AIPROD peut gagner demain** : si et seulement si les modèles propriétaires sont entraînés, atteignent la qualité production, et trouvent un marché (souveraineté, on-premise, B2B Europe).

**Le défi existentiel d'AIPROD** : transformer une architecture remarquable en un produit fonctionnel avant que le financement ne s'épuise et que les modèles open-source ne comblent l'écart.

**Le défi existentiel de TapNow** : survivre dans un marché d'agrégateurs où les providers (Google, ByteDance) peuvent lancer leurs propres interfaces, éliminant l'intermédiaire.

---

<sub><em>Audit réalisé le 20 février 2026. Basé sur l'exploration exhaustive du code source AIPROD et l'analyse publique de TapNow.ai. Les informations sur TapNow sont limitées aux éléments publiquement observables — l'architecture technique interne de TapNow n'est pas connue.</em></sub>
