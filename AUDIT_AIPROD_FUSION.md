# AUDIT TECHNIQUE COMPLET — AIPROD FUSION

**Date** : 15 février 2026  
**Auditeur** : Principal AI Infrastructure Auditor  
**Scope** : Projet complet AIPROD (monorepo `aiprod-core` + `aiprod-pipelines` + `aiprod-trainer`)  
**Méthode** : Lecture et analyse du code source réel uniquement — zéro supposition, zéro interprétation optimiste  

---

## 1. Nature réelle du système

### Architecture réellement implémentée

Le projet est un **monorepo à 3 packages** :

| Package | Lignes de code | Rôle réel |
|---------|---------------|-----------|
| `aiprod-core` | ~9 200 | Architectures neuronales (transformer, VAE, TTS, audio codec, lip-sync, mixer) |
| `aiprod-pipelines` | ~12 000+ | Pipelines de diffusion + API SaaS + inference graph + post-production |
| `aiprod-trainer` | ~8 500+ | Framework d'entraînement LoRA/fine-tuning |

### Modularité réelle ou monolithe déguisé ?

**Modularité réelle.** Les trois packages ont des responsabilités distinctes et des frontières claires :
- `aiprod-core` : zéro dépendance vers les deux autres
- `aiprod-pipelines` : dépend de `aiprod-core` uniquement
- `aiprod-trainer` : dépend de `aiprod-core` uniquement

Cependant, **deux architectures parallèles coexistent sans partage** :
- 5 pipelines de diffusion classiques (`ti2vid_one_stage`, `ti2vid_two_stages`, `distilled`, `keyframe_interpolation`, `ic_lora`)
- 1 système `inference/` graph-based avec DAG (300+ exports, ~5 000 lignes) qui duplique la même fonctionnalité

🟠 **Double architecture non consolidée = dette technique significative.**

### Orchestrateur réel ou simple enchaînement de scripts ?

**Deux orchestrateurs distincts et déconnectés** :
1. **API Orchestrator** (`api/orchestrator.py`) : Machine à 11 états avec boucle `while`, checkpoint/restore, retry. **Fonctionnel mais avec backends mockés.**
2. **Inference Graph** (`inference/graph.py`) : DAG avec tri topologique de Kahn, détection de cycles. **Fonctionnel mais déconnecté de l'API.**

🟠 Ces deux systèmes ne communiquent pas entre eux.

### Présence d'une vraie state machine ?

**Oui.** L'orchestrateur API implémente une state machine à 11 états (INIT → ANALYSIS → CREATIVE_DIRECTION → VISUAL_TRANSLATION → FINANCIAL_OPTIMIZATION → RENDER_EXECUTION → QA_TECHNICAL → QA_SEMANTIC → FINALIZE + ERROR + FAST_TRACK) avec transitions, checkpoint JSON, et retry policy.

### Couplage entre modules

**Faible entre packages** (bien). **Fort à l'intérieur de chaque package** (normal). Le problème est l'existence de deux architectures parallèles dans `aiprod-pipelines` qui ne se connaissent pas.

---

## 2. Vérification du moteur vidéo (LTX-2 intégré)

### LTX-2 réellement intégré ?

**🔴 NON. LTX-2 n'est pas "intégré" — le projet EST un fork renommé de LTX-Video 2.0.**

Preuves :
- `AIPRODModel` = alias de `SHDTModel` (dans `model/transformer/__init__.py`)
- `AIPRODModelConfigurator` = alias de `SHDTConfigurator` (dans `model/configurators.py`)
- Le template de model card référence encore `Lightricks/AIPROD` et `https://github.com/Lightricks/AIPROD`
- L'audit interne du trainer admet : *"It is NOT training a proprietary model from scratch — it fine-tunes an existing open-source diffusion model."*

### Fork modifié ou wrapper superficiel ?

**Fork substantiel.** Les architectures `SHDTModel` et `AIPRODv3Model` sont des implémentations complètes (~1 300 lignes de code transformer) avec :
- Grouped Query Attention (GQA) avec Flash Attention
- Spatial + Temporal attention factorées
- Cross-modal attention (vidéo ↔ texte ↔ audio)
- Adaptive RMS Norm avec modulation conditionnelle
- 3D positional encoding apprise

Le code est architecturalement réel, mais les poids entraînés n'existent pas.

### Seed global contrôlé ?

**Oui.** `seed_everything()` dans `utils.py` propage le seed via `random.seed()`, `torch.manual_seed()`, `torch.cuda.manual_seed_all()`. Les pipelines acceptent un paramètre `seed` propagé au `torch.Generator`.

### Reproductibilité possible ?

**En théorie oui, en pratique non.** Voir section 7.

### Temporal consistency réelle ?

**Oui.** Le transformer traite les dimensions spatiales et temporelles via `SpatialAttention` + `TemporalAttention` factorées. L'architecture `AIPRODv3Model` utilise `AxialAttention` avec factorisation spatiale/temporelle explicite.

### Latent reuse implémenté ?

**Oui.** `VideoConditionByLatentIndex` permet le remplacement de frames spécifiques dans l'espace latent. `VideoConditionByReferenceLatent` permet le blending d'un latent de référence complet.

### Identity locking réel ?

**Oui.** Pipeline IC-LoRA (`ic_lora.py`) avec reference video conditioning. LoRA weights séparés par stage. Facteur de downscale de référence extrait des métadonnées safetensors.

### Keyframe anchoring présent ?

**Oui.** `KeyframeInterpolationPipeline` avec `VideoConditionByKeyframeIndex` (soft blend sans zeroing du masque) et `image_conditionings_by_adding_guiding_latent`.

### Gestion VRAM ?

**Oui.** Composants implémentés :
- `GPUHealthMonitor` : `torch.cuda.mem_get_info()` + pynvml
- `OOMFallback` : chaîne de résolution (1080p → 720p → 512p)
- `cleanup_memory()` : `gc.collect()` + `torch.cuda.empty_cache()`
- Tiled VAE decoding avec blending de chevauchement

### Checkpointing ?

**Oui.** Checkpointing d'entraînement via Accelerate + safetensors. Checkpointing d'orchestration via JSON.

### Mixed precision ?

**Oui.** `bfloat16` partout, support FP8 via `optimum-quanto`, quantification INT2/INT4/INT8 bloc par bloc.

### Multi-GPU support ?

**Oui.** DDP et FSDP via HuggingFace Accelerate, configurations `accelerate/ddp.yaml` et `accelerate/fsdp.yaml` présentes.

### Appels simulés / fonctions stub / placeholders / incohérences GPU

| Élément | Statut |
|---------|--------|
| `LatentUpsampler` | 🟡 **Stub documenté** — bilinéaire 2× en attendant upsampler appris |
| `RenderExecutorAdapter._render_with_backend()` | 🔴 **Mock complet** — génère des URLs fictives `gs://aiprod-assets/...` avec `random.random()` pour simuler des échecs |
| `TechnicalQAGateAdapter` | 🟠 **Valide des dicts en mémoire**, pas des fichiers vidéo réels (ffprobe wrapper existe mais non connecté) |
| `SemanticQAGateAdapter` | 🟠 **Mock scoring** quand pas de vision LLM client |
| Supervisor block | 🔴 **Inexistant** — décrit dans le JSON V33, zéro code correspondant |
| Veo-3 API client | 🔴 **Inexistant** — le string "veo3" n'est qu'un label dans la logique de sélection de coût |
| Runway Gen-3 API client | 🔴 **Inexistant** — zéro code d'intégration |
| Replicate API client | 🔴 **Inexistant** — zéro code d'intégration |

---

## 3. Orchestration & Agents

### State machine réelle ?

**Oui.** 11 états, transitions définies, boucle d'exécution `while` avec checkpoint/restore.

### Gestion dépendances inter-agents ?

**Oui.** Les handlers reçoivent un `memory` dict partagé. Chaque handler lit ses inputs et écrit ses outputs dans `memory`. Le JSON V33 déclare un `memorySchema` avec champs requis/optionnels.

### Retry logic ?

**Partielle.** `retryPolicy` déclaré dans le config (maxRetries: 3, backoff: 15s). L'orchestrateur a une transition `ERROR → ANALYSIS` pour relance. Mais la logique de retry est **simpliste** — simple compteur sans backoff exponentiel, sans circuit breaker à ce niveau.

🟡 Le module `resilience/resilience.py` a un `CircuitBreaker` (CLOSED → OPEN → HALF_OPEN), mais il n'est **pas connecté** à l'orchestrateur API.

### Fallback strategy ?

**Oui pour les handlers** — chaque handler a un branch `else` avec données par défaut si l'adapter est absent. **Non pour les backends de rendu** — la chaîne `veo3 → runway_gen3 → replicate_wan25` n'est qu'un label dans le `FinancialOrchestratorAdapter`, aucun client API n'existe derrière.

### Idempotence ?

**Non.** Aucun mécanisme de déduplication de jobs. Relancer un job produit un nouveau run sans vérification de doublon.

🟡 Mineur — acceptable en phase initiale.

### Logging structuré ?

**Oui.** `StructuredLogger` avec formatage JSON, rédaction de champs sensibles, intégration `structlog` (fallback stdlib), context binding, `timed()` context manager. Production-ready.

### Traçabilité d'un job complet ?

**Oui.** `TracingManager` avec OpenTelemetry + export OTLP (Jaeger/Tempo). `pipeline_trace()` crée un span racine. Dégradation gracieuse (`_NoOpSpan` / `_NoOpTracer` si OTEL absent).

### Reprise après crash ?

**Partielle.** L'orchestrateur sauvegarde l'état dans un checkpoint JSON à chaque transition. `resume_job()` recharge l'état et reprend depuis le dernier état sauvegardé. **Mais** : le checkpoint ne capture que l'état de la state machine, pas l'état des tenseurs/modèles en mémoire GPU.

🟠 Suffisant pour les jobs SaaS, insuffisant pour les jobs GPU longs.

---

## 4. Pipeline Audio Propriétaire

### TTS interne réel ou wrapper externe ?

**Réel.** Système TTS complet en ~1 250 lignes :
- `TextEncoder` (transformer)
- `MelDecoder` (transformer + PostNet)
- `ProsodyModeler` (VariancePredictor + LengthRegulator + PitchPredictor + EnergyPredictor)
- `SpeakerEmbedding` (lookup + LSTM d-vector pour zero-shot cloning)
- `VocoderGenerator` (HiFi-GAN avec MPD 5-period)
- Text frontend complet : phonèmes IPA, nombres → mots, G2P avec 25+ règles

🟠 **Architectures définies, mais AUCUN poids entraîné disponible.** Les modèles s'instancient avec des poids aléatoires.

### Synchronisation voix / timeline ?

**Oui.** `LipSyncModel` avec AudioEncoder + BiLSTM + FacialDecoder (52 blend-shapes FLAME). Sync loss (MSE + LSE-D + LSE-C).

🟠 **Même problème : pas de poids entraînés.**

### Gestion des stems ?

**Oui.** `AudioMixer` avec :
- Equal-power pan law
- Biquad peaking EQ (coefficients corrects)
- Compresseur feed-forward avec envelope follower
- Réverbe algorithmique Schroeder-Moorer (4 comb + 2 allpass)
- Hard limiter
- `SpatialAudio` (stéréo ↔ 5.1 avec ITU-R BS.775, binaural ITD+ILD)

**Code mathématiquement correct et fonctionnel.**

### Mastering automatisé ?

**Partiellement.** Le mixer fait EQ + compression + limiting + reverb, ce qui constitue une chaîne de mastering basique. Pas de loudness normalization (LUFS).

### Alignement audio / vidéo vérifié ?

**Module `multimodal_coherence/`** avec scoring de cohérence audio/vidéo. Implémenté avec traitement du signal réel.

---

## 5. Pipeline Montage & Rendu

### Timeline réellement générée ?

**Oui.** `TimelineGenerator` avec :
- `PacingEngine` : durées de plans basées sur l'émotion
- Export CMX 3600 EDL
- Export FCPXML v1.11

### Stitching vidéo cohérent ?

**Oui.** `TransitionsLib` avec :
- Cross-fade (alpha blending tensor)
- Wipe (masque directionnel)
- Match-cut (micro-dissolve)

### Transitions automatisées ?

**Oui.** Sélection automatique basée sur le type de scène.

### Gestion multi-format ?

**Oui.** `ExportEngine` avec :
- Vidéo : H.264, H.265, ProRes 422/4444, DNxHR, VP9, AV1
- Audio : AAC, Opus, FLAC, PCM, Dolby
- Séquences d'images : EXR, DPX
- Via subprocess FFmpeg réel

### Export déterministe ?

**Non garanti.** L'encodage FFmpeg n'est pas déterministe par défaut (dépend du threading). Aucun flag `ffmpeg -threads 1` ou `-deterministic` n'est appliqué.

🟡 Mineur.

### Encodage optimisé GPU ?

**Non.** FFmpeg est appelé via subprocess CPU (`subprocess.Popen`). Aucun flag NVENC n'est utilisé.

🟡 Optimisation manquante.

---

## 6. Infrastructure & GPU Scaling

### Architecture Kubernetes-ready ?

**En théorie oui, en pratique non.** Les manifestes K8s existent et sont bien structurés :
- Namespace + ResourceQuota (64 CPU, 256Gi RAM, 20 GPUs)
- Gateway Deployment (2 replicas) + HPA (2-100 pods) + PDB
- GPU Worker avec nvidia-tesla-a100 node selector + VRAM liveness probe
- DCGM Exporter DaemonSet + ServiceMonitor
- 4 priority classes (system/enterprise/pro/free)

**Mais :**
- 🔴 Aucune image Docker n'a jamais été buildée (`gcr.io/aiprod/gateway:latest` n'existe pas)
- 🔴 Aucun cluster GKE n'est provisionné
- 🔴 `deploy/scripts/` est **vide** — aucun script de déploiement
- 🟠 Le PVC `model-cache` déclare `ReadOnlyMany` avec `standard-rwo` (incompatible)
- 🟠 Les deux Dockerfiles ont des entrypoints différents (`endpoints:app` vs `gateway:create_fastapi_app`)

### Gestion workers GPU ?

**Décrite dans K8s** (scaling 0-20 pods, nvidia-tesla-a100). **Jamais testée.**

### Queue manager ?

**Implémenté dans le code.** `multi_tenant_saas/` contient un scheduler batch avec sizing mémoire-aware, dispatch par timeout/batch-size. Backends in-memory (pas de Redis/RabbitMQ).

🟠 Queue in-memory = perte de jobs au restart.

### Priorité par budget ?

**Oui.** Priority classes K8s (system: 1M, enterprise: 100K, pro: 10K, free: 1K). Billing service avec plans par tier.

### Monitoring VRAM ?

**Oui dans le code.** `GPUHealthMonitor` (`torch.cuda.mem_get_info()` + pynvml). Prometheus metrics (17 métriques dont `gpu_utilization`, `vram_usage`, `gpu_temperature`).

### Limites mémoire ?

**Oui dans K8s** (limits: 48Gi RAM, `nvidia.com/gpu: 1`). **Oui dans le code** (`OOMFallback` avec chaîne de résolution).

### Backpressure ?

**Non.** Aucun mécanisme de backpressure implémenté. Le rate limiter API (sliding window) est le seul contrôle de flux.

🟠 Absence de backpressure = saturation possible.

### Saturation testée ?

**Non.** Aucun test de charge, aucun benchmark, aucun profiling GPU enregistré.

🔴 Aucune preuve de fonctionnement sous charge.

---

## 7. Reproductibilité & Déterminisme

### Seed unique propagé partout ?

**Partiellement.** `seed_everything()` couvre `random` + `torch` + `torch.cuda`. Les pipelines acceptent un seed. **Mais** : `numpy` n'est pas seedé. CUDA convolutions non-déterministes par défaut (`torch.backends.cudnn.deterministic` non forcé).

🟠 Reproductibilité approximative.

### Hash job reproductible ?

**Non.** Aucun hash de job incluant config + seed + versions de modèles. L'inference graph a un SHA-256 de config pour le cache de presets, mais pas de hash de job end-to-end.

🔴 Critique.

### Versioning modèles ?

**Oui.** `ModelRegistry` avec `register()`, `promote()`, `rollback()`, `compare_canary()`, quality gates (FID, CLIP-Score, latence). Backend JSON local + MLflow.

### Versioning weights ?

**En structure oui** (SHA-256 des artifacts dans le registry). **En pratique non** — les répertoires `models/pretrained/`, `models/checkpoints/`, `models/gemma-3/` sont **vides**.

🔴 Aucun poids de production disponible.

### Snapshot environnement ?

**Non.** Aucun fichier lock (`uv.lock`, `pip freeze`), aucun snapshot d'environnement reproductible.

🔴 Critique.

### Freeze des dépendances ?

**Non.** `requirements.txt` liste 40+ dépendances **sans aucune version pin**. Builds non-reproductibles.

🔴 Critique.

---

## 8. Viabilité économique réelle

### Estimation réaliste du coût GPU par vidéo 30s

**Aucune mesure réelle n'existe.** Extrapolation basée sur l'architecture :

| Composant | Estimation |
|-----------|-----------|
| Diffusion transformer (1.9B params, ~100 steps, bfloat16) | A100 80GB : ~2-5 min → $0.10-0.25 |
| VAE decode | ~10-30s → $0.01-0.05 |
| TTS (si modèle entraîné) | ~5-10s → $0.01-0.02 |
| Audio codec | ~2-5s → $0.005-0.01 |
| Upsampling stage 2 | ~1-3 min → $0.05-0.15 |
| **Total GPU (estimation)** | **$0.17-0.48 par vidéo 30s** |
| Stockage (S3/GCS) | ~$0.02/GB |
| CPU orchestration | Négligeable |
| **Worst-case avec retry 3x** | **$0.51-1.44** |

🟠 Le `FinancialOrchestrator` déclare `maxCostPerMinute: $1.20`. La réalité (avec retries, échecs, stockage) dépasse probablement ce plafond en régime dégradé.

### Sous-estimations identifiées

- 🔴 Coût GPU du TTS propriétaire non estimé (modèle pas entraîné)
- 🔴 Coût d'entraînement des modèles propriétaires non budgété
- 🟠 Coût du traffic réseau (transfert de vidéos entre services) absent
- 🟠 Coût Gemini API pour CreativeDirector/SemanticQA non inclus dans les estimations pipeline
- 🟡 Le `dynamicPricing` dans le config V33 référence `market_rate_api` — ne existe pas

### Absence de métriques / monitoring

- 🟠 17 métriques Prometheus définies mais jamais collectées (aucun cluster en production)
- 🟠 Aucun dashboard réel (Grafana/Datadog non déployé)
- 🟠 Zero data de production pour calibrer les estimations

---

## 9. Robustesse en cas d'échec

| Scénario | Comportement |
|----------|-------------|
| **GPU OOM** | `OOMFallback` : résolution downgrade (1080p → 720p → 512p). **Implémenté mais jamais testé sous charge réelle.** |
| **Crash diffusion** | Checkpoint d'état de l'orchestrateur (JSON). **Pas de checkpoint du tenseur latent en cours de débruitage.** Un crash mid-diffusion perd tout le travail du step courant. |
| **Timeout** | `DeadlineManager` avec exception `DeadlineExceeded` par stage. **Implémenté.** SLA dans le config : fast-track 300s, standard 900s, premium 1800s. |
| **Fichier corrompu** | `DataIntegrity` : vérification SHA-256 des artifacts. **Implémenté.** |
| **Audio échoue** | Pas de fallback audio spécifique. Si le TTS échoue, le pipeline ne produit pas de vidéo avec audio silencieux — il échoue complètement. |
| **Job interrompu** | Reprise via checkpoint JSON (état orchestrateur). **Pas de reprise des calculs GPU.** |

### Pipeline transactionnel ou non ?

**Non transactionnel.** Pas de commit/rollback atomique. Un échec en QA_SEMANTIC après un rendu réussi laisse des fichiers orphelins sans nettoyage garanti.

🟠 Risque de fuite de ressources (stockage, mémoire).

---

## 10. Failles critiques identifiées

### 🔴 Critique

| # | Faille | Impact |
|---|--------|--------|
| C1 | **Aucun backend de rendu vidéo n'est implémenté** — Veo-3, Runway, Replicate sont des labels sans code. Le `RenderExecutorAdapter` génère des URLs fictives. | Le système **ne peut pas produire de vidéo**. Illusion technique totale sur le composant central. |
| C2 | **Aucun poids de modèle de production** — `models/pretrained/`, `models/gemma-3/`, `models/checkpoints/` sont vides. | Les modèles propriétaires (TTS, lip-sync, audio mixer) s'instancient avec des poids aléatoires = sortie = bruit. |
| C3 | **Le projet est un fork renommé de LTX-Video 2.0** présenté comme propriétaire — aliases `AIPRODModel = SHDTModel`. | Risque juridique (licence MIT de Lightricks non respectée si rebranding commercial). Illusion de propriété intellectuelle. |
| C4 | **Dépendances non versionnées** — `requirements.txt` sans pins, aucun lockfile. | Builds non-reproductibles. Régression silencieuse possible à tout moment. |
| C5 | **Le Supervisor Agent décrit dans AIPROD_V33.json n'existe pas dans le code** — zéro ligne de code. | Incohérence config ↔ code. Le gate d'approbation final est absent. |
| C6 | **Aucune infrastructure déployée** — cluster GKE inexistant, images Docker jamais buildées, `deploy/scripts/` vide. | Le système n'a **jamais tourné** en dehors d'un environnement local de développement. |
| C7 | **Le `quickstart.py` référence des répertoires vides** (`models/aiprod2`, `models/gemma-3`) et un repo HuggingFace inexistant. | Point d'entrée démonstratif non fonctionnel. |

### 🟠 Majeur

| # | Faille | Impact |
|---|--------|--------|
| M1 | **Double architecture non consolidée** — 5 pipelines classiques + 1 inference graph DAG font la même chose sans partage de code. | ~5 000 lignes de dette technique. Maintenance double. |
| M2 | **Queue manager in-memory** — pas de Redis/RabbitMQ. | Perte de jobs au restart du service. |
| M3 | **Pipeline non transactionnel** — pas de commit/rollback, pas de nettoyage de fichiers orphelins. | Fuite de ressources en cas d'échec partiel. |
| M4 | **Handlers avec fallback mock** — chaque handler fonctionne "normalement" sans adapter réel en produisant des données fictives. | Bugs masqués en développement. Le pipeline "tourne" mais ne fait rien de réel. |
| M5 | **CircuitBreaker non connecté à l'orchestrateur API.** | Mécanisme de résilience implémenté mais inutilisé. |
| M6 | **`CurriculumScheduler` et `StreamingDatasetAdapter` non connectés au trainer.** | Code complet mais jamais appelé — dead code fonctionnel. |
| M7 | **PVC K8s `ReadOnlyMany` avec StorageClass `standard-rwo`** — incompatible. | Déploiement K8s échouerait au provisioning. |
| M8 | **Tests `test_aiprod_core_components.py` silencieusement skippés** via `try/except: pytest.skip()`. | Fausse impression de suite de tests verte. |
| M9 | **`pyproject.toml` omet `aiprod-trainer`** des sources UV workspace. | Build monorepo incomplet. |

### 🟡 Mineur

| # | Faille | Impact |
|---|--------|--------|
| m1 | `LatentUpsampler` est un bilinéaire 2× (placeholder documenté) au lieu d'un upsampler appris. | Qualité d'upsampling sous-optimale. |
| m2 | FFmpeg appelé en CPU (`subprocess.Popen`) sans NVENC. | Encodage vidéo plus lent que nécessaire. |
| m3 | Export vidéo non déterministe (FFmpeg threading). | Bitstream non reproductible. |
| m4 | `numpy` non seedé dans `seed_everything()`. | Reproductibilité incomplète. |
| m5 | Ruff `known-first-party` utilise `AIPROD_core` (majuscule) vs `aiprod_core` réel. | Tri des imports incorrect. |
| m6 | Dockerfile CPU inclut `pytest` en production. | Image de production inutilement lourde. |
| m7 | Pas de loudness normalization LUFS dans l'audio mixer. | Non conforme aux standards de diffusion (EBU R128). |

---

## 11. Top 7 corrections obligatoires avant production

### 1. Implémenter au minimum UN vrai backend de rendu vidéo
**Directive :** Créer un client API fonctionnel pour au moins un backend (LTX-2 local via les pipelines de diffusion existants dans `aiprod-pipelines`, OU un client Replicate/Runway). Connecter ce client au `RenderExecutorAdapter`. Supprimer les mock URLs.

**Effort estimé :** 2-3 jours pour connecter les pipelines locaux existants au render adapter.

### 2. Obtenir / entraîner les poids de modèle
**Directive :** Télécharger les poids LTX-2 depuis Lightricks (déjà dans `ltx2_research/` mais marqués "research only"), entraîner les LoRA propriétaires, entraîner le TTS et le lip-sync, ou utiliser des modèles pré-entraînés existants (Bark, Coqui TTS).

**Effort estimé :** Semaines à mois pour l'entraînement. Heures pour intégrer un TTS open-source existant.

### 3. Verrouiller TOUTES les dépendances
**Directive :** Générer un `uv.lock` ou `pip freeze > requirements.lock`. Ajouter des version pins dans `requirements.txt` (`torch>=2.5.0,<2.6`). Commit le lockfile.

**Effort estimé :** 1 heure.

### 4. Consolider les deux architectures (pipelines classiques vs inference graph)
**Directive :** Choisir UNE architecture (recommandation : inference graph DAG) et migrer les 5 pipelines classiques en tant que presets du graph. Supprimer le code dupliqué.

**Effort estimé :** 1-2 semaines.

### 5. Connecter les composants de résilience à l'orchestrateur
**Directive :** Wirer `CircuitBreaker`, `DeadlineManager`, `DriftDetector` dans la boucle de l'orchestrateur API. Intégrer `CurriculumScheduler` dans le trainer.

**Effort estimé :** 2-3 jours.

### 6. Implémenter un vrai système de queue persistant
**Directive :** Remplacer la queue in-memory par Redis (via `rq` ou `celery`) ou un broker de messages (RabbitMQ/Cloud Tasks). Assurer la persistance des jobs.

**Effort estimé :** 3-5 jours.

### 7. Buildée et tester AU MOINS une image Docker fonctionnelle
**Directive :** Unifier les deux Dockerfiles (ou en choisir un). Builder l'image GPU localement. Lancer un `docker run` qui exécute un job de bout en bout. Fixer les problèmes découverts.

**Effort estimé :** 2-3 jours.

---

## 12. Score final

| Dimension | Score /10 | Justification |
|-----------|-----------|---------------|
| **Solidité architecturale** | 7/10 | Architectures transformer/VAE/TTS réelles et bien codées. Double architecture non consolidée. Couplage adapter/handler bien pensé. |
| **Cohérence technique** | 4/10 | Déconnexion majeure entre config V33 (SaaS multi-backend) et code réel (backends mockés). Supervisor absent. Deux architectures parallèles. |
| **Reproductibilité** | 2/10 | Seeds gérés mais dépendances non versionnées, pas de lockfile, pas de hash job, pas de snapshot environnement, CUDA non-déterministe. |
| **Scalabilité GPU** | 5/10 | Code multi-GPU réel (DDP/FSDP/Accelerate). K8s bien structuré mais jamais déployé. Queue in-memory. Aucun test de charge. |
| **Viabilité économique** | 3/10 | Aucune donnée de production, aucune mesure réelle de coût, dynamicPricing référence des APIs inexistantes, coûts d'entraînement non budgétés. |

### Score global : 4.2 / 10

### Probabilité de survie 12 mois en production : < 10%

**Motifs :** Aucun backend de rendu fonctionnel, aucun poids de modèle de production, aucune infrastructure déployée, zéro donnée de production, builds non-reproductibles.

### Verdict

> **👉 Expérimental — tendance Illusion Technique**

Le projet contient **~30 000 lignes de code réel et bien écrit**. Les architectures neuronales (transformer, VAE, audio codec, TTS, lip-sync) sont des implémentations substantielles, pas des stubs. Le framework d'entraînement est production-grade.

**Mais le cœur du produit — la génération de vidéo — est un mock.** Le `RenderExecutorAdapter` génère des URLs fictives. Aucun des trois backends de rendu déclarés (Veo-3, Runway, Replicate) n'a de client implémenté. Les modèles audio (TTS, lip-sync, vocoder) n'ont pas de poids entraînés. L'infrastructure K8s est du boilerplate jamais déployé.

Le système est une **coquille architecturale impressionnante** avec des fondations solides, mais dont le composant central (production vidéo) est absent. Le passage de l'état actuel à la production nécessite au minimum les 7 corrections listées ci-dessus, ce qui représente plusieurs semaines à plusieurs mois de travail.

---

*Fin de l'audit — 15 février 2026*
