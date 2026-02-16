# AUDIT REPORT — AIPROD-PIPELINES

**Date :** 14 février 2026  
**Package :** `c:\Users\averr\AIPROD\packages\aiprod-pipelines`  
**Total fichiers Python :** 224  
**Total lignes de code Python :** ~63 876  
**Méthode :** Lecture intégrale de chaque fichier source

---

## RÉSUMÉ EXÉCUTIF

`aiprod-pipelines` est la couche d'orchestration et d'inférence du projet AIPROD. Il contient **deux systèmes distincts** :

1. **5 pipelines d'inférence réels** (~2 000 lignes) — code de production fonctionnel orchestrant le moteur `aiprod_core` (fork LTX-Video 2.0)
2. **Infrastructure d'inférence étendue** (~62 000 lignes) — modules SaaS, tensor parallelism, distributed LoRA, edge deployment, reward modeling, etc. — **structurellement complets mais majoritairement non connectés** au pipeline réel

**Verdict : Ce package est un wrapper/orchestrateur autour de modèles open-source existants, pas un moteur de génération propriétaire.**

---

## SECTION 1 : MODULES PIPELINE PRINCIPAUX (`src/aiprod_pipelines/`)

### 1.1 `__init__.py` (28 lignes)
- Re-exporte les 5 classes pipeline : `DistilledPipeline`, `ICLoraPipeline`, `KeyframeInterpolationPipeline`, `TI2VidOneStagePipeline`, `TI2VidTwoStagesPipeline`
- **Statut :** Code fonctionnel

### 1.2 `distilled.py` (178 lignes)
- Pipeline de génération vidéo distillée en deux étapes
- Stage 1 : génération basse résolution via diffusion Euler avec sigma pré-calculés
- Stage 2 : upsampling 2× et raffinement
- Encode le texte via AIPROD text encoder, les images via video VAE, boucle de débruitage, décodage VAE
- **Statut : Implémentation réelle fonctionnelle**
- **Imports externes :** `aiprod_core` (diffusion steps, noisers, video/audio VAE, AIPROD text encoder, upsampler, transformer)
- **Modèles hardcodés :** AIPROD text encoder (via `text_encoder_root`), `AIPRODV_LORA_COMFY_RENAMING_MAP` (ComfyUI)
- **Wraps des modèles existants : OUI** — orchestre les composants `aiprod_core` qui encapsulent LTX-Video 2.0

### 1.3 `ic_lora.py` (366 lignes)
- Pipeline deux étapes avec In-Context LoRA conditioning
- Supporte les signaux de conditionnement vidéo (depth, pose, edges)
- Stage 1 : transformer avec LoRA, Stage 2 : transformer sans LoRA
- Lit `reference_downscale_factor` depuis les métadonnées safetensors du LoRA
- **Statut : Implémentation réelle fonctionnelle**
- **Imports externes :** `safetensors` (lecture métadonnées LoRA), stack `aiprod_core`
- **Wraps des modèles existants : OUI**

### 1.4 `keyframe_interpolation.py` (271 lignes)
- Pipeline d'interpolation de keyframes en deux étapes
- Utilise `AIPROD2Scheduler` pour le scheduling sigma, `MultiModalGuider` pour guidance CFG + STG
- Stage 1 : demi-résolution, Stage 2 : upsampling avec LoRA distillé
- **Statut : Implémentation réelle fonctionnelle**
- **Wraps des modèles existants : OUI**

### 1.5 `ti2vid_one_stage.py` (185 lignes)
- Pipeline texte/image → vidéo en une étape
- Pass de débruitage complet à la résolution cible avec guidance CFG
- **Statut : Implémentation réelle fonctionnelle**

### 1.6 `ti2vid_two_stages.py` (263 lignes)
- Pipeline deux étapes avec guidance CFG en stage 1, upsampling LoRA distillé en stage 2
- **Statut : Implémentation réelle fonctionnelle**

---

## SECTION 2 : UTILITAIRES (`src/aiprod_pipelines/utils/`)

### 2.1 `args.py` (~300 lignes)
- Parsers CLI `argparse` pour tous les modes pipeline
- Définit `ImageAction`, `LoraAction`, `VideoConditioningAction`
- 3 variantes parser (1-stage, 2-stage, 2-stage-distilled)
- **Statut : Code fonctionnel**

### 2.2 `constants.py` (98 lignes)
- Valeurs par défaut : schedules sigma (`DISTILLED_SIGMA_VALUES`, `STAGE_2_DISTILLED_SIGMA_VALUES`), résolutions (512×768 → 1024×1536), paramètres guider (CFG=3.0/7.0, STG=1.0), prompt négatif, constantes architecture VAE
- **Valeurs hardcodées notables :** `AUDIO_SAMPLE_RATE = 24000`, `VIDEO_LATENT_CHANNELS = 128`
- **Statut : Code fonctionnel — hyperparamètres calibrés pour le modèle sous-jacent**

### 2.3 `helpers.py` (589 lignes)
- **Cœur du moteur d'inférence.** Implémente :
  - `euler_denoising_loop` — boucle de débruitage Euler standard
  - `gradient_estimating_euler_denoising_loop` — avec correction de vélocité
  - `denoise_audio_video` — débruitage conjoint audio-vidéo
  - `simple_denoising_func`, `guider_denoising_func`, `multi_modal_guider_denoising_func` — guidance CFG + STG + isolation modalité
  - Helpers conditionnement image, enhancement prompt, validation résolution
- **Statut : Implémentation réelle fonctionnelle — logique de guidance multi-modale sophistiquée**
- **Imports : Usage intensif de** `aiprod_core` (guiders, patchifiers, latent tools, transformers, perturbation system)

### 2.4 `media_io.py` (~320 lignes)
- I/O vidéo/audio/image via `av` (PyAV/FFmpeg)
- `encode_video` (H.264 + AAC muxing), `decode_video_from_file`, `decode_audio_from_file`
- Simulation artefacts compression CRF, redimensionnement aspect-ratio-preserving avec center crop
- **Statut : Code production — I/O média de bonne qualité**

### 2.5 `model_ledger.py` (~230 lignes)
- **Hub central de chargement des modèles.** Câble les `SingleGPUModelBuilder` pour : transformer, video encoder/decoder, audio decoder, vocoder, text encoder (AIPROD LLMBridge), spatial upsampler
- Chaque `build()` crée une instance fraîche depuis les poids checkpoint
- Support quantization FP8 pour le transformer
- `with_loras()` crée des variantes partageant le registre de poids
- **Références modèles hardcodées :**
  - `AIPRODV_MODEL_COMFY_RENAMING_MAP` — mappages clés state dict ComfyUI
  - `AIPRODModelConfigurator` — config architecture transformer
  - `VideoDecoderConfigurator`, `VideoEncoderConfigurator` — config VAE
  - `AudioDecoderConfigurator`, `VocoderConfigurator` — config modèles audio
  - `AIPRODTextEncoderModelConfigurator` — config text encoder **AIPROD LLMBridge**
  - `LatentUpsamplerConfigurator` — config upsampler
  - `AIPROD_TEXT_ENCODER_OPS` — opérations modèle AIPROD text encoder
- **Verdict : PREUVE PRINCIPALE** — toutes les architectures modèle viennent de `aiprod_core` (fork LTX-Video 2.0)

### 2.6 `types.py` (76 lignes)
- Définitions Protocol pour `DenoisingFunc`, `DenoisingLoopFunc`, et conteneur `PipelineComponents`
- **Statut : Code fonctionnel**

---

## SECTION 3 : COUCHE API (`src/aiprod_pipelines/api/`)

### 3.1 `orchestrator.py` (~250 lignes)
- Machine à états pipeline production 11 états avec checkpoint/resume :
  - INIT → ANALYSIS → CREATIVE_DIRECTION → VISUAL_TRANSLATION → FINANCIAL_OPTIMIZATION → RENDER_EXECUTION → QA_TECHNICAL → QA_SEMANTIC → FINALIZE (+ FAST_TRACK + ERROR)
- **Statut : Implémentation réelle de la logique d'orchestration. Mais AUCUNE inférence modèle directe — délègue aux objets adapter.**

### 3.2 `handlers.py` (~310 lignes)
- 11 fonctions handler async pour chaque état
- Chaque handler appelle un adapter ou tombe sur une logique stub (ex: `handle_financial_optimization` default backend = `"runway_gen3"` à $1/min)
- **Statut : MIXTE — structure d'orchestration réelle avec fallbacks stub**
- **Références hardcodées :** `"runway_gen3"`, `"replicate_wan25"` comme noms de backend, URLs mock `gs://aiprod-assets/`

### 3.3 Adapters (`api/adapters/`)

| Fichier | Lignes | Résumé | Statut |
|---------|--------|--------|--------|
| `base.py` | 91 | ABC abstrait avec validation de contexte | ABC réel |
| `creative.py` | 442 | Gemini + pipeline distillé pour direction créative, caching, génération scènes | **Implémentation partielle** — structure réelle mais dépend de `gemini_client` et `distilled_pipeline` injectés |
| `render.py` | 312 | Exécuteur de rendu avec retry + chaîne fallback | **Partiel** — logique retry/batch réelle mais `backends` injectés |
| `financial*.py` | ~300 chaque | Adapters estimation coûts | Implémentations partielles |
| `qa*.py` | ~300 chaque | Adapters validation QA | Implémentations partielles |
| `input_sanitizer.py` | ~200 | Validation/sanitization input | Implémentation partielle |

### 3.4 Autres sous-répertoires API

| Répertoire | Fichiers clé | Résumé |
|-----------|-------------|--------|
| `checkpoint/` | manager.py (202), recovery.py | **Implémentation réelle** — save/restore JSON sur disque |
| `schema/` | schemas.py (66), transformer.py, aiprod_schemas.py | **TypedDicts réels** pour contexte/requête/réponse pipeline |
| `integrations/` | gemini_client.py (362) | **Client API Gemini réel** avec rate limiting. Importe `google.generativeai`. Hardcode modèle `"gemini-1.5-pro"` |
| `optimization/` | performance.py | Utilitaires optimisation performance |

---

## SECTION 4 : SYSTÈME INFERENCE GRAPH (`src/aiprod_pipelines/inference/`)

### 4.1 Infrastructure Core

| Fichier | Lignes | Résumé | Statut |
|---------|--------|--------|--------|
| `__init__.py` | 673 | Re-export massif. Exporte 300+ symboles depuis ~20 sous-modules | Imports réels |
| `graph.py` | 374 | `GraphNode` ABC, `GraphContext` dataclass, `InferenceGraph` exécuteur DAG avec tri topologique Kahn, détection de cycles | **Implémentation réelle fonctionnelle** — moteur d'exécution DAG correct |
| `nodes.py` | 420 | `TextEncodeNode`, `DenoiseNode`, `UpsampleNode`, `DecodeVideoNode`, `AudioEncodeNode`, `CleanupNode` | 🔴 **MOCK/STUB** — toutes les méthodes `_encode_single`, `_denoise_step`, `_upsample`, `_decode_tiled` retournent `torch.randn(...)`. Commentaires : "Mock implementation" |
| `presets.py` | 1 719 | `PresetFactory` avec 5 standard + 5 adaptatifs + 5 quantizés preset builders, `PresetCache` LRU | **Logique construction graphe réelle** mais construit sur les nodes mock de nodes.py |

### 4.2 Sous-modules d'inférence (expansion massive)

| Sous-module | Lignes | Résumé | Statut |
|-------------|--------|--------|--------|
| `caching/` | ~845 | Cache d'inférence avec cache nodes | Implémentation standalone |
| `guidance/` | ~817 | Nodes guidance adaptive, analyseur prompt, prédicteur qualité, scaler timestep | **Algorithmes réels** pour ajustement dynamique guidance |
| `kernel_fusion/` | ~1 204 | Attention fusionnée, conv, groupnorm, détection capabilities GPU | **Implémentations structurelles** — définitions opérations réelles, CUDA kernel fusion mockée |
| `quality_metrics/` | ~1 288 | FVVR, LPIPS, optical flow/motion metrics, monitoring qualité | **Implémentations partielles** — formules métriques définies, calculs simplifiés |
| `prompt_understanding/` | ~1 864 | Tokenization prompt, reconnaissance entités, extraction concepts, construction graphe sémantique | **Implémentations NLP réelles** — extraction entités regex, construction graphe |
| `lora_tuning/` | ~1 389 | Implémentations couche LoRA (Linear, Conv2d), trainer, inférence, composition | **Implémentations PyTorch réelles** — `LoRALinear` avec vrai forward pass décomposition low-rank |
| `multimodal_coherence/` | ~2 432 | Analyse audio/vidéo, scoring cohérence, moteur sync, monitoring | **Structurel** — structures données réelles, algorithmes analyse simplifiés |
| `multi_tenant_saas/` | ~2 471 | Plateforme SaaS complète : tenant management, auth JWT, RBAC, billing, API gateway, rate limiting, job scheduling, feature flags, monitoring | **Implémentations complètes** — toutes les classes ont de la logique mais aucune intégration backend réelle |
| `tensor_parallelism/` | ~1 756 | Stratégies sharding, primitives communication, config distribuée, accumulation gradient, sharding modèle | **Structurel** — plans et configs réels, exécution distribuée non connectée |
| `distributed_lora/` | ~1 521 | Training LoRA distribué, federated learning, registre LoRA, fusion modèle | **Structurel** — dataclasses et squelette trainer, pas de training distribué réel |
| `tiling/` | ~1 044 | Tiling spatial/temporal/hybride, moteur tiling adaptatif, blending | **Implémentations algorithmes réelles** |
| `latent_distillation/` | ~719 | Techniques distillation latente | Structurel |
| `quantization/` | ~1 205 | Moteur quantization INT8/BF16/FP8, calibration | **Partiel** — config/métriques réels, ops quantization pas entièrement intégrées |
| `dynamic_batch_sizing/` | ~1 253 | Batch sizing adaptatif, profiling mémoire, estimation performance | Structurel |
| `edge_deployment/` | ~1 342 | Runtime mobile, moteur pruning, optimisation modèle edge | Structurel |
| `reward_modeling/` | ~513 | A/B testing, reward model | Structurel |
| `video_editing/` | ~1 034 | Analyse contenu, validation dataset, vérification qualité | Structurel |
| `validation/` | ~586 | API gateway, validation backend | Structurel |

---

## SECTION 5 : FICHIERS TOP-LEVEL

| Fichier | Lignes | Résumé | Statut |
|---------|--------|--------|--------|
| `pyproject.toml` | 12 | Métadonnées package. Dépendances : `aiprod-core`, `av`, `tqdm`, `pillow` | Config réelle |
| `validate_inference_graph.py` | 375 | Script validation testant imports inference graph, GraphContext, GraphNode, InferenceGraph, presets | Script test/validation |
| `validate_phase1.py` | ~110 | Charge et valide adapters PHASE 1 via `exec()` — validation hacky | Script test |
| `run_tests.py` | 25 | Configure sys.path et lance pytest sur test_foundation.py | Lanceur de tests |
| `UNIFIED_INFERENCE_GRAPH_GUIDE.md` | — | Guide documentant le système de graphe d'inférence | Documentation |
| `scripts/validate_production.py` | 631 | Validation déploiement Cloud Run avec health checks, load testing, connectivité GCP | **Outillage opérationnel réel** mais cible `https://aiprod-merger-__PROJECT_ID__.run.app` (placeholder template) |

---

## SECTION 6 : TESTS (`tests/`)

### Tests top-level

| Fichier | Lignes | Résumé |
|---------|--------|--------|
| `test_foundation.py` | 1 084 | Tests checkpoint, schema, orchestrateur avec adapters mockés. Mock `torch` entièrement |
| `test_phase1.py` | 637 | Tests adapters PHASE 1 avec mocking lourd. Mock `torch`, `diffusers`, `transformers` |
| `test_phase2.py` | ~400 | Tests optimisation financière, sélection backend |
| `test_phase4.py` | ~350 | Tests client Gemini, intégration |
| `test_e2e_integration.py` | 492 | Tests intégration pipeline complet avec tous adapters |
| `test_integration_matrix.py` | 630 | 13 transitions d'état × 8 scénarios d'échec |

### Tests inference (`tests/inference/`)

| Répertoire | Fichiers | Évaluation |
|-----------|---------|-----------|
| racine | test_graph.py (383), test_nodes.py (298), test_integration.py (294), test_presets.py (324), conftest.py (99) | **Réels** — tests infrastructure graphe avec assertions concrètes |
| `analytics/` | test_analytics.py | Réel |
| `caching/` | conftest.py, test_caching.py, test_caching_node.py, test_preset_cache.py | Réel |
| `guidance/` | conftest.py + 4 fichiers test | Réel |
| `kernel_fusion/` | 4 fichiers test (437 lignes pour opérations seules) | **Réel** — tests correction numérique détaillés |
| `latent_distillation/` | conftest.py + 2 fichiers test | Réel |
| `quantization/` | conftest.py + 2 fichiers test (421 lignes) | **Réel** — validation config, assertions méthodes |
| `reward_modeling/` | test_reward_model.py (265 lignes) | **Réel** — RewardNet, UserFeedback, ABTestingFramework |
| `tiling/` | 4 fichiers test | Réel |
| `validation/` | test_validation_system.py | Réel |
| `video_editing/` | test_editor.py | Réel |

**Verdict : aiprod-pipelines possède la suite de tests la plus étendue du projet** (~5 000+ lignes, 30+ fichiers). Les tests importent depuis `aiprod_pipelines.api.orchestrator`, `.adapters`, `.schema`, `.inference` avec patterns mock réels, assertions concrètes, tests async, et workarounds import complexes.

---

## SECTION 7 : DÉPENDANCES MODÈLES EXTERNES

| Modèle/Bibliothèque | Où référencé | Utilisation |
|---------------------|-------------|-------------|
| **AIPROD Text Encoder** (LLMBridge) | `model_ledger.py`, 5 pipelines | Encodage texte via `aiprod_core.model.text_encoder` |
| **Transformer diffusion** (pattern AIPRODV/LTX-V) | `model_ledger.py` via `AIPRODModelConfigurator` | Modèle débruitage vidéo — chargé depuis checkpoint utilisateur |
| **Video VAE** (encoder + decoder) | `model_ledger.py` | Encodage/décodage espace latent |
| **Audio VAE + Vocoder** | `model_ledger.py` | Génération/décodage audio |
| **Spatial Upsampler** | `model_ledger.py` via `LatentUpsamplerConfigurator` | Upsampling 2× espace latent |
| **Google Gemini 1.5 Pro** | `api/integrations/gemini_client.py` | Génération texte direction créative via API Google |
| **Runway Gen-3** | `api/handlers.py`, adapters | Backend vidéo fallback (nom string, pas de SDK) |
| **Replicate WAN-2.5** | `api/handlers.py`, adapters | Backend vidéo fallback (nom string, pas de SDK) |
| **PyAV/FFmpeg** | `media_io.py` | Encodage/décodage vidéo/audio |
| **ComfyUI** | Maps renommage clés partout | Compatibilité state dict via constantes `COMFY_RENAMING_MAP` |

---

## SECTION 8 : NOMS/CHEMINS MODÈLES HARDCODÉS

- `"gemini-1.5-pro"` — dans `gemini_client.py`
- `"runway_gen3"` — backend défaut dans handlers.py et adapters
- `"replicate_wan25"` — backend fallback dans render adapter
- `"gs://aiprod-assets/"`, `"gs://aiprod-merger-assets"` — templates buckets GCS
- Maps clés ComfyUI (`AIPRODV_MODEL_COMFY_RENAMING_MAP`, `AIPRODV_LORA_COMFY_RENAMING_MAP`) — définis dans `aiprod_core`

---

## SECTION 9 : ÉVALUATION RÉEL vs STUB

| Couche | Implémentation réelle | Stub/Mock |
|--------|----------------------|-----------|
| **5 classes pipeline** (distilled, ic_lora, keyframe, t2v_1stage, t2v_2stage) | **100% réel** — orchestration qualité production | — |
| **Utils** (helpers, media_io, model_ledger, args, constants, types) | **100% réel** — moteur d'inférence core | — |
| **API orchestrateur + state machine** | **~80% réel** — machine à états fonctionnelle avec checkpoint/resume | Chemins fallback utilisent données mock |
| **API adapters** | **~60% réel** — structure et logique présentes | Dépendent de dépendances injectées pouvant ne pas exister |
| **Inference graph (graph.py)** | **100% réel** — exécuteur DAG correct | — |
| **Inference nodes (nodes.py)** | Structure seule | **100% mock** — tout calcul retourne `torch.randn()` |
| **Inference presets** | Construction graphe réelle | S'appuie sur les nodes mock |
| **20+ sous-modules inference** (multi_tenant_saas, tensor_parallelism, distributed_lora, lora_tuning, etc.) | Structures données et algorithmes partiellement réels | **Aucune intégration avec l'inférence modèle réelle** — modules standalone |

---

## SECTION 10 : VERDICT WRAPPER vs PROPRIÉTAIRE

**Ce package est une couche wrapper/orchestration autour de modèles open-source existants.**

1. **Toutes les architectures modèle** sont définies dans le package frère `aiprod-core`, pas ici. Le package pipelines orchestre uniquement le chargement et l'exécution.

2. L'architecture modèle sous-jacente (transformer + video VAE + AIPROD text encoder) suit le pattern **LTX-Video 2.0** (Lightricks) — attesté par les maps de compatibilité ComfyUI, le nommage `AIPRODV`, et la structure architecturale.

3. Le text encoder est **AIPROD LLMBridge** — un encodeur propriétaire.

4. La couche API référence **Gemini 1.5 Pro** (API propriétaire Google), **Runway Gen-3**, et **Replicate** comme services externes.

5. **Aucune architecture modèle propriétaire n'est implémentée** dans ce package. Tout le code original est de l'orchestration, la gestion pipeline, et l'infrastructure (machine à états, checkpointing, outillage SaaS).

6. Environ **~2 000 lignes** sont du vrai code pipeline production (5 pipelines + utils). Les **~62 000 lignes restantes** sont des modules infrastructure (système graphe d'inférence, plateforme SaaS, framework training distribué, etc.) structurellement complets mais majoritairement non connectés à l'inférence modèle réelle.

---

## SECTION 11 : FAILLES CRITIQUES

### 🔴 Critique

1. **Inference nodes entièrement mockées.** `nodes.py` — censé être le cœur de l'exécution du graphe d'inférence — retourne `torch.randn()` pour TOUTES les opérations. Les 1 719 lignes de presets construisent des graphes sur des nodes factices.

2. **~62 000 lignes d'infrastructure non connectée.** Les modules SaaS multi-tenant, tensor parallelism, distributed LoRA, edge deployment, reward modeling sont des structures sans backend. Code volumétrique mais non fonctionnel.

3. **Dépendance totale aux modèles LTX-Video 2.0 via aiprod_core.** Aucune architecture propre au package pipelines. Si LTX-Video change de licence ou d'API, tout le code d'orchestration est invalidé.

4. **Client Gemini hardcodé.** Direction créative entièrement dépendante de l'API Google Gemini — pas d'alternative locale.

### 🟠 Majeur

5. **Fallbacks de rendu sur APIs tierces.** Le handler de rendu fallback sur Runway Gen3 et Replicate WAN-2.5 comme strings — pas de SDK intégré, pas de gestion d'erreur API.

6. **Pas de batching inference.** Chaque requête est traitée séquentiellement — impact direct throughput SaaS.

7. **Templates placeholders non résolus.** `validate_production.py` cible `https://aiprod-merger-__PROJECT_ID__.run.app` — jamais remplacé.

8. **Tests mockent torch entièrement.** Les tests de foundation et phase1 patchent `torch`, `diffusers`, `transformers` — empêchant la validation GPU réelle.

### 🟡 Mineur

9. **Format export unique.** H.264 + AAC seulement via `media_io.py`. Pas de ProRes, DNxHR, ou formats professionnels.

10. **Validation hacky.** `validate_phase1.py` utilise `exec()` pour charger et valider les adapters — pattern fragile et non sécurisé.

11. **Pas de monitoring intégré.** Les modules monitoring du SaaS existent mais ne sont connectés à aucun backend (Prometheus, Grafana, etc.).

---

## SECTION 12 : SCORES

| Critère | Score | Justification |
|---------|-------|---------------|
| Qualité code pipeline (5 pipelines + utils) | **7/10** | Code d'orchestration propre, bien structuré, fonctionnel. Hérité de/compatible avec LTX-Video. |
| Qualité infrastructure étendue | **2/10** | Volume massif (~62K lignes) mais non connecté. Nodes mockées. Pas d'intégration backend. |
| Couverture tests | **5/10** | Suite de tests étendue (30+ fichiers) mais mock torch entièrement — aucune validation GPU. |
| Valeur originale vs wrapper | **3/10** | Orchestration et state machine sont originaux. Tout le reste dépend de modèles/APIs tiers. |

---

*Fin du rapport d'audit — 14 février 2026*
