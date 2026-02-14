# PLAN MASTER — AIPROD 10/10

**Date :** 14 février 2026  
**Objectif :** Transformer AIPROD en un système IA propriétaire de génération vidéo cinématographique end-to-end, cohérent, 100% fonctionnel, 100% opérationnel, défendable technologiquement et économiquement viable.  
**Basé sur :** Audit Architectural + 3 Audit Reports (aiprod-core, aiprod-pipelines, aiprod-trainer)  
**Score actuel :** Modèle 2/10, Infrastructure 3/10, Viabilité 2/10  
**Score cible :** 10/10 sur chaque axe

---

## TABLE DES MATIÈRES

1. [Diagnostic synthétique — État des lieux](#1-diagnostic-synthétique)
2. [Vision architecturale cible](#2-vision-architecturale-cible)
3. [Phase 0 — Assainissement juridique & technique (Semaines 1-2)](#3-phase-0--assainissement)
4. [Phase 1 — Fondations propriétaires (Semaines 3-12)](#4-phase-1--fondations-propriétaires)
5. [Phase 2 — Pipeline cinématographique complet (Semaines 13-24)](#5-phase-2--pipeline-cinématographique-complet)
6. [Phase 3 — Infrastructure production (Semaines 25-36)](#6-phase-3--infrastructure-production)
7. [Phase 4 — SaaS & scalabilité (Semaines 37-48)](#7-phase-4--saas--scalabilité)
8. [Phase 5 — Excellence & différenciation (Semaines 49-72)](#8-phase-5--excellence--différenciation)
9. [Matrice de traçabilité — Chaque faille → sa correction](#9-matrice-de-traçabilité)
10. [Budget & ressources](#10-budget--ressources)
11. [KPIs de validation par phase](#11-kpis-de-validation)
12. [Score cible détaillé 10/10](#12-score-cible-détaillé)

---

## 1. Diagnostic synthétique

### Failles critiques à résoudre (17 identifiées)

| # | Gravité | Faille | Source audit |
|---|---------|--------|-------------|
| F1 | 🔴 | Modèle fondamental = fork LTX-Video 2.0 renommé, pas propriétaire | Architectural §1, Core §Critical Finding |
| F2 | 🔴 | Aucune capacité d'entraînement from scratch | Architectural §2, Trainer §Executive Summary |
| F3 | 🔴 | SaaS déployé sans GPU (Cloud Run CPU-only) | Architectural §8 |
| F4 | 🔴 | ~62K lignes infrastructure non connectée (nodes mockées `torch.randn()`) | Pipelines §4.1, §9 |
| F5 | 🔴 | Risque juridique IP — licences Apache 2.0 supprimées | Architectural §10-1 |
| F6 | 🔴 | Aucun TTS, lip-sync, montage, étalonnage, HDR | Architectural §6, §7 |
| F7 | 🟠 | Dualité architecturale non résolue (local GPU vs SaaS API) | Architectural §10-7 |
| F8 | 🟠 | Zéro trace d'exécution réelle (logs vides, 0 run WandB) | Architectural §10-8 |
| F9 | 🟠 | Tests unitaires quasi inexistants pour aiprod-core | Core §Key Observations 1 |
| F10 | 🟠 | Inference nodes retournent `torch.randn()` | Pipelines §4.1 |
| F11 | 🟠 | Monitoring et observabilité absents | Architectural §9 |
| F12 | 🟠 | Pas de versioning modèle (MLflow, registry) | Architectural §9 |
| F13 | 🟡 | Monkey-patching `torch._dynamo` dans curriculum.py | Core §Key Observations 6 |
| F14 | 🟡 | Répertoires scaffolding vides | Architectural §10-15 |
| F15 | 🟡 | Pas de batching inference | Architectural §3 |
| F16 | 🟡 | Format export unique (H.264 seulement) | Architectural §7 |
| F17 | 🟡 | Prototype toy model = dead code (1 870 lignes) | Core §Prototype |

### Actifs réutilisables

| Actif | Lignes | Qualité | Réutiliser ? |
|-------|--------|---------|-------------|
| 5 pipelines d'inférence (distilled, ic_lora, keyframe, t2v_1stage, t2v_2stage) | ~2 000 | Haute | ✅ Oui — adapter au modèle propriétaire |
| Utils inférence (helpers, media_io, model_ledger, constants, types) | ~1 600 | Haute | ✅ Oui — cœur du moteur d'exécution |
| Orchestrateur state machine 11 états | ~560 | Haute | ✅ Oui — garder et connecter |
| Checkpoint/recovery manager | ~400 | Bonne | ✅ Oui |
| Trainer LoRA complet | ~5 000 | Haute | ✅ Oui — étendre pour training from scratch |
| Streaming data pipeline (cache, prefetcher, adapter) | ~1 500 | Haute | ✅ Oui |
| Scripts preprocessing (process_videos, captions, split_scenes) | ~3 900 | Haute | ✅ Oui |
| Scheduler flow-matching, guiders (CFG, STG, APG) | ~500 | Haute | ✅ Oui — réutilisable avec tout modèle diffusion |
| Tiled VAE decoding | ~370 | Haute | ✅ Oui |
| Graph inference engine (graph.py) | ~374 | Haute | ✅ Oui — connecter aux vrais nodes |
| Architecture transformer (BasicAVTransformerBlock) | ~7 500 | Haute (fork) | ⚠️ À légaliser puis étendre |

---

## 2. Vision architecturale cible

### Architecture 10/10

```
┌──────────────────────────────────────────────────────────────────┐
│                      AIPROD PLATFORM                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │ PROMPT       │→│ SCÉNARISTE   │→│ DIRECTEUR CRÉATIF       │  │
│  │ Utilisateur  │  │ LLM interne  │  │ Découpage scènes,      │  │
│  │              │  │ (fine-tuné)  │  │ caméra, timing, mood    │  │
│  └─────────────┘  └──────────────┘  └────────────────────────┘  │
│         │                                      │                 │
│         ▼                                      ▼                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              MOTEUR DE GÉNÉRATION PROPRIÉTAIRE              │ │
│  │                                                             │ │
│  │  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌──────────┐  │ │
│  │  │VIDEO GEN │  │ AUDIO GEN │  │ TTS/VOIX │  │ MUSIQUE  │  │ │
│  │  │Diffusion │  │Audio VAE  │  │Propriét. │  │Propriét. │  │ │
│  │  │Transform.│  │+ Vocoder  │  │          │  │          │  │ │
│  │  └──────────┘  └───────────┘  └──────────┘  └──────────┘  │ │
│  │         │              │             │            │         │ │
│  │         ▼              ▼             ▼            ▼         │ │
│  │  ┌─────────────────────────────────────────────────────┐   │ │
│  │  │           SYNCHRONISATION CROSS-MODALE              │   │ │
│  │  │     lip-sync • timing audio/vidéo • cohérence       │   │ │
│  │  └─────────────────────────────────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────┘ │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              POST-PRODUCTION AUTOMATISÉE                    │ │
│  │                                                             │ │
│  │  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌──────────┐  │ │
│  │  │ MONTAGE  │  │ÉTALONNAGE │  │ MIXAGE   │  │  EXPORT  │  │ │
│  │  │Timeline  │  │Color grade│  │Audio mix │  │Multi-fmt │  │ │
│  │  │Cuts,trans│  │LUT, HDR   │  │5.1/stereo│  │ProRes,H26│  │ │
│  │  └──────────┘  └───────────┘  └──────────┘  └──────────┘  │ │
│  └─────────────────────────────────────────────────────────────┘ │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              CONTRÔLE QUALITÉ AUTOMATISÉ                    │ │
│  │  QA technique • QA sémantique • A/B test • Reward model     │ │
│  └─────────────────────────────────────────────────────────────┘ │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              INFRASTRUCTURE SaaS                            │ │
│  │  API Gateway • Auth • Billing • GPU Cluster • Monitoring    │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Principes architecturaux

1. **Tout modèle utilisé en production est soit propriétaire, soit légalement attribué et licencié**
2. **Chaque module est fonctionnel, testé, connecté, et mesurable**
3. **Zéro code mock en production — zéro `torch.randn()` hors tests**
4. **Pipeline end-to-end exécutable : prompt → vidéo finale exportée**
5. **Infrastructure GPU réelle avec autoscaling**
6. **Monitoring exhaustif : GPU, latence, qualité, coûts, drift**
7. **Versioning complet : code (git), modèles (registry), data (DVC)**

---

## 3. Phase 0 — Assainissement (Semaines 1-2)

**Objectif : Nettoyer la dette, sécuriser le juridique, établir les fondations saines.**

### 0.1 Conformité juridique LTX-Video 2.0 [F1, F5]

| Action | Livrable | Critère de validation |
|--------|----------|----------------------|
| Restaurer les notices Apache 2.0 dans tous les fichiers issus de LTX-Video | Headers de licence dans chaque fichier `.py` de `aiprod_core/` | `grep -r "Apache" src/aiprod_core/` retourne un résultat par fichier |
| Créer un fichier `NOTICE` à la racine | `NOTICE` listant Lightricks/LTX-Video 2.0, PixArt-Alpha, Google Gemma 3 | Fichier présent et complet |
| Créer un fichier `THIRD_PARTY_LICENSES.md` | Licences de chaque dépendance critique | Validé par revue juridique |
| Documenter explicitement les modifications apportées au fork | `MODIFICATIONS.md` dans `aiprod-core/` | Diff documenté entre LTX-Video original et AIPROD |
| Décider de la stratégie IP : fork attribué → modèle propriétaire progressif | Document stratégique interne | Approuvé par direction |

### 0.2 Nettoyage dead code [F13, F14, F17]

| Action | Livrable | Impact |
|--------|----------|--------|
| Supprimer ou archiver le prototype toy model (`src/models/`, `src/training/`, `src/data/`) | 1 870 lignes supprimées de `aiprod-core` | Zéro confusion entre code prod et prototype |
| Supprimer le monkey-patching `torch._dynamo` dans `curriculum.py` | Code nettoyé | Stabilité améliorée |
| Peupler ou supprimer les répertoires scaffolding vides | Soit implémentation, soit suppression | Arborescence honnête |
| Remplacer `config/templates/pyproject.template.toml` dupliqué | Vrai template avec variables Jinja2 ou suppression | Config propre |

### 0.3 Première exécution réelle [F8]

| Action | Livrable | Critère de validation |
|--------|----------|----------------------|
| Télécharger les poids LTX-Video 2.0 officiels dans `models/aiprod2/` | Fichiers `.safetensors` (~19 GB) | `ls models/aiprod2/*.safetensors` |
| Télécharger les poids Gemma 3 dans `models/gemma-3/` | Poids + tokenizer | `ls models/gemma-3/` |
| Exécuter `examples/quickstart.py` de bout en bout | Vidéo MP4 générée | Fichier `.mp4` lisible et cohérent |
| Logger les métriques (latence, VRAM, qualité) | Première entrée dans `logs/` | Fichiers de log non vides |
| Créer un run WandB documenté | Dashboard WandB avec métriques | URL du run accessible |

---

## 4. Phase 1 — Fondations propriétaires (Semaines 3-12)

**Objectif : Construire les bases d'un modèle et d'une infrastructure réellement propriétaires.**

### 1.1 Dataset propriétaire [F2]

| Action | Détail | Livrable |
|--------|--------|----------|
| Définir la politique de données | Types de vidéos, licences acceptables, résolutions, durées, langues audio | `docs/DATA_GOVERNANCE.md` |
| Constituer un dataset initial sous licence | 10 000-50 000 clips vidéo avec audio, CC-BY ou licences commerciales | `datasets/v1/` avec métadonnées |
| Pipeline d'ingestion automatisé | Download → validation → scene split → caption → embedding → latent | Script exécutable end-to-end |
| Audit qualité dataset | Distribution des durées, résolutions, catégories, langues | Rapport avec histogrammes |
| Data versioning | DVC ou équivalent pour tracer les versions du dataset | `.dvc` fichiers trackés |

**Volume cible minimum :** 10 000 heures de vidéo sous licence pour fine-tuning avancé, 100 000+ heures pour training from scratch.

### 1.2 Training from scratch — Phase préparatoire [F2]

| Action | Détail | Livrable |
|--------|--------|----------|
| Étendre `aiprod-trainer` pour supporter le training complet (pas seulement LoRA) | Mode `full_finetune` + mode `pretrain_from_scratch` | Config YAML `pretrain_full.yaml` |
| Implémenter le training du Video VAE | Loss reconstruction + KL divergence + perceptual loss (LPIPS) | Module `vae_trainer.py` |
| Implémenter le training de l'Audio VAE + Vocoder | Loss spectrale + adversarial (discriminateur) | Module `audio_trainer.py` |
| Implémenter le training du transformer diffusion | Flow matching loss + multi-GPU FSDP | Extension de `trainer.py` |
| Curriculum training multi-phase | Phase 1: basse résolution → Phase 2: haute résolution → Phase 3: longue durée | Config curriculum documentée |
| Estimer précisément le budget compute | Benchmark sur 1% du dataset, extrapoler | Tableau avec coûts A100/H100 |

**Budget estimé :**

| Composant | GPU-heures | Coût ($2/h A100) |
|-----------|-----------|------------------|
| Video VAE pré-training | 200-500h | $400-1 000 |
| Audio VAE + Vocoder | 100-300h | $200-600 |
| Transformer 1.9B pré-training (100K h vidéo) | 2 000-8 000h | $4 000-16 000 |
| Fine-tuning spécialisé LoRA | 50-200h | $100-400 |
| **Total Phase 1** | **2 350-9 000h** | **$4 700-18 000** |

> **Note :** Ces estimations supposent l'utilisation de spots instances et d'optimisations (gradient checkpointing, mixed precision). Un training from scratch compétitif avec Sora/Kling nécessiterait 10-100× plus.

### 1.3 Architecture modèle propriétaire [F1]

Stratégie en deux temps :

**Court terme (Semaines 3-12) — "Fork légitime augmenté" :**

| Action | Détail |
|--------|--------|
| Respecter la licence Apache 2.0 (Phase 0) | Attribution complète |
| Développer des extensions architecturales originales | Nouvelles couches, mécanismes d'attention, conditionnement |
| Documenter chaque modification vs LTX-Video original | `MODIFICATIONS.md` mis à jour |
| Publier les modifications conformément à Apache 2.0 | Transparence |

Extensions architecturales originales à développer :

| Extension | Description | Innovation |
|-----------|-------------|------------|
| `SceneConsistencyModule` | Mémoire inter-scènes pour cohérence narrative | Attention cross-scène avec banque de features |
| `CameraControlConditioning` | Contrôle caméra paramétrique (pan, tilt, zoom, dolly) | ControlNet caméra intégré au transformer |
| `EmotionConditioningLayer` | Conditionnement émotionnel des scènes | Embedding émotion → AdaLN |
| `TemporalSuperResolution` | Interpolation temporelle apprise | Module entre les blocs transformer |
| `AdaptiveComputeBlock` | Allocation compute dynamique par complexité de scène | Early exit + routing |

**Long terme (post-Phase 5) — "Modèle AIPROD v3 from scratch" :**

| Action | Détail |
|--------|--------|
| Architecture novel inspirée mais non dérivée | Nouveau design basé sur les learnings |
| Training sur dataset propriétaire massif | 100K+ heures |
| Benchmark vs state-of-the-art | Métriques FVD, CLIP-Score, qualité humaine |

### 1.4 Tests unitaires [F9]

| Package | Tests à écrire | Couverture cible |
|---------|---------------|-----------------|
| `aiprod-core` — components | `test_schedulers.py`, `test_guiders.py`, `test_patchifiers.py`, `test_diffusion_steps.py` | 90%+ |
| `aiprod-core` — model/transformer | `test_transformer_block.py`, `test_attention.py`, `test_rope.py`, `test_adaln.py` | 85%+ |
| `aiprod-core` — model/video_vae | `test_video_vae.py`, `test_tiling.py`, `test_convolutions.py` | 85%+ |
| `aiprod-core` — model/audio_vae | `test_audio_vae.py`, `test_vocoder.py`, `test_audio_ops.py` | 85%+ |
| `aiprod-core` — loader | `test_registry.py`, `test_sd_ops.py`, `test_builder.py` | 90%+ |
| `aiprod-core` — text_encoders | `test_gemma_encoder.py`, `test_tokenizer.py`, `test_connector.py` | 85%+ |
| `aiprod-core` — conditioning | `test_keyframe_cond.py`, `test_latent_cond.py`, `test_reference_cond.py` | 90%+ |
| `aiprod-trainer` | Compléter tests hors streaming | 80%+ |
| `aiprod-pipelines` — pipelines | Tests GPU réels (pas torch mocké) | 75%+ |

**Infrastructure de test :**

| Outil | Usage |
|-------|-------|
| `pytest` + `pytest-cov` | Exécution + couverture |
| `pytest-benchmark` | Benchmarks performance |
| `pytest-gpu` (custom marker) | Tests nécessitant GPU |
| CI/CD GitHub Actions | Exécution automatique sur push |
| GPU runner (self-hosted) | Tests GPU dans la CI |

---

## 5. Phase 2 — Pipeline cinématographique complet (Semaines 13-24)

**Objectif : Implémenter TOUS les composants manquants pour une vidéo "cinématographique" end-to-end.**

### 2.1 Module TTS propriétaire [F6]

| Action | Détail | Livrable |
|--------|--------|----------|
| Choisir l'architecture TTS | VITS2, NaturalSpeech 3, ou StyleTTS2 comme base (open-source) | Document de choix architectural |
| Implémenter `aiprod_core/model/tts/` | Modèle TTS multi-locuteur, multi-langue | Module fonctionnel |
| Training sur dataset voix sous licence | LibriTTS, Common Voice, ou dataset commercial | Modèle TTS entraîné |
| Intégrer au pipeline principal | Node TTS dans le graph d'inférence | Pipeline prompt → voix |
| Qualité cible | MOS ≥ 4.0 (comparable à ElevenLabs) | Benchmark MOS |

**Structure module :**
```
aiprod_core/model/tts/
├── __init__.py
├── model.py              # Architecture TTS principale
├── model_configurator.py  # Configurateur + SDOps
├── text_frontend.py       # G2P, normalisation texte
├── prosody.py             # Modélisation prosodie
├── vocoder_tts.py         # Vocoder spécialisé voix
└── speaker_embedding.py   # Embeddings multi-locuteur
```

### 2.2 Module Lip-Sync [F6]

| Action | Détail | Livrable |
|--------|--------|----------|
| Implémenter `aiprod_core/model/lip_sync/` | Synchronisation audio → mouvements lèvres | Module fonctionnel |
| Architecture | Wav2Lip-inspired ou SyncNet-based | Code + poids entraînés |
| Intégration post-génération | Appliqué comme post-processing sur la vidéo générée | Pipeline connecté |
| Métriques | LSE-D ≤ 7.0, LSE-C ≥ 6.0 | Benchmark validé |

### 2.3 Module Musique & Sound Design [F6]

| Action | Détail | Livrable |
|--------|--------|----------|
| Étendre l'Audio VAE existant | Branche conditionnelle pour musique vs ambiance vs FX | Module étendu |
| Contrôle granulaire audio | Paramètres : genre musical, tempo, intensité, mood | API de contrôle |
| Mixage multi-piste | Voix + musique + ambiance + FX → mix stéréo/5.1 | Module `audio_mixer.py` |
| Sound design procédural | Bibliothèque de sons d'ambiance contextuels | Module + assets audio |

**Structure module :**
```
aiprod_core/model/audio_mixer/
├── __init__.py
├── mixer.py               # Mixage multi-piste
├── spatial_audio.py        # Spatialisation 5.1/binaural
├── dynamics.py             # Compression, EQ, limiting
├── music_controller.py     # Contrôle musique conditionnelle
└── sound_design.py         # FX procéduraux
```

### 2.4 Module Montage automatisé [F6]

| Action | Détail | Livrable |
|--------|--------|----------|
| Implémenter `aiprod_pipelines/editing/` | Système de montage automatisé | Module fonctionnel |
| Génération timeline | Multi-scènes → timeline avec cuts, transitions, timing | `timeline_generator.py` |
| Transitions | Coupe franche, fondu, wipe, dissolve, match cut | `transitions.py` |
| Rythme narratif | Pacing basé sur l'émotion/action de chaque scène | `pacing_engine.py` |
| Format EDL/XML | Export timeline dans des formats standard (EDL, FCPXML) | `timeline_export.py` |

**Structure module :**
```
aiprod_pipelines/editing/
├── __init__.py
├── timeline.py            # Structure de données timeline
├── timeline_generator.py  # Génération automatique depuis scénario
├── transitions.py         # Bibliothèque de transitions
├── pacing_engine.py       # Contrôle du rythme narratif
├── continuity_checker.py  # Vérification raccords
└── timeline_export.py     # Export EDL, FCPXML, AAF
```

### 2.5 Module Étalonnage & Color Science [F6]

| Action | Détail | Livrable |
|--------|--------|----------|
| Implémenter `aiprod_pipelines/color/` | Pipeline color grading automatisé | Module fonctionnel |
| Gestion LUT | Bibliothèque LUT (cinématique, documentaire, corporate, etc.) | 20+ LUT intégrées |
| Color science | ACES workflow, espaces couleur (Rec.709, Rec.2020, DCI-P3) | Support multi-espace |
| HDR pipeline | Tone mapping, PQ/HLG, métadonnées HDR10/Dolby Vision | Export HDR fonctionnel |
| Color matching inter-scènes | Cohérence colorimétrique automatique entre les scènes | Algorithme + tests |

**Structure module :**
```
aiprod_pipelines/color/
├── __init__.py
├── color_pipeline.py      # Pipeline principal
├── lut_manager.py         # Gestion des LUT (load, apply, blend)
├── color_space.py         # Conversions espaces couleur
├── hdr.py                 # Pipeline HDR (PQ, HLG, tone mapping)
├── auto_grade.py          # Color grading automatisé par IA
├── scene_matching.py      # Cohérence colorimétrique inter-scènes
└── luts/                  # Bibliothèque LUT intégrée
    ├── cinematic_warm.cube
    ├── cinematic_cold.cube
    ├── documentary.cube
    └── ...
```

### 2.6 Export multi-format [F16]

| Format | Usage | Priorité |
|--------|-------|----------|
| H.264 + AAC (.mp4) | Web, réseaux sociaux | ✅ Existe |
| H.265/HEVC (.mp4) | Streaming haute qualité | P1 |
| ProRes 422/4444 (.mov) | Post-production professionnelle | P1 |
| DNxHR (.mxf) | Avid / broadcast | P2 |
| VP9/AV1 (.webm) | Web optimisé | P2 |
| EXR séquence | VFX compositing | P3 |
| DPX séquence | Cinéma numérique (DCP) | P3 |

### 2.7 Connecter les inference nodes [F4, F10]

| Action | Détail | Impact |
|--------|--------|--------|
| Remplacer `torch.randn()` dans `nodes.py` | Appels réels aux 5 pipelines via `model_ledger.py` | ~62K lignes deviennent fonctionnelles |
| `TextEncodeNode` → `AVGemmaTextEncoderModel` | Encodage texte réel | Node fonctionnelle |
| `DenoiseNode` → `euler_denoising_loop` / `denoise_audio_video` | Débruitage réel | Node fonctionnelle |
| `UpsampleNode` → `LatentUpsampler` | Upsampling réel | Node fonctionnelle |
| `DecodeVideoNode` → `VideoDecoder` (tiled) | Décodage VAE réel | Node fonctionnelle |
| `AudioEncodeNode` → `AudioDecoder` + `Vocoder` | Décodage audio réel | Node fonctionnelle |
| Ajouter nodes pour TTS, lip-sync, montage, étalonnage | Nouveaux modules Phase 2 | Pipeline complet |

### 2.8 Cohérence inter-scènes [F6]

| Action | Détail | Livrable |
|--------|--------|----------|
| Implémenter `SceneMemoryBank` | Banque de features partagée entre scènes | Module fonctionnel |
| Attention cross-scène | Mécanisme d'attention entre la scène courante et les scènes précédentes | Extension du transformer |
| Cohérence des personnages | Embeddings personnages persistants | `character_consistency.py` |
| Cohérence de l'environnement | Features d'environnement partagées | `environment_consistency.py` |
| Tests de cohérence | Métriques CLIP-Score inter-scènes, FID inter-scènes | Suite de benchmarks |

---

## 6. Phase 3 — Infrastructure production (Semaines 25-36)

**Objectif : Passer d'un prototype local à une infrastructure production-ready.**

### 3.1 Architecture de déploiement unifiée [F3, F7]

**Choix architectural : GPU-native SaaS avec orchestrateur intégré.**

```
┌─────────────────────────────────────────────────┐
│              Load Balancer (L7)                  │
│         (GCP / AWS ALB / Cloudflare)             │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────┐
│           API Gateway (FastAPI)                  │
│  Auth • Rate Limiting • Request Validation       │
│  Deployed on: Cloud Run (CPU, autoscale 1-100)   │
└─────────────────┬───────────────────────────────┘
                  │ gRPC / Message Queue
┌─────────────────▼───────────────────────────────┐
│         Job Orchestrator (Celery/Ray)            │
│  Queue management • Priority scheduling          │
│  Deployed on: GKE (CPU node pool)                │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────┐
│         GPU Worker Pool (Inference)              │
│  Model loaded in memory • Batched inference      │
│  Deployed on: GKE (GPU node pool)                │
│  Nodes: 2-20× A100/H100, autoscale on queue      │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────┐
│         Storage Layer                            │
│  Models: GCS/S3 • Videos: GCS/S3 • Logs: BQ     │
│  Cache: Redis • State: PostgreSQL                │
└─────────────────────────────────────────────────┘
```

### 3.2 Dockerfile GPU [F3]

| Action | Livrable |
|--------|----------|
| Nouveau `Dockerfile.gpu` basé sur `nvidia/cuda:12.4-devel` | Image avec PyTorch + CUDA + tous les modèles |
| Multi-stage build (compile → runtime) | Image optimisée (~15-20 GB) |
| Health check GPU intégré | `/health` endpoint avec VRAM check |
| Modèles pré-chargés dans l'image ou montés en volume | Startup time < 60s |

### 3.3 Kubernetes GPU [F3, F14]

| Action | Livrable |
|--------|----------|
| Manifestes Kubernetes (deployment, service, HPA, PDB) | `deploy/kubernetes/*.yaml` complets |
| GPU node pool avec autoscaling | Scale 0 → N basé sur queue length |
| Pod priority classes | Premium > Standard > Free tier |
| Resource quotas par namespace | Isolation multi-tenant |
| GPU health monitoring | DaemonSet nvidia-dcgm-exporter |

### 3.4 Monitoring & observabilité [F11]

| Composant | Outil | Métriques |
|-----------|-------|-----------|
| Infrastructure | Prometheus + Grafana | CPU, RAM, GPU util, VRAM, température |
| Application | OpenTelemetry | Latence par étape, throughput, error rate |
| Modèle | Custom metrics → Prometheus | Qualité score, FID, CLIP-Score, drift |
| Business | Custom dashboards | Coût/vidéo, revenu/vidéo, marge |
| Alerting | Grafana Alerting / PagerDuty | SLO violations, GPU crash, OOM, drift |
| Logs | Loki / CloudWatch | Structured logging JSON |
| Tracing | Jaeger / Tempo | Trace distribuée end-to-end |

### 3.5 Model Registry & Versioning [F12]

| Action | Livrable |
|--------|----------|
| Déployer MLflow ou DVC Model Registry | Instance MLflow accessible |
| Versionner chaque modèle (transformer, VAE, audio, TTS, etc.) | Versions sémantiques vX.Y.Z |
| Pipeline de promotion : dev → staging → production | Workflow CI/CD avec gates de qualité |
| Rollback automatique si dégradation qualité | Canary deployment avec métriques |
| Stockage des artefacts sur GCS/S3 | Modèles versionnés dans le cloud |

### 3.6 Robustesse & résilience [F11]

| Scénario | Mécanisme | Implémentation |
|----------|-----------|----------------|
| GPU crash | Health check + restart + migration | Kubernetes liveness probe GPU |
| OOM | Fallback résolution inférieure + retry | try/catch VRAM + config dégradée |
| Timeout | Deadline par étape du pipeline | Timeout configurable par Node |
| Corruption dataset | Checksum SHA-256 + validation post-download | Module `data_integrity.py` |
| Model drift | Monitoring qualité automatique + alerte | Métriques FID/CLIP-Score périodiques |
| Décroissance qualité | A/B testing + rollback automatique | Canary avec comparaison métriques |

---

## 7. Phase 4 — SaaS & scalabilité (Semaines 37-48)

**Objectif : Lancer un SaaS fonctionnel, sécurisé, facturable.**

### 4.1 API Gateway complète

| Fonctionnalité | Détail | Livrable |
|----------------|--------|----------|
| Authentification | JWT + API keys + OAuth2 | Module auth fonctionnel |
| Rate limiting | Par tier (Free: 5/jour, Pro: 100/jour, Enterprise: illimité) | Config par plan |
| Validation requêtes | Schémas Pydantic stricts | Validation complète |
| Versioning API | `/v1/`, `/v2/` routes | Rétrocompatibilité |
| Documentation | OpenAPI/Swagger auto-générée | `/docs` endpoint |
| Webhooks | Notification de complétion | Callbacks configurables |
| SDK clients | Python, JavaScript, REST | Packages publiés |

### 4.2 Billing & métriques financières

| Fonctionnalité | Détail |
|----------------|--------|
| Metering | Comptage précis : durée vidéo, résolution, features utilisées |
| Pricing | Par seconde de vidéo générée + suppléments (4K, HDR, TTS, etc.) |
| Intégration Stripe | Subscriptions + usage-based billing |
| Dashboard client | Consommation, historique, factures |
| Alertes budget | Notification quand le client approche sa limite |
| Coût interne par vidéo | Tracking GPU-hours × coût/h par vidéo |

### 4.3 Multi-tenant réel [F4]

| Action | Livrable |
|--------|----------|
| Connecter le module `multi_tenant_saas/` à PostgreSQL | Persistence réelle des tenants |
| Isolation des jobs par namespace Kubernetes | Sécurité inter-tenant |
| Queue prioritaire par tier | Redis/RabbitMQ avec priority queues |
| Quotas de stockage par tenant | Limites GCS par tenant |
| Audit trail | Logging de chaque action par tenant |

### 4.4 Batching inference [F15]

| Action | Détail | Gain |
|--------|--------|------|
| Implémenter le dynamic batching | Regroupement des requêtes par résolution/durée similaire | Throughput ×2-4 |
| Request queuing avec timeout | Attente max 5s avant exécution même si batch incomplet | Latence contrôlée |
| Connecter le module `dynamic_batch_sizing/` | Memory-aware batch size | Utilisation GPU optimale |

### 4.5 Optimisation inference [F15]

| Optimisation | Gain latence estimé | Priorité |
|-------------|--------------------|---------| 
| TensorRT compilation du transformer | ×2-3 | P1 |
| ONNX Runtime pour VAE decoder | ×1.5-2 | P1 |
| torch.compile (Inductor) end-to-end | ×1.3-1.5 | P2 |
| Speculative decoding (fewer denoising steps) | ×2-4 | P2 |
| KV-cache pour attention | ×1.2-1.5 | P3 |
| INT4 quantization (GPTQ/AWQ) | ×1.5-2 (VRAM ÷2) | P3 |

---

## 8. Phase 5 — Excellence & différenciation (Semaines 49-72)

**Objectif : Dépasser le marché, construire un moat durable.**

### 5.1 Reward model & amélioration continue

| Action | Détail |
|--------|--------|
| Connecter le module `reward_modeling/` à de vrais feedbacks utilisateur | Collecte de préférences humaines |
| Entraîner un reward model sur les préférences | Modèle de scoring qualité |
| RLHF / DPO sur le transformer diffusion | Alignement avec les préférences humaines |
| A/B testing automatisé en production | Comparaison modèles candidats |

### 5.2 Scénariste IA interne (remplacement Gemini)

| Action | Détail |
|--------|--------|
| Fine-tuner un LLM open-source (Llama/Mistral) pour le scénario | Spécialisé découpage en scènes, direction caméra, émotions |
| Éliminer la dépendance à l'API Gemini | LLM local déployé sur GPU |
| Prompts → découpage scènes structuré (JSON) | Output directement consommable par le pipeline |
| Contrôle créatif avancé | Style, genre, public cible, ton |

### 5.3 Contrôle caméra avancé

| Fonctionnalité | Détail |
|----------------|--------|
| ControlNet caméra | Pan, tilt, zoom, dolly, crane, steadicam |
| Trajectoires caméra paramétriques | Courbes de Bézier pour mouvements fluides |
| Templates cinématographiques | "Plan séquence", "champ-contrechamp", "travelling" |
| Camera shake simulation | Handheld, action cam, stabilisé |

### 5.4 Modèle AIPROD v3 from scratch (objectif long terme)

| Étape | Détail | Timeline |
|-------|--------|----------|
| Architecture novel | DiT amélioré avec innovations AIPROD | Mois 18-24 |
| Training sur dataset propriétaire 100K+ h | Multi-node A100/H100 | Mois 24-36 |
| Benchmark vs SOTA | FVD, CLIP-Score, human eval vs Sora/Kling/Runway | Mois 36 |
| Dépréciation fork LTX-Video | Migration progressive | Mois 36-48 |

### 5.5 Edge deployment & on-premise

| Action | Détail |
|--------|--------|
| Connecter le module `edge_deployment/` | Modèles quantisés pour RTX 4090/5090 |
| Version embarquée | Desktop app avec inférence locale |
| Plugin DaVinci Resolve / Premiere Pro | Intégration post-production pro |
| API on-premise pour enterprises | Déploiement dans le datacenter client |

---

## 9. Matrice de traçabilité — Chaque faille → sa correction

| Faille | Gravité | Phase de correction | Actions clé | Validation |
|--------|---------|--------------------|-----------|-----------| 
| F1 — Fork non attribué | 🔴 | Phase 0 (S1-2) | Licences Apache 2.0, NOTICE, MODIFICATIONS.md | Revue juridique OK |
| F2 — Pas de training from scratch | 🔴 | Phase 1 (S3-12) | Dataset, trainers VAE/transformer, curriculum | Checkpoint >5 GB généré |
| F3 — SaaS sans GPU | 🔴 | Phase 3 (S25-36) | Dockerfile.gpu, K8s GPU, worker pool | Pod GPU déployé + vidéo générée via API |
| F4 — 62K lignes non connectées | 🔴 | Phase 2 (S13-24) | Connexion nodes, suppression dead code | `torch.randn` absent du code prod |
| F5 — Licences supprimées | 🔴 | Phase 0 (S1-2) | NOTICE, headers, THIRD_PARTY_LICENSES | Compliance audit OK |
| F6 — TTS/lip-sync/montage/étalonnage absents | 🔴 | Phase 2 (S13-24) | 5 nouveaux modules implémentés | Pipeline end-to-end fonctionnel |
| F7 — Dualité architecturale | 🟠 | Phase 3 (S25-36) | Architecture unifiée GPU-native SaaS | Un seul Dockerfile, un seul deploy path |
| F8 — Zéro exécution réelle | 🟠 | Phase 0 (S1-2) | quickstart.py exécuté, WandB logging | Vidéo MP4 + run WandB |
| F9 — Tests absents core | 🟠 | Phase 1 (S3-12) | 30+ fichiers test, CI/CD | Coverage >85% |
| F10 — Nodes mockées | 🟠 | Phase 2 (S13-24) | Remplacement torch.randn par appels réels | Tests d'intégration GPU |
| F11 — Monitoring absent | 🟠 | Phase 3 (S25-36) | Prometheus, Grafana, OpenTelemetry | Dashboard live avec alertes |
| F12 — Pas de versioning modèle | 🟠 | Phase 3 (S25-36) | MLflow, model registry, promotion pipeline | Modèle v1.0.0 enregistré |
| F13 — Monkey-patching torch._dynamo | 🟡 | Phase 0 (S1-2) | Suppression code + fix propre | Code nettoyé |
| F14 — Répertoires vides | 🟡 | Phase 0 (S1-2) | Peupler ou supprimer | Zéro dossier vide non justifié |
| F15 — Pas de batching | 🟡 | Phase 4 (S37-48) | Dynamic batching + queue | Throughput ×2-4 mesuré |
| F16 — Export H.264 seulement | 🟡 | Phase 2 (S13-24) | 7 formats export | ProRes, H.265, AV1 fonctionnels |
| F17 — Dead code prototype | 🟡 | Phase 0 (S1-2) | Suppression 1 870 lignes | Code supprimé |

---

## 10. Budget & ressources

### Ressources humaines (équipe cible)

| Rôle | Nombre | Phase de recrutement |
|------|--------|---------------------|
| ML Engineer senior (training/modèles) | 2 | Phase 0-1 |
| ML Engineer (inference/optimisation) | 1 | Phase 1-2 |
| Backend Engineer senior (infra/SaaS) | 1 | Phase 2-3 |
| Audio/DSP Engineer | 1 | Phase 2 |
| DevOps/MLOps Engineer | 1 | Phase 3 |
| QA / Test Engineer | 1 | Phase 1 |
| **Total équipe technique** | **7** | |

### Budget compute

| Poste | Coût annuel estimé |
|-------|-------------------|
| Training (fine-tuning + expérimentations) | $20 000 - $50 000 |
| Training from scratch (si poursuivi) | $100 000 - $500 000 |
| Inference SaaS (100-1 000 vidéos/jour) | $50 000 - $200 000 |
| Infrastructure (K8s, networking, storage) | $20 000 - $50 000 |
| Monitoring & outils (WandB, MLflow, etc.) | $5 000 - $15 000 |
| **Total annuel compute** | **$195 000 - $815 000** |

### Budget data

| Poste | Coût estimé |
|-------|-------------|
| Acquisition dataset vidéo sous licence | $50 000 - $200 000 |
| Annotation / captioning | $10 000 - $50 000 |
| Stockage dataset (GCS/S3) | $5 000 - $20 000/an |
| **Total data** | **$65 000 - $270 000** |

### Budget total Phase 0 → Phase 4 (12 mois)

| Poste | Estimation |
|-------|-----------|
| Salaires (7 personnes × 12 mois) | $700 000 - $1 200 000 |
| Compute | $195 000 - $815 000 |
| Data | $65 000 - $270 000 |
| Outils & licences | $20 000 - $50 000 |
| **TOTAL 12 MOIS** | **$980 000 - $2 335 000** |

---

## 11. KPIs de validation par phase

### Phase 0 — Assainissement ✓

| KPI | Cible | Méthode de mesure |
|-----|-------|-------------------|
| Compliance juridique | 100% fichiers avec headers licence | Script de vérification automatique |
| Dead code supprimé | 0 fichier prototype dans le build | `import aiprod_core` ne charge aucun toy model |
| Première vidéo générée | 1 vidéo MP4 cohérente | Validation visuelle + WandB artifact |
| Logs non vides | ≥1 run documenté | `ls logs/` + WandB dashboard |

### Phase 1 — Fondations ✓

| KPI | Cible | Méthode de mesure |
|-----|-------|-------------------|
| Dataset constitué | ≥10 000 clips vidéo | Comptage + rapport distribution |
| Coverage tests aiprod-core | ≥85% | `pytest --cov` |
| Training pipeline fonctionnel | VAE + transformer entraînables | Checkpoint sauvegardé + loss convergente |
| Modifications architecturales | ≥3 extensions originales documentées | Code + MODIFICATIONS.md |

### Phase 2 — Pipeline cinématographique ✓

| KPI | Cible | Méthode de mesure |
|-----|-------|-------------------|
| TTS fonctionnel | MOS ≥ 4.0 | Test écoute humain |
| Lip-sync fonctionnel | LSE-D ≤ 7.0 | Benchmark SyncNet |
| Montage automatisé | Timeline ≥3 scènes | Export EDL valide |
| Color grading | 3+ looks disponibles (LUT) | Vidéo exportée avec LUT |
| `torch.randn()` en prod | 0 occurrence | `grep -r "torch.randn" src/ --include="*.py"` hors tests |
| Export multi-format | ≥4 formats | ProRes, H.265, H.264, AV1 testés |
| Pipeline end-to-end | Prompt → vidéo finale avec voix + musique + étalonnage | Vidéo de démonstration |

### Phase 3 — Infrastructure ✓

| KPI | Cible | Méthode de mesure |
|-----|-------|-------------------|
| Déploiement GPU K8s | Pod GPU opérationnel | `kubectl get pods` avec GPU allocated |
| Vidéo via API | Requête HTTP → vidéo retournée | Test curl/httpie |
| Monitoring live | Dashboard Grafana avec ≥10 métriques | Screenshot dashboard |
| Model registry | ≥3 versions de modèle enregistrées | MLflow UI |
| Failover GPU | Récupération après kill de pod en <60s | Test chaos engineering |
| Latence P95 | ≤ 5 min pour vidéo 10s | Métriques Prometheus |

### Phase 4 — SaaS ✓

| KPI | Cible | Méthode de mesure |
|-----|-------|-------------------|
| API publique documentée | Swagger complet | `/docs` accessible |
| Auth fonctionnelle | JWT + API key | Tests d'authentification |
| Billing | Facturation par vidéo | Transaction Stripe test |
| Throughput | ≥50 vidéos/jour sur cluster minimal | Load test |
| Coût/vidéo 30s | ≤ $1.50 | Mesure GPU-hours × tarif |
| Uptime | ≥99.5% sur 30 jours | Monitoring uptime |

### Phase 5 — Excellence ✓

| KPI | Cible | Méthode de mesure |
|-----|-------|-------------------|
| Reward model actif | Corrélation ≥0.7 avec préférences humaines | Benchmark sur 200 paires |
| LLM scénariste interne | 0 appel API Gemini | Monitoring API calls |
| Contrôle caméra | 6+ types de mouvements | Demo vidéo |
| Qualité vs SOTA | FVD ≤ SOTA ×1.2 | Benchmark standardisé |

---

## 12. Score cible détaillé 10/10

### Solidité modèle — De 2/10 à 10/10

| Critère | État actuel (2/10) | Cible 10/10 |
|---------|-------------------|-------------|
| Propriété intellectuelle | Fork non attribué | Fork légalement attribué + extensions propriétaires documentées + roadmap v3 from scratch |
| Training | LoRA uniquement | Full training pipeline (VAE + transformer + TTS + audio mixer) |
| Dataset | Inexistant | ≥10 000h vidéo sous licence, versionné (DVC), audité |
| Architecture | Copie LTX-Video | LTX-Video attribué + 5 extensions originales + roadmap AIPROD v3 |
| Qualité | Non mesurée | FVD, CLIP-Score, MOS benchmarkés vs SOTA |
| Son | Ambiance basique | TTS multi-langue + lip-sync + mixage multi-piste + musique conditionnelle |
| Tests | 0 test core | Coverage ≥85% avec tests GPU réels |

### Solidité infrastructure — De 3/10 à 10/10

| Critère | État actuel (3/10) | Cible 10/10 |
|---------|-------------------|-------------|
| Déploiement | Cloud Run CPU sans GPU | K8s GPU autoscalé + multi-région |
| Pipeline | 5 pipelines fonctionnels mais disconnectés | Pipeline end-to-end: prompt → vidéo finale exportée |
| Nodes inference | `torch.randn()` partout | Chaque node connectée au vrai modèle |
| Monitoring | Scripts nvidia-smi | Prometheus + Grafana + OpenTelemetry + alertes |
| Model versioning | Filesystem | MLflow + promotion pipeline + rollback automatique |
| Résilience | Aucune | Health checks GPU, OOM fallback, retry, migration jobs |
| Tests | Mocked torch | Tests GPU réels + CI/CD avec GPU runner |
| Code quality | 62K lignes non connectées | Zéro dead code, zéro mock en production |

### Viabilité économique — De 2/10 à 10/10

| Critère | État actuel (2/10) | Cible 10/10 |
|---------|-------------------|-------------|
| Moat technologique | Aucun (fork reproductible) | Extensions propriétaires + LoRA spécialisés + pipeline end-to-end unique |
| Dépendances | Totales (LTX, Gemini, Runway, etc.) | LTX attribué, LLM interne, inférence autonome, 0 API externe en prod |
| Coût/vidéo | Non mesuré | ≤$1.50/vidéo 30s, optimisé (TensorRT, batching, distillation) |
| Revenue model | Inexistant | SaaS facturé avec Stripe, 3 tiers (Free/Pro/Enterprise) |
| Scalabilité | Non testée | Cluster GPU autoscalé, 50-1 000 vidéos/jour |
| Compétitivité | Wrapper sans valeur ajoutée | Pipeline cinématographique complet unique (TTS + montage + étalonnage + HDR) |

---

## Chronogramme synthétique

```
Semaine  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
         ├──────┤
         Phase 0
         Juridique
         Nettoyage
         1ère exéc.
                  ├────────────────────────────────────────┤
                  Phase 1 — Fondations propriétaires
                  Dataset │ Training pipelines │ Extensions archi │ Tests

                                                            ├────────────────────────────────────────┤
                                                            Phase 2 — Pipeline cinématographique
                                                            TTS │ Lip-sync │ Montage │ Étalonnage │ Nodes réelles

Semaine  25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48
         ├────────────────────────────────────────┤
         Phase 3 — Infrastructure production
         K8s GPU │ Monitoring │ Model registry │ Résilience

                                                  ├────────────────────────────────────────┤
                                                  Phase 4 — SaaS & scalabilité
                                                  API │ Billing │ Multi-tenant │ Batching │ Optim

Semaine  49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72
         ├────────────────────────────────────────────────────────────────────────────────┤
         Phase 5 — Excellence & différenciation
         Reward model │ LLM interne │ Contrôle caméra │ AIPROD v3 R&D │ Edge deploy
```

---

## Conclusion

Ce plan transforme AIPROD d'un **fork renommé avec une couche d'orchestration partiellement implémentée** (score 2-3/10) en un **système de production vidéo cinématographique end-to-end, juridiquement sain, techniquement solide, économiquement défendable** (score 10/10).

Les prérequis non négociables sont :
1. **Honnêteté juridique** sur l'origine LTX-Video (Phase 0, immédiat)
2. **Preuve d'exécution** — générer une vraie vidéo, pas des `torch.randn()` (Phase 0, immédiat)
3. **Investissement significatif** — équipe de 7 personnes, ~$1-2.3M sur 12 mois
4. **Patience stratégique** — 18 mois minimum avant un produit compétitif avec le marché

La différenciation à terme ne viendra pas du modèle de diffusion (commoditisé) mais du **pipeline cinématographique end-to-end** (TTS + lip-sync + montage + étalonnage + HDR + export multi-format) — c'est là que la valeur propriétaire se construit.

---

*Plan Master AIPROD — 14 février 2026*
