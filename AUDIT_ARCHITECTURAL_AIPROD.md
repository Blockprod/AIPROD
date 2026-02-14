# AUDIT ARCHITECTURAL — AIPROD

**Date :** 14 février 2026  
**Auditeur :** Principal AI Systems Architect & Production Infrastructure Auditor  
**Périmètre :** Analyse complète du code source — `C:\Users\averr\AIPROD`  
**Méthode :** Lecture intégrale de chaque fichier source, config, script, notebook et checkpoint  
**Tolérance à l'illusion technologique : zéro**

---

## 1. Nature réelle du système

### Description exacte inférée depuis le code

AIPROD est constitué de **trois packages Python** dans un monorepo :

| Package | Lignes de code | Nature réelle |
|---------|---------------|---------------|
| `aiprod-core` | ~7 500 (moteur) + ~1 870 (prototype) | **Fork renommé de LTX-Video 2.0 (Lightricks)** avec remplacement systématique des chaînes "LTX" → "AIPROD" |
| `aiprod-pipelines` | ~63 800 | Couche d'orchestration/pipeline autour du moteur LTX-Video. ~2 000 lignes de pipelines réels, ~62 000 lignes d'infrastructure (SaaS, tensor parallelism, distributed LoRA, etc.) largement non connectées |
| `aiprod-trainer` | ~5 000+ | **Toolkit de fine-tuning LoRA** sur le modèle LTX-Video 2.0 pré-entraîné. Ne fait PAS d'entraînement from scratch |

### Modèles réellement propriétaires ?

🔴 **NON.**

Le moteur de production (`aiprod_core/model/`) est un **fork direct de LTX-Video 2.0 de Lightricks** :

- Architecture transformer : 48 couches, 32 têtes d'attention, head_dim=128, cross_attention_dim=4096 — **identique à LTX-Video 2.0**
- Video VAE : compression spatiale 32×, temporelle 8×, contrainte `1 + 8k` frames — **identique à LTX-Video 2.0**
- Audio VAE + Vocoder HiFi-GAN — **branche audio de LTX-Video 2.0**
- Scheduler flow-matching avec `flux_time_shift` — **LTX-Video 2.0**
- Text encoder : Google Gemma 3 (hidden_size=3840, 48 layers, vocab=262 208) — modèle open-source tiers
- Renommages : `LTXRopeType` → `AIPRODRopeType`, `LTXModel` → `AIPRODModel`, etc.
- Classes `PixArtAlphaTextProjection` et `PixArtAlphaCombinedTimestepSizeEmbeddings` conservées avec URLs GitHub d'origine
- **Aucune notice de copyright, aucun header de licence, aucune attribution Lightricks/LTX-Video** dans le code source

Un second code "prototype" existe dans `src/models/backbone.py` (768-D, attention+CNN interleaved) : original mais **toy model** non connecté au moteur de production, inutilisable pour la génération vidéo.

### Stack réellement internalisée ?

🔴 **NON.**

| Composant | Réalité |
|-----------|---------|
| Modèle diffusion vidéo | Fork LTX-Video 2.0 (Lightricks) |
| Text encoder | Google Gemma 3 (open-source) |
| Audio VAE + Vocoder | Fork LTX-Video 2.0 |
| Direction créative | Google Gemini 1.5 Pro (API externe) |
| Rendu vidéo SaaS | Google Veo-3, Runway Gen3, Replicate WAN-2.5 (APIs externes) |
| Captioning vidéo | Qwen2.5-Omni-7B ou Gemini Flash (modèles tiers) |
| Déploiement | Google Cloud Run (pas de GPU dans le Dockerfile) |

### Autonomie réelle ou dépendance cachée ?

🔴 **Dépendance totale.** Le système dépend de :

1. **Lightricks** pour l'architecture et les poids du modèle fondamental
2. **Google** pour le text encoder (Gemma 3), la direction créative (Gemini), et potentiellement le rendu (Veo-3)
3. **Runway / Replicate** comme backends de rendu alternatifs dans le pipeline SaaS
4. **HuggingFace** pour l'hébergement et le téléchargement des modèles

### Cohérence globale

Le projet présente une **dualité architecturale non résolue** :

1. **Pipeline local GPU** : inference text-to-video via les 5 pipelines (`TI2VidOneStage`, `TI2VidTwoStages`, `Distilled`, `ICLora`, `KeyframeInterpolation`) — code fonctionnel et de bonne qualité
2. **Pipeline SaaS/orchestrateur** : state machine 11 états avec API Cloud Run, qui délègue le rendu à des APIs externes (Veo-3, Runway, etc.) — **aucun GPU dans le Dockerfile de déploiement**

Ces deux architectures coexistent sans lien clair. Le SaaS ne déploie pas le modèle local.

---

## 2. Modèles propriétaires

### Types de modèles utilisés

| Type | Modèle réel | Propriétaire ? |
|------|-------------|---------------|
| LLM interne | Aucun — utilise Gemini 1.5 Pro (API Google) | 🔴 Non |
| Diffusion vidéo | Fork LTX-Video 2.0 (~1.9B params) | 🔴 Non — fork renommé |
| VAE vidéo | Fork LTX-Video 2.0 VAE (3D causal) | 🔴 Non |
| Audio VAE | Fork LTX-Video 2.0 audio branch | 🔴 Non |
| Vocoder | HiFi-GAN (fork LTX-Video) | 🔴 Non |
| TTS | Inexistant | 🔴 N/A |
| Musique | Inexistant | 🔴 N/A |
| Cohérence inter-scènes | Inexistant en production | 🔴 N/A |
| Text encoder | Google Gemma 3 (4B params) | 🔴 Non — open-source tiers |

### Architecture des modèles

- **Transformer diffusion** : Flow-matching avec AdaLN (PixArt-Alpha), RoPE 3D, attention multi-modale (vidéo + audio + cross-modal), STG guidance
- **Video VAE** : Encoder/decoder causal 3D, blocs ResNet, attention spatiale, compression 32×32×8
- **Audio VAE** : Encoder/decoder mel-spectrogram, convolutions causales, normalisation par canal
- **Vocoder** : Style HiFi-GAN pour reconstruction waveform
- **Tout issu de LTX-Video 2.0**

### Taille estimée des modèles

| Modèle | Paramètres estimés |
|--------|-------------------|
| Transformer (AIPRODModel) | ~1.9B (48 layers × 32 heads × 128 head_dim) |
| Video VAE | ~150-300M |
| Audio VAE + Vocoder | ~50-100M |
| Gemma 3 Text Encoder | ~4B |
| **Total inference** | **~6-7B paramètres** |

### Pipeline d'entraînement

🔴 **Aucun entraînement from scratch n'est réalisé ni réalisable dans l'état actuel.**

**Ce qui existe :**

- Un toolkit de **fine-tuning LoRA** (`aiprod-trainer`) pour adapter le modèle LTX-Video 2.0 pré-entraîné
- LoRA rank 16-32 sur attention vidéo, audio, et cross-modale
- Configs Accelerate (DDP, FSDP) pour multi-GPU
- Pipeline de pré-processing : captioning → embeddings → latents
- Validation avec sampling périodique
- Tracking via Weights & Biases

**Ce qui manque :**

- 🔴 Aucun entraînement du transformer fondamental (1.9B params)
- 🔴 Aucun entraînement du VAE vidéo
- 🔴 Aucun entraînement de l'audio VAE
- 🔴 Aucun entraînement du text encoder (Gemma 3 gelé)
- 🔴 Aucun dataset propriétaire identifiable (répertoires vides : `models/aiprod2/`, `models/gemma-3/`, `models/pretrained/`)
- 🔴 Le prototype `train.py` dans `aiprod-core/src/training/` entraîne un **toy model** non connecté au moteur de production (backbone 768-D + VAE simpliste avec données synthétiques rectangles)
- 🔴 Le seul checkpoint existant (`PHASE_1_SIMPLE_epoch_0.pt`, 152 MB) est celui du toy model — un modèle LTX-Video réel ferait 18-40 GB

**Dataset réel ou théorique ?**

🔴 Théorique. Les scripts de preprocessing (`process_videos.py`, `process_captions.py`, `split_scenes.py`) existent et sont fonctionnels, mais :
- Le répertoire `datasets/` est dans `.gitignore` (aucune donnée commitable)
- Aucun log de training dans `logs/` (répertoire vide)
- Aucune trace d'exécution réelle du pipeline de training

**Coût estimé d'entraînement from scratch :**

| Phase | Estimation |
|-------|-----------|
| Pré-training transformer 1.9B (video + audio) | 500-2 000 heures A100 (~$1-4M) |
| Pré-training Video VAE | 100-500 heures A100 (~$200K-1M) |
| Pré-training Audio VAE + Vocoder | 50-200 heures A100 (~$100-400K) |
| Fine-tuning LoRA (ce qui est couvert) | 10-50 heures A100 (~$20-100K) |
| **Total from scratch** | **$1.3-5.4M minimum** |

🔴 Le fine-tuning LoRA (seule capacité actuelle) ne constitue pas un "modèle propriétaire". Il produit un adaptateur de quelques dizaines de MB sur un modèle tiers de 19 GB.

---

## 3. Pipeline d'inférence

### Optimisation GPU

| Optimisation | Implémentée ? |
|-------------|--------------|
| FP8 quantization transformer | ✅ Oui (via `optimum-quanto`) |
| Tiled VAE decoding | ✅ Oui (blending trapézoïdal) |
| Multi-backend attention (PyTorch, xFormers, FlashAttention3) | ✅ Oui |
| Batching | 🟠 Non — inférence single-request |
| Distillation | 🟠 Pipeline `distilled.py` existe mais utilise des sigma pré-calculés, pas de vrai modèle distillé |
| torch.compile | ✅ Configs Accelerate avec backend Inductor |

### Latence estimée

- Vidéo 30s @ 25fps (750 frames) sur A100 80GB : **~5-15 minutes** (estimation basée sur LTX-Video 2.0 benchmarks)
- Vidéo 30s @ 25fps sur GTX 1070 : commentaire dans le code dit "15-45 minutes"
- 🟠 Aucun benchmark réel exécuté (logs vides)

### Stabilité des générations

- CFG + STG guidance implémentés
- APG (Adaptive Projected Guidance) implémenté
- Multi-modal guidance (audio/vidéo isolation) implémenté
- 🟡 Pas de métriques de qualité collectées en production

### Maintien cohérence multi-scènes

🔴 **Non implémenté.** Aucun module de cohérence inter-scènes fonctionnel. Le module `multimodal_coherence/` dans inference (~2 400 lignes) est structurel uniquement — algorithmes simplifiés, pas connecté au pipeline réel.

---

## 4. Orchestration distribuée

### State machine réelle

✅ Oui — `Orchestrator` avec 11 états (INIT → ANALYSIS → CREATIVE_DIRECTION → VISUAL_TRANSLATION → FINANCIAL_OPTIMIZATION → RENDER_EXECUTION → QA_TECHNICAL → QA_SEMANTIC → FINALIZE + FAST_TRACK + ERROR). Checkpoint/resume JSON fonctionnel.

### Gestion multi-GPU

| Capacité | État |
|----------|------|
| Training multi-GPU DDP | ✅ Configs Accelerate fonctionnelles |
| Training FSDP | ✅ Config avec wrapping `BasicAVTransformerBlock` |
| Inference multi-GPU | 🔴 Non implémenté |
| tensor_parallelism (inference) | 🔴 Module existe (~1 756 lignes) mais **non connecté** — structures de données et configs seulement |

### Multi-node training

🟠 Théorique via Accelerate — configs DDP supportent multi-process mais aucune trace de training multi-node.

### Scheduler interne / Priorités SaaS

Le module `multi_tenant_saas/` (~2 471 lignes) implémente :
- Tenant management, auth JWT, RBAC
- Billing, rate limiting, job scheduling
- Feature flags, monitoring

🔴 **Aucune intégration backend réelle.** Toutes les classes ont de la logique interne mais pas de connexion à une base de données, un message broker, ou un système de queue.

### Isolation des jobs clients

🔴 Non implémentée en production. Le module SaaS est structurel.

### Retry logic

✅ Implémentée dans l'adaptateur de rendu (`render.py`) avec chaîne de fallback Veo-3 → Runway → Replicate.

### Reproductibilité déterministe

🟠 Seeds configurables dans les pipelines mais pas de garantie de reproductibilité cross-GPU/cross-version.

---

## 5. Pipeline Vidéo interne

### Modèle vidéo réellement compétitif ?

Le modèle est **identique à LTX-Video 2.0**, donc :
- ✅ Compétitif au niveau de LTX-Video 2.0 (modèle open-source de bonne qualité)
- 🔴 Pas un avantage compétitif — n'importe qui peut utiliser LTX-Video 2.0

### Résolution native

- Stage 1 : 512×768 (demi-résolution)
- Stage 2 (upsampling) : 1024×1536
- Défini dans `constants.py`

### Frame interpolation

✅ Pipeline `KeyframeInterpolationPipeline` implémenté — interpolation entre keyframes via diffusion.

### Cohérence temporelle

✅ Assurée par le RoPE 3D (positional encoding spatial + temporel) du transformer. Hérité de LTX-Video 2.0.

### Contrôle caméra réel ou simulé ?

🔴 **Non implémenté.** Aucun système de contrôle caméra (pas de ControlNet caméra, pas de paramètres de mouvement caméra dans les pipelines).

### Risque d'artefacts

🟠 Standard pour un modèle diffusion 1.9B — artefacts possibles sur les mouvements complexes, les mains, les visages. Même profil de risque que LTX-Video 2.0 vanilla.

---

## 6. Pipeline Audio interne

### Modèle TTS propriétaire ?

🔴 **Inexistant.** Aucun modèle TTS dans le code. L'audio VAE génère de l'audio ambiant/musique conjointement avec la vidéo (hérité de LTX-Video 2.0 audio branch), mais :
- Pas de synthèse vocale
- Pas de dialogue
- Pas de voix-off

### Qualité comparable au marché ?

🟠 L'audio VAE de LTX-Video 2.0 est fonctionnel mais **pas au niveau** de ElevenLabs, Bark, ou XTTS pour le TTS. Il produit des ambiances sonores synchronisées avec la vidéo, pas de la parole.

### Lip-sync interne ou post-processing ?

🔴 **Inexistant.** Aucun module de lip-sync.

### Mixage automatisé crédible ?

🔴 Le module `multimodal_coherence/` est structurel. Pas de mixage audio automatisé fonctionnel.

### Gestion dynamique musique / ambiance ?

🟠 La génération audio conjointe (via les cross-modal attention blocks `audio_to_video_attn`, `video_to_audio_attn`) produit de l'audio contextualisé, mais sans contrôle granulaire musique/ambiance/dialogue.

---

## 7. Montage & Étalonnage

### Timeline générée automatiquement ?

🔴 **Non implémenté.** Le module `video_editing/` (~1 034 lignes) dans inference contient :
- Content analysis
- Dataset validation  
- Quality checking

Mais **aucun montage automatisé** : pas de cuts, pas de transitions, pas de timeline multi-clips.

### Gestion LUT

🔴 Inexistante.

### Color science maîtrisée

🔴 Inexistante. Pas de color grading, pas d'étalonnage, pas de color spaces (les vidéos sont générées en sRGB sans post-processing couleur).

### Pipeline HDR

🔴 Inexistant.

### Formats export

✅ Export H.264 + AAC via PyAV/FFmpeg (`media_io.py`). Format unique, pas de ProRes, DNxHR, EXR, ou formats cinématographiques.

---

## 8. Scalabilité & Compute

### Coût training initial estimé

| Scénario | Coût |
|----------|------|
| Fine-tuning LoRA (capacité actuelle) | ~$20-100K (10-50h A100) |
| Entraînement from scratch du transformer 1.9B | ~$1-4M (500-2000h A100) |
| Stack complète from scratch (transformer + VAE + audio) | ~$1.3-5.4M |
| **Avec itérations R&D réalistes (3-5 tentatives)** | **$5-20M** |

### Coût inference par vidéo 30s

| GPU | Coût estimé |
|-----|-------------|
| A100 80GB (cloud) | ~$0.50-2.00 par vidéo 30s (5-15 min @ $2/h) |
| H100 (cloud) | ~$0.30-1.50 par vidéo 30s |
| Consumer GPU (RTX 4090) | Temps × coût électricité local |

### Risque d'explosion GPU

🟠 **Modéré.** Le modèle complet (6-7B params total) nécessite :
- Minimum 24 GB VRAM pour inference FP8
- 40-80 GB VRAM sans quantization
- Le tiled VAE decoding et FP8 quantization réduisent significativement la charge mémoire

### Besoin estimé en A100 / H100

| Usage | Besoin |
|-------|--------|
| Fine-tuning LoRA | 1× A100/H100 (batch_size=1, gradient accumulation) |
| Inference SaaS (100 req/jour) | 2-4× A100 |
| Inference SaaS (1000 req/jour) | 10-20× A100 |
| Training from scratch | 32-128× A100 pendant 1-4 semaines |

### Optimisations possibles

- ✅ FP8 quantization (déjà implémenté)
- ✅ Tiled VAE (déjà implémenté)
- 🟡 Distillation du transformer (pipeline existe, pas de modèle distillé)
- 🟡 Batching inference (non implémenté)
- 🟡 Speculative decoding (non implémenté)
- 🟡 TensorRT/ONNX export (non implémenté)

### Viabilité SaaS sans levée massive

🔴 **Non viable sans levée significative.**

- Le Dockerfile Cloud Run **n'alloue aucun GPU** (4 CPU, 8 GB RAM) — le SaaS actuel est un orchestrateur API, pas un service de rendu
- Si rendu via APIs externes (Veo-3, Runway) : coûts variables mais dépendance tier totale
- Si rendu local : infrastructure GPU dédiée nécessaire (~$50-200K/an pour un cluster modeste)
- Budget .env.example : $10K/mois, $500/jour — insuffisant pour un volume SaaS significatif avec modèles locaux

---

## 9. Robustesse réelle

### Comportement si node GPU crash

🔴 **Aucun mécanisme.** Pas de health check GPU au niveau service, pas de failover, pas de migration de job.

### Comportement si OOM

🟠 Le tiled VAE decoding et FP8 réduisent le risque. Pas de gestion OOM gracieuse dans les pipelines (pas de try/catch autour de l'allocation VRAM, pas de fallback résolution inférieure).

### Comportement si timeout

✅ Le pipeline SaaS a des timeouts configurables (Cloud Run : 3600s max). 🔴 Pas de timeout côté inference locale.

### Comportement si corruption dataset

🟠 Les scripts de preprocessing valident les formats vidéo mais pas de checksum ou de vérification d'intégrité post-processing.

### Comportement si drift modèle

🔴 **Aucun monitoring de drift.** Aucune métrique de qualité collectée en continu. Le module `reward_modeling/` (~513 lignes) est structurel.

### Comportement si décroissance qualité

🔴 Pas de détection automatique. Pas de A/B testing fonctionnel (module existe mais non connecté).

### Capacité de monitoring

🟡 GPU monitoring via `nvidia-smi` scripts. Pas de monitoring applicatif Prometheus/Grafana. WandB configuré mais 0 run logged.

### Capacité de rollback

✅ Checkpoint manager JSON pour l'orchestrateur SaaS. 🔴 Pas de rollback modèle (pas de model registry, pas de versioning des poids).

### Versioning modèle

🔴 **Inexistant.** Pas de MLflow, pas de model registry, pas de versioning des checkpoints beyond le filesystem.

---

## 10. Failles critiques identifiées

### 🔴 Critique

1. **Le modèle fondamental n'est pas propriétaire.** L'intégralité du moteur de production (`aiprod_core`) est un fork de LTX-Video 2.0 (Lightricks) avec remplacement de chaînes. Les notices de copyright et licences ont été supprimées. Cela pose un risque juridique majeur et invalide l'argument commercial de "modèle propriétaire".

2. **Aucune capacité d'entraînement from scratch.** Le seul training fonctionnel est du fine-tuning LoRA sur un modèle pré-entraîné tiers. Le prototype `train.py` entraîne un toy model inutilisable. Aucun dataset propriétaire n'existe.

3. **Le SaaS déployé n'a pas de GPU.** Le Dockerfile Cloud Run déploie un orchestrateur CPU-only qui délègue le rendu à des APIs tierces (Veo-3, Runway, Replicate). Ce n'est pas un système d'inférence proprietaire.

4. **Gap massif entre le code réel et l'ambition.** ~62 000 lignes de code d'infrastructure dans `aiprod-pipelines/inference/` (SaaS multi-tenant, tensor parallelism, distributed LoRA, edge deployment, reward modeling) sont des **structures non connectées** : classes avec logique interne mais sans intégration au pipeline réel. Les inference nodes retournent `torch.randn()`.

5. **Risque juridique sur la propriété intellectuelle.** LTX-Video 2.0 est distribué sous licence Apache 2.0 qui requiert : préservation des notices de copyright, attribution dans les fichiers NOTICE, indication des modifications. **Aucune de ces obligations n'est respectée.**

6. **Aucune capacité TTS, lip-sync, montage, étalonnage, HDR.** Pour un produit qui "doit produire des vidéos cinématographiques 100% finalisées automatiquement", ces composants sont entièrement absents.

### 🟠 Majeur

7. **Dualité architecturale non résolue.** Deux systèmes coexistent sans lien : un pipeline local GPU (fonctionnel mais non déployé) et un SaaS Cloud Run (déployé mais sans GPU). Pas de stratégie claire sur lequel constitue le produit.

8. **Aucune trace d'exécution réelle.** Répertoire `logs/` vide, 0 run WandB, le seul checkpoint est un toy model de 152 MB. Aucune preuve que le système a jamais généré une vidéo avec le moteur de production.

9. **Tests unitaires incomplets pour le cœur.** `aiprod-core/tests/` ne contient qu'un `conftest.py` — les tests décrits dans le README n'existent pas. Les tests de `aiprod-pipelines` mockent `torch` entièrement, empêchant la validation GPU réelle.

10. **Inference nodes mockées.** Le système d'inference graph (`aiprod-pipelines/inference/nodes.py`) — censé être le cœur de l'exécution — retourne `torch.randn()` pour toutes les opérations (encode, denoise, decode, upsample). Les presets construisent des graphes sur des nodes factices.

11. **Monitoring et observabilité absents.** Pas de Prometheus, Grafana, alerting. Pas de model drift detection. Pas de quality metrics collection en production.

12. **Pas de versioning modèle.** Aucun MLflow, model registry, ou mécanisme de rollback des poids du modèle.

### 🟡 Mineur

13. **Monkey-patching dangereux.** `curriculum.py` injecte des fake modules dans `sys.modules` pour contourner `torch._dynamo`. Fragile et source potentielle de bugs silencieux.

14. **Config template dupliquée.** `config/templates/pyproject.template.toml` est une copie exacte du `pyproject.toml` racine — pas de templating réel.

15. **Répertoires scaffolding vides.** `scripts/data/`, `scripts/deployment/`, `scripts/dev/`, `scripts/maintenance/`, `scripts/testing/`, `deploy/kubernetes/`, `deploy/scripts/` — pure structure sans contenu.

16. **Pas de batching inference.** Chaque requête est traitée séquentiellement. Impact direct sur le throughput SaaS.

17. **Format d'export unique.** H.264 + AAC uniquement. Pas de ProRes, DNxHR, ou formats professionnels attendus pour un produit "cinématographique".

---

## 11. Recommandations prioritaires

### Top 5 corrections obligatoires avant tout lancement

1. **Résoudre le statut juridique du fork LTX-Video 2.0.** Soit restaurer les attributions Apache 2.0 et communiquer honnêtement que le modèle est basé sur LTX-Video, soit effectivement entraîner un modèle from scratch (coût : $5-20M). Ne pas lever de fonds en présentant un fork renommé comme un modèle propriétaire — c'est un risque de due diligence fatal.

2. **Choisir une architecture de déploiement unique.** Pipeline local GPU ou SaaS orchestrateur API ? Les deux approches sont légitimes mais incompatibles dans leur état actuel. Si SaaS avec GPU propre : budgetiser l'infrastructure. Si SaaS avec APIs tierces : assumer la dépendance et optimiser les coûts.

3. **Connecter les inference nodes.** Remplacer les `torch.randn()` dans `nodes.py` par les appels réels aux pipelines (`TI2VidTwoStagesPipeline`, etc.). Sans cela, le graph d'inférence (qui représente ~60% de la codebase) est non-fonctionnel.

4. **Implémenter les composants manquants pour le "cinématographique".** TTS, lip-sync, montage automatisé, étalonnage, color grading, export multi-format sont des prérequis non négociables pour la proposition de valeur annoncée.

5. **Exécuter le pipeline de bout en bout et sauvegarder les preuves.** Générer des vidéos avec le moteur réel, logger les métriques, collecter des benchmarks de qualité/latence/coût. Sans cela, toute démonstration ou levée repose sur du fictif.

### Optimisations court terme (0-3 mois)

- Implémenter le batching inference
- Ajouter TensorRT/ONNX export pour réduction latence 2-5×
- Configurer Prometheus + Grafana pour monitoring GPU/inference
- Écrire les tests unitaires du core (`aiprod-core/tests/`)
- Nettoyer les ~62 000 lignes de code infrastructure non connecté

### Stratégie compute moyen terme (3-12 mois)

- Si modèle propriétaire : constituer un dataset vidéo/audio sous licence ($500K-2M), lancer l'entraînement sur cluster A100/H100 ($2-5M)
- Si fork LTX-Video (honnête) : développer des LoRA spécialisées de haute qualité comme différenciateur, investir dans le pipeline de post-production
- Implémenter la distillation pour réduire le coût inference de 3-5×
- Déployer sur une infrastructure GPU auto-scalable (GKE avec GPU nodes, ou RunPod/Lambda)

### Stratégie R&D long terme (12-36 mois)

- Développer un modèle architectural réellement propriétaire si le positionnement l'exige
- Investir dans TTS/lip-sync propriétaire pour la différenciation
- Construire un pipeline de montage/étalonnage automatisé end-to-end
- Développer des capacités multi-scènes avec cohérence narrative
- Implémenter l'A/B testing et le reward modeling pour l'amélioration continue

---

## 12. Score final

| Critère | Score | Justification |
|---------|-------|---------------|
| **Solidité modèle** | 2/10 | Pas de modèle propriétaire. Fork renommé sans attribution. Aucun training from scratch. Toy model non fonctionnel. |
| **Solidité infrastructure** | 3/10 | Pipeline local fonctionnel hérité de LTX-Video. SaaS orchestrateur sans GPU. ~62K lignes d'infrastructure non connectée. Monitoring absent. |
| **Viabilité économique** | 2/10 | Pas de modèle propre = pas de moat technologique. Dépendance totale sur modèles/APIs tiers. Coûts training from scratch : $5-20M. Coûts infra GPU SaaS : $50-200K/an minimum. |

### Probabilité que le SaaS survive 12 mois sans levée massive

**< 5%** dans la configuration actuelle.

- Sans modèle propriétaire, le produit est un wrapper sur LTX-Video 2.0 que n'importe qui peut reproduire
- Sans GPU dans le déploiement, le SaaS dépend d'APIs tierces dont les coûts et la disponibilité ne sont pas contrôlés
- La dette technique (62K lignes non connectées, tests manquants, composants critiques absents) nécessite 6-12 mois de travail d'ingénierie avant un MVP crédible
- Le risque juridique sur le fork non attribué est un showstopper pour toute due diligence

### Verdict

> 👉 **Irréaliste sans capitaux massifs ET réorientation stratégique fondamentale.**
>
> AIPROD dans son état actuel est un **fork renommé de LTX-Video 2.0 enveloppé dans une couche d'orchestration SaaS partiellement implémentée**. Il ne possède aucun modèle propriétaire, n'a jamais entraîné de modèle from scratch, et n'a produit aucune preuve d'exécution réelle du pipeline de bout en bout.
>
> Le projet présente une ambition de niveau Big Tech (modèles fondamentaux propriétaires, pipeline cinématographique end-to-end, SaaS mondial) sans les ressources correspondantes (pas de dataset, pas de compute, pas d'équipe ML visible, pas de modèle entraîné).
>
> La voie la plus réaliste vers un produit viable serait d'**assumer honnêtement la base LTX-Video 2.0**, se concentrer sur la **valeur ajoutée de l'orchestration et du post-processing**, et investir dans les **composants manquants** (TTS, montage, étalonnage) plutôt que de prétendre à un modèle propriétaire inexistant.

---

*Fin de l'audit — 14 février 2026*
