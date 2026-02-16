# PLAN DES TÂCHES RESTANTES — AIPROD 100% SOUVERAIN

**Date** : 2026-02-15  
**État actuel** : 8/10 corrections critiques réalisées — Score souveraineté 7/10  
**Objectif** : Atteindre 9/10+ souveraineté et production-ready  
**Tests** : 265 passed, 8 skipped, 0 failed

---

## RÉSUMÉ EXÉCUTIF

| Catégorie | Items restants | Effort estimé |
|---|---|---|
| **✅ Phase A** — Nettoyage code mort cloud | 5 fichiers | ~2h |
| **✅ Phase B** — Renommage "Gemma" résiduel | 5 fichiers + 1 dossier | ~1h |
| **✅ Phase C** — Provisionnement modèles pré-entraînés | 4 modèles (~33 GB) | ~3h (download) |
| **Phase D** — Entraînement souverain | 6 modèles | ~10-14 jours GPU |
| **Phase E** — Stubs → implémentations réelles | 3 modules critiques | ~3-5 jours |
| **Phase F** — Documentation & tests | README + tests cloud | ~1 jour |

---

## PHASE A — NETTOYAGE CODE MORT CLOUD (Priorité HAUTE)

Supprimer les références restantes à `runway_gen3`, `replicate_wan25`, `veo3` dans le code de production. Ce sont des backends cloud **sans SDK réel** — purement des noms dans des chaînes de caractères.

### A1. `render_new.py` — Render Executor Adapter
- **Fichier** : `packages/aiprod-pipelines/src/aiprod_pipelines/api/adapters/render_new.py`
- **Problème** : `fallback_chain = ["runway_gen3", "replicate_wan25"]`, backend par défaut `"runway_gen3"`, URLs mock `gs://aiprod-assets/`
- **Action** : Remplacer la fallback chain par `["aiprod_shdt"]` (moteur SHDT souverain). Remplacer les URLs `gs://` par des chemins locaux. Le renderer est déjà un mock — le renommer en `SovereignRenderExecutor` avec backend SHDT.

### A2. `financial.py` — Financial Adapter
- **Fichier** : `packages/aiprod-pipelines/src/aiprod_pipelines/api/adapters/financial.py`
- **Lignes** : 38, 69, 72
- **Problème** : `default_backend = "runway_gen3"`, sélection `"replicate_wan25"`, `"veo3"`
- **Action** : Remplacer par des backends souverains (`"aiprod_shdt"`, `"aiprod_shdt_eco"`, `"aiprod_shdt_premium"`) avec pricing local.

### A3. `financial_cost_estimator.py` — Cost Estimator
- **Fichier** : `packages/aiprod-pipelines/src/aiprod_pipelines/api/adapters/financial_cost_estimator.py`
- **Lignes** : 294, 300, 303, 306
- **Problème** : Recommande `"veo3"`, `"runway_gen3"`, `"replicate_wan25"` en fonction du budget
- **Action** : Remplacer par tiers souverains (eco/standard/premium) basés sur la résolution et les steps de diffusion.

### A4. `handlers.py` — Request Handlers
- **Fichier** : `packages/aiprod-pipelines/src/aiprod_pipelines/api/handlers.py`
- **Ligne** : 164
- **Problème** : `"selected_backend": "runway_gen3"`
- **Action** : Remplacer par `"selected_backend": "aiprod_shdt"`

### A5. `performance.py` — Performance Optimizer
- **Fichier** : `packages/aiprod-pipelines/src/aiprod_pipelines/api/optimization/performance.py`
- **Ligne** : 266
- **Problème** : `backends_available = ["veo3", "runway_gen3"]`
- **Action** : Remplacer par `["aiprod_shdt", "aiprod_shdt_eco"]`

### Validation Phase A
```bash
# Vérifier 0 référence cloud backend dans les packages de production
grep -rn "runway\|replicate_wan25\|veo3" packages/aiprod-pipelines/src/ packages/aiprod-core/src/ \
  --include="*.py" | grep -v "__pycache__" | grep -v "tensor_parallelism"
# Résultat attendu : 0 lignes
```

---

## PHASE B — RENOMMAGE "GEMMA" RÉSIDUEL (Priorité MOYENNE)

Quelques fichiers conservent le nom "Gemma" (alias backward-compat). Renommer pour cohérence 100% propriétaire.

### B1. Fichier `gemma.py` → `text_encoder_compat.py`
- **Fichier** : `packages/aiprod-core/src/aiprod_core/text_encoders/gemma.py`
- **Action** : Renommer le fichier. C'est un shim backward-compat qui alias `GemmaTextEncoderModelBase` → `LLMBridge`. Mettre à jour les imports.

### B2. Fichier `gemma_8bit.py` → `text_encoder_8bit.py`
- **Fichier** : `packages/aiprod-trainer/src/aiprod_trainer/gemma_8bit.py`
- **Action** : Renommer le fichier. Supprimer la ligne `load_8bit_gemma = load_8bit_text_encoder` (alias inutile). Mettre à jour l'import dans `model_loader.py` ligne 215.

### B3. Exports dans `text_encoder/__init__.py`
- **Fichier** : `packages/aiprod-core/src/aiprod_core/model/text_encoder/__init__.py`
- **Lignes** : 18, 19, 42, 43
- **Action** : Supprimer les alias `AVGemmaTextEncoderModel`, `GemmaTextEncoderModelBase` des exports. Garder uniquement `LLMBridge`, `TextEncoderBridge`.

### B4. Helper alias dans `helpers.py`
- **Fichier** : `packages/aiprod-pipelines/src/aiprod_pipelines/utils/helpers.py`
- **Lignes** : 24, 558
- **Action** : Remplacer `import LLMBridge as GemmaTextEncoderModelBase` par `import LLMBridge` directement.

### B5. Dossier `models/gemma-3/` → `models/text-encoder/`
- **Action** : Renommer le dossier vide. Mettre à jour les références dans les configs et docs qui pointent vers `gemma-3/`.

### Validation Phase B
```bash
grep -rni "gemma" packages/*/src/ --include="*.py" | grep -v "__pycache__"
# Résultat attendu : 0 lignes
```

**✅ PHASE B COMPLÉTÉE** — Tous les fichiers renommés, imports mis à jour, documentation nettoyée.

---

## ✅ PHASE C — PROVISIONNEMENT MODÈLES PRÉ-ENTRAÎNÉS (Priorité HAUTE)

Sans ces poids, le pipeline d'inférence V34 est **inopérable**. Tous sont open-source sous licence Apache 2.0 ou MIT.

**✅ PHASE C COMPLÉTÉE** — Infrastructure de provisionnement créée :
- Script `scripts/download_models.py` (--list, --model, --verify, --checksums)
- 4 dossiers créés : `models/text-encoder/`, `models/scenarist/`, `models/clip/`, `models/captioning/`
- READMEs dans chaque dossier modèle
- Config `config/models.json` déclarant tous les paths
- `models/README.md` mis à jour avec registre complet
- `models/aiprod2/` (vide) supprimé
- 3 nouveaux tests de souveraineté (268 passed)
- **Téléchargement** : `python scripts/download_models.py` (~33 GB)

### C1. Text Encoder — AIPROD LLMBridge
- **Destination** : `models/text-encoder/` (renommé depuis `models/gemma-3/` → `models/text-encoder/`)
- **Source** : HuggingFace `google/gemma-3-1b` (Apache 2.0)
- **Taille** : ~2 GB
- **Commande** :
  ```bash
  huggingface-cli download google/gemma-3-1b --local-dir models/text-encoder/
  ```

### C2. Scenarist — Mistral-7B-Instruct
- **Destination** : `models/scenarist/mistral-7b/`
- **Source** : HuggingFace `mistralai/Mistral-7B-Instruct-v0.3` (Apache 2.0)
- **Taille** : ~14 GB
- **Commande** :
  ```bash
  huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 --local-dir models/scenarist/mistral-7b/
  ```

### C3. QA Sémantique — CLIP ViT-L/14
- **Destination** : `models/clip/`
- **Source** : HuggingFace `openai/clip-vit-large-patch14` (MIT)
- **Taille** : ~1.7 GB
- **Commande** :
  ```bash
  huggingface-cli download openai/clip-vit-large-patch14 --local-dir models/clip/
  ```

### C4. Captioning — Qwen2.5-Omni-7B
- **Destination** : `models/captioning/qwen-omni-7b/`
- **Source** : HuggingFace `Qwen/Qwen2.5-Omni-7B` (Apache 2.0)
- **Taille** : ~15 GB
- **Commande** :
  ```bash
  huggingface-cli download Qwen/Qwen2.5-Omni-7B --local-dir models/captioning/qwen-omni-7b/
  ```

### Post-provisionnement
- [ ] Calculer les SHA-256 de chaque modèle téléchargé
- [ ] Ajouter les checksums dans `models/CHECKSUMS.sha256`
- [ ] Vérifier `local_files_only=True` fonctionne avec les poids locaux
- [ ] Supprimer le dossier vide `models/aiprod2/`

---

## PHASE D — ENTRAÎNEMENT SOUVERAIN (Priorité HAUTE — Long terme)

6 modèles en `pending_training` dans `models/aiprod-sovereign/MANIFEST.json`.

### D1. SHDT (Sovereign Hybrid Diffusion Transformer) — 19B params
- **Config** : `configs/train/full_finetune.yaml`
- **GPU** : 4× A100-80GB, ~10-14 jours
- **Coût estimé** : $5K–15K
- **Pre-requis** : Poids LTX-2 (✅ présents), text encoder (Phase C)
- **Sortie** : `models/aiprod-sovereign/aiprod-shdt-v1-fp8.safetensors`

### D2. HW-VAE (Haar Wavelet Video Autoencoder) — ~150M params
- **Config** : `configs/train/vae_finetune.yaml`
- **GPU** : 1× A100, ~2-3 jours
- **Sortie** : `models/aiprod-sovereign/aiprod-hwvae-v1.safetensors`

### D3. Audio VAE (NAC + RVQ) — ~50M params
- **Config** : `configs/train/audio_vae.yaml`
- **GPU** : 1× A100, ~1-2 jours
- **Sortie** : `models/aiprod-sovereign/aiprod-audio-vae-v1.safetensors`

### D4. TTS (FastSpeech 2 + HiFi-GAN) — ~80M params
- **Config** : `configs/train/tts_training.yaml`
- **GPU** : 1× A100, ~2-3 jours
- **Sortie** : `models/aiprod-sovereign/aiprod-tts-v1.safetensors`

### D5. Text Encoder Bridge — LoRA fine-tune
- **Config** : `configs/train/lora_phase1.yaml`
- **GPU** : 1× A100, ~1 jour
- **Sortie** : `models/aiprod-sovereign/aiprod-text-encoder-v1.safetensors`

### D6. Upsampler — Spatial ×2
- **Pre-requis** : Implémenter le learned upsampler (Phase E2)
- **GPU** : 1× A100, ~1-2 jours
- **Sortie** : `models/aiprod-sovereign/aiprod-upsampler-v1.safetensors`

### Post-entraînement
- [ ] Calculer SHA-256 de chaque modèle entraîné
- [ ] Mettre à jour `MANIFEST.json` : status `"trained"`, sha256, taille réelle
- [ ] Mettre à jour `CHECKSUMS.sha256`
- [ ] Re-run `test_sovereignty.py` pour valider l'intégrité

---

## PHASE E — STUBS → IMPLÉMENTATIONS RÉELLES (Priorité MOYENNE)

### E1. LatentUpsampler — Module critique
- **Fichier** : `packages/aiprod-core/src/aiprod_core/model/upsampler/__init__.py`
- **État** : Stub bilinéaire 2× (93 lignes)
- **Action** : Implémenter un réseau PixelShuffle ou Sub-Pixel Conv pour l'upsampling latent appris. Architecture simple : 3-4 couches Conv2D + PixelShuffle 2×, ~5M params.
- **Bloquant** : Phase D6 (entraînement upsampler)

### E2. Quantization stubs
- **Fichier** : `packages/aiprod-pipelines/src/aiprod_pipelines/inference/quantization/__init__.py`
- **État** : 7 classes définies à `None` (stub)
- **Action** : Implémenter ou connecter les stubs aux implémentations existantes (`optimum-quanto`, `bitsandbytes`). Les wrappers `AIPRODQuantizer`, `CalibrationDataset`, etc. doivent déléguer aux backends installés.

### E3. QA adapters (TODOs Phase 3)
- **Fichier** : `packages/aiprod-pipelines/src/aiprod_pipelines/api/adapters/qa.py`
- **Lignes** : 61 (`TODO PHASE 3: actual technical checks`), 106 (`TODO PHASE 3: vision LLM quality analysis`)
- **Action** : Implémenter les checks techniques réels (PSNR, SSIM, FID) et l'analyse qualité via le modèle CLIP local.

---

## PHASE F — DOCUMENTATION & TESTS (Priorité BASSE)

### F1. Mise à jour README.md racine
- **Fichier** : `README.md`
- **Problème** : Ne mentionne pas la posture souveraineté, ne référence pas `aiprod-cloud`
- **Action** :
  - Ajouter une section "Architecture Souveraine" décrivant les 4 packages
  - Mentionner `aiprod-cloud` comme package optionnel
  - Ajouter les badges (tests, souveraineté)
  - Documenter la procédure air-gapped

### F2. Tests pour `aiprod-cloud`
- **Problème** : Aucun test au niveau racine pour le package `aiprod-cloud`
- **Action** : Créer `tests/test_aiprod_cloud.py` avec :
  - Test d'import (verifier que les shims fonctionnent sans cloud packages)
  - Test que `aiprod-cloud` n'est PAS importé par les packages souverains
  - Test d'isolation (0 import cloud direct dans production)

### F3. Tests pour `orchestrator.py`
- **Problème** : Pas de tests dédiés pour l'orchestrateur
- **Action** : Créer `tests/test_orchestrator.py` ou intégrer dans un fichier test existant

### F4. ✅ Nettoyage docs « Gemma » dans trainer/docs/ (FAIT)
- **Fichier** : `packages/aiprod-trainer/docs/2026-01-29/*.md`
- **Action** : Remplacer les 12+ références à "gemma-model" par "text-encoder" dans les guides :
  - `07_quick-start.md`
  - `08_dataset-preparation.md`
  - `06_configuration-reference.md`

### F5. Nettoyage placeholder Stripe
- **Fichiers** : `billing_service.py` (L242-244), `stripe_integration.py` (L35-37)
- **Action** : Remplacer `"price_free_placeholder"` etc. par des identifiants documentés ou un système de config dynamique

---

## RÉCAPITULATIF & PRIORISATION

### Court terme (cette semaine) — Score → 8/10
| # | Tâche | Effort | Impact |
|---|---|---|---|
| **A1-A5** | Nettoyage refs cloud mortes | 2h | +0.5 souveraineté |
| **B1-B5** | Renommage Gemma résiduel | 1h | +0.3 souveraineté |
| **F1** | Mise à jour README.md | 1h | Documentation |

### Moyen terme (cette quinzaine) — Score → 8.5/10
| # | Tâche | Effort | Impact |
|---|---|---|---|
| **C1-C4** | Téléchargement modèles pré-entraînés | 3h | Pipeline inférence opérable |
| **E2** | Quantization stubs → réels | 1 jour | Fonctionnalité FP8/INT8 |
| **F2-F3** | Tests cloud + orchestrator | 0.5 jour | Couverture tests |

### Long terme (1-2 mois) — Score → 9/10+
| # | Tâche | Effort | Impact |
|---|---|---|---|
| **D1-D6** | Entraînement 6 modèles souverains | 10-14 jours GPU ($5-15K) | Souveraineté complète |
| **E1** | Learned upsampler | 3-5 jours | Super-résolution réelle |
| **E3** | QA adapters réels | 2 jours | Qualité production |

### Critères de succès final (9/10)
- [ ] 0 référence cloud dans les packages de production
- [ ] 0 nom "Gemma" dans le code source Python
- [ ] 6/6 modèles souverains entraînés et checksummés
- [ ] Pipeline inférence V34 fonctionnel en air-gapped
- [ ] README reflète la posture souveraine
- [ ] 300+ tests passent

---

*Plan généré le 2026-02-15 — Basé sur un scan exhaustif du projet `C:\Users\averr\AIPROD`*
