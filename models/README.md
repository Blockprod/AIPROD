# Models — AIPROD

> **⚠️ PROPRIETARY — STRICTLY CONFIDENTIAL**  
> © 2026 Blockprod. All rights reserved.  
> Modèles propriétaires. Ne pas distribuer.

Gestion des poids de modèles et checkpoints du projet.
Tous les modèles fonctionnent **100% hors-ligne** (`local_files_only=True`).

## 📁 Structure

```
models/
├── text-encoder/            # Text encoder base weights (LLMBridge)
├── scenarist/mistral-7b/    # Storyboard LLM (Mistral-7B-Instruct)
├── clip/                    # CLIP ViT-L/14 — QA sémantique
├── captioning/qwen-omni-7b/ # Captioning audio-visuel (Qwen2.5-Omni)
├── aiprod-sovereign/        # Modèles entraînés souverains (Phase D)
├── ltx2_research/           # Poids de recherche LTX-2
├── cache/                   # Cache local
├── checkpoints/             # Snapshots de training
├── pretrained/              # Modèles pré-entraînés divers
└── CHECKSUMS.sha256         # Intégrité des poids
```

## 📥 Provisionnement

```bash
# Télécharger tous les modèles pré-entraînés (~33 GB)
python scripts/download_models.py

# Télécharger un modèle spécifique
python scripts/download_models.py --model text-encoder
python scripts/download_models.py --model scenarist
python scripts/download_models.py --model clip
python scripts/download_models.py --model captioning

# Voir le statut de provisionnement
python scripts/download_models.py --list

# Vérifier les checksums SHA-256
python scripts/download_models.py --verify
```

## 📋 Registre des modèles

| Modèle | Destination | Source | Licence | Taille |
|---|---|---|---|---|
| Text Encoder | `models/text-encoder/` | `google/gemma-3-1b-pt` | Apache 2.0 | ~2 GB |
| Scenarist LLM | `models/scenarist/mistral-7b/` | `mistralai/Mistral-7B-Instruct-v0.3` | Apache 2.0 | ~14 GB |
| CLIP QA | `models/clip/` | `openai/clip-vit-large-patch14` | MIT | ~1.7 GB |
| Captioning | `models/captioning/qwen-omni-7b/` | `Qwen/Qwen2.5-Omni-7B` | Apache 2.0 | ~15 GB |

## 💾 Checkpoint Management

```python
from aiprod_pipelines.api.checkpoint.manager import CheckpointManager

mgr = CheckpointManager({"storage_path": "./models/checkpoints"})
mgr.save_checkpoint(context)
```

---

*© 2026 Blockprod. All rights reserved. Proprietary and confidential.*
