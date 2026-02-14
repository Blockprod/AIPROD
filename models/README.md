# Models - AIPROD

Gestion des checkpoints et modèles du projet.

## 📁 Structure

```
models/
├── cache/                   # Modèles téléchargés pour cache local
│   ├── gemini/
│   ├── veo3/
│   └── runway/
├── checkpoints/             # Snapshots du training
│   ├── phase_0/
│   ├── phase_1/
│   └── latest.pt
└── pretrained/              # Modèles pré-entraînés
    ├── AIPROD-19b-dev.safetensors
    ├── spatial-upscaler-x2.safetensors
    └── README.md
```

## 📥 Modèles Pré-requis

### AIPROD Model
```bash
# Télécharger depuis HuggingFace
huggingface-cli download Lightricks/AIPROD \
  --local-dir models/pretrained/ \
  --repo-type model
```

**Variantes disponibles:**
- `AIPROD-19b-dev.safetensors` (full precision, 40GB)
- `AIPROD-19b-dev-fp8.safetensors` (quantized, 20GB)
- `AIPROD-19b-distilled.safetensors` (distilled, 10GB)

### Upscalers
```bash
# Spatial Upscaler
wget https://huggingface.co/Lightricks/AIPROD/resolve/main/AIPROD-spatial-upscaler-x2-1.0.safetensors \
  -O models/pretrained/spatial-upscaler-x2-1.0.safetensors

# Temporal Upscaler
wget https://huggingface.co/Lightricks/AIPROD/resolve/main/AIPROD-temporal-upscaler-x2-1.0.safetensors \
  -O models/pretrained/temporal-upscaler-x2-1.0.safetensors
```

## 💾 Checkpoint Management

### Sauvegarder un checkpoint
```python
from aiprod_pipelines.api.checkpoint.manager import CheckpointManager

mgr = CheckpointManager({"storage_path": "./models/checkpoints"})
mgr.save_checkpoint(context)
```

### Restaurer un checkpoint
```python
restored_context = mgr.restore_checkpoint("request_id")
```

## 🗑️ Nettoyage

```bash
# Supprimer les anciens checkpoints
find models/checkpoints -mtime +30 -delete

# Vider le cache local
rm -rf models/cache/*
```

## 📊 Espace disque

| Type | Taille | Location |
|------|--------|----------|
| AIPROD 19b | 40GB | models/pretrained/ |
| AIPROD FP8 | 20GB | models/pretrained/ |
| Upscalers | 5GB | models/pretrained/ |
| Cache local | ~50GB | models/cache/ |
| Checkpoints | Variable | models/checkpoints/ |

**Total estimé**: 100-150GB (dépend de configuration)

---

*Created: 2026-02-10*
