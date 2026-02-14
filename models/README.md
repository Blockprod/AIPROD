# Models - AIPROD

> **⚠️ PROPRIETARY - STRICTLY CONFIDENTIAL**  
> © 2026 Blockprod. All rights reserved.  
> Modèles propriétaires. Ne pas distribuer.

Gestion des checkpoints et modèles du projet.

## 📁 Structure

```
models/
├── cache/                   # Cache local
├── checkpoints/             # Snapshots du training
└── pretrained/              # Modèles pré-entraînés
```

## 📥 Accès aux modèles

> **CONFIDENTIEL** — Contacter l'équipe Blockprod pour l'accès aux modèles et checkpoints.

## 💾 Checkpoint Management

```python
from aiprod_pipelines.api.checkpoint.manager import CheckpointManager

mgr = CheckpointManager({"storage_path": "./models/checkpoints"})
mgr.save_checkpoint(context)
```

---

*© 2026 Blockprod. All rights reserved. Proprietary and confidential.*
