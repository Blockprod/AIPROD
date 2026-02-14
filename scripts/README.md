# Scripts - AIPROD

Scripts utilitaires organisés par catégorie.

## 📁 Structure

```
scripts/
├── deployment/              # Déploiement et infrastructure
│   ├── deploy_cloud_run.sh
│   └── deploy_kubernetes.sh
├── maintenance/             # Maintenance et nettoyage
│   ├── caption_videos.py
│   ├── compute_reference.py
│   └── cleanup.sh
├── testing/                 # Tests et validation
│   ├── load_test.py
│   ├── validate_production.py
│   └── integration_test.sh
├── data/                    # Traitement de données
│   ├── process_dataset.py
│   ├── process_videos.py
│   ├── process_captions.py
│   └── split_scenes.py
└── dev/                     # Développement
    ├── validate_streaming.py
    ├── setup_dev.sh
    └── generate_docs.sh
```

## 🚀 Quick Start

### Déploiement
```bash
# Déployer sur Cloud Run
bash scripts/deployment/deploy_cloud_run.sh
```

### Tests
```bash
# Test de charge
python scripts/testing/load_test.py --requests 100

# Validation de production
python scripts/testing/validate_production.py \
  --url https://aiprod-merger-xxx.run.app
```

### Maintenance
```bash
# Générer des captions
python scripts/maintenance/caption_videos.py \
  --input videos/ --output captions/
```

### Données
```bash
# Traiter un dataset
python scripts/data/process_dataset.py \
  --input raw/ --output processed/
```

---

*Created: 2026-02-10*
