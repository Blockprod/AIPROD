# Configuration - AIPROD

Ce dossier centralise toute la configuration du projet AIPROD.

## 📁 Structure

```
config/
├── AIPROD.json              # Configuration principale du projet
├── env/                     # Configurations par environnement
│   ├── development.yaml     # Développement local
│   ├── staging.yaml         # Environnement de staging
│   └── production.yaml      # Production (GCP)
├── cloud/                   # Configuration GCP
│   ├── cloud-run.yaml       # Cloud Run deployment
│   ├── monitoring.yaml      # Cloud Monitoring
│   └── logging.yaml         # Cloud Logging
└── templates/               # Fichiers modèles
    └── pyproject.template.toml
```

## 🔧 Usage

### Configuration par environnement

```bash
# Développement
export ENV=development
export CONFIG_PATH=./config/env/development.yaml

# Staging
export ENV=staging
export CONFIG_PATH=./config/env/staging.yaml

# Production
export ENV=production
export CONFIG_PATH=./config/env/production.yaml
```

### GCP Configuration

Pour déployer sur Google Cloud Platform :

```bash
# Cloud Run deployment
gcloud run deploy aiprod-merger \
  --config=config/cloud/cloud-run.yaml
```

## 📝 Variables d'environnement

### Core
- `GCP_PROJECT_ID`: ID du projet GCP
- `BUCKET_NAME`: Nom du bucket Cloud Storage
- `LOG_LEVEL`: Niveau de logging (INFO, WARNING, ERROR)

### Services
- `GEMINI_API_KEY`: Clé API Gemini
- `REPLICATE_API_KEY`: Clé API Replicate (optionnel)

### Performance
- `CACHE_TTL_SECONDS`: TTL du cache (défaut: 86400)
- `MAX_WORKERS`: Nombre de workers (défaut: 4)

## 🔐 Sécurité

- Ne jamais commiter `.env` ou clés d'API
- Utiliser `config/env/development.yaml.example` comme template
- Les variables sensibles doivent être en env vars ou secrets manager

---

*Created: 2026-02-10*
