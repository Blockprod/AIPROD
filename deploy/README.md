# Deployment - AIPROD

> **⚠️ PROPRIETARY - CONFIDENTIAL**  
> Infrastructure de déploiement propriétaire **Blockprod**.

Ce dossier contient tous les fichiers et scripts de déploiement.

## 📁 Structure

```
deploy/
├── docker/                  # Configuration Docker
│   ├── Dockerfile           # Image container Python 3.13
│   ├── .dockerignore        # Exclusions build
│   └── README.md
├── kubernetes/              # Configuration Kubernetes (optionnel)
│   ├── deployment.yaml
│   ├── service.yaml
│   └── ingress.yaml
└── scripts/                 # Scripts de déploiement
    ├── deploy.sh            # Déploiement automatisé
    ├── rollback.sh          # Rollback de version
    ├── validate.sh          # Validation post-deploy
    └── migrate.sh           # Migration de données
```

## 🐳 Docker Build

```bash
# Build de l'image
docker build -f deploy/docker/Dockerfile \
  -t gcr.io/PROJECT_ID/aiprod-merger:v1.0 .

# Push vers Container Registry
docker push gcr.io/PROJECT_ID/aiprod-merger:v1.0
```

## 🚀 Déploiement Cloud Run

```bash
# Déploiement automatisé
bash deploy/scripts/deploy.sh

# Déploiement manuel
gcloud run deploy aiprod-merger \
  --image gcr.io/PROJECT_ID/aiprod-merger:v1.0 \
  --platform managed \
  --region us-central1 \
  --memory 8Gi \
  --cpu 4 \
  --min-instances 1 \
  --max-instances 100
```

## 🔄 Rollback

```bash
# Retour à la version précédente
bash deploy/scripts/rollback.sh

# Vérifier la révision active
gcloud run revisions list --service aiprod-merger
```

## ✅ Validation Post-Déploiement

```bash
# Validation automatisée
bash deploy/scripts/validate.sh

# Health check manuel
curl https://aiprod-merger-xxx.run.app/health
```

---

*© 2026 Blockprod. All rights reserved.*
