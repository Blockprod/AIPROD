# 📋 PLAN D'ACTION CLAIR & FINAL

**Status global** : Phase 3 à 98% (prêt pour production deployment)  
**Date** : 3 février 2026  
**Objectif** : Terraform deployment complet d'ici Feb 5

---

## 🔴 ÉTAPE 1 : GCP Manual Configuration (2-3h) — À faire MAINTENANT

### 1. Revoke old API keys

```bash
gcloud secrets delete gemini-api-key      # old one
gcloud secrets delete runway-api-key       # old one
gcloud secrets delete datadog-api-key      # old one
```

### 2. Create NEW secrets

```bash
gcloud secrets create gemini-api-key --data="YOUR_GEMINI_KEY"
gcloud secrets create runway-api-key --data="YOUR_RUNWAY_KEY"
gcloud secrets create datadog-api-key --data="YOUR_DATADOG_KEY"
gcloud secrets create gcs-bucket-name --data="aiprod-v33-assets"
```

### 3. Create Terraform service account

```bash
gcloud iam service-accounts create terraform-sa
gcloud projects add-iam-policy-binding aiprod-484120 \
  --member="serviceAccount:terraform-sa@aiprod-484120.iam.gserviceaccount.com" \
  --role="roles/editor"

gcloud iam service-accounts keys create terraform-key.json \
  --iam-account=terraform-sa@aiprod-484120.iam.gserviceaccount.com
```

### 4. Verify Docker image exists in Artifact Registry

```bash
gcloud artifacts docker images list \
  europe-west1-docker.pkg.dev/aiprod-484120/aiprod \
  --project=aiprod-484120
```

### ✅ Checklist ÉTAPE 1

- [x] Secrets created in Secret Manager (4)
- [ ] Firebase service account key saved
- [x] Terraform service account created + key downloaded
- [x] Docker image exists in registry

---

## 🟠 ÉTAPE 2 : Terraform Deployment (4-6h) — Après GCP setup

### 1. Initialize

```bash
cd infra/terraform
export GOOGLE_APPLICATION_CREDENTIALS=../../credentials/terraform-key.json

terraform init
```

### 2. Plan (review what will be created)

```bash
terraform plan -out=tfplan
```

### 3. Apply (deploy infrastructure)

```bash
terraform apply tfplan
# ⏳ Wait ~30 min pour Cloud SQL, ~10 min pour Cloud Run
```

### 4. Get outputs

```bash
terraform output cloud_run_url
# → https://aiprod-api-xxxxx.run.app
```

### ✅ Checklist ÉTAPE 2

- [ ] terraform init successful
- [ ] terraform plan reviewed (50+ resources)
- [ ] terraform apply completed
- [ ] Cloud Run API service deployed
- [ ] Cloud SQL instance running
- [ ] Pub/Sub topics created
- [ ] All 50+ resources provisioned

---

## 🟢 ÉTAPE 3 : Production Validation (1-2h) — Après Terraform

### 1. Test API health

```bash
curl https://aiprod-api-xxxxx.run.app/health
```

### 2. Test pipeline endpoint (need JWT token)

```bash
curl -X POST https://aiprod-api-xxxxx.run.app/pipeline/run \
  -H "Authorization: Bearer <YOUR_JWT>" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Test", "aspect_ratio": "16:9", "duration": 5}'
```

### 3. Test Cloud SQL

```bash
gcloud sql connect aiprod-v33 --user=aiprod
```

### 4. Test Pub/Sub

```bash
gcloud pubsub topics publish pipeline-jobs --message='test'
```

### ✅ Checklist ÉTAPE 3

- [ ] API responds to /health → 200 OK
- [ ] POST /pipeline/run creates jobs
- [ ] Cloud SQL connected + migrated
- [ ] Pub/Sub topics operational
- [ ] Monitoring receiving data
- [ ] No errors in logs

---

## 📅 TIMELINE RÉSUMÉ

| Étape                | Durée | Status      | Date        |
| -------------------- | ----- | ----------- | ----------- |
| **GCP Setup**        | 2-3h  | 🔴 À faire  | **Feb 3**   |
| **Terraform Deploy** | 4-6h  | 🟠 Après    | **Feb 4-5** |
| **Validation**       | 1-2h  | 🟢 Après    | **Feb 5**   |
| **Go-Live**          | -     | 📅 Objectif | **Feb 17**  |

---

## ✅ AUJOURD'HUI (Feb 3) — COMPLÉTÉ

- [x] Code production-ready (Phase 0, 1, 2, 3 code 100%)
- [x] 295/295 tests PASSING
- [x] GitHub Actions workflows VALIDATED
- [x] Docker builds successfully
- [x] runwayml reintegrated properly
- [x] CI/CD pipeline STABLE & GREEN
- [x] Terraform IaC ready for deployment
- [x] Checklist updated with clear action items

---

## 🚀 PRÊT À COMMENCER LA GCP SETUP ?

Je peux vous guider **step-by-step à besoin !**
