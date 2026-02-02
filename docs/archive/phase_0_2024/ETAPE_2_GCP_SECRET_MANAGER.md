# ✅ ÉTAPE 2 - GCP SECRET MANAGER SETUP

**Date**: 2 Février 2026  
**Statut**: EN COURS  
**Durée Estimée**: 1-1.5 heures  
**Owner**: DevOps Engineer  
**Tool**: gcloud CLI (DÉJÀ INSTALLÉ ✅)

---

## 📋 RÉSUMÉ ÉTAPE 2

**Objectif**: Créer 5 secrets dans GCP Secret Manager pour que main.py puisse les charger au démarrage.

**Secrets à Créer**:

1. GEMINI_API_KEY
2. RUNWAY_API_KEY
3. DATADOG_API_KEY
4. DATADOG_APP_KEY
5. GCS_BUCKET_NAME

**Configuration Requise**:

- ✅ gcloud CLI: v551.0.0 (INSTALLÉ)
- ✅ Projet: aiprod-484120 (CONFIGURÉ)
- ⏳ Authentification gcloud (À VÉRIFIER)

---

## 🔐 STEP 2.1: Vérifier Authentification GCP

**C'est quoi?** Vérifier que vous êtes connecté à GCP avec les bonnes permissions.

**Commande**:

```powershell
gcloud auth list
```

**Résultat Attendu**:

```
                  Credentialed Accounts
ACTIVE  ACCOUNT
*       your-email@gmail.com or your-email@company.com
```

Si **pas d'account** → Run:

```powershell
gcloud auth login
```

---

## 🔐 STEP 2.2: Créer les 5 Secrets dans Secret Manager

**C'est quoi?** Créer les "conteneurs" vides pour les secrets.

**Commandes** (copier/coller dans PowerShell):

```powershell
# 1. Gemini API Key
gcloud secrets create GEMINI_API_KEY --replication-policy="automatic"

# 2. Runway API Key
gcloud secrets create RUNWAY_API_KEY --replication-policy="automatic"

# 3. Datadog API Key
gcloud secrets create DATADOG_API_KEY --replication-policy="automatic"

# 4. Datadog APP Key
gcloud secrets create DATADOG_APP_KEY --replication-policy="automatic"

# 5. GCS Bucket Name
gcloud secrets create GCS_BUCKET_NAME --replication-policy="automatic"
```

**Résultat Attendu** (après chaque commande):

```
Created secret [GEMINI_API_KEY] with replication policy AUTOMATIC.
```

**Vérifier** que les 5 secrets ont bien été créés:

```powershell
gcloud secrets list
```

**Doit afficher** (environ):

```
NAME                    CREATED             REPLICATION_POLICY
DATADOG_API_KEY         2026-02-02T15:30:00  automatic
DATADOG_APP_KEY         2026-02-02T15:30:00  automatic
GEMINI_API_KEY          2026-02-02T15:30:00  automatic
GCS_BUCKET_NAME         2026-02-02T15:30:00  automatic
RUNWAY_API_KEY          2026-02-02T15:30:00  automatic
```

---

## 🔐 STEP 2.3: Ajouter les Valeurs aux Secrets

**C'est quoi?** Mettre les vraies valeurs dans chaque secret.

**Valeurs à Utiliser** (des fichier `.env` du projet):

```powershell
# Lire les valeurs du .env local
cat .env | Select-String "GEMINI_API_KEY|RUNWAY_API_KEY|DD_API_KEY|GCS_BUCKET_NAME"
```

**Puis ajouter chaque valeur** (exécuter chaque ligne):

```powershell
# 1. GEMINI API Key (valeur du .env ligne 13)
echo "AIzaSyAUdogIIbGavH9gvZi7SvteGKcdfz9tRbw" | gcloud secrets versions add GEMINI_API_KEY --data-file=-

# 2. RUNWAY API Key (valeur du .env ligne 16)
echo "key_50d32d6432d622ec0c7c95f1aa0a68cf781192bd531ff1580c3f4853755c5edba0b52fb49426d07aa6b4356e505ab6e1b80987b501aa08f37000fa51f76796b7" | gcloud secrets versions add RUNWAY_API_KEY --data-file=-

# 3. DATADOG API Key (valeur du .env ligne 33)
echo "f987c9c2933619d8df6f928121549394" | gcloud secrets versions add DATADOG_API_KEY --data-file=-

# 4. DATADOG APP Key (à trouver dans .env)
# Chercher: DD_APP_KEY=... dans .env
echo "<VALUE_FROM_ENV>" | gcloud secrets versions add DATADOG_APP_KEY --data-file=-

# 5. GCS Bucket Name (valeur du .env)
# Chercher: GCS_BUCKET_NAME=... dans .env
echo "aiprod-484120-assets" | gcloud secrets versions add GCS_BUCKET_NAME --data-file=-
```

**Résultat Attendu** (après chaque commande):

```
Created secret version [1] for secret [GEMINI_API_KEY].
```

**Vérifier** que toutes les valeurs sont bien sauvegardées:

```powershell
gcloud secrets versions list GEMINI_API_KEY
gcloud secrets versions list RUNWAY_API_KEY
gcloud secrets versions list DATADOG_API_KEY
gcloud secrets versions list DATADOG_APP_KEY
gcloud secrets versions list GCS_BUCKET_NAME
```

---

## 🔐 STEP 2.4: Configurer IAM Permissions

**C'est quoi?** Donner à Cloud Run le droit de lire les secrets.

**À Faire** (une seule fois):

```powershell
# 1. Vérifier si le service account existe
gcloud iam service-accounts describe aiprod-sa@aiprod-484120.iam.gserviceaccount.com --quiet 2>$null
# Si erreur → créer:
gcloud iam service-accounts create aiprod-sa --display-name="AIPROD Service Account"

# 2. Donner permissions sur CHAQUE secret
gcloud secrets add-iam-policy-binding GEMINI_API_KEY `
  --member=serviceAccount:aiprod-sa@aiprod-484120.iam.gserviceaccount.com `
  --role=roles/secretmanager.secretAccessor

gcloud secrets add-iam-policy-binding RUNWAY_API_KEY `
  --member=serviceAccount:aiprod-sa@aiprod-484120.iam.gserviceaccount.com `
  --role=roles/secretmanager.secretAccessor

gcloud secrets add-iam-policy-binding DATADOG_API_KEY `
  --member=serviceAccount:aiprod-sa@aiprod-484120.iam.gserviceaccount.com `
  --role=roles/secretmanager.secretAccessor

gcloud secrets add-iam-policy-binding DATADOG_APP_KEY `
  --member=serviceAccount:aiprod-sa@aiprod-484120.iam.gserviceaccount.com `
  --role=roles/secretmanager.secretAccessor

gcloud secrets add-iam-policy-binding GCS_BUCKET_NAME `
  --member=serviceAccount:aiprod-sa@aiprod-484120.iam.gserviceaccount.com `
  --role=roles/secretmanager.secretAccessor
```

**Résultat Attendu**:

```
Updated IAM policy for secret [GEMINI_API_KEY].
Updated IAM policy for secret [RUNWAY_API_KEY].
...
```

---

## 🔐 STEP 2.5: Tester l'Accès aux Secrets

**C'est quoi?** Vérifier que les secrets ont bien été créés et sont accessibles.

**Test 1: Lire un secret**:

```powershell
gcloud secrets versions access latest --secret="GEMINI_API_KEY"
```

**Doit afficher**:

```
AIzaSyAUdogIIbGavH9gvZi7SvteGKcdfz9tRbw
```

**Test 2: Lister tous les secrets**:

```powershell
gcloud secrets list --format="table(name,created,replication.automatic)"
```

**Doit afficher tous les 5 secrets** avec `automatic` replication.

---

## ✅ CHECKLIST ÉTAPE 2

- [ ] **Step 2.1**: Authentification gcloud vérifiée
- [ ] **Step 2.2**: 5 secrets créés dans GCP (gcloud secrets list)
- [ ] **Step 2.3**: Toutes les valeurs ajoutées (5 versions créées)
- [ ] **Step 2.4**: IAM permissions configurées pour aiprod-sa
- [ ] **Step 2.5**: Accès aux secrets testé (gcloud secrets versions access)

---

## 📍 Status de Progression

```
ÉTAPE 2: P0.1.2 - GCP Secret Manager Setup
├─ Step 2.1: Auth gcloud ..................... 🟡 À VÉRIFIER
├─ Step 2.2: Créer 5 secrets ................. 🟡 À FAIRE
├─ Step 2.3: Ajouter valeurs ................. 🟡 À FAIRE
├─ Step 2.4: IAM Permissions ................. 🟡 À FAIRE
└─ Step 2.5: Tester accès .................... 🟡 À FAIRE
```

---

## 🚀 APRÈS ÉTAPE 2

Une fois tous les checkboxes cochés ✅:

```
ÉTAPE 3: Intégrer auth dans main.py
├─ Ajouter imports (15 LOC)
├─ Ajouter startup hooks (20 LOC)
├─ Ajouter middleware (1 LOC)
├─ Protéger endpoints (10 LOC)
├─ Exception handlers (15 LOC)
└─ Test local (curl + pytest)
```

**Durée ÉTAPE 3**: 1-2 heures

---

## 📝 Notes

- **Secrets**: Stockés en GCP, jamais en local après ÉTAPE 2
- **Service Account**: Utilisé par Cloud Run pour accéder aux secrets
- **Replication**: "automatic" = répliqué dans toutes les régions GCP automatiquement
- **Sécurité**: Les secrets ne s'affichent JAMAIS dans les logs gcloud (sauf avec `versions access`)
