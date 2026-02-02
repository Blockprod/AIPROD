---
# ⚡ PHASE 0 - PLAN D'EXÉCUTION FINALE

**Objectif**: Complétez Phase 0 à **100%** avant de commencer Phase 1

**Status Actuel**: 63% - Code ✅ | Actions Manuelles 🟡 | Intégration 🟡

**Timeline**: Aujourd'hui (2 février) 2026 → Demain soir (3 février) 2026

---

## 🎯 ÉTAPES DE COMPLÉTION PHASE 0

### ÉTAPE 1: P0.1.1 - Audit & Révocation Clés (60% → 100%)

**Durée**: 2 heures | **Owner**: DevOps/Cloud Engineer

#### Checklist de Révocation

- [ ] **Step 1.1**: Auditer git history pour exposures

  ```bash
  # Chercher les clés exposées dans git
  git log -p --all -S "AIzaSy" | head -50
  git log -p --all -S "key_5" | head -50
  git log -p --all -S "dd_api" | head -50

  # Documenter résultat: Combien de commits exposés?
  ```

- [ ] **Step 1.2**: Révoquer Gemini API Key
  1. Aller à: https://console.cloud.google.com/apis/credentials
  2. Trouver la clé: `AIzaSyAUdogIIbGavH9gvZi7SvteGKcdfz9tRbw`
  3. Cliquer "Delete"
  4. Attendre confirmation
  5. Générer une nouvelle clé Gemini
  6. Sauvegarder nouvelle clé temporairement

- [ ] **Step 1.3**: Révoquer Runway ML Key
  1. Aller à: https://app.runwayml.com/settings/api
  2. Trouver la clé Runway
  3. Cliquer "Revoke"
  4. Générer une nouvelle clé
  5. Sauvegarder temporairement

- [ ] **Step 1.4**: Révoquer Datadog API Keys
  1. Aller à: https://app.datadoghq.com/organization/settings/api-keys
  2. Trouver les 2 clés:
     - DD_API_KEY: `f987c9c2933619d8df6f928121549394`
     - DD_APP_KEY: `588df46400fff53495e3a77cbfeaf6289d2f1a44`
  3. Cliquer "Revoke" sur chacune
  4. Générer de nouvelles clés
  5. Sauvegarder temporairement

- [ ] **Step 1.5**: Nettoyer git history (si exposé)

  ```bash
  # Si clés trouvées dans git history, utiliser git filter-branch
  git filter-branch --tree-filter 'grep -r "AIzaSyAUdogIIb" . && rm -f found_files' -- --all
  # (Ou utiliser BFG Repo-Cleaner)
  ```

- [ ] **Step 1.6**: Vérifier `.gitignore`
  ```bash
  cat .gitignore | grep ".env"
  # Doit avoir: .env (sans commentaire)
  ```

**Validation**:

```bash
# Vérifier aucune clé dans le repo
git grep "AIzaSyAUdogIIb" -- ':!.env.example'  # Doit être vide ✅
git grep "key_50d32" -- ':!.env.example'       # Doit être vide ✅
git grep "f987c9c2" -- ':!.env.example'        # Doit être vide ✅

# Vérifier .env.example n'a que des placeholders
grep -i "api_key\|secret" .env.example         # Doit avoir des <...> placeholders ✅
```

**Résultat Attendu**: P0.1.1 passe de 60% → 100% ✅

---

### ÉTAPE 2: P0.1.2 - Secret Manager Setup (70% → 100%)

**Durée**: 1-1.5 heures | **Owner**: DevOps/Cloud Engineer

#### Checklist GCP Secret Manager

- [ ] **Step 2.1**: Créer les secrets dans GCP

  ```bash
  # Vérifier que vous êtes connecté
  gcloud auth login
  gcloud config set project aiprod-484120

  # Créer les secrets (replication automatique)
  gcloud secrets create GEMINI_API_KEY --replication-policy="automatic"
  gcloud secrets create RUNWAY_API_KEY --replication-policy="automatic"
  gcloud secrets create DATADOG_API_KEY --replication-policy="automatic"
  gcloud secrets create DATADOG_APP_KEY --replication-policy="automatic"
  gcloud secrets create GCS_BUCKET_NAME --replication-policy="automatic"

  # Vérifier création
  gcloud secrets list
  ```

- [ ] **Step 2.2**: Ajouter les valeurs des secrets

  ```bash
  # Gemini (remplacer par nouvelle clé de Step 1.2)
  echo "AIzaSyAUdogIIbGavH9gvZi7SvteGKcdfz9tRbw_NEW" | \
    gcloud secrets versions add GEMINI_API_KEY --data-file=-

  # Runway (remplacer par nouvelle clé de Step 1.3)
  echo "key_50d32d6432d622ec0c7c95f1aa0a68cf781192bd531ff1580c3f4853755c5edba0b52fb49426d07aa6b4356e505ab6e1b80987b501aa08f37000fa51f76796b7_NEW" | \
    gcloud secrets versions add RUNWAY_API_KEY --data-file=-

  # Datadog API (nouvelle clé)
  echo "f987c9c2933619d8df6f928121549394_NEW" | \
    gcloud secrets versions add DATADOG_API_KEY --data-file=-

  # Datadog APP (nouvelle clé)
  echo "588df46400fff53495e3a77cbfeaf6289d2f1a44_NEW" | \
    gcloud secrets versions add DATADOG_APP_KEY --data-file=-

  # Bucket (existant)
  echo "aiprod-484120-assets" | \
    gcloud secrets versions add GCS_BUCKET_NAME --data-file=-
  ```

- [ ] **Step 2.3**: Configurer IAM pour Cloud Run

  ```bash
  # Créer service account (si n'existe pas)
  gcloud iam service-accounts create aiprod-sa --display-name="AIPROD Service Account"

  # Donner accès aux secrets
  gcloud secrets add-iam-policy-binding GEMINI_API_KEY \
    --member=serviceAccount:aiprod-sa@aiprod-484120.iam.gserviceaccount.com \
    --role=roles/secretmanager.secretAccessor

  gcloud secrets add-iam-policy-binding RUNWAY_API_KEY \
    --member=serviceAccount:aiprod-sa@aiprod-484120.iam.gserviceaccount.com \
    --role=roles/secretmanager.secretAccessor

  gcloud secrets add-iam-policy-binding DATADOG_API_KEY \
    --member=serviceAccount:aiprod-sa@aiprod-484120.iam.gserviceaccount.com \
    --role=roles/secretmanager.secretAccessor

  gcloud secrets add-iam-policy-binding DATADOG_APP_KEY \
    --member=serviceAccount:aiprod-sa@aiprod-484120.iam.gserviceaccount.com \
    --role=roles/secretmanager.secretAccessor

  gcloud secrets add-iam-policy-binding GCS_BUCKET_NAME \
    --member=serviceAccount:aiprod-sa@aiprod-484120.iam.gserviceaccount.com \
    --role=roles/secretmanager.secretAccessor
  ```

- [ ] **Step 2.4**: Tester accès aux secrets
  ```bash
  # Tester lecture depuis Secret Manager
  gcloud secrets versions access latest --secret="GEMINI_API_KEY"
  # Doit retourner: AIzaSyAUdogIIbGavH9gvZi7SvteGKcdfz9tRbw_NEW
  ```

**Résultat Attendu**: P0.1.2 passe de 70% → 100% ✅

---

### ÉTAPE 3: P0.2.3 - Intégrer Auth dans main.py (90% → 100%)

**Durée**: 1-2 heures | **Owner**: Backend Engineer

#### Checklist Intégration Middleware

- [ ] **Step 3.1**: Lire le guide complet

  ```bash
  cat docs/INTEGRATION_P0_SECURITY.md
  # Suivre étapes 1-8
  ```

- [ ] **Step 3.2**: Appliquer les 8 étapes (environ 1-1.5h)
  - Étape 1: Ajouter imports (15 lignes)
  - Étape 2: Ajouter startup hooks (20 lignes)
  - Étape 3: Ajouter middleware (1 ligne)
  - Étape 4-7: Protéger endpoints
  - Étape 8: Exception handlers

- [ ] **Step 3.3**: Tester localement

  ```bash
  # Mode développement (auth désactivée pour test)
  export FIREBASE_ENABLED=false
  python -m uvicorn src.api.main:app --reload --port 8000

  # Dans un autre terminal, tester

  # Sans token → doit échouer si auth obligatoire
  curl -X POST http://localhost:8000/pipeline/run \
    -H "Content-Type: application/json" \
    -d '{"content": "test", "preset": "quick_social"}'
  # Expected: 401 ou 200 selon endpoint

  # Vérifier les logs
  # Doit voir: "GET /health", "Security initialized"
  ```

**Résultat Attendu**: P0.2.3 passe de 90% → 100% ✅

---

### ÉTAPE 4: P0.3.1 - Sécuriser docker-compose (90% → 100%)

**Durée**: 30 minutes | **Owner**: DevOps Engineer

#### Checklist Docker-Compose

- [ ] **Step 4.1**: Mettre à jour docker-compose.yml

  ```bash
  # Ouvrir le fichier
  cat docker-compose.yml | grep -A 5 "grafana:"

  # Trouver la section Grafana et modifier:
  # AVANT:
  # GF_SECURITY_ADMIN_PASSWORD=admin

  # APRÈS:
  # GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
  ```

- [ ] **Step 4.2**: Générer mot de passe fort

  ```bash
  # Générer 16+ caractères
  openssl rand -base64 16
  # Copier le résultat

  # Ou utiliser Python
  python -c "import secrets; print(secrets.token_urlsafe(16))"
  ```

- [ ] **Step 4.3**: Mettre à jour .env.local

  ```bash
  # Créer .env.local (git ignored)
  cat >> .env.local << 'EOF'
  GRAFANA_PASSWORD=your_generated_password_here
  GRAFANA_ADMIN_USER=admin
  EOF
  ```

- [ ] **Step 4.4**: Tester

  ```bash
  # Redémarrer Grafana
  docker-compose up -d grafana

  # Vérifier accès avec nouveau password
  curl http://localhost:3030 -u admin:your_password
  # Should work without hardcoded "admin" password
  ```

**Résultat Attendu**: P0.3.1 passe de 90% → 100% ✅

---

### ÉTAPE 5: P0.4.1 & P0.4.3 - Activer Audit Logger (100% → 100% + Vérification)

**Durée**: 1 heure | **Owner**: Backend Engineer

#### Checklist Audit Logger

- [ ] **Step 5.1**: Vérifier `src/security/audit_logger.py` existe

  ```bash
  ls -la src/security/audit_logger.py
  # Doit exister et avoir 240+ LOC
  wc -l src/security/audit_logger.py
  ```

- [ ] **Step 5.2**: Vérifier imports dans main.py

  ```bash
  grep "from src.security.audit_logger import" src/api/main.py
  # Si vide, ajouter dans étape 3
  ```

- [ ] **Step 5.3**: Ajouter audit logging dans endpoints critiques

  ```bash
  # Dans src/api/main.py, ajouter après chaque endpoint important:
  audit_logger = get_audit_logger()
  audit_logger.log_api_call(
      endpoint="/pipeline/run",
      method="POST",
      user_id=user.get("email"),
      status_code=200,
      duration_ms=elapsed_ms
  )
  ```

- [ ] **Step 5.4**: Tester audit logs localement

  ```bash
  # Lancer l'API
  FIREBASE_ENABLED=false python -m uvicorn src.api.main:app --reload

  # Dans un autre terminal, faire une requête
  curl -X POST http://localhost:8000/pipeline/run \
    -H "Content-Type: application/json" \
    -d '{"content": "test"}'

  # Vérifier les logs JSON
  # Doit voir: {"timestamp": "...", "event_type": "API_CALL", ...}
  ```

**Résultat Attendu**: P0.4 passe de 100% → 100% ✅ (Vérification complète)

---

## ✅ VALIDATION PHASE 0 - 100% COMPLETE

### Checklist Finale

**P0.1 - Secrets Exposés**:

- [x] Code créé: src/config/secrets.py ✅
- [ ] Clés API révoquées ✅
- [ ] Secrets dans GCP Secret Manager ✅
- [ ] Test chargement depuis Secret Manager ✅

**P0.2 - Auth API**:

- [x] Code créé: src/auth/firebase_auth.py ✅
- [x] Code créé: src/api/auth_middleware.py ✅
- [ ] Middleware intégré dans main.py ✅
- [ ] Tests locaux passants (curl test) ✅

**P0.3 - Passwords en Dur**:

- [ ] docker-compose.yml mis à jour ✅
- [ ] Grafana password changé ✅
- [ ] Vars d'env configurées ✅

**P0.4 - Audit Logging**:

- [x] Code créé: src/security/audit_logger.py ✅
- [ ] Audit logging dans main.py ✅
- [ ] Logs JSON vérifiés ✅

**Code Quality**:

- [x] 22/22 tests unitaires passants ✅
- [x] 2,000+ LOC documentation ✅
- [x] requirements.txt mis à jour ✅
- [ ] .gitignore a `.env` ✅

**Security**:

- [ ] Aucune clé dans git history ✅
- [ ] .env.example contient que des placeholders ✅
- [ ] Endpoints sans token retournent 401 ✅

---

## 🎯 RÉSULTAT FINAL PHASE 0

```
Avant: 63% (Code ✅ | Actions 🟡 | Intégration 🟡)
Après: 100% (Tout ✅)

Timeline: 4-5 heures de travail
  - Révocation clés: 2h
  - Setup GCP: 1-1.5h
  - Intégration code: 1-2h
  - Vérification: 30min-1h
```

---

## 🚀 APRÈS PHASE 0 - PHASE 1 READY

Une fois Phase 0 à 100%, vous pouvez **immédiatement** commencer **Phase 1**:

**P1.1**: Persistance (PostgreSQL) - 10h
**P1.2**: Queue Pub/Sub - 16h
**P1.3**: Remplacer mocks - 11h
**P1.4**: CI/CD Pipeline - 4h

Total Phase 1: ~41h (1-2 semaines)

---

**Mode Stricte**: Chaque étape numérotée doit avoir sa checkbox cochée ✅ avant de passer à la suivante.

👉 **Commencez par**: ÉTAPE 1 (P0.1.1 - Révocation Clés)
