# ✅ ÉTAPE 1 - AUDIT & RÉVOCATION CLÉS - LOG D'EXÉCUTION

**Date**: 2 Février 2026  
**Statut**: EN COURS  
**Owner**: DevOps/Cloud Engineer

---

## 📊 RÉSULTATS DU SCAN AUTOMATISÉ

### Step 1.1: Audit Git History ✅

**Résultat**: ✅ **PAS DE GIT REPO ACTIF**

- Le repo n'est pas un git repository (`.git` absent)
- **Impact**: Pas de nettoyage git history requis
- **Action**: Passer au Step 1.2

---

### Step 1.2: Clés Exposées Locales ⚠️ **TROUVÉES**

**Fichier**: `.env`  
**3 clés exposées détectées**:

#### Clé 1 - GEMINI API KEY

```
Line 13: GEMINI_API_KEY=AIzaSyAUdogIIbGavH9gvZi7SvteGKcdfz9tRbw
```

- **Service**: Google Cloud Gemini API
- **Status**: ❌ **EXPOSÉE** dans `.env`
- **Action Requise**: RÉVOQUER et générer nouvelle clé
- **Source**: https://console.cloud.google.com/apis/credentials

#### Clé 2 - RUNWAY ML API KEY

```
Line 16: RUNWAY_API_KEY=key_50d32d6432d622ec0c7c95f1aa0a68cf781192bd531ff1580c3f4853755c5edba0b52fb49426d07aa6b4356e505ab6e1b80987b501aa08f37000fa51f76796b7
```

- **Service**: Runway ML (vidéo generation)
- **Status**: ❌ **EXPOSÉE** dans `.env`
- **Action Requise**: RÉVOQUER et générer nouvelle clé
- **Source**: https://app.runwayml.com/settings/api

#### Clé 3 - DATADOG API KEY

```
Line 33: DD_API_KEY=f987c9c2933619d8df6f928121549394
```

- **Service**: Datadog Monitoring
- **Status**: ❌ **EXPOSÉE** dans `.env`
- **Action Requise**: RÉVOQUER et générer nouvelle clé
- **Source**: https://app.datadoghq.com/organization/settings/api-keys

---

## 🎯 PLAN D'ACTION DÉTAILLÉ

### ACTION 1️⃣ : RÉVOQUER GEMINI API KEY (15-20 min)

**Étapes**:

1. Ouvrir: https://console.cloud.google.com/apis/credentials
2. **Chercher la clé**:
   - Filtrer par: "API Keys"
   - Trouver: `AIzaSyAUdogIIbGavH9gvZi7SvteGKcdfz9tRbw`
3. **Cliquer sur la clé** → Menu contextuel
4. **Sélectionner** "Delete" ou "Revoke"
5. **Confirmer** la suppression
6. **Attendre** (quelques secondes pour synchronisation)
7. **Générer nouvelle clé**:
   - Cliquer "+ CREATE CREDENTIALS"
   - Sélectionner "API Key"
   - Copier la nouvelle clé
   - **Sauvegarder temporairement** dans un fichier texte sécurisé

**Validation**:

```
✓ Ancienne clé: AIzaSyAUdogIIbGavH9gvZi7SvteGKcdfz9tRbw → DELETED
✓ Nouvelle clé: AIzaSy_______________ → GÉNÉRÉE
```

---

### ACTION 2️⃣ : RÉVOQUER RUNWAY API KEY (15-20 min)

**Étapes**:

1. Ouvrir: https://app.runwayml.com/settings/api
2. **Chercher la clé**:
   - Chercher: `key_50d32d6432d622ec0c...`
3. **Cliquer sur la clé** → Menu options
4. **Sélectionner** "Revoke" ou "Delete"
5. **Confirmer** la révocation
6. **Attendre** confirmation
7. **Générer nouvelle clé**:
   - Cliquer "+ Generate New Key" ou "Create API Token"
   - Copier la nouvelle clé
   - **Sauvegarder temporairement**

**Validation**:

```
✓ Ancienne clé: key_50d32d6432d622ec0c... → REVOKED
✓ Nouvelle clé: key___________________ → GÉNÉRÉE
```

---

### ACTION 3️⃣ : RÉVOQUER DATADOG API KEYS (30 min pour 2 clés)

**Étapes pour API Key**:

1. Ouvrir: https://app.datadoghq.com/organization/settings/api-keys
2. **Chercher l'ancienne clé**:
   - Chercher: `f987c9c2933619d8df6f928121549394` (DD_API_KEY)
3. **Clicker sur la clé** → Menu
4. **Sélectionner** "Revoke" ou "Delete"
5. **Confirmer** révocation
6. **Générer nouvelle clé**:
   - Cliquer "+ New Key"
   - Nommer: `aiprod-api-key-v2`
   - Copier la clé
   - **Sauvegarder temporairement**

**Étapes pour APP Key**:

1. Ouvrir: https://app.datadoghq.com/organization/settings/application-keys
2. **Chercher l'ancienne clé**:
   - Chercher: `588df46400fff53495e3a77cbfeaf6289d2f1a44` (DD_APP_KEY)
3. **Clicker sur la clé** → Menu
4. **Sélectionner** "Revoke" ou "Delete"
5. **Confirmer** révocation
6. **Générer nouvelle clé**:
   - Cliquer "+ New Key"
   - Nommer: `aiprod-app-key-v2`
   - Copier la clé
   - **Sauvegarder temporairement**

**Validation**:

```
✓ Ancienne DD_API_KEY: f987c9c2933619d8df6f928121549394 → REVOKED
✓ Ancienne DD_APP_KEY: 588df46400fff53495e3a77cbfeaf... → REVOKED
✓ Nouvelle DD_API_KEY: ________________________________ → GÉNÉRÉE
✓ Nouvelle DD_APP_KEY: ________________________________ → GÉNÉRÉE
```

---

## 📝 NOUVELLES CLÉS À SAUVEGARDER

**Créer un fichier temporaire sécurisé** (ex: `NEW_KEYS_TEMP.txt`) avec:

```
# NOUVELLES CLÉS GÉNÉRÉES LE 2 FEV 2026
# À UTILISER DANS ÉTAPE 2 (GCP Secret Manager)

GEMINI_API_KEY_NEW=                    # Collez ici après génération
RUNWAY_API_KEY_NEW=                    # Collez ici après génération
DATADOG_API_KEY_NEW=                   # Collez ici après génération
DATADOG_APP_KEY_NEW=                   # Collez ici après génération
```

⚠️ **SÉCURITÉ**: Supprimer ce fichier après Step 2 (GCP Secret Manager)

---

## ✅ CHECKLIST FINALE - ÉTAPE 1

- [ ] **Step 1.1**: Git history vérifié (PAS DE GIT REPO - OK)
- [ ] **Step 1.2**: Clés exposées identifiées (3 trouvées ✅)
- [ ] **Action 1**: Gemini key révoquée + nouvelle générée
- [ ] **Action 2**: Runway key révoquée + nouvelle générée
- [ ] **Action 3a**: Datadog API key révoquée + nouvelle générée
- [ ] **Action 3b**: Datadog APP key révoquée + nouvelle générée
- [ ] **Action 4**: Nouvelles clés sauvegardées dans fichier temporaire
- [ ] **Step 1.5**: Nettoyer `.env` (garder structure, retirer valeurs)
- [ ] **Step 1.6**: Vérifier `.gitignore` a `.env`

---

## 🔒 APRÈS RÉVOCATION - NETTOYAGE LOCAL

Une fois toutes les clés révoquées, **nettoyer les fichiers locaux**:

```powershell
# Vider .env des anciennes clés (garder structure)
# Remplacer les valeurs par des placeholders:
GEMINI_API_KEY=<PLACEHOLDER>
RUNWAY_API_KEY=<PLACEHOLDER>
DD_API_KEY=<PLACEHOLDER>
DD_APP_KEY=<PLACEHOLDER>
```

---

## 🚀 ÉTAPE 1 COMPLÈTE QUAND

✅ **Tous les checkboxes ci-dessus cochés**

**Temps total estimé**: 60-90 minutes  
**Prochaine étape**: ÉTAPE 2 (GCP Secret Manager)

---

## 📍 Statut de Progression

```
ÉTAPE 1: P0.1.1 - Audit & Révocation Clés
├─ Step 1.1: Audit Git ............................ ✅ FAIT (Pas de repo)
├─ Step 1.2: Clés Exposées Détectées ............. ✅ FAIT (3 trouvées)
├─ Action 1: Gemini Key .......................... 🟡 ATTENTE ACTION
├─ Action 2: Runway Key .......................... 🟡 ATTENTE ACTION
├─ Action 3a: Datadog API Key .................... 🟡 ATTENTE ACTION
├─ Action 3b: Datadog APP Key .................... 🟡 ATTENTE ACTION
├─ Step 1.5: Nettoyer git history ............... ✅ N/A (pas de git)
└─ Step 1.6: Vérifier .gitignore ................ ⏳ À VÉRIFIER
```
