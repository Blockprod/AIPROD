# 🔐 AUDIT DE SÉCURITÉ - Fixes Recommandées

**Date**: 7 février 2026  
**Priorité**: 🔴 **CRITIQUE**

---

## 📋 Résumé des Problèmes

| Problème                       | Sévérité    | Status   | Fix                       |
| ------------------------------ | ----------- | -------- | ------------------------- |
| API Keys hardcodées dans tests | 🔴 CRITIQUE | ❌ ACTIF | Supprimer, utiliser mocks |
| Passwords par défaut           | 🔴 CRITIQUE | ❌ ACTIF | Utiliser variables env    |
| Noms variables incohérents     | 🟠 HAUT     | ❌ ACTIF | Standardiser              |
| Firebase path incohérent       | 🟠 HAUT     | ❌ ACTIF | Unifier                   |
| Email exposé publiquement      | 🟡 MOYEN    | ❌ ACTIF | Redirection sécurisée     |
| API keys dans les logs         | 🔴 CRITIQUE | ❌ ACTIF | Masquer avec truncate     |

---

## 🔴 FIX #1: Supprimer les API Keys des Tests

### Fichiers à Corriger:

- `tests/test_security.py` → Ligne 39
- `tests/test_runway.py` → Ligne 9
- `tests/test_gemini.py` → Ligne 3

### Avant:

```python
# ❌ DANGEREUX!
api_key = "AIzaSyAUdogIIbGavH9gvZi7SvteGKcdfz9tRbw"
client = RunwayML(api_key="key_50d32d6432d6...")
```

### Après:

```python
# ✅ SÉCURISÉ
from unittest.mock import MagicMock, patch

@patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-12345"})
def test_gemini():
    # Utiliser la clé de test
    key = os.getenv("GEMINI_API_KEY")  # "test-key-12345"

# OU utiliser mocks:
with patch("src.agents.creative_director.CreativeDirector.generate") as mock_gen:
    mock_gen.return_value = "Mocked response"
```

---

## 🔴 FIX #2: Standardiser les Noms de Variables

### Standardisation Proposée:

```env
# ✅ NOMS STANDARDS (utiliser PARTOUT dans le code)

# Video APIs
RUNWAY_API_KEY=key_50d3...
REPLICATE_API_KEY=r8_...

# AI Models
GEMINI_API_KEY=AIzaSy...
ELEVENLABS_API_KEY=sk_...

# GCP
GOOGLE_CLOUD_PROJECT=aiprod-484120

# Database
DATABASE_URL=postgresql://...

# Credentials
FIREBASE_CREDENTIALS=credentials/firebase-credentials.json
```

### Fichiers à Mettre à Jour:

1. **src/agents/render_executor.py** (lignes 79, 88, 285)

   ```python
   # AVANT:
   runway_api_key = os.getenv("RUNWAYML_API_SECRET") or os.getenv("RUNWAY_API_KEY")
   replicate_api_key = os.getenv("REPLICATE_API_TOKEN")

   # APRÈS:
   runway_api_key = os.getenv("RUNWAY_API_KEY")
   replicate_api_key = os.getenv("REPLICATE_API_KEY")
   ```

2. **src/config/secrets.py** (si personnalisé)
   - Utiliser UNIQUEMENT les noms standards

3. **scripts/** (tous les scripts)
   - Remplacer `RUNWAYML_API_SECRET` → `RUNWAY_API_KEY`
   - Remplacer `REPLICATE_API_TOKEN` → `REPLICATE_API_KEY`
   - Remplacer `GCP_PROJECT_ID` → `GOOGLE_CLOUD_PROJECT`

---

## 🔴 FIX #3: Supprimer les Passwords Hardcodés

### Fichiers à Corriger:

- `src/workers/pipeline_worker.py` (ligne 64)
- `src/api/main.py` (ligne 100)
- `migrations/env.py` (lignes 23, 48)

### Pattern Correct:

```python
# ❌ DANGEREUX - Ne JAMAIS faire ça:
db_url = os.getenv("DATABASE_URL", "postgresql://aiprod:password@localhost:5432/AIPROD")

# ✅ SÉCURISÉ:
db_url = os.getenv("DATABASE_URL")
if not db_url:
    raise ValueError("DATABASE_URL must be set in environment")
```

---

## 🟠 FIX #4: Firebase Credentials Path

### Unifier la Variable:

```env
# .env
FIREBASE_CREDENTIALS_PATH=credentials/firebase-credentials.json
```

### Utiliser Partout:

```python
# src/auth/firebase_auth.py
credentials_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
if not credentials_path:
    raise ValueError("FIREBASE_CREDENTIALS_PATH not configured")

# Initialize Firebase
service_account = firebase_admin.credentials.Certificate(credentials_path)
```

---

## 🟡 FIX #5: Email Sécurisé

### Remplacer l'Email Partout:

Créer une variable d'env:

```env
CONTACT_EMAIL=support@aiprod.com
ALERT_EMAIL=alerts@aiprod.com
REPORT_EMAIL=reports@aiprod.com
```

Remplacer dans:

- `README.md` → Utiliser `CONTACT_EMAIL`
- Documentation → URL de formulaire sécurisé
- `src/api/main.py` → Variable env, pas hardcoding

---

## 🔴 FIX #6: Masquer les secrets dans les Logs

### Avant:

```
ERROR: Client error '429' for url 'https://...?key=AIzaSyAUdogIIbGavH9gvZi7SvteGKcdfz9tRbw'
```

### Après:

```
ERROR: Client error '429' for url 'https://...?key=AIzaSy****...'
```

### Implementation:

```python
# src/utils/structured_logging.py
import re

def mask_secrets(message: str) -> str:
    """Masquer les API keys dans les logs."""
    patterns = [
        (r"key=([a-zA-Z0-9_]+)", r"key=\1[:20]***"),  # Runway
        (r"AIzaSy[a-zA-Z0-9_-]+", r"AIzaSy****..."),  # Gemini
        (r"sk_[a-zA-Z0-9]+", r"sk_****..."),  # ElevenLabs
        (r"r8_[a-zA-Z0-9]+", r"r8_****..."),  # Replicate
    ]
    for pattern, replacement in patterns:
        message = re.sub(pattern, replacement, message)
    return message
```

---

## ✅ Checklist des Fixes

### IMMÉDIAT (Critical):

- [ ] Supprimer API keys des fichiers de test
- [ ] Supp passwords hardcodés
- [ ] Standardiser les noms de variables
- [ ] Tester que tout fonctionne

### COURT TERME (1 semaine):

- [ ] Auditer tous les logs pour secrets exposés
- [ ] Implémenter masking des secrets
- [ ] Rotationner les clés Gemini/Runway/Replicate
- [ ] Auditer l'historique git

### LONG TERME (production):

- [ ] Utiliser GCP Secret Manager
- [ ] Implementer Key Rotation automatique
- [ ] Audit de sécurité externe
- [ ] CI/CD security checks

---

## 🛡️ Validation

Après fixes, exécuter:

```bash
# Scanner les API keys hardcodées
git grep "AIzaSy" -- ':!.env.example' ':!.env' ':!*.md'
git grep "key_50d" -- ':!.env.example' ':!.env' ':!*.md'
git grep "sk_a0" -- ':!.env.example' ':!.env' ':!*.md'
git grep "r8_" -- ':!.env.example' ':!.env' ':!*.md'

# Vérifier qu'aucun secret n'est exposé
grep -r "password=" src/ scripts/ migrations/ | grep -v ".example"
```

Résultat attendu: **AUCUNE CORRESPONDANCE** ✅

---

## 📞 Support

Contactez le responsable de la sécurité avant de déployer!
