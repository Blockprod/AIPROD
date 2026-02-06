# 🎬 AIPROD - BACKEND ACTIVATION GUIDE

# Pour utiliser les vrais backends vidéo au lieu des mocks

## ========================================

## 1. VÉRIFIER LES CLÉS API

## ========================================

Vos clés sont DÉJÀ dans `.env`:

- ✅ RUNWAY_API_KEY = key_50d32d...
- ✅ GEMINI_API_KEY = AIzaSyAU...

## ========================================

## 2. DÉMARRER L'API AVEC VRAIES CLÉS

## ========================================

OPTION A: Charger .env explicitement AVANT de lancer

```powershell
cd c:\Users\averr\AIPROD
.venv311\Scripts\python.exe -c "from dotenv import load_dotenv; load_dotenv('.env'); import os; print(f'RUNWAY_API_KEY loaded: {os.getenv(\"RUNWAY_API_KEY\")[:20]}...')"
.venv311\Scripts\python.exe -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

OPTION B: Utiliser le script de démarrage

```powershell
.venv311\Scripts\python.exe scripts/start_api.py
```

## ========================================

## 3. APPELER L'API AVEC MODE RÉEL

## ========================================

Envoyez un appel à `/pipeline/run` avec `"_use_real_backend": true`:

```bash
curl -X POST http://localhost:8000/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{
    "content": "A beautiful sunset over the ocean",
    "duration_sec": 5,
    "preset": "brand_campaign",
    "_use_real_backend": true
  }'
```

Ou utiliser le script Python:

```powershell
.venv311\Scripts\python.exe scripts/generate_promo_via_api.py
```

## ========================================

## 4. COÛTS DES BACKENDS

## ========================================

| Backend   | Durée 5s | Couût API | Notes                    |
| --------- | -------- | --------- | ------------------------ |
| Runway    | 30s      | $30       | Meilleure qualité (0.95) |
| Veo-3     | 40s      | $2.60     | Premium (0.92)           |
| Replicate | 20s      | $0.26     | Budget (0.75)            |

AIPROD optimise automatiquement le coût !

## ========================================

## 5. LOGS/MONITORING

## ========================================

Regardez les logs pour voir le backend sélectionné:

```
grep -i "RenderExecutor" logs/AIPROD.log
grep -i "backend" logs/AIPROD.log
```

## ========================================

## 6. STATUS DE VOTRE SETUP

## ========================================

✅ Runway ML: INSTALLÉ et PRÊT
✅ Clés API: CONFIGURÉES
✅ .env Loading: FIXÉ
✅ Cost Optimization: AUTOMATIQUE

👉 PROCHAINE ÉTAPE: Relancer l'API et réappeler /pipeline/run
