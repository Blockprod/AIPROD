# Phase 2: Intégration Suno API - Guide Complet

## 🎵 Objectif

Intégrer l'API Suno pour générer de la musique de qualité professionnelle au lieu d'utiliser des mocks.

## ✅ Étapes Complétées

### 1. Implémentation Code (FAIT ✅)

- ✅ Créé `src/agents/music_composer.py` v2 avec support Suno
- ✅ Ajouté `generate_music_suno()` avec API calls
- ✅ Implémenté `_build_music_prompt()` pour prompts optimisés
- ✅ Ajouté stratégie fallback (Suno → Soundful → Mock)
- ✅ Intégré logging pour tracking
- ✅ Ajouté `SUNO_API_KEY` aux secrets optionnels dans `src/config/secrets.py`
- ✅ Créé script d'initialisation: `scripts/setup_suno_secret.py`

### 2. Structure MusicComposer Actuelle

```python
MusicComposer(provider="suno")
├── generate_music_suno()         # API Suno réelle
├── generate_music_soundful()     # API Soundful (fallback)
├── generate_music()              # Orchestration + fallback logic
├── _build_music_prompt()         # Optimisation prompts
└── run()                         # Interface principale
```

## 🔐 Configuration Suno API Key

### Étape 1: Créer un compte Suno

1. Allez sur https://suno.ai
2. Créez un compte (gratuit ou payant)
3. Accédez à https://suno.ai/api-keys
4. Générez une nouvelle clé API
5. Copiez la clé (format: `sk-...`)

### Étape 2: Ajouter la clé à GCP Secret Manager (Production)

#### Option A: Avec Google Cloud CLI

```bash
# Créer le secret
gcloud secrets create SUNO_API_KEY \
  --replication-policy="automatic" \
  --data-file=- << EOF
sk-your-actual-suno-api-key-here
EOF

# Vérifier
gcloud secrets describe SUNO_API_KEY
gcloud secrets versions list SUNO_API_KEY
```

#### Option B: Avec Python Script

```bash
# Activer venv
.\.venv\Scripts\Activate.ps1

# Configurer authentification GCP
gcloud auth application-default login

# Ajouter le secret
python scripts/setup_suno_secret.py "sk-your-actual-suno-api-key-here"
```

#### Option C: Via Google Cloud Console

1. Aller à https://console.cloud.google.com/security/secret-manager
2. Cliquer "Create Secret"
3. Nom: `SUNO_API_KEY`
4. Valeur: `sk-...` (votre clé Suno)
5. Cliquer "Create Secret"

### Étape 3: Configuration Locale (.env)

Pour tester localement, créez `.env`:

```env
GCP_PROJECT_ID=aiprod-484120
SUNO_API_KEY=sk-your-test-api-key-here
ENVIRONMENT=development
```

## 🧪 Tests & Validation

### Test 1: Vérifier le chargement du secret

```python
import os
from src.config.secrets import get_secret

# Charger le secret
suno_key = get_secret("SUNO_API_KEY")
print(f"Suno Key loaded: {suno_key is not None}")
```

### Test 2: Tester MusicComposer avec Suno

```python
from src.agents.music_composer import MusicComposer

# Créer instance
composer = MusicComposer(provider="suno")

# Test avec mock (si pas de clé API)
manifest = {
    "script": "Beautiful sunset scene with birds flying",
    "music_style": "cinematic",
    "duration": 30,
    "mood": "peaceful"
}

result = composer.run(manifest)
print(f"Provider: {result.get('music', {}).get('provider')}")
print(f"Status: {result.get('music', {}).get('status')}")
```

### Test 3: Exécuter les tests unitaires

```bash
# Activer venv
.\.venv\Scripts\Activate.ps1

# Tester uniquement MusicComposer
pytest tests/unit/test_music_composer.py -v

# Tester pipeline complète
pytest tests/unit/test_state_machine.py::test_run_success -v
```

## 📊 Suno API Reference

### Endpoint: POST /api/generate

```json
{
  "prompt": "Cinematic background music for video content about: Beautiful sunset...",
  "duration": 30,
  "style": "cinematic",
  "gpt_description_prompt": "Generate background music for: Beautiful sunset scene..."
}
```

### Response (200 OK)

```json
{
  "id": "song-12345abc",
  "music_url": "https://cdn.suno.ai/...",
  "url": "https://suno.ai/song/12345",
  "title": "Cinematic Music",
  "duration_seconds": 30,
  "status": "completed"
}
```

### Response (202 Accepted - Async)

```json
{
  "id": "job-12345abc",
  "status": "pending",
  "message": "Generation in progress..."
}
```

## 🔄 Workflow de Génération Musique

```
1. User Request
   ↓
2. StateMachine.run()
   ├─ ScriptGenerator → script.txt
   ├─ ImageGenerator → images (FATTO ✅)
   ├─ VideoRenderer  → video.mp4 (FATTO ✅)
   ├─ AudioGenerator → voice.mp3 (FATTO ✅)
   └─ MusicComposer  → music.mp3 (NUOVO - Phase 2)
   ↓
3. PostProcessor
   └─ Mix audio + music + video
   ↓
4. Output: Complete Video with Sound
```

## 🚨 Gestion d'Erreurs

### Scenario 1: Pas de clé API

```
SUNO_API_KEY = None
→ Fallback: generate_music_soundful()
→ Si pas SOUNDFUL: Fallback mock
→ Logs: "SUNO_API_KEY not configured, falling back to mock"
```

### Scenario 2: API Error (5xx)

```
response.status_code = 500
→ Logs: "Suno API error 500"
→ Fallback: Soundful ou Mock
→ Status: "failed"
```

### Scenario 3: Timeout

```
requests.exceptions.Timeout
→ Logs: "Suno API timeout, falling back to mock"
→ Status: "failed"
→ Duration: 30s max

```

### Scenario 4: Async Processing (202)

```
response.status_code = 202
→ Return: {"status": "pending", "song_id": "..."}
→ Client peut faire polling pour résultat
→ Utile pour générations longues (> 30s)
```

## 📈 Monitoring & Metrics

### Logs à Observer

```
[INFO] MusicComposer initialized with provider: suno
[INFO] Suno: Calling API to generate music - style=cinematic, duration=30s, mood=peaceful
[INFO] Suno: Music generated successfully - song_id=song-12345abc
```

### Métriques à Tracker

- Nombre d'appels Suno
- Taux de succès / Taux d'erreur
- Temps de réponse moyen
- Fallback rate (% utilisant mock)

## 🔗 Documentation Officielle

- Suno API Docs: https://api.suno.ai/docs
- Suno Dashboard: https://suno.ai/api-keys
- Pricing: https://suno.ai/pricing

## 📋 Checklist Phase 2

- [x] Implémenter `generate_music_suno()` dans MusicComposer
- [x] Ajouter gestion d'erreurs et fallbacks
- [x] Intégrer logging pour debugging
- [x] Ajouter support pour async processing (202)
- [x] Configurer SUNO_API_KEY dans secrets.py
- [x] Créer script setup_suno_secret.py
- [x] Documenter étapes configuration
- [ ] Créer tests unitaires Suno (Phase 3)
- [ ] Intégrer avec PostProcessor (Phase 4)
- [ ] Déployer à production (Phase 6)

## 🎯 Prochaines Étapes (Phases Suivantes)

### Phase 3: SoundEffectsAgent

- Créer agent pour bruitages/SFX
- Intégrer avec orchestrator
- Tester avec effets vidéo

### Phase 4: PostProcessor Integration

- Mixer audio + musique + vidéo avec ffmpeg
- Ajouter transitions et effets
- Optimiser qualité audio

### Phase 5: Tests Complets

- Tests unitaires audio/video
- Tests d'intégration pipeline
- Performance testing

### Phase 6: Production

- Déployer Suno secrets à GCP
- Monitor en production
- Optimiser costs
