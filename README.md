# AIPROD V33 - Pipeline de Génération Vidéo IA

## 🎯 Description

AIPROD V33 est une plateforme cloud-native de génération vidéo IA avec :

- **Orchestration asynchrone** des agents spécialisés
- **Double QA System** (technique + sémantique)
- **Optimisation financière** déterministe (sans LLM)
- **Fast Track** pour les requêtes simples (< 20s)
- **Cache de cohérence** TTL 168h
- **API REST** complète avec monitoring

## 🚀 Démarrage rapide

### Prérequis

- Python 3.10+
- pip ou pip3

### Installation

1. **Cloner ou télécharger le projet**

```bash
cd AIPROD_V33
```

2. **Créer et activer l'environnement virtuel**

```bash
python -m venv .venv
.venv/Scripts/Activate.ps1  # Windows PowerShell
source .venv/bin/activate   # macOS/Linux
```

3. **Installer les dépendances**

```bash
pip install -r requirements.txt
```

### Démarrer l'API

```bash
uvicorn src.api.main:app --reload --port 8000
```

L'API sera disponible à `http://localhost:8000`

### Documentation interactive

```
http://localhost:8000/docs
```

## 📋 Exemple d'utilisation

### Lancez le pipeline

```bash
curl -X POST http://localhost:8000/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Créer une vidéo d'\''une danse",
    "priority": "high",
    "lang": "fr"
  }'
```

### Consulter les métriques

```bash
curl http://localhost:8000/metrics
```

### Vérifier les alertes

```bash
curl http://localhost:8000/alerts
```

## 🧪 Tests

### Exécuter tous les tests

```bash
pytest
```

### Tests unitaires uniquement

```bash
pytest tests/unit/
```

### Tests d'intégration

```bash
pytest tests/integration/
```

### Tests de performance

```bash
pytest tests/performance/
```

### Couverture de code

```bash
pytest --cov=src --cov-report=html
```

## 📁 Structure du projet

```
AIPROD_V33/
├── src/
│   ├── orchestrator/          # State Machine du pipeline
│   │   ├── state_machine.py
│   │   └── transitions.py
│   ├── agents/                # Agents spécialisés
│   │   ├── creative_director.py
│   │   ├── fast_track_agent.py
│   │   ├── render_executor.py
│   │   ├── semantic_qa.py
│   │   └── visual_translator.py
│   ├── api/                   # API REST et fonctions métier
│   │   ├── main.py            # Endpoints FastAPI
│   │   └── functions/
│   │       ├── financial_orchestrator.py
│   │       ├── technical_qa_gate.py
│   │       └── input_sanitizer.py
│   ├── memory/                # Gestion de la mémoire partagée
│   │   ├── memory_manager.py
│   │   ├── schema_validator.py
│   │   └── exposed_memory.py
│   └── utils/                 # Utilitaires
│       ├── monitoring.py      # Logging structuré
│       ├── metrics_collector.py
│       ├── cache_manager.py
│       └── gcp_client.py
├── tests/
│   ├── unit/                  # Tests unitaires (14+ tests)
│   ├── integration/           # Tests d'intégration (3+ tests)
│   └── performance/           # Tests de performance (2+ tests)
├── docs/
│   ├── architecture.md        # Documentation technique
│   └── api_documentation.md   # Documentation API
├── config/
│   └── v33.json              # Configuration du projet
├── logs/                      # Fichiers de logs
├── requirements.txt           # Dépendances Python
├── pyproject.toml            # Configuration pytest
└── README.md                 # Ce fichier
```

## 🏗️ Architecture

Pour une vue d'ensemble détaillée de l'architecture, consultez [`docs/architecture.md`](docs/architecture.md).

### Composants principaux

1. **Orchestrator** : Gère les états et les transitions du pipeline
2. **Memory Manager** : Mémoire partagée avec cache TTL 168h
3. **Creative Director** : Fusion des agents avec fallback Gemini
4. **Fast Track Agent** : Pipeline simplifié (< 20s)
5. **Render Executor** : Exécution du rendu
6. **Semantic QA** : Validation sémantique (LLM)
7. **Visual Translator** : Adaptation multilingue
8. **Financial Orchestrator** : Optimisation coût/qualité (déterministe)
9. **Technical QA Gate** : Vérifications binaires

## 📊 Métriques & Monitoring

L'API expose en temps réel :

- **Latence moyenne** du pipeline
- **Coût moyen** par exécution
- **Score de qualité moyen**
- **Nombre d'exécutions** et d'**erreurs**
- **Alertes** sur les seuils critiques

Endpoints :

- `GET /metrics` : Métriques agrégées
- `GET /alerts` : Alertes actives
- `GET /icc/data` : Données ICC (Interface Client)

## 🔧 Configuration

Configuration externalisée dans `config/v33.json` :

```json
{
  "retry": { "maxRetries": 3, "backoffSec": 15 },
  "cache": { "ttl": 168 },
  "fastTrack": { "maxDurationSec": 30, "maxScenes": 3, "costCeiling": 0.3 },
  "financial": { "updateIntervalHours": 24 }
}
```

## 📚 Documentation

- [Architecture technique](docs/architecture.md)
- [Documentation API](docs/api_documentation.md)
- [Configuration](config/v33.json)

## 🚢 Déploiement

### Docker

```bash
docker build -t aiprod-v33 .
docker run -p 8000:8000 aiprod-v33
```

### Google Cloud Platform

```bash
gcloud functions deploy aiprod-v33 \
  --runtime python311 \
  --trigger-http
```

## 📝 Logging

Les logs sont stockés dans `logs/aiprod_v33.log` avec rotation automatique :

- Taille max : 5MB par fichier
- Backups : 5 fichiers historiques
- Format : `[timestamp] LEVEL module: message`

## ✅ Conformité aux spécifications

- ✅ Pipeline complet fonctionnel
- ✅ Agents asynchrones intégrés
- ✅ Double QA System (technique + sémantique)
- ✅ Optimisation financière déterministe
- ✅ Cache de cohérence TTL 168h
- ✅ API REST FastAPI documentée
- ✅ Monitoring et métriques
- ✅ Tests unitaires, intégration et performance
- ✅ Logging structuré
- ✅ Documentation complète

## 🔄 État du projet

**Statut** : ✅ Implémentation complète et validée

**Modules implémentés** :

- ✅ Memory Manager
- ✅ Orchestrator (State Machine)
- ✅ Creative Director, Fast Track, Render Executor, Semantic QA, Visual Translator
- ✅ Financial Orchestrator, Technical QA Gate, Input Sanitizer
- ✅ API REST FastAPI avec tous les endpoints
- ✅ Monitoring & Métriques
- ✅ Logging structuré
- ✅ Tests (unitaires, intégration, performance)
- ✅ Documentation

## 📞 Support

Pour toute question ou problème :

1. Consultez la [documentation API](docs/api_documentation.md)
2. Vérifiez les logs dans `logs/aiprod_v33.log`
3. Lancez les tests pour diagnostiquer les problèmes

## 📄 Licence

Propriétaire - AIPROD V33 (2026)

## 🎯 Prochaines étapes (Roadmap)

- [ ] Intégration GCP (Cloud Run, Cloud Functions)
- [ ] Authentification et autorisation (JWT)
- [ ] Rate limiting et quotas
- [ ] Webhooks asynchrones
- [ ] Persistence en Firestore/BigQuery
- [ ] Support Sora pour génération vidéo native
- [ ] Multi-backends LLM (Claude, GPT-4, etc.)
- [ ] Dashboard d'administration
- [ ] Notifications en temps réel
