# CONTEXTE : CRÉATION DU PROJET AIPROD ULTIMATE

## 🎯 OBJECTIF

Créer une implémentation complète et fonctionnelle du pipeline de génération vidéo IA "AIPROD" basé sur le fichier de configuration `AIPROD.json` que j'ai joint.

## 📁 STRUCTURE DU PROJET À CRÉER

aiprod-v33/
├── .vscode/
│ ├── extensions.json
│ ├── settings.json
│ ├── launch.json
│ └── templates.code-snippets
├── src/
│ ├── orchestrator/
│ │ ├── init.py
│ │ ├── state_machine.py
│ │ └── transitions.py
│ ├── agents/
│ │ ├── init.py
│ │ ├── creative_director.py
│ │ ├── visual_translator.py
│ │ ├── semantic_qa.py
│ │ ├── fast_track_agent.py
│ │ └── render_executor.py
│ ├── functions/
│ │ ├── init.py
│ │ ├── financial_orchestrator.py
│ │ ├── technical_qa_gate.py
│ │ └── input_sanitizer.py
│ ├── memory/
│ │ ├── init.py
│ │ ├── memory_manager.py
│ │ ├── schema_validator.py
│ │ └── exposed_memory.py
│ ├── utils/
│ │ ├── init.py
│ │ ├── gcp_client.py
│ │ ├── llm_wrappers.py
│ │ ├── cache_manager.py
│ │ └── monitoring.py
│ └── api/
│ ├── init.py
│ └── main.py
├── config/
│ └── v33.json (déjà fourni)
├── tests/
│ ├── unit/
│ │ ├── test_memory_manager.py
│ │ ├── test_financial_orchestrator.py
│ │ └── test_creative_director.py
│ ├── integration/
│ │ └── test_full_pipeline.py
│ └── performance/
│ └── test_pipeline_performance.py
├── scripts/
│ ├── setup_gcp.sh
│ ├── deploy.sh
│ └── monitor.py
├── docs/
│ ├── architecture.md
│ └── api_documentation.md
├── deployments/
│ ├── cloudrun.yaml
│ ├── cloudfunctions.yaml
│ └── monitoring.yaml
├── credentials/
│ └── .gitkeep
├── .env.example
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
└── README.md

## 🔧 CONTRAINTES TECHNIQUES

- **Python 3.10+** avec typage fort (type hints)
- **Architecture asynchrone** (async/await) pour les agents
- **Configuration externalisée** depuis v33.json
- **Google Cloud Platform** comme infrastructure principale
- **FastAPI** pour l'API REST
- **Pydantic** pour la validation des données
- **Tests unitaires et d'intégration** complets

## 🎨 STYLE DE CODE

- **Documentation complète** avec docstrings Google style
- **Logging structuré** avec différents niveaux
- **Gestion d'erreurs élégante** avec retry mechanisms
- **Code modulaire** avec séparation des responsabilités
- **Performance optimisée** avec caching et batching

## 📋 EXIGENCES FONCTIONNELLES (basées sur v33.json)

### 1. ORCHESTRATOR (État Machine)

- Implémenter les états : INIT, ANALYSIS, CREATIVE_DIRECTION, VISUAL_TRANSLATION, FINANCIAL_OPTIMIZATION, RENDER_EXECUTION, QA_TECHNICAL, QA_SEMANTIC, FINALIZE, ERROR, FAST_TRACK
- Gérer les transitions conditionnelles (fast vs full pipeline)
- Implémenter retry policy (maxRetries: 3, backoffSec: 15)

### 2. MEMORY MANAGER

- Système de mémoire partagée avec validation de schéma
- Mémoire exposée pour ICC (Interface Client Collaboratif)
- Cache de cohérence avec TTL 168h

### 3. CREATIVE DIRECTOR (Agent principal)

- Fusion de 4 agents : Reasoner + ICRL + ACT + ScriptMind
- Génère ProductionManifest avec consistency_markers
- Intègre le cache de cohérence
- Utilise Gemini 1.5 Pro avec fallback vers Flash

### 4. FINANCIAL ORCHESTRATOR (Déterministe)

- Décisions financières SANS LLM
- Optimisation coût/qualité basée sur rules
- Dynamic pricing avec updateIntervalHours: 24
- Certification des coûts avec audit trail

### 5. DOUBLE QA SYSTEM

- QA Technique : vérifications binaires déterministes
- QA Sémantique : évaluation par vision LLM (Gemini 1.5 Pro Vision)
- Rapports interactifs pour ICC

### 6. FAST TRACK AGENT

- Pipeline simplifié pour complexité < 0.3
- Contraintes : maxDurationSec: 30, maxScenes: 3, noDialogue: true
- Performance target : maxLatencySec: 20, costCeiling: 0.3

## 🚀 COMME DÉMARRER

1. **Pour chaque fichier** : Commence par le code le plus critique
2. **Approche incrémentale** : Implémente un composant, teste, passe au suivant
3. **Priorité des composants** :
   - Memory Manager + Orchestrator (fondation)
   - Creative Director (cœur métier)
   - Financial Orchestrator (différenciation)
   - API + Déploiement (livrable)
4. **Tests en parallèle** : Écrire les tests pendant le développement

## 💡 CONSEILS D'IMPLÉMENTATION

- Utiliser `@dataclass` pour les DTOs
- Implémenter `__str__` et `__repr__` pour le debugging
- Configurer le logging avec rotation des fichiers
- Ajouter des métriques de performance (latence, coût, qualité)
- Prévoir l'extension avec de nouveaux backends (Sora, etc.)

## 🎯 LIVRABLE FINAL

Une application Cloud-Native prête pour le déploiement sur Google Cloud Platform avec :

- ✅ Pipeline complet fonctionnel
- ✅ API REST documentée
- ✅ Interface Client Collaboratif (ICC)
- ✅ Monitoring et alertes
- ✅ Tests automatisés
- ✅ Documentation technique

## ❓ QUESTIONS À SE POSER POUR CHAQUE COMPOSANT

1. Quelles sont les entrées/sorties définies dans v33.json ?
2. Comment gérer les erreurs et retries ?
3. Quels logs sont nécessaires pour le debugging ?
4. Comment exposer cette fonctionnalité à l'ICC ?
5. Comment tester ce composant de manière isolée ?

---

**NOTE AU DÉVELOPPEUR** : Ce prompt est conçu pour être utilisé avec GitHub Copilot dans VS Code. Attache le fichier `AIPROD.json` comme contexte. Commence par créer la structure de dossiers, puis génère chaque fichier en suivant les spécifications du JSON. Pose-moi des questions si une partie du fichier de configuration n'est pas claire.
