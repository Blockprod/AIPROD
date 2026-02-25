# Audit professionnel du projet AIPROD (proposition opérationnelle)

_Date : 2026-02-25_

## 1) Objectif

Mettre en place un **audit global, structuré et actionnable** du monorepo AIPROD afin de :

- réduire le risque technique et opérationnel,
- sécuriser la conformité “souveraineté”,
- améliorer la maintenabilité et la vélocité delivery,
- prioriser un plan de remédiation avec ROI clair.

---

## 2) Périmètre couvert

### 2.1 Périmètre technique

- Monorepo racine (`packages/aiprod-core`, `packages/aiprod-pipelines`, `packages/aiprod-trainer`, `packages/aiprod-cloud`).
- Outillage transversal : `requirements.txt`, `pyproject.toml`, `pytest.ini`, `.github/workflows/`.
- Scripts d’exploitation (`scripts/`, `deploy/`, `config/`).
- Documentation critique (`README.md`, `docs/`).

### 2.2 Périmètre qualité et risques

- Architecture & modularité.
- Qualité de code (linting, typing, complexité, duplication).
- Robustesse tests (unitaires, intégration, souveraineté).
- CI/CD et supply-chain.
- Sécurité applicative et secret management.
- Performance et coût d’exécution GPU/CPU.
- Exploitabilité (observabilité, runbooks, recovery).
- Gouvernance documentaire.

---

## 3) Méthodologie d’audit (niveau cabinet)

## Phase A — Cadrage (J0/J1)

- Interviews ciblées (Tech Lead, MLOps, Ops, Produit).
- Définition des objectifs mesurables (KPI qualité/sécurité/perf).
- Validation du périmètre des environnements (dev, CI, training, prod).

## Phase B — Audit statique et dynamique (J1 à J3)

- Revue structure du repo + conventions.
- Exécution checks automatisés (tests, lint, règles souveraineté).
- Scan surface de risque (TODO/FIXME sensibles, zones non testées, dépendances).
- Contrôle workflows CI/CD (gates, reproductibilité, garde-fous).

## Phase C — Cartographie des risques (J3)

- Classification par gravité : **Critique / Haute / Moyenne / Faible**.
- Probabilité × impact (matrice de risque).
- Détection de “single points of failure” (techniques et process).

## Phase D — Plan de remédiation (J4)

- Quick wins (0–2 semaines).
- Actions structurantes (1–2 mois).
- Chantiers de transformation (trimestre).
- Estimation effort, dépendances et ordre d’exécution.

## Phase E — Restitution exécutive (J5)

- Rapport complet + version COMEX (2 pages).
- Backlog prêt à implémenter (tickets priorisés).
- Plan de gouvernance (revue mensuelle qualité/sécurité).

---

## 4) Livrables attendus

1. **Rapport d’audit détaillé** (forces, risques, preuves, recommandations).
2. **Matrice de risques** avec priorités et propriétaires.
3. **Roadmap de remédiation 30/60/90 jours**.
4. **Tableau de bord KPI** (tests, dette, stabilité, sécurité, perf).
5. **Plan de contrôle continu** intégré à la CI.

---

## 5) Premières observations factuelles (pré-audit rapide)

Ces constats sont issus d’un pré-scan technique et servent d’amorce pour l’audit complet.

### Points positifs

- Monorepo structuré en packages métier cohérents (`core`, `pipelines`, `trainer`, `cloud`).
- Politique souveraineté déjà présente (tests dédiés + workflow CI explicite).
- Couverture tests non triviale (racine + packages).
- Outillage qualité défini (ruff/pytest/pre-commit).

### Points d’attention

- Dépendances majoritairement en `>=` (risque de non-reproductibilité et drift).  
  → À traiter via lock strict et stratégie d’upgrade contrôlée.
- Existence de TODO fonctionnels dans des zones API/pipeline (indicateur de dette active).
- Pipeline CI concentré autour d’un workflow principal : opportunité de renforcer la granularité des gates (lint/type/test/perf/sécu séparés).

---

## 6) Plan recommandé (30/60/90 jours)

## J+30 (stabilisation)

- Figer la chaîne de dépendances (lockfile reproductible, politique de mise à jour mensuelle).
- Élargir les quality gates CI : lint + typing + tests + règles sécurité basiques.
- Créer un registre de risques projet vivant (owner + échéance + statut).

## J+60 (industrialisation)

- Couverture tests par package + seuils minimaux progressifs.
- Audit des performances critiques (latence inference, mémoire GPU, throughput training).
- Standardiser runbooks incidents (training fail, OOM, corruption artefacts).

## J+90 (gouvernance continue)

- KPI hebdo automatisés (qualité/sécu/perf) avec tendance.
- Revue architecture trimestrielle (dette & simplification).
- Re-certification souveraineté avec preuve d’audit traçable.

---

## 7) Indicateurs de succès (KPI)

- Taux de succès CI (objectif > 95%).
- Temps médian de correction incident critique.
- Taux de tests passants sur branches protégées.
- Nombre de vulnérabilités critiques ouvertes (objectif 0).
- Écart de performance avant/après remédiation (GPU memory, latence, coût).

---

## 8) Proposition d’exécution

Si validé, je peux livrer l’audit en deux niveaux :

1. **Audit flash (48h)** : cartographie risques + quick wins priorisés.
2. **Audit complet (5 jours ouvrés)** : analyse exhaustive + roadmap 90 jours + tableau de bord KPI.

Dans les deux cas, la restitution sera orientée exécution avec backlog directement exploitable par l’équipe.
