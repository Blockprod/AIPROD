# Documentation API AIPROD

## URL de base

```
http://localhost:8000
```

## Endpoints

### Health Check

**Endpoint** : `GET /health`

Vérifie la disponibilité de l'API.

**Response** (200):

```json
{
  "status": "ok"
}
```

---

### Lancer le pipeline

**Endpoint** : `POST /pipeline/run`

Lance l'exécution complète du pipeline de génération vidéo.

**Request Body**:

```json
{
  "content": "Description du contenu vidéo",
  "priority": "low", // ou "high" pour Fast Track
  "lang": "en" // code langue (en, fr, etc.)
}
```

**Response** (200):

```json
{
  "status": "success",
  "state": "DELIVERED",
  "data": {
    "fast_track": {...},
    "fusion": {...},
    "render": {...},
    "semantic_qa": {...},
    "visual_translation": {...}
  }
}
```

**Response** (500): Erreur lors de l'exécution

```json
{
  "detail": "Error message"
}
```

---

### Statut du pipeline

**Endpoint** : `GET /pipeline/status`

Récupère l'état actuel du pipeline.

**Response** (200):

```json
{
  "state": "DELIVERED"
}
```

---

### Données ICC (Interface Client Collaboratif)

**Endpoint** : `GET /icc/data`

Récupère les données exposées à l'interface client.

**Response** (200):

```json
{
  "fast_track": {...},
  "fusion": {...},
  "render": {...},
  "semantic_qa": {...},
  "visual_translation": {...}
}
```

---

### Métriques de performance

**Endpoint** : `GET /metrics`

Récupère les métriques agrégées du pipeline.

**Response** (200):

```json
{
  "pipeline_executions": 5,
  "pipeline_errors": 0,
  "total_latency_ms": 8500,
  "total_cost": 3.75,
  "total_quality_score": 4.25,
  "avg_latency_ms": 1700,
  "avg_cost": 0.75,
  "avg_quality": 0.85
}
```

---

### Alertes actives

**Endpoint** : `GET /alerts`

Récupère les alertes déclenchées basées sur les seuils.

**Seuils** :

- high_latency: > 5000ms
- high_cost: > $1
- low_quality: < 0.6 (60%)
- high_error_rate: > 10 erreurs

**Response** (200):

```json
{
  "alerts": {
    "high_latency": true
  }
}
```

Ou vide si aucune alerte :

```json
{
  "alerts": {}
}
```

---

### Optimisation financière

**Endpoint** : `POST /financial/optimize`

Optimise le coût et la qualité selon les règles métier.

**Request Body**:

```json
{
  "complexity_score": 0.7,
  "assets": ["img1.png", "video1.mp4"]
}
```

**Response** (200):

```json
{
  "optimized_cost": 1.35,
  "quality": "optimal",
  "certification": "CERT-2026-01-12T12:00:00.000000"
}
```

---

### Validation technique

**Endpoint** : `POST /qa/technical`

Valide un manifeste selon les règles techniques déterministes.

**Request Body**:

```json
{
  "assets": ["img1.png", "video1.mp4"],
  "complexity_score": 0.5,
  "cost": 0.75,
  "quality_score": 0.85
}
```

**Response** (200):

```json
{
  "technical_valid": true,
  "checks": {
    "asset_count": true,
    "manifest_complete": true,
    "cost_valid": true,
    "quality_acceptable": true
  },
  "details": "All checks passed"
}
```

---

## Codes HTTP

| Code | Signification                          |
| ---- | -------------------------------------- |
| 200  | OK - Succès                            |
| 422  | Validation Error - Données invalides   |
| 500  | Internal Server Error - Erreur serveur |

## Formats de données

### Priorités

- `"low"` : Pipeline complet (full)
- `"high"` : Fast Track (latence minimale)

### Langues supportées

- `"en"` : Anglais
- `"fr"` : Français
- (extensible)

## Exemples de requêtes

### Fast Track simple

```bash
curl -X POST http://localhost:8000/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Créer une vidéo d'\''une personne qui danse",
    "priority": "high",
    "lang": "fr"
  }'
```

### Pipeline complet

```bash
curl -X POST http://localhost:8000/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Créer une vidéo cinématographique d'\''un paysage montagneux au coucher du soleil",
    "priority": "low",
    "lang": "en"
  }'
```

### Récupérer les métriques

```bash
curl http://localhost:8000/metrics
```

### Vérifier les alertes

```bash
curl http://localhost:8000/alerts
```

## Documentation interactive

Une documentation Swagger interactive est disponible à :

```
http://localhost:8000/docs
```

## Authentification

Pas d'authentification pour la version de développement.
À implémenter en production (API keys, JWT, etc.).

## Limites de débit

Aucune limite actuelle.
À implémenter en production avec rate limiting.

## Support et débogage

- **Logs** : Consultez `logs/AIPROD.log`
- **État du pipeline** : Utilisez `/pipeline/status`
- **Métriques** : Consultez `/metrics`
- **Alertes** : Consultez `/alerts`
