"""
AIPROD V33 - Phase 3 Integration Guide

Guide d'intégration des features Phase 3 dans l'application existante.
"""

# ============================================================================

# 1. INTEGRER LE SYSTÈME DE MÉTRIQUES

# ============================================================================

"""
Dans n'importe quel agent, importer et utiliser le collecteur de métriques:
"""

# Exemple dans RenderExecutor

from src.utils.custom_metrics import (
get_metrics_collector,
report_metric,
report_pipeline_complete,
report_error
)

# Collecter une métrique simple

collector = get_metrics_collector()
collector.report_metric("pipeline_duration", 45.2, {
"preset": "quick_social",
"job_id": "abc123"
})

# Reporter un pipeline complet

report_pipeline_complete(
job_id="abc123",
preset="quick_social",
duration_sec=45.2,
quality_score=0.87,
cost=30.0,
backend="runway"
)

# Reporter une erreur

report_error(
error_type="render_failed",
job_id="abc123",
backend="runway",
details="API timeout after 30 seconds"
)

# ============================================================================

# 2. UTILISER LE SYSTÈME MULTI-BACKEND

# ============================================================================

"""
Dans les jobs qui nécessitent la génération vidéo:
"""

from src.agents.render_executor import RenderExecutor, VideoBackend

# Créer un executor avec backend spécifique

executor = RenderExecutor(preferred_backend=VideoBackend.RUNWAY)

# Ou avec sélection automatique basée sur contraintes

executor = RenderExecutor()
backend = executor.\_select_backend(
budget_remaining=50.0, # Budget restant en dollars
quality_required=0.8, # Qualité minimale requise
speed_priority=False # True pour priorité vitesse (Replicate)
)

# Estimer le coût avant d'exécuter

cost = executor.\_estimate_cost(backend, duration=5)
print(f"Estimated cost: ${cost}")

# Exécuter le job avec le backend sélectionné

result = await executor.run(
prompt_bundle={
"text_prompt": "A beautiful sunset over the ocean",
"quality_required": 0.9 # Haute qualité
},
backend=backend,
budget_remaining=50.0
)

print(f"Backend utilisé: {result['backend']}")
print(f"Durée: {result['duration_seconds']:.1f}s")
print(f"Coût estimé: ${result['cost_estimate']:.2f}")

# ============================================================================

# 3. LIRE ET INTERPRÉTER LES ALERTES

# ============================================================================

"""
Les alertes sont créées automatiquement dans Cloud Monitoring.
Pour les consulter:
"""

# Via GCP Console

# https://console.cloud.google.com/monitoring/alerting

# Via gcloud CLI

# gcloud alpha monitoring policies list

# gcloud alpha monitoring policies describe <POLICY_ID>

# Les 5 alertes créées:

ALERTS = {
"Budget Warning": {
"threshold": "$90/day",
"action": "Notifier admin, limiter jobs premium",
"severity": "WARNING"
},
"Budget Critical": {
"threshold": "$100/day",
"action": "Bloquer nouveaux jobs",
"severity": "CRITICAL"
},
"Quality Low": {
"threshold": "quality_score < 0.6",
"action": "Switch vers backend premium",
"severity": "CRITICAL"
},
"Latency High": {
"threshold": "P95 > 900s",
"action": "Augmenter concurrence, activer fallback",
"severity": "CRITICAL"
},
"Runway Errors": {
"threshold": "> 5 errors/hour",
"action": "Activer fallback Replicate",
"severity": "WARNING"
}
}

# ============================================================================

# 4. MONITORER LES JOBS EN TEMPS RÉEL

# ============================================================================

"""
Le dashboard Cloud Monitoring affiche en temps réel:
"""

DASHBOARD_WIDGETS = {
"Pipeline Duration": ["P50", "P95", "P99"],
"Quality Score": "Average with 0.6 threshold",
"Daily Cost": "Stacked with $90 warning + $100 critical",
"Errors by Type": "Breakdown par type d'erreur",
"Jobs Today": "Scorecard avec count",
"Cost Today": "Scorecard avec total"
}

# ============================================================================

# 5. INTÉGRER DANS LE SUPERVISEUR

# ============================================================================

"""
Le Supervisor reçoit maintenant les métriques et peut les utiliser:
"""

# Exemple d'intégration dans supervisor.py

async def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]: # ... code existant ...

    from src.utils.custom_metrics import get_metrics_collector

    collector = get_metrics_collector()

    # Reporter la supervision
    if final_approval:
        collector.report_metric("jobs_completed", 1, {
            "preset": inputs.get("preset"),
            "quality_score": quality_score,
            "backend": inputs.get("backend", "unknown")
        })
    else:
        collector.report_metric("jobs_failed", 1, {
            "reason": rejection_reason,
            "preset": inputs.get("preset")
        })

    # ... rest of code ...

# ============================================================================

# 6. CONFIGURATION DE PRODUCTION

# ============================================================================

"""
Variables d'environnement requises:
"""

REQUIRED_ENV_VARS = {
"GCP_PROJECT_ID": "aiprod-484120",
"RUNWAYML_API_SECRET": "<your-runway-api-key>",
"REPLICATE_API_TOKEN": "<your-replicate-token>",
"GCS_BUCKET_NAME": "aiprod-484120-assets"
}

"""
Déployer les alertes:
"""

# Commande

DEPLOY_COMMAND = """
gcloud monitoring policies create \
 --policy-from-file=deployments/monitoring.yaml \
 --project=aiprod-484120
"""

"""
Créer les notification channels:
"""

CREATE_CHANNEL = """
gcloud beta monitoring channels create \\
--display-name="AIPROD Alerts" \\
--type="email" \\
--channel-labels=email_address=alerts@example.com \\
--project=aiprod-484120
"""

# ============================================================================

# 7. TESTER LES FEATURES PHASE 3

# ============================================================================

"""
Exécuter les tests:
"""

# Tests de concurrence (46 tests)

# pytest tests/load/test_concurrent_jobs.py -v

# Tests de coût (27 tests)

# pytest tests/load/test_cost_limits.py -v

# Tous les tests Phase 3

# pytest tests/load/ -v

# Avec coverage

# pytest tests/ --cov=src --cov-report=html

# ============================================================================

# 8. WORKFLOW D'UTILISATION COMPLÈTE

# ============================================================================

"""
Workflow complet d'un job avec Phase 3 features:
"""

import asyncio
from src.agents.render_executor import RenderExecutor, VideoBackend
from src.utils.custom_metrics import report_pipeline_complete, report_error

async def process_job(job_spec: Dict[str, Any]) -> Dict[str, Any]: # 1. Extraire le budget
budget_remaining = job_spec.get("budget", 100.0)

    # 2. Sélectionner le backend optimal
    executor = RenderExecutor()
    backend = executor._select_backend(
        budget_remaining=budget_remaining,
        quality_required=job_spec.get("quality_required", 0.8)
    )

    # 3. Vérifier le coût estimé
    cost_estimate = executor._estimate_cost(backend, 5)
    if cost_estimate > budget_remaining:
        return {
            "status": "rejected",
            "reason": f"Cost ${cost_estimate:.2f} exceeds budget ${budget_remaining:.2f}",
            "backend": backend.value
        }

    # 4. Exécuter le job
    import time
    start_time = time.time()

    try:
        result = await executor.run(
            prompt_bundle=job_spec.get("prompt_bundle", {}),
            backend=backend,
            budget_remaining=budget_remaining
        )

        duration = time.time() - start_time

        # 5. Reporter les métriques
        if result["status"] == "rendered":
            report_pipeline_complete(
                job_id=job_spec.get("job_id", "unknown"),
                preset=job_spec.get("preset", "custom"),
                duration_sec=duration,
                quality_score=job_spec.get("quality_score", 0.85),
                cost=cost_estimate,
                backend=backend.value
            )

        return result

    except Exception as e:
        # 6. Reporter les erreurs
        report_error(
            error_type="render_failed",
            job_id=job_spec.get("job_id"),
            backend=backend.value,
            details=str(e)
        )

        return {
            "status": "error",
            "error": str(e),
            "backend": backend.value
        }

# ============================================================================

# 9. MÉTRIQUES DISPONIBLES

# ============================================================================

"""
Métriques envoyées à Cloud Monitoring:
"""

AVAILABLE_METRICS = {
"Performance": [
"custom.googleapis.com/aiprod/pipeline_duration_seconds",
"custom.googleapis.com/aiprod/agent_duration_seconds",
"custom.googleapis.com/aiprod/render_duration_seconds"
],
"Quality": [
"custom.googleapis.com/aiprod/quality_score",
"custom.googleapis.com/aiprod/semantic_qa_score",
"custom.googleapis.com/aiprod/technical_qa_score"
],
"Cost": [
"custom.googleapis.com/aiprod/cost_total",
"custom.googleapis.com/aiprod/cost_per_job",
"custom.googleapis.com/aiprod/cost_per_minute"
],
"Counters": [
"custom.googleapis.com/aiprod/jobs_completed",
"custom.googleapis.com/aiprod/jobs_failed",
"custom.googleapis.com/aiprod/cache_hits",
"custom.googleapis.com/aiprod/cache_misses"
],
"Backend": [
"custom.googleapis.com/aiprod/backend_requests",
"custom.googleapis.com/aiprod/backend_errors",
"custom.googleapis.com/aiprod/backend_fallbacks"
]
}

# ============================================================================

# 10. TROUBLESHOOTING

# ============================================================================

"""
Problèmes courants et solutions:
"""

TROUBLESHOOTING = {
"Metrics not appearing": {
"causes": [
"Credentials GCP not configured",
"google-cloud-monitoring not installed",
"Project ID incorrect"
],
"solutions": [
"Run: gcloud auth application-default login",
"Run: pip install google-cloud-monitoring>=2.19.0",
"Check: export GCP_PROJECT_ID=aiprod-484120"
]
},
"Backend always selecting Replicate": {
"causes": [
"Budget too low",
"Quality threshold too high",
"Other backends unhealthy"
],
"solutions": [
"Increase budget_remaining parameter",
"Lower quality_required to 0.75 or less",
"Check backend error counts in logs"
]
},
"Alerts not triggering": {
"causes": [
"Alert policies not deployed",
"Notification channels not configured",
"Metrics not being sent"
],
"solutions": [
"Deploy: gcloud monitoring policies create ...",
"Create channels: gcloud beta monitoring channels create ...",
"Check logs: gcloud logging read 'severity>=WARNING'"
]
},
"Tests failing": {
"causes": [
"Mock mode not activated",
"Async/await issues",
"Import errors"
],
"solutions": [
"Tests automatically use mock for missing API keys",
"Use pytest-asyncio: pytest -v tests/load/",
"Run: pip install -r requirements.txt"
]
}
}
