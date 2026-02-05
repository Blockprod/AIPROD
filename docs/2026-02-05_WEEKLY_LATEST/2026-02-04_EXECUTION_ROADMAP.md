# 🚀 ROADMAP D'EXÉCUTION — PLAN STRUCTURÉ & PRIORISÉ

**Document de planification** : 4 février 2026  
**Horizon** : 4 février — 31 mai 2026  
**Responsable** : DevOps/SRE team  
**Status** : 🟢 **Production LIVE — 41 tâches restantes**

---

## 📊 Vue d'ensemble par priorité

| Phase        | Catégorie             | Tâches | Durée | Deadline  | Status     |
| ------------ | --------------------- | ------ | ----- | --------- | ---------- |
| **CRITIQUE** | Production Validation | 6      | 1h    | Feb 5     | 🔴 À faire |
| **PHASE 1**  | Sécurité Avancée      | 9      | 4h    | Feb 6-9   | 🟡 À faire |
| **PHASE 2**  | Infrastructure DB     | 5      | 3h    | Feb 17-28 | 🟡 À faire |
| **PHASE 3**  | API & Features        | 5      | 4h    | Feb 17-28 | 🟡 À faire |
| **PHASE 4**  | Documentation         | 5      | 4h    | Feb 17-28 | 🟡 À faire |
| **PHASE 5**  | Optimisations         | 11     | 6h    | Mar-Mai   | 📝 À faire |

**Total** : 41 tâches | **~22h** | **Feb 5 — May 31**

---

# 🔴 PHASE CRITIQUE — Production Validation (Feb 5)

**Deadline** : 5 février 2026  
**Durée totale** : ~1 heure  
**Objectif** : Confirmer que tout fonctionne en production  
**Dépendances** : None (production already live)  
**Success Criteria** : 100% des 6 validations ✅

---

## TÂCHE CRITIQUE 1.1 — Valider les endpoints API

**ID** : `CRIT-1.1`  
**Titre** : Confirmer tous les endpoints fonctionnels  
**Priorité** : 🔴 CRITIQUE  
**Durée** : 15 min  
**Dépendance** : Aucune

### Description

Tester que les 8 endpoints principaux répondent correctement et retournent le statut HTTP attendu.

### Checklist de validation

- [ ] GET /health → 200 OK avec `{"status": "ok"}`
- [ ] GET /docs → 200 OK (Swagger UI chargé)
- [ ] GET /metrics → 200 OK (Prometheus metrics exposées)
- [ ] POST /pipeline/run → 200 OK (test avec payload valide)
- [ ] GET /pipeline/{id} → 200 OK (avec un job valide)
- [ ] GET /pipeline/{id}/result → 200 OK ou 202 Accepted
- [ ] POST /auth/login → 200 OK (Firebase token validation)
- [ ] GET /presets → 200 OK (cost presets retournés)

### Commandes d'exécution

```bash
# 1. Health check
curl -s https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/health | jq .

# 2. Test Swagger docs
curl -s -I https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/docs | head -1

# 3. Check metrics endpoint
curl -s https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/metrics | head -20

# 4. Test OpenAPI schema
curl -s https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/openapi.json | jq '.info'

# 5. Full validation script
#!/bin/bash
endpoints=(
  "https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/health"
  "https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/docs"
  "https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/metrics"
  "https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/openapi.json"
)

for endpoint in "${endpoints[@]}"; do
  status=$(curl -s -o /dev/null -w "%{http_code}" "$endpoint")
  echo "$endpoint : $status"
done
```

### Métriques de succès

| Métrique           | Target | Validation |
| ------------------ | ------ | ---------- |
| HTTP 200 responses | 8/8    | ✅         |
| Response time      | <1s    | ✅         |
| Error rate         | 0%     | ✅         |

### Notes

- Si un endpoint échoue, vérifier Cloud Run logs immédiatement
- Vérifier que les certificats TLS sont valides
- Documenter tout problème trouvé

---

## TÂCHE CRITIQUE 1.2 — Vérifier Cloud SQL

**ID** : `CRIT-1.2`  
**Titre** : Vérifier intégrité de la base de données  
**Priorité** : 🔴 CRITIQUE  
**Durée** : 10 min  
**Dépendance** : Aucune

### Description

Confirmer que Cloud SQL PostgreSQL 14 est opérationnel, accessible et contient les données attendues.

### Checklist de validation

- [ ] Cloud SQL instance status = RUNNABLE
- [ ] Database accessible depuis Cloud Run
- [ ] Tables créées (jobs, results, pipeline_jobs, etc.)
- [ ] 0 erreurs de connexion dans logs
- [ ] Backups configurés et actifs
- [ ] Replicas configurés (si applicable)

### Commandes d'exécution

```bash
# 1. List Cloud SQL instances
gcloud sql instances list --project=aiprod-484120

# 2. Check specific instance
gcloud sql instances describe aiprod-postgres --project=aiprod-484120

# 3. Check instance status (should be RUNNABLE)
gcloud sql instances describe aiprod-postgres \
  --format="value(state)" \
  --project=aiprod-484120

# 4. List databases
gcloud sql databases list --instance=aiprod-postgres --project=aiprod-484120

# 5. Check recent logs for errors
gcloud sql operations list --instance=aiprod-postgres \
  --project=aiprod-484120 \
  --limit=10

# 6. Test connection from Cloud Run
gcloud sql connect aiprod-postgres \
  --user=postgres \
  --project=aiprod-484120 << EOF
SELECT version();
SELECT table_name FROM information_schema.tables WHERE table_schema='public';
EOF
```

### Métriques de succès

| Métrique          | Target   | Validation |
| ----------------- | -------- | ---------- |
| Instance status   | RUNNABLE | ✅         |
| Tables count      | ≥5       | ✅         |
| Connection errors | 0        | ✅         |
| Last backup age   | <24h     | ✅         |

### Notes

- Si problème de connexion, vérifier le VPC peering
- Vérifier que les credentials sont dans Secret Manager
- Vérifier la durée des requêtes (p95 < 50ms idéalement)

---

## TÂCHE CRITIQUE 1.3 — Confirmer Pub/Sub opérationnel

**ID** : `CRIT-1.3`  
**Titre** : Confirmer Pub/Sub opérationnel (async jobs)  
**Priorité** : 🔴 CRITIQUE  
**Durée** : 10 min  
**Dépendance** : Aucune

### Description

Vérifier que Google Cloud Pub/Sub fonctionne correctement pour les jobs asynchrones.

### Checklist de validation

- [ ] 3 topics existent et sont actifs
  - [ ] `aiprod-pipeline-jobs`
  - [ ] `aiprod-job-results`
  - [ ] `aiprod-job-notifications`
- [ ] 2 subscriptions existent et sont actives
  - [ ] `aiprod-job-processor` (pulls from pipeline-jobs)
  - [ ] `aiprod-result-notifier` (pulls from job-results)
- [ ] 0 messages en Dead Letter Queue (DLQ)
- [ ] Lag < 5 minutes sur les subscriptions
- [ ] Aucune souscription "stale"

### Commandes d'exécution

```bash
# 1. List topics
gcloud pubsub topics list --project=aiprod-484120

# 2. List subscriptions
gcloud pubsub subscriptions list --project=aiprod-484120

# 3. Check topic details
gcloud pubsub topics describe aiprod-pipeline-jobs --project=aiprod-484120

# 4. Check subscription details
gcloud pubsub subscriptions describe aiprod-job-processor \
  --project=aiprod-484120

# 5. Check for unacked messages (lag)
gcloud pubsub subscriptions pull aiprod-job-processor \
  --auto-ack \
  --limit=10 \
  --project=aiprod-484120

# 6. Full monitoring script
#!/bin/bash
echo "=== Topics ==="
gcloud pubsub topics list --project=aiprod-484120 --format="table(name)"
echo ""
echo "=== Subscriptions ==="
gcloud pubsub subscriptions list --project=aiprod-484120 \
  --format="table(name, topic, numUnackedMessages)"
```

### Métriques de succès

| Métrique         | Target    | Validation |
| ---------------- | --------- | ---------- |
| Topics           | 3 actifs  | ✅         |
| Subscriptions    | 2 actives | ✅         |
| Unacked messages | <1000     | ✅         |
| DLQ messages     | 0         | ✅         |

### Notes

- Si messages en DLQ, investiguer les jobs échoués
- Vérifier les logs de Cloud Run pour les erreurs de traitement
- Vérifier que les workers pullent les messages

---

## TÂCHE CRITIQUE 1.4 — Valider Prometheus metrics collection

**ID** : `CRIT-1.4`  
**Titre** : Valider Prometheus metrics collection  
**Priorité** : 🔴 CRITIQUE  
**Durée** : 10 min  
**Dépendance** : CRIT-1.1 (endpoints validés)

### Description

Confirmer que Prometheus expose les métriques correctement et que les métriques principales sont collectées.

### Checklist de validation

- [ ] `/metrics` endpoint répond en <100ms
- [ ] Métriques standard FastAPI exposées
  - [ ] `http_requests_total`
  - [ ] `http_request_duration_seconds`
  - [ ] `http_requests_in_progress`
- [ ] Métriques métier exposées
  - [ ] `jobs_completed_total`
  - [ ] `job_processing_duration_seconds`
  - [ ] `api_calls_by_endpoint`
- [ ] Aucune erreur de parsing Prometheus
- [ ] Format correct (OpenMetrics)

### Commandes d'exécution

```bash
# 1. Get metrics endpoint
curl -s https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/metrics | head -50

# 2. Check metric types
curl -s https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/metrics | grep "# TYPE"

# 3. Check http_requests_total
curl -s https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/metrics | grep "http_requests_total"

# 4. Check custom metrics
curl -s https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/metrics | grep "jobs_"

# 5. Test metrics parsing with promtool
promtool check metrics <<< $(curl -s https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/metrics)
```

### Métriques de succès

| Métrique       | Target | Validation |
| -------------- | ------ | ---------- |
| Response time  | <100ms | ✅         |
| Metric count   | >50    | ✅         |
| Parse errors   | 0      | ✅         |
| Custom metrics | >5     | ✅         |

### Notes

- Si metrics ne s'exposent pas, redémarrer le pod Cloud Run
- Vérifier que Prometheus config pointe vers cet endpoint
- Vérifier les logs pour les erreurs de collecte

---

## TÂCHE CRITIQUE 1.5 — Confirmer Cloud Logging live

**ID** : `CRIT-1.5`  
**Titre** : Confirmer Cloud Logging live  
**Priorité** : 🔴 CRITIQUE  
**Durée** : 10 min  
**Dépendance** : Aucune

### Description

Vérifier que Google Cloud Logging reçoit les logs de Cloud Run en temps réel.

### Checklist de validation

- [ ] Logs entrant dans Cloud Logging (dernières 10 minutes)
- [ ] Tous les niveaux représentés (DEBUG, INFO, WARNING, ERROR)
- [ ] Pas d'erreurs non traitées (error rate < 1%)
- [ ] Structured logging fonctionne (JSON format)
- [ ] Logs retention configuré (minimum 30 jours)
- [ ] Log exclusion filters fonctionnent (si applicable)

### Commandes d'exécution

```bash
# 1. Get recent logs (last 10 minutes)
gcloud logging read --project=aiprod-484120 \
  --limit=20 \
  --format=json

# 2. Filter by severity (ERROR)
gcloud logging read "severity=ERROR" \
  --project=aiprod-484120 \
  --limit=10

# 3. Check Cloud Run logs specifically
gcloud logging read "resource.type=cloud_run_revision" \
  --project=aiprod-484120 \
  --limit=10

# 4. Stream logs in real-time
gcloud logging read --project=aiprod-484120 \
  --follow

# 5. Count logs by level
gcloud logging read --project=aiprod-484120 \
  --format="value(severity)" \
  --limit=1000 | sort | uniq -c
```

### Métriques de succès

| Métrique        | Target      | Validation |
| --------------- | ----------- | ---------- |
| Recent logs     | >0 in 10min | ✅         |
| Error rate      | <1%         | ✅         |
| Structured logs | 100%        | ✅         |
| Retention       | ≥30 jours   | ✅         |

### Notes

- Si aucun log visible, vérifier IAM permissions
- Vérifier que l'application envoie les logs correctement
- Vérifier le buffer de logs (peut avoir 1-2min de délai)

---

## TÂCHE CRITIQUE 1.6 — Vérifier TLS/HTTPS enforcement

**ID** : `CRIT-1.6`  
**Titre** : Vérifier TLS/HTTPS enforcement  
**Priorité** : 🔴 CRITIQUE  
**Durée** : 10 min  
**Dépendance** : Aucune

### Description

Confirmer que tous les accès HTTP sont redirigés vers HTTPS avec un certificat valide.

### Checklist de validation

- [ ] HTTP → HTTPS redirect fonctionne (301/302)
- [ ] Certificat TLS valide (expiration > 30 jours)
- [ ] TLS version ≥ 1.2
- [ ] Cipher suites forts (pas de RC4, DES, etc.)
- [ ] HSTS headers présents
- [ ] Pas de mixed content (HTTP assets en HTTPS)

### Commandes d'exécution

```bash
# 1. Test HTTP redirect
curl -I -L http://aiprod-v33-api-hxhx3s6eya-ew.a.run.app 2>&1 | head -5

# 2. Check certificate details
openssl s_client -connect aiprod-v33-api-hxhx3s6eya-ew.a.run.app:443 \
  -servername aiprod-v33-api-hxhx3s6eya-ew.a.run.app < /dev/null 2>/dev/null | \
  openssl x509 -noout -dates -subject

# 3. Check TLS version
curl -v --tlsv1.2 https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/health 2>&1 | grep "TLS"

# 4. Check HSTS header
curl -I https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/health 2>&1 | grep -i "strict"

# 5. SSL Labs test (external, requires public domain)
# Visit: https://www.ssllabs.com/ssltest/analyze.html?d=aiprod-v33-api-hxhx3s6eya-ew.a.run.app

# 6. Quick security check
#!/bin/bash
url="https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app"
echo "Testing $url"
echo "1. HTTP Redirect:"
curl -I http://${url#https://} 2>&1 | grep -i "location"
echo ""
echo "2. Certificate:"
openssl s_client -connect ${url#https://}:443 -servername ${url#https://} </dev/null 2>/dev/null | openssl x509 -noout -dates
echo ""
echo "3. HSTS Header:"
curl -I $url 2>&1 | grep -i "strict-transport"
```

### Métriques de succès

| Métrique      | Target  | Validation |
| ------------- | ------- | ---------- |
| HTTP redirect | 301/302 | ✅         |
| Cert validity | >30j    | ✅         |
| TLS version   | ≥1.2    | ✅         |
| HSTS header   | Present | ✅         |

### Notes

- Cloud Run gère automatiquement les certificats (Google managed)
- S'assurer que HTTP est redirigé (pas de content servi en HTTP)
- Vérifier que les clients internes acceptent le cert

---

# 🟡 PHASE 1 — Sécurité Avancée (Feb 6-9)

**Deadline** : 9 février 2026  
**Durée totale** : ~4 heures  
**Objectif** : Hardener la sécurité en production  
**Dépendances** : PHASE CRITIQUE complétée  
**Success Criteria** : Toutes les mesures de sécurité avancées déployées

---

## TÂCHE 2.1 — Secret rotation policy (90 days)

**ID** : `SEC-2.1`  
**Titre** : Implement secret rotation policy (90 days)  
**Priorité** : 🟡 HAUTE  
**Durée** : 45 min  
**Dépendance** : CRIT-1.2 (Cloud SQL validé)

### Description

Créer une politique automatisée de rotation des secrets toutes les 90 jours via Cloud Scheduler.

### Checklist d'implémentation

- [ ] Cloud Scheduler job créé
- [ ] Cloud Function pour rotation écrite
- [ ] Secret Manager permissions configurées
- [ ] Rotation schedule = 90 jours
- [ ] Notification email lors de rotation
- [ ] Rollback mechanism en place
- [ ] Tested au moins une fois manuellement
- [ ] Documentation mise à jour

### Implémentation détaillée

```bash
# 1. Create Cloud Function for secret rotation
gcloud functions deploy rotate-secrets \
  --runtime python311 \
  --trigger-topic secret-rotation \
  --entry-point rotate_secrets \
  --project=aiprod-484120

# 2. Create Cloud Scheduler job
gcloud scheduler jobs create pubsub rotate-secrets-job \
  --schedule="0 0 1 */3 *" \
  --topic=secret-rotation \
  --message-body='{"action": "rotate"}' \
  --time-zone="UTC" \
  --location=europe-west1 \
  --project=aiprod-484120

# 3. Grant Cloud Function permissions
gcloud projects add-iam-policy-binding aiprod-484120 \
  --member=serviceAccount:rotate-secrets@aiprod-484120.iam.gserviceaccount.com \
  --role=roles/secretmanager.secretAccessor
```

### Code Cloud Function

```python
# functions/rotate_secrets.py
import functions_framework
from google.cloud import secretmanager
import os
from datetime import datetime, timedelta

@functions_framework.cloud_event
def rotate_secrets(cloud_event):
    """Rotate secrets older than 90 days"""
    client = secretmanager.SecretManagerServiceClient()
    project_id = "aiprod-484120"

    secrets_to_rotate = [
        "suno-api-key",
        "freesound-api-key",
        "google-cloud-api-key",
        "elevenlabs-api-key"
    ]

    for secret_name in secrets_to_rotate:
        try:
            secret = client.get_secret(
                request={"name": f"projects/{project_id}/secrets/{secret_name}"}
            )

            # Check if older than 90 days
            created = secret.created.timestamp()
            now = datetime.now().timestamp()
            age_days = (now - created) / (24 * 3600)

            if age_days > 90:
                # Trigger manual rotation
                # (In practice: alert human to rotate OR auto-rotate if applicable)
                print(f"Secret {secret_name} is {age_days} days old - needs rotation")
                # Send alert to ops team
        except Exception as e:
            print(f"Error checking {secret_name}: {e}")

    return "Rotation check completed"
```

### Métriques de succès

| Métrique              | Target       | Validation |
| --------------------- | ------------ | ---------- |
| Scheduler job created | Yes          | ✅         |
| Rotation frequency    | 90 days      | ✅         |
| Last rotation         | <90 days ago | ✅         |
| Function execution    | 0 errors     | ✅         |

### Notes

- Implémenter une vraie rotation (pas juste une alerte)
- Tester le rollback en cas d'erreur
- Configurer des alerts Slack/email
- Documenter le processus de rotation

---

## TÂCHE 2.2 — Create KMS keys for secret encryption

**ID** : `SEC-2.2`  
**Titre** : Create KMS keys for secret encryption  
**Priorité** : 🟡 HAUTE  
**Durée** : 30 min  
**Dépendance** : Aucune

### Description

Créer des clés KMS (Key Management Service) pour chiffrer les secrets sensibles.

### Checklist d'implémentation

- [ ] KMS keyring créé (`aiprod-keyring`)
- [ ] KMS key créée (`aiprod-secrets-key`)
- [ ] Key rotation automatique activée (90 jours)
- [ ] IAM bindings configurés pour Service Accounts
- [ ] Logging & monitoring du KMS activé
- [ ] Secrets chiffrés avec cette clé
- [ ] Backup keys testées

### Commandes d'implémentation

```bash
# 1. Create KMS keyring
gcloud kms keyrings create aiprod-keyring \
  --location=europe-west1 \
  --project=aiprod-484120

# 2. Create KMS key with automatic rotation
gcloud kms keys create aiprod-secrets-key \
  --location=europe-west1 \
  --keyring=aiprod-keyring \
  --purpose=encryption \
  --rotation-period=7776000s \
  --next-rotation-time=2026-05-04T00:00:00Z \
  --project=aiprod-484120

# 3. Grant Cloud Run service account permission to use key
gcloud kms keys add-iam-policy-binding aiprod-secrets-key \
  --location=europe-west1 \
  --keyring=aiprod-keyring \
  --member=serviceAccount:aiprod-cloud-run@aiprod-484120.iam.gserviceaccount.com \
  --role=roles/cloudkms.cryptoKeyEncrypterDecrypter \
  --project=aiprod-484120

# 4. Verify key creation
gcloud kms keys list --location=europe-west1 --keyring=aiprod-keyring --project=aiprod-484120

# 5. Check key details
gcloud kms keys versions list aiprod-secrets-key \
  --location=europe-west1 \
  --keyring=aiprod-keyring \
  --project=aiprod-484120
```

### Intégration avec Secret Manager

```python
# src/config/secrets.py
from google.cloud import secretmanager, kms_v1

def get_encrypted_secret(secret_name: str) -> str:
    """Get secret decrypted with KMS"""
    client = secretmanager.SecretManagerServiceClient()
    project_id = "aiprod-484120"

    # Get the secret
    response = client.access_secret_version(
        request={
            "name": f"projects/{project_id}/secrets/{secret_name}/versions/latest"
        }
    )

    # The secret is automatically decrypted by KMS
    return response.payload.data.decode("UTF-8")
```

### Métriques de succès

| Métrique        | Target     | Validation |
| --------------- | ---------- | ---------- |
| Keyring created | Yes        | ✅         |
| Key created     | Yes        | ✅         |
| Auto-rotation   | Every 90d  | ✅         |
| IAM bindings    | Configured | ✅         |

### Notes

- KMS key rotation = création auto de nouvelles versions
- Conserver les vieilles versions pour decrypt des anciens secrets
- Tester le decrypt avec chaque version
- Audit toutes les opérations de clé

---

## TÂCHE 2.3 — Enable Cloud Armor for DDoS protection

**ID** : `SEC-2.3`  
**Titre** : Enable Cloud Armor for DDoS protection  
**Priorité** : 🟡 HAUTE  
**Durée** : 30 min  
**Dépendance** : Aucune

### Description

Créer une security policy Cloud Armor pour protéger l'API contre les attaques DDoS.

### Checklist d'implémentation

- [ ] Security policy créée
- [ ] Policy attachée à Cloud Run service
- [ ] Règles par défaut créées (allow, deny rules)
- [ ] Logging activé
- [ ] Alert configurée pour attaques
- [ ] Test de blocage effectué
- [ ] Whitelist IPs configurée (si applicable)
- [ ] Rate limiting rules définies

### Commandes d'implémentation

```bash
# 1. Create Cloud Armor security policy
gcloud compute security-policies create aiprod-security-policy \
  --description="DDoS protection for AIPROD API" \
  --type=CLOUD_ARMOR \
  --project=aiprod-484120

# 2. Add default allow rule
gcloud compute security-policies rules create 65534 \
  --security-policy=aiprod-security-policy \
  --action=allow \
  --description="Default allow rule" \
  --project=aiprod-484120

# 3. Add geo-blocking rule (optional - block certain countries)
gcloud compute security-policies rules create 100 \
  --security-policy=aiprod-security-policy \
  --action=deny-403 \
  --description="Block high-risk countries" \
  --origin-region-list=KP,IR \
  --project=aiprod-484120

# 4. Add rate limiting rule
gcloud compute security-policies rules create 101 \
  --security-policy=aiprod-security-policy \
  --action=rate-based-ban \
  --rate-limit-options-enforce-on-key=IP \
  --rate-limit-options-rate-limit-threshold-count=1000 \
  --rate-limit-options-rate-limit-threshold-interval-sec=60 \
  --rate-limit-options-ban-duration-sec=600 \
  --description="Rate limit: 1000 req/min per IP, 10min ban" \
  --project=aiprod-484120

# 5. Enable logging
gcloud compute security-policies update aiprod-security-policy \
  --enable-layer7-ddos-defense \
  --project=aiprod-484120

# 6. Attach to Cloud Run (via Terraform or manually)
# See Terraform section below
```

### Terraform integration

```hcl
# infra/terraform/security.tf
resource "google_compute_security_policy" "aiprod_policy" {
  name        = "aiprod-security-policy"
  description = "DDoS protection for AIPROD API"

  rules {
    action   = "allow"
    priority = 65534
    description = "Default allow rule"
    match {
      versioned_expr = "LATEST"
    }
  }

  rules {
    action   = "deny(403)"
    priority = 100
    description = "Block high-risk countries"
    match {
      versioned_expr = "LATEST"
      origin_region_code = ["KP", "IR"]
    }
  }

  # Rate limiting rule
  rules {
    action   = "rate-based-ban"
    priority = 101
    description = "Rate limit protection"
    match {
      versioned_expr = "LATEST"
    }
    rate_limit_options {
      conform_action = "allow"
      exceed_action = "deny(429)"
      rate_limit_threshold {
        count = 1000
        interval_sec = 60
      }
      ban_duration_sec = 600
      enforce_on_key = "IP"
    }
  }
}

# Attach to backend service
resource "google_compute_backend_service" "aiprod" {
  security_policy = google_compute_security_policy.aiprod_policy.id
}
```

### Monitoring & Alerts

```bash
# View security policy logs
gcloud logging read \
  "resource.type=http_load_balancer AND resource.labels.policy_name=aiprod-security-policy" \
  --project=aiprod-484120 \
  --limit=20

# Check blocked requests
gcloud logging read \
  "resource.type=http_load_balancer AND severity=WARNING" \
  --project=aiprod-484120 \
  --limit=20
```

### Métriques de succès

| Métrique        | Target | Validation |
| --------------- | ------ | ---------- |
| Policy created  | Yes    | ✅         |
| Policy attached | Yes    | ✅         |
| Logging enabled | Yes    | ✅         |
| Test block      | Works  | ✅         |

### Notes

- Tester les rules sans bloquer le trafic d'abord (preview mode)
- Monitorer les faux positifs
- Ajuster les seuils de rate limiting selon le trafic réel
- Garder un whitelist pour les services critiques

---

## TÂCHE 2.4 — Implement SlowAPI rate limiting

**ID** : `SEC-2.4`  
**Titre** : Implement SlowAPI rate limiting  
**Priorité** : 🟡 HAUTE  
**Durée** : 30 min  
**Dépendance** : Aucune

### Description

Ajouter le rate limiting applicatif avec SlowAPI pour limiter les requêtes par IP/user.

### Checklist d'implémentation

- [ ] SlowAPI installé (`pip install slowapi`)
- [ ] Rate limiter middleware ajouté au FastAPI app
- [ ] Limites définies par endpoint
- [ ] Storage backend configuré (in-memory ou Redis)
- [ ] Retry-After headers retournés
- [ ] Exceptions gérées correctement
- [ ] Tests unitaires écrits
- [ ] Métriques de rate limiting exposées

### Installation et configuration

```bash
# 1. Install SlowAPI
pip install slowapi>=0.1.9

# 2. Update requirements.txt
echo "slowapi>=0.1.9" >> requirements.txt
pip install -r requirements.txt

# 3. Commit changes
git add requirements.txt src/api/main.py
git commit -m "Add SlowAPI rate limiting"
```

### Implémentation dans FastAPI

```python
# src/api/rate_limiting.py
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request
from slowapi.middleware import SlowAPIMiddleware

# Initialize limiter
limiter = Limiter(key_func=get_remote_address)

# Rate limit configurations by endpoint
RATE_LIMITS = {
    "free": "10/minute",      # 10 req/min
    "pro": "100/minute",      # 100 req/min
    "enterprise": "unlimited"  # No limit
}

def get_user_tier(request: Request) -> str:
    """Determine user tier from token/header"""
    # Implement based on your auth system
    return "free"  # default

async def rate_limit_key_builder(request: Request) -> str:
    """Custom key builder using user ID if available"""
    user_id = request.headers.get("X-User-ID")
    if user_id:
        return f"user:{user_id}"
    return get_remote_address(request)

def create_rate_limiter():
    return Limiter(
        key_func=rate_limit_key_builder,
        default_limits=["1000/hour"]  # Global default
    )

# src/api/main.py
from fastapi import FastAPI
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from src.api.rate_limiting import limiter, RATE_LIMITS

app = FastAPI()

# Add limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

async def _rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded"},
        headers={"Retry-After": "60"}
    )

# Apply limits to endpoints
@app.post("/pipeline/run")
@limiter.limit("10/minute")  # Default: 10 per minute
async def run_pipeline(request: Request, payload: PipelineRequest):
    # Implementation
    pass

@app.post("/pipeline/batch")
@limiter.limit("5/minute")  # Stricter: 5 per minute for batch
async def batch_pipeline(request: Request, payload: BatchRequest):
    # Implementation
    pass

@app.get("/metrics")
@limiter.limit("100/minute")  # Looser: 100 per minute for metrics
async def get_metrics(request: Request):
    # Implementation
    pass
```

### Tests unitaires

```python
# tests/unit/test_rate_limiting.py
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_rate_limit_exceeded():
    """Test that rate limit is enforced"""
    # Make 11 requests (limit is 10/min)
    for i in range(11):
        response = client.post("/pipeline/run", json={"test": "data"})

    # 11th request should be rate limited
    assert response.status_code == 429
    assert "Retry-After" in response.headers

def test_rate_limit_reset():
    """Test that rate limit resets after time window"""
    response1 = client.post("/pipeline/run", json={"test": "data"})
    assert response1.status_code in [200, 400]  # Not rate limited

    # Wait 61 seconds (limit is per minute)
    time.sleep(61)

    response2 = client.post("/pipeline/run", json={"test": "data"})
    assert response2.status_code in [200, 400]  # Not rate limited after reset
```

### Métriques de succès

| Métrique           | Target       | Validation |
| ------------------ | ------------ | ---------- |
| SlowAPI installed  | Yes          | ✅         |
| Limiter configured | Yes          | ✅         |
| Endpoints limited  | All critical | ✅         |
| Tests passing      | 100%         | ✅         |

### Notes

- Adapter les limites selon le trafic réel
- Implémenter tiered pricing avec limites différentes
- Utiliser Redis pour un distributed rate limiting
- Exclure certaines IPs (monitoring, healthchecks)

---

## TÂCHE 2.5 — Configure WAF rules

**ID** : `SEC-2.5`  
**Titre** : Configure WAF rules in Cloud Armor  
**Priorité** : 🟡 HAUTE  
**Durée** : 30 min  
**Dépendance** : SEC-2.3 (Cloud Armor créé)

### Description

Ajouter des règles WAF (Web Application Firewall) pour bloquer les attaques Web courantes.

### Checklist d'implémentation

- [ ] XSS protection enabled
- [ ] SQL injection detection
- [ ] Path traversal blocking
- [ ] Size limits sur payloads
- [ ] Protocol version enforcement
- [ ] Method whitelist (GET, POST, PUT, DELETE)
- [ ] Custom rules ajoutées
- [ ] Rules testées

### Règles WAF à ajouter

```bash
# 1. XSS protection rule
gcloud compute security-policies rules create 110 \
  --security-policy=aiprod-security-policy \
  --action=deny-403 \
  --description="Block potential XSS attacks" \
  --rules-enabled=owasp-crs-v030101-xss-rule-230000-xss_filter-1 \
  --project=aiprod-484120

# 2. SQL injection protection
gcloud compute security-policies rules create 111 \
  --security-policy=aiprod-security-policy \
  --action=deny-403 \
  --description="Block SQL injection attempts" \
  --rules-enabled=owasp-crs-v030101-sqli-rule-341330-sqli_filter-1 \
  --project=aiprod-484120

# 3. Protocol enforcement (only TLS 1.2+)
gcloud compute security-policies rules create 112 \
  --security-policy=aiprod-security-policy \
  --action=deny-403 \
  --description="Enforce modern TLS versions" \
  --expression="origin.ssl_version < 'TLSV1_2'" \
  --project=aiprod-484120

# 4. HTTP method whitelist (GET, POST, PUT, DELETE, OPTIONS)
gcloud compute security-policies rules create 113 \
  --security-policy=aiprod-security-policy \
  --action=deny-403 \
  --description="Whitelist HTTP methods" \
  --expression="request.method NOT IN ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'HEAD']" \
  --project=aiprod-484120

# 5. Payload size limit (max 10MB)
gcloud compute security-policies rules create 114 \
  --security-policy=aiprod-security-policy \
  --action=deny-413 \
  --description="Limit payload size to 10MB" \
  --expression="int(request.headers['content-length']) > 10485760" \
  --project=aiprod-484120

# 6. Custom rule: Block suspicious User-Agents
gcloud compute security-policies rules create 115 \
  --security-policy=aiprod-security-policy \
  --action=deny-403 \
  --description="Block suspicious user agents" \
  --expression="request.headers['user-agent'] CONTAINS 'sqlmap' || request.headers['user-agent'] CONTAINS 'nikto'" \
  --project=aiprod-484120
```

### Terraform WAF rules

```hcl
# infra/terraform/waf_rules.tf
resource "google_compute_security_policy" "aiprod_policy" {
  # ... existing config ...

  # Rule 110: XSS Protection
  rules {
    action = "deny(403)"
    priority = 110
    description = "Block potential XSS attacks"
    match {
      versioned_expr = "LATEST"
      expr {
        expression = "evaluatePreconfiguredExpr('xss-v33-stable')"
      }
    }
  }

  # Rule 111: SQL Injection Protection
  rules {
    action = "deny(403)"
    priority = 111
    description = "Block SQL injection attempts"
    match {
      versioned_expr = "LATEST"
      expr {
        expression = "evaluatePreconfiguredExpr('sqli-v33-stable')"
      }
    }
  }

  # Rule 113: HTTP Method Whitelist
  rules {
    action = "deny(403)"
    priority = 113
    description = "Whitelist HTTP methods"
    match {
      versioned_expr = "LATEST"
      expr {
        expression = "request.method NOT IN ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'HEAD']"
      }
    }
  }

  # Rule 114: Payload Size Limit
  rules {
    action = "deny(413)"
    priority = 114
    description = "Limit payload to 10MB"
    match {
      versioned_expr = "LATEST"
      expr {
        expression = "int(request.headers['content-length']) > 10485760"
      }
    }
  }
}
```

### Testing WAF rules

```bash
# 1. Test XSS detection
curl "https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/test?input=<script>alert('xss')</script>"
# Should return 403 Forbidden

# 2. Test SQL injection detection
curl "https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/test?id=1' OR '1'='1"
# Should return 403 Forbidden

# 3. Test large payload
curl -X POST "https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/test" \
  -H "Content-Type: application/json" \
  -d "$(python -c 'print("{\"data\": \"" + "x"*11000000 + "\"}")')"
# Should return 413 Payload Too Large

# 4. Test invalid HTTP method
curl -X PATCH "https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/test"
# Should return 403 Forbidden
```

### Monitoring WAF alerts

```bash
# View WAF-blocked requests
gcloud logging read \
  "resource.type=http_load_balancer AND httpRequest.status=403 AND httpRequest.requestUrl=~'.*aiprod.*'" \
  --project=aiprod-484120 \
  --limit=50 \
  --format=json | jq '.[] | {
    timestamp: .timestamp,
    client_ip: .httpRequest.clientIp,
    method: .httpRequest.requestMethod,
    path: .httpRequest.requestUrl,
    status: .httpRequest.status
  }'
```

### Métriques de succès

| Métrique          | Target | Validation |
| ----------------- | ------ | ---------- |
| XSS rule enabled  | Yes    | ✅         |
| SQLi rule enabled | Yes    | ✅         |
| Method whitelist  | Yes    | ✅         |
| Size limit        | Yes    | ✅         |
| Tests blocking    | 100%   | ✅         |

### Notes

- Tester les rules en "preview" d'abord
- Monitorer les faux positifs
- Documenter les exceptions
- Audit les règles trimestriellement

---

## TÂCHE 2.6 — Setup email alerts for critical errors

**ID** : `MON-2.6`  
**Titre** : Setup email alerts for critical errors  
**Priorité** : 🟡 HAUTE  
**Durée** : 45 min  
**Dépendance** : CRIT-1.5 (Cloud Logging validé)

### Description

Configurer des alertes email automatiques pour les erreurs critiques.

### Checklist d'implémentation

- [ ] Alert policy créée pour Error rate > 1%
- [ ] Alert policy créée pour API latency > 1s
- [ ] Alert policy créée pour Database errors
- [ ] Alert policy créée pour OOM/crash
- [ ] Email channels configurés
- [ ] Notification template personnalisé
- [ ] Alert testing effectué
- [ ] Documentation du runbook

### Commandes d'implémentation

```bash
# 1. Create notification channel for email
gcloud alpha monitoring channels create \
  --display-name="DevOps Team Email" \
  --type=email \
  --channel-labels=email_address=devops@aiprod.ai \
  --project=aiprod-484120

# 2. Create alert policy for error rate
gcloud monitoring policies create \
  --notification-channels=CHANNEL_ID \
  --display-name="High Error Rate Alert" \
  --condition-display-name="Error rate > 1%" \
  --condition-threshold-value=1 \
  --condition-threshold-duration=300s \
  --condition-threshold-filter='resource.type="cloud_run_revision" AND metric.type="run.googleapis.com/request_count" AND metric.labels.response_code_class="5xx"' \
  --project=aiprod-484120
```

### Terraform configuration

```hcl
# infra/terraform/monitoring.tf
resource "google_monitoring_notification_channel" "devops_email" {
  display_name = "DevOps Team Email"
  type         = "email"
  labels = {
    email_address = "devops@aiprod.ai"
  }
  enabled = true
}

resource "google_monitoring_alert_policy" "high_error_rate" {
  display_name = "High Error Rate Alert"
  combiner     = "OR"

  conditions {
    display_name = "Error rate > 1%"
    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND metric.type=\"run.googleapis.com/request_count\" AND metric.labels.response_code_class=\"5xx\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 1.0
      aggregations {
        alignment_period  = "60s"
        per_series_aligner = "ALIGN_RATE"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.devops_email.id]
  documentation {
    content = "High error rate detected. Check Cloud Logging for details."
    mime_type = "text/markdown"
  }
}

resource "google_monitoring_alert_policy" "high_latency" {
  display_name = "High API Latency Alert"
  combiner     = "OR"

  conditions {
    display_name = "p95 latency > 1s"
    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND metric.type=\"run.googleapis.com/request_latencies\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 1000  # milliseconds
      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_PERCENTILE_95"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.devops_email.id]
}

resource "google_monitoring_alert_policy" "database_errors" {
  display_name = "Database Connection Errors"
  combiner     = "OR"

  conditions {
    display_name = "DB errors > 5 in 5min"
    condition_threshold {
      filter          = "resource.type=\"cloudsql_database\" AND metric.type=\"cloudsql.googleapis.com/database/network/connections\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 5
    }
  }

  notification_channels = [google_monitoring_notification_channel.devops_email.id]
}
```

### Alert Policy Templates

```yaml
# configs/alert-rules.yaml
alert_policies:
  - name: "High Error Rate"
    threshold: 1.0 # percent
    duration: 300 # seconds
    channels: ["email", "slack"]
    runbook: "docs/runbooks/high-error-rate.md"

  - name: "High Latency"
    threshold: 1000 # milliseconds (p95)
    duration: 300
    channels: ["email", "slack"]
    runbook: "docs/runbooks/high-latency.md"

  - name: "Database Errors"
    threshold: 5
    duration: 300
    channels: ["email"]
    runbook: "docs/runbooks/database-errors.md"

  - name: "OOM/Memory Pressure"
    metric: "memory_usage"
    threshold: 95 # percent
    duration: 60
    channels: ["email", "sms"]
    runbook: "docs/runbooks/oom.md"
```

### Métriques de succès

| Métrique        | Target | Validation |
| --------------- | ------ | ---------- |
| Email channels  | 1+     | ✅         |
| Alert policies  | 3+     | ✅         |
| Test alert sent | Yes    | ✅         |
| Runbooks linked | All    | ✅         |

### Notes

- Tester les alertes manuellement
- Implémenter une escalade
- Définir des durées appropriées (éviter les faux positifs)
- Documenter l'action à prendre pour chaque alerte

---

## TÂCHE 2.7 — Configure Slack webhook integration

**ID** : `MON-2.7`  
**Titre** : Configure Slack webhook for notifications  
**Priorité** : 🟡 HAUTE  
**Durée** : 30 min  
**Dépendance** : MON-2.6 (Email alerts configurés)

### Description

Intégrer Slack pour recevoir les alertes et notifications en temps réel.

### Checklist d'implémentation

- [ ] Slack app créée
- [ ] Webhook URL générée
- [ ] Notification channel Slack créé dans Cloud Monitoring
- [ ] Alert policies attachées à Slack
- [ ] Test alert envoyé
- [ ] Formatage des messages optimisé
- [ ] Auto-update de Slack app (Python script)

### Setup Slack

```bash
# 1. Create Slack app (https://api.slack.com/apps)
# - Name: AIPROD Alerts
# - Scopes: chat:write, incoming-webhook
# - Create webhook
# - Copy webhook URL

# 2. Store webhook in Secret Manager
gcloud secrets create slack-webhook-url \
  --replication-policy="automatic" \
  --data-file=- << EOF
https://hooks.slack.com/services/YOUR/WEBHOOK/URL
EOF

# 3. Grant Cloud Function permission
gcloud projects add-iam-policy-binding aiprod-484120 \
  --member=serviceAccount:alert-sender@aiprod-484120.iam.gserviceaccount.com \
  --role=roles/secretmanager.secretAccessor
```

### Cloud Function for Slack notifications

````python
# functions/send_slack_alert.py
import functions_framework
from google.cloud import secretmanager
import requests
import json
from datetime import datetime

def get_slack_webhook():
    """Get Slack webhook URL from Secret Manager"""
    client = secretmanager.SecretManagerServiceClient()
    secret = client.access_secret_version(
        request={
            "name": "projects/aiprod-484120/secrets/slack-webhook-url/versions/latest"
        }
    )
    return secret.payload.data.decode("UTF-8")

@functions_framework.cloud_event
def send_slack_alert(cloud_event):
    """Send alert to Slack"""
    import base64

    # Get alert details from Pub/Sub message
    pubsub_message = base64.b64decode(cloud_event.data["message"]["data"]).decode()
    alert_data = json.loads(pubsub_message)

    webhook_url = get_slack_webhook()

    # Format Slack message
    message = {
        "text": f":warning: *AIPROD Alert*",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"🚨 {alert_data['policy_name']}"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Severity:*\n{alert_data['severity']}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Time:*\n{datetime.now().isoformat()}"
                    }
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Details:*\n```{alert_data['condition']}```"
                }
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "View in Cloud Monitoring"
                        },
                        "url": alert_data['runbook_url']
                    }
                ]
            }
        ]
    }

    # Send to Slack
    response = requests.post(webhook_url, json=message)
    return f"Alert sent: {response.status_code}"
````

### Créer notification channel dans Terraform

```hcl
# infra/terraform/slack_integration.tf
resource "google_monitoring_notification_channel" "slack" {
  display_name = "AIPROD Slack #alerts"
  type         = "slack"
  labels = {
    channel_name = "#alerts"
  }
  enabled = true
}

# Attach to alert policies
resource "google_monitoring_alert_policy" "example" {
  # ... alert config ...
  notification_channels = [google_monitoring_notification_channel.slack.id]
}
```

### Message Template pour Slack

```python
def format_alert_for_slack(alert):
    """Format CloudMonitoring alert for Slack"""
    return {
        "text": f":warning: *{alert['policy_name']}*",
        "attachments": [
            {
                "color": "danger" if alert['severity'] == "CRITICAL" else "warning",
                "fields": [
                    {
                        "title": "Condition",
                        "value": alert['condition_name'],
                        "short": False
                    },
                    {
                        "title": "Threshold",
                        "value": f"{alert['threshold']} {alert['unit']}",
                        "short": True
                    },
                    {
                        "title": "Current Value",
                        "value": f"{alert['current_value']} {alert['unit']}",
                        "short": True
                    }
                ],
                "actions": [
                    {
                        "type": "button",
                        "text": "View Details",
                        "url": alert['details_url']
                    },
                    {
                        "type": "button",
                        "text": "View Runbook",
                        "url": alert['runbook_url']
                    }
                ]
            }
        ]
    }
```

### Métriques de succès

| Métrique             | Target | Validation |
| -------------------- | ------ | ---------- |
| Slack app created    | Yes    | ✅         |
| Webhook configured   | Yes    | ✅         |
| Notification channel | Yes    | ✅         |
| Test message sent    | Yes    | ✅         |

### Notes

- Tester le webhook avant production
- Implémenter le throttling pour éviter le spam
- Ajouter des emojis pour meilleure visibilité
- Créer des channels par severity

---

## TÂCHE 2.8 — Create incident escalation policy

**ID** : `MON-2.8`  
**Titre** : Create incident escalation policy  
**Priorité** : 🟡 HAUTE  
**Durée** : 30 min  
**Dépendance** : MON-2.6 (Alerts configurés)

### Description

Documenter et mettre en place une politique d'escalade des incidents.

### Checklist d'implémentation

- [ ] Runbook créé: `docs/incident-response.md`
- [ ] On-call schedule défini
- [ ] Escalation tiers documentés
- [ ] Response time SLAs définis
- [ ] War room procedure écrite
- [ ] Post-mortem template créé
- [ ] Notification channels configurés par severity

### Fichier d'escalade

```markdown
# Incident Response & Escalation Policy

## Severity Levels

### P1 - CRITICAL (Escalate immediately)

- Production API down
- Data loss/corruption
- Security breach
- **Response SLA**: 5 minutes
- **Resolution SLA**: 30 minutes
- **Escalation**: To CTO + DevOps lead

### P2 - HIGH (Escalate within 15 min)

- Significant performance degradation
- Feature unavailable
- High error rate (>5%)
- **Response SLA**: 15 minutes
- **Resolution SLA**: 2 hours
- **Escalation**: To team lead

### P3 - MEDIUM (Escalate within 1 hour)

- Minor issues
- Low error rate (<1%)
- Slow performance
- **Response SLA**: 1 hour
- **Resolution SLA**: 8 hours

### P4 - LOW (Track and prioritize)

- Documentation issues
- Minor bugs
- Enhancement requests
- **No SLA**

## Escalation Paths
```

On-Call Engineer (primary)
↓ (15 min no response)
Team Lead
↓ (30 min no resolution)
CTO/Engineering Manager
↓ (1 hour critical, no resolution)
CEO/COO

```

## War Room Protocol

1. **Alert received** → Immediately join Slack #incidents
2. **Investigation** → Log findings in ticket
3. **Status updates** → Every 5 minutes to #incidents (P1), 15 min (P2)
4. **Resolution** → Document fix, test, deploy
5. **Post-mortem** → Schedule within 48 hours

## Response Checklist

- [ ] Acknowledge alert in Slack
- [ ] Create incident ticket (JIRA/GitHub Issues)
- [ ] Investigate root cause
- [ ] Implement fix/workaround
- [ ] Deploy fix
- [ ] Verify resolution
- [ ] Close ticket
- [ ] Schedule post-mortem
```

### On-Call Schedule

```yaml
# config/oncall_schedule.yaml
week_1:
  monday_friday: "engineer@aiprod.ai"
  weekend: "backup-engineer@aiprod.ai"

week_2:
  monday_friday: "backup-engineer@aiprod.ai"
  weekend: "engineer@aiprod.ai"

backup_contacts:
  level_1: "team-lead@aiprod.ai"
  level_2: "cto@aiprod.ai"

notification:
  p1: ["slack", "email", "sms"]
  p2: ["slack", "email"]
  p3: ["slack"]
  p4: ["email"]
```

### Post-Mortem Template

```markdown
# Post-Mortem Report

**Date**: [Date]
**Duration**: [Start] — [End]
**Severity**: P[1-4]
**Status**: Resolved

## Summary

[Brief description of the incident]

## Timeline

- **HH:MM** - Alert triggered
- **HH:MM** - On-call notified
- **HH:MM** - Root cause identified
- **HH:MM** - Fix deployed
- **HH:MM** - Verified resolved

## Root Cause

[Technical analysis of what went wrong]

## Impact

- **Users affected**: [Number]
- **Duration**: [Time]
- **Data affected**: [Yes/No]

## Actions Taken

1. [Action]
2. [Action]

## Follow-up Actions

- [ ] [Action] - Owner: [Name] - Deadline: [Date]
- [ ] [Action] - Owner: [Name] - Deadline: [Date]

## Lessons Learned

- [Learning 1]
- [Learning 2]

## Prevention

[How to prevent this in the future]
```

### Métriques de succès

| Métrique             | Target     | Validation |
| -------------------- | ---------- | ---------- |
| Runbook created      | Yes        | ✅         |
| On-call schedule     | Defined    | ✅         |
| SLAs documented      | All levels | ✅         |
| War room procedure   | Written    | ✅         |
| Post-mortem template | Ready      | ✅         |

### Notes

- Conduire drill mensuel
- Tester escalation avec équipe
- Mettre à jour on-call schedule hebdo
- Archiver post-mortems pour analyse

---

## TÂCHE 2.9 — Setup Grafana dashboards for production metrics

**ID** : `MON-2.9`  
**Titre** : Setup Grafana dashboards for production metrics  
**Priorité** : 🟡 HAUTE  
**Durée** : 30 min  
**Dépendance** : CRIT-1.4 (Prometheus validé)

### Description

Créer des dashboards Grafana pour visualiser les métriques de production.

### Checklist d'implémentation

- [ ] Datasource Prometheus connectée à Grafana
- [ ] Dashboard "API Overview" créé
- [ ] Dashboard "Infrastructure" créé
- [ ] Dashboard "Business Metrics" créé
- [ ] Alerts intégrées dans dashboards
- [ ] Annotations pour deployments activées
- [ ] Dashboard templates sauvegardés
- [ ] Public dashboards (read-only) créés

### Installation Grafana

```bash
# 1. Install Grafana (si pas déjà installé)
docker pull grafana/grafana:latest

# 2. Run Grafana
docker run -d -p 3000:3000 \
  -e GF_SECURITY_ADMIN_PASSWORD=admin \
  -v grafana-storage:/var/lib/grafana \
  grafana/grafana

# Ou via Kubernetes/Cloud Run si applicable
```

### Configuration Prometheus Datasource

```bash
# 1. Access Grafana: http://localhost:3000
# 2. Go to Configuration → Data Sources
# 3. Add Prometheus
# 4. URL: http://prometheus:9090 (ou GCP URL)
# 5. Save & Test
```

### Dashboard JSON Templates

```json
{
  "dashboard": {
    "title": "AIPROD API Overview",
    "tags": ["production", "api"],
    "panels": [
      {
        "title": "Requests per Second",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total[5m]))"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Error Rate (%)",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{status_code=~'5..'}[5m])) / sum(rate(http_requests_total[5m])) * 100"
          }
        ],
        "alert": {
          "name": "High Error Rate",
          "conditions": [
            {
              "evaluator": { "type": "gt", "params": [1] },
              "operator": "and"
            }
          ]
        }
      },
      {
        "title": "API Latency (p95)",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "Database Connections",
        "targets": [
          {
            "expr": "pg_stat_activity_count"
          }
        ]
      },
      {
        "title": "Cloud Run Instances",
        "targets": [
          {
            "expr": "cloud_run_instance_count"
          }
        ]
      }
    ]
  }
}
```

### Dashboards à créer

**1. API Overview Dashboard**

```
- Requests/sec
- Error rate
- Latency (p50, p95, p99)
- Top endpoints (by calls)
- Top errors
```

**2. Infrastructure Dashboard**

```
- Cloud Run instances
- CPU usage
- Memory usage
- Network I/O
- Disk usage
```

**3. Database Dashboard**

```
- Connections
- Query latency
- Slow queries
- Connection pool usage
- Query errors
```

**4. Business Metrics Dashboard**

```
- Jobs completed
- Revenue per job
- User signups
- Conversion rate
- Cost per job
```

### Prometheus Queries

```promql
# API Metrics
sum(rate(http_requests_total[5m]))                           # RPS
sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))   # Error rate
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))  # p95 latency

# Infrastructure
node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes  # Memory %
rate(node_cpu_seconds_total[5m]) * 100                       # CPU %
node_network_receive_bytes_total                              # Network in
node_network_transmit_bytes_total                             # Network out

# Database
pg_stat_activity_count                                        # DB connections
avg(rate(pg_stat_statements_exec_time[5m])) / 1000           # Query latency (ms)
```

### Grafana Provisioning (Terraform)

```hcl
# infra/terraform/grafana.tf
resource "grafana_folder" "aiprod" {
  title = "AIPROD"
}

resource "grafana_dashboard" "api_overview" {
  folder = grafana_folder.aiprod.id
  config_json = templatefile("${path.module}/dashboards/api-overview.json", {
    datasource_uid = grafana_data_source.prometheus.uid
  })
}

resource "grafana_dashboard" "infrastructure" {
  folder = grafana_folder.aiprod.id
  config_json = file("${path.module}/dashboards/infrastructure.json")
}

resource "grafana_data_source" "prometheus" {
  type       = "prometheus"
  name       = "Prometheus"
  url        = "http://prometheus:9090"
  is_default = true
}
```

### Métriques de succès

| Métrique    | Target | Validation |
| ----------- | ------ | ---------- |
| Datasources | 1+     | ✅         |
| Dashboards  | 4+     | ✅         |
| Panels      | 20+    | ✅         |
| Alerts      | 5+     | ✅         |

### Notes

- Exporter dashboards en JSON pour versionning
- Configurer auto-refresh (30s pour live, 1m pour historical)
- Ajouter annotations pour deployments
- Créer dashboards read-only pour clients

---

# 🟡 PHASE 2 — Infrastructure Database (Feb 17-28)

**Deadline** : 28 février 2026  
**Durée totale** : ~3 heures  
**Objectif** : Optimiser la base de données  
**Dépendances** : PHASE 1 complétée  
**Success Criteria** : Tous les optimisations DB appliquées

---

## TÂCHE 3.1 — Add performance indexes to Cloud SQL

**ID** : `DB-3.1`  
**Titre** : Add performance indexes to Cloud SQL  
**Priorité** : 🟡 HAUTE  
**Durée** : 45 min  
**Dépendance** : CRIT-1.2 (Cloud SQL validé)

### Description

Ajouter des index de performance sur les tables principales pour accélérer les requêtes.

### Checklist d'implémentation

- [ ] Index sur `jobs(status)` créé
- [ ] Index sur `jobs(created_at DESC)` créé
- [ ] Index composé sur `jobs(user_id, status)` créé
- [ ] Index sur `results(job_id)` créé
- [ ] Index sur `pipeline_jobs(id, status)` créé
- [ ] Autres indexes créés selon slow query log
- [ ] Query plans analysés et validés
- [ ] Performance amélioration mesurée

### Commandes SQL

```bash
# Connecter à Cloud SQL
gcloud sql connect aiprod-postgres \
  --user=postgres \
  --project=aiprod-484120

# Exécuter les index creation statements:
```

```sql
-- 1. Index on jobs table
CREATE INDEX CONCURRENTLY idx_jobs_status ON jobs(status);
CREATE INDEX CONCURRENTLY idx_jobs_created_at ON jobs(created_at DESC);
CREATE INDEX CONCURRENTLY idx_jobs_user_status ON jobs(user_id, status);

-- 2. Index on results table
CREATE INDEX CONCURRENTLY idx_results_job_id ON results(job_id);
CREATE INDEX CONCURRENTLY idx_results_created_at ON results(created_at DESC);

-- 3. Index on pipeline_jobs
CREATE INDEX CONCURRENTLY idx_pipeline_jobs_status ON pipeline_jobs(status);
CREATE INDEX CONCURRENTLY idx_pipeline_jobs_created_at ON pipeline_jobs(created_at DESC);

-- 4. Index on pipeline_runs (if exists)
CREATE INDEX CONCURRENTLY idx_pipeline_runs_job_id ON pipeline_runs(job_id);
CREATE INDEX CONCURRENTLY idx_pipeline_runs_created_at ON pipeline_runs(created_at DESC);

-- 5. Analyze query plans
EXPLAIN ANALYZE
SELECT * FROM jobs WHERE user_id = '123' AND status = 'COMPLETED' ORDER BY created_at DESC LIMIT 10;

-- 6. Check index usage
SELECT schemaname, tablename, indexname, idx_scan
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- 7. Check table sizes
SELECT schemaname, tablename,
  pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

### Terraform migration

```hcl
# infra/terraform/database_indexes.tf
resource "google_sql_database_instance" "aiprod" {
  # ... existing config ...

  # Note: Indexes are created via migration scripts, not Terraform
  # See migrations/versions/002_add_performance_indexes.py
}

# This would be in Alembic migration:
# def upgrade():
#     op.create_index('idx_jobs_status', 'jobs', ['status'], unique=False)
#     op.create_index('idx_jobs_user_status', 'jobs', ['user_id', 'status'], unique=False)
#     ...
```

### Performance validation

```bash
# Run performance tests before and after
# Compare query execution times

# Before indexes:
# SELECT * FROM jobs WHERE user_id='123' ORDER BY created_at DESC LIMIT 10
# Execution time: 500ms

# After indexes:
# SELECT * FROM jobs WHERE user_id='123' ORDER BY created_at DESC LIMIT 10
# Execution time: 50ms  (10x improvement)
```

### Monitoring index performance

```sql
-- Check index effectiveness (should be >0)
SELECT schemaname, tablename, indexname, idx_scan
FROM pg_stat_user_indexes
WHERE idx_scan > 0
ORDER BY idx_scan DESC;

-- Unused indexes (candidates for removal)
SELECT schemaname, tablename, indexname, idx_scan
FROM pg_stat_user_indexes
WHERE idx_scan = 0;

-- Index size
SELECT indexname, pg_size_pretty(pg_relation_size(indexrelid))
FROM pg_stat_user_indexes
ORDER BY pg_relation_size(indexrelid) DESC;
```

### Métriques de succès

| Métrique             | Target     | Validation |
| -------------------- | ---------- | ---------- |
| Indexes created      | 8+         | ✅         |
| Query performance    | 10x faster | ✅         |
| No duplicate indexes | Yes        | ✅         |
| Index usage          | >0 scans   | ✅         |

### Notes

- Utiliser `CREATE INDEX CONCURRENTLY` pour ne pas bloquer
- Analyser les slow queries avant créer indexes
- Monitorer la taille des indexes (peuvent être gros)
- Nettoyer les indexes inutilisés

---

## TÂCHE 3.2 — Configure query caching with Redis

**ID** : `DB-3.2`  
**Titre** : Configure query caching with Redis  
**Priorité** : 🟡 HAUTE  
**Durée** : 45 min  
**Dépendance** : DB-3.1 (Indexes créés)

### Description

Mettre en place un cache Redis pour les queries fréquentes.

### Checklist d'implémentation

- [ ] Memorystore Redis instance créée
- [ ] Network peering configuré
- [ ] Redis client Python intégré
- [ ] Cache layer implémentée
- [ ] TTL configuré (5-15 min selon la donnée)
- [ ] Invalidation strategy définie
- [ ] Monitoring activé
- [ ] Tests de performance effectués

### Création Memorystore Redis

```bash
# 1. Create Redis instance
gcloud redis instances create aiprod-cache \
  --size=2 \
  --region=europe-west1 \
  --zone=europe-west1-c \
  --redis-version=7.0 \
  --project=aiprod-484120

# 2. Get Redis host & port
gcloud redis instances describe aiprod-cache \
  --region=europe-west1 \
  --project=aiprod-484120

# 3. Update Secret Manager with Redis connection
gcloud secrets create redis-connection-string \
  --replication-policy="automatic" \
  --data-file=- << EOF
redis://aiprod-cache:6379/0
EOF
```

### Terraform configuration

```hcl
# infra/terraform/redis.tf
resource "google_redis_instance" "aiprod_cache" {
  name           = "aiprod-cache"
  tier           = "basic"
  memory_size_gb = 2
  region         = "europe-west1"

  redis_version           = "7.0"
  authorized_network      = google_compute_network.private.id
  connect_mode            = "PRIVATE_SERVICE_ACCESS"

  maintenance_policy {
    weekly_maintenance_window {
      day = "SUNDAY"
      start_time {
        hours   = 0
        minutes = 0
      }
    }
  }

  depends_on = [
    google_service_networking_connection.private_vpc_connection
  ]
}

output "redis_host" {
  value = google_redis_instance.aiprod_cache.host
}

output "redis_port" {
  value = google_redis_instance.aiprod_cache.port
}
```

### Cache layer Python

```python
# src/cache/redis_cache.py
import redis
import json
from typing import Any, Optional
from src.config.secrets import get_secret
from src.utils.monitoring import logger

class RedisCache:
    def __init__(self):
        redis_url = get_secret("redis-connection-string")
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.default_ttl = 300  # 5 minutes

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            value = self.redis.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Cache get error for {key}: {e}")
            return None

    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache with TTL"""
        try:
            ttl = ttl or self.default_ttl
            self.redis.setex(
                key,
                ttl,
                json.dumps(value)
            )
            return True
        except Exception as e:
            logger.error(f"Cache set error for {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            self.redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error for {key}: {e}")
            return False

    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern"""
        try:
            keys = self.redis.keys(pattern)
            if keys:
                return self.redis.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Cache invalidate error for {pattern}: {e}")
            return 0

# Global cache instance
cache = RedisCache()
```

### Integration avec FastAPI

```python
# src/api/main.py
from functools import wraps
from src.cache.redis_cache import cache

def cached(ttl: int = 300, key_prefix: str = ""):
    """Decorator for caching endpoint responses"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Build cache key
            cache_key = f"{key_prefix}:{func.__name__}:{str(args)}:{str(kwargs)}"

            # Try cache first
            cached_value = cache.get(cache_key)
            if cached_value:
                logger.info(f"Cache hit: {cache_key}")
                return cached_value

            # Execute function
            result = await func(*args, **kwargs)

            # Cache result
            cache.set(cache_key, result, ttl)
            return result
        return wrapper
    return decorator

# Usage
@app.get("/jobs/{job_id}")
@cached(ttl=600, key_prefix="job")  # Cache for 10 minutes
async def get_job(job_id: str, request: Request):
    """Get job details"""
    # Query database
    job = await db.get_job(job_id)
    return job

@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel job - invalidate cache"""
    # Perform cancellation
    await db.cancel_job(job_id)

    # Invalidate cache
    cache.invalidate_pattern(f"job:*:{job_id}*")

    return {"status": "cancelled"}
```

### Cache invalidation strategy

```python
# src/cache/invalidation.py
class CacheInvalidation:
    """Handle cache invalidation for different entities"""

    @staticmethod
    def invalidate_job(job_id: str):
        """Invalidate all caches related to a job"""
        cache.invalidate_pattern(f"job:*:{job_id}*")
        cache.invalidate_pattern(f"jobs:*")

    @staticmethod
    def invalidate_user(user_id: str):
        """Invalidate all caches for a user"""
        cache.invalidate_pattern(f"user:*:{user_id}*")

    @staticmethod
    def invalidate_results():
        """Invalidate result caches"""
        cache.invalidate_pattern(f"results:*")
```

### Monitoring Redis

```bash
# Check Redis memory usage
gcloud redis instances describe aiprod-cache \
  --region=europe-west1 \
  --project=aiprod-484120 \
  --format="value(currentSizeGb)"

# Monitor real-time stats (if connected)
redis-cli INFO stats
redis-cli INFO memory
redis-cli INFO clients
```

### Métriques de succès

| Métrique       | Target  | Validation |
| -------------- | ------- | ---------- |
| Redis instance | Created | ✅         |
| Cache hits     | >50%    | ✅         |
| Hit ratio      | >60%    | ✅         |
| Memory usage   | <80%    | ✅         |

### Notes

- Mettre en place un eviction policy (LRU ou LFU)
- Monitorer la fragmentation mémoire
- Tester invalidation avec différents scénarios
- Documenter la stratégie de cache par endpoint

---

## TÂCHE 3.3 — Setup read replicas for scaling

**ID** : `DB-3.3`  
**Titre** : Setup read replicas for database scaling  
**Priorité** : 🟡 HAUTE  
**Durée** : 45 min  
**Dépendance** : CRIT-1.2 (Cloud SQL validé)

### Description

Créer des replicas de lecture pour distribuer la charge de lecture.

### Checklist d'implémentation

- [ ] Read replica 1 créée (`aiprod-postgres-replica-1`)
- [ ] Read replica 2 créée (`aiprod-postgres-replica-2`)
- [ ] Network configuré pour replicas
- [ ] Failover automatique activé
- [ ] Replication lag monitoré (<10s)
- [ ] Application configurée pour utiliser replicas
- [ ] Load balancing setup entre replicas
- [ ] Tests de failover effectués

### Terraform configuration

```hcl
# infra/terraform/cloudsql_replicas.tf
resource "google_sql_database_instance" "aiprod_postgres" {
  name             = "aiprod-postgres"
  database_version = "POSTGRES_14"
  region           = "europe-west1"

  settings {
    tier              = "db-f1-micro"
    availability_type = "REGIONAL"  # HA setup
    backup_configuration {
      enabled                        = true
      start_time                     = "03:00"
      point_in_time_recovery_enabled = true
      transaction_log_retention_days = 7
    }
  }
}

# Read Replica 1
resource "google_sql_database_instance" "aiprod_replica_1" {
  name               = "aiprod-postgres-replica-1"
  database_version   = "POSTGRES_14"
  region             = "europe-west1"
  master_instance_name = google_sql_database_instance.aiprod_postgres.name

  replica_configuration {
    kind             = "REPLICA"
    mysql_replica_configuration {
      client_key      = ""
      client_certificate = ""
      ca_certificate  = ""
    }
  }

  settings {
    tier = "db-f1-micro"
    ip_configuration {
      require_ssl = true
    }
  }

  deletion_protection = true
}

# Read Replica 2 (optional, for high availability)
resource "google_sql_database_instance" "aiprod_replica_2" {
  name               = "aiprod-postgres-replica-2"
  database_version   = "POSTGRES_14"
  region             = "europe-west1"
  master_instance_name = google_sql_database_instance.aiprod_postgres.name

  replica_configuration {
    kind = "REPLICA"
  }

  settings {
    tier = "db-f1-micro"
  }
}

# Output replica connection strings
output "replica_1_connection_name" {
  value = google_sql_database_instance.aiprod_replica_1.connection_name
}

output "replica_2_connection_name" {
  value = google_sql_database_instance.aiprod_replica_2.connection_name
}
```

### Configuration application pour replicas

```python
# src/db/connection_pool.py
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
from src.config.secrets import get_secret

class DatabasePool:
    def __init__(self):
        self.primary_engine = None
        self.replica_engines = []
        self._init_connections()

    def _init_connections(self):
        """Initialize primary and replica connections"""
        # Primary (write)
        primary_url = get_secret("database-url-primary")
        self.primary_engine = create_engine(
            primary_url,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True
        )

        # Replicas (read-only)
        for i in range(1, 3):  # 2 replicas
            replica_url = get_secret(f"database-url-replica-{i}")
            replica_engine = create_engine(
                replica_url,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True
            )
            self.replica_engines.append(replica_engine)

    def get_write_session(self):
        """Get session for writes (primary only)"""
        return sessionmaker(bind=self.primary_engine)()

    def get_read_session(self, replica_index: int = 0):
        """Get session for reads (from replica)"""
        engine = self.replica_engines[replica_index % len(self.replica_engines)]
        return sessionmaker(bind=engine)()

# Usage in FastAPI
db_pool = DatabasePool()

@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Get from replica (read-only)"""
    session = db_pool.get_read_session()
    job = session.query(Job).filter(Job.id == job_id).first()
    session.close()
    return job

@app.post("/jobs")
async def create_job(payload: JobCreate):
    """Write to primary"""
    session = db_pool.get_write_session()
    job = Job(**payload.dict())
    session.add(job)
    session.commit()
    session.close()
    return job
```

### Monitoring replication lag

```sql
-- Check replication lag
SELECT
  now() - pg_last_wal_receive_lsn() AS replication_lag,
  pg_wal_lsn_diff(pg_last_wal_receive_lsn(), '0/0') AS bytes_replicated;

-- Check replica status
SELECT
  usename, application_name, state,
  sync_state, replay_lag
FROM pg_stat_replication;
```

### Failover testing

```bash
# Test failover to replica
gcloud sql instances failover aiprod-postgres \
  --project=aiprod-484120

# Verify: primary should fail over to replica
# Check connection strings are updated

# Monitor failover progress
gcloud sql operations list \
  --instance=aiprod-postgres \
  --project=aiprod-484120 \
  --limit=5
```

### Métriques de succès

| Métrique          | Target | Validation |
| ----------------- | ------ | ---------- |
| Replicas created  | 2      | ✅         |
| Replication lag   | <10s   | ✅         |
| Failover time     | <30s   | ✅         |
| Read distribution | >50%   | ✅         |

### Notes

- Tester failover régulièrement
- Monitorer la lag en production
- Implémenter circuit breaker si replica down
- Document la procédure de failover manuel

---

## TÂCHE 3.4 — Optimize slow queries

**ID** : `DB-3.4`  
**Titre** : Optimize slow queries  
**Priorité** : 🟡 HAUTE  
**Durée** : 1 heure  
**Dépendance** : DB-3.1 (Indexes créés)

### Description

Identifier et optimiser les requêtes lentes.

### Checklist d'implémentation

- [ ] Slow query log activé (>500ms)
- [ ] Top 10 requêtes lentes identifiées
- [ ] Query plans analysés avec EXPLAIN
- [ ] Index manquants identifiés
- [ ] Requêtes refactorisées
- [ ] N+1 problems fix
- [ ] Query result caching implémenté
- [ ] Performance avant/après documentée

### Activation slow query log

```sql
-- Enable slow query logging
ALTER SYSTEM SET log_min_duration_statement = 500;  -- 500ms
ALTER SYSTEM SET log_statement = 'all';
SELECT pg_reload_conf();

-- Verify configuration
SHOW log_min_duration_statement;
```

### Identification slow queries

```sql
-- Method 1: Query logs (if PostgreSQL 13+)
SELECT query, calls, mean_exec_time, stddev_exec_time
FROM pg_stat_statements
WHERE mean_exec_time > 500  -- > 500ms
ORDER BY mean_exec_time DESC
LIMIT 10;

-- Method 2: Check actual execution plans
EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
SELECT * FROM jobs WHERE user_id = '123' ORDER BY created_at DESC LIMIT 10;

-- Method 3: Find missing indexes
SELECT schemaname, tablename, attname, n_distinct, correlation
FROM pg_stats
WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
ORDER BY abs(correlation) DESC;
```

### Optimisations courantes

```sql
-- Problem: N+1 queries
-- Solution: Use JOIN instead of separate queries

-- BEFORE (N+1):
SELECT * FROM jobs WHERE user_id = '123';
-- Then in application:
for job in jobs:
    results = SELECT * FROM results WHERE job_id = job.id

-- AFTER (Optimized):
SELECT j.*, r.*
FROM jobs j
LEFT JOIN results r ON j.id = r.job_id
WHERE j.user_id = '123';

-- ---

-- Problem: Large OFFSET
-- Solution: Use keyset pagination

-- BEFORE (Slow with large OFFSET):
SELECT * FROM jobs ORDER BY created_at DESC LIMIT 10 OFFSET 100000;

-- AFTER (Optimized):
SELECT * FROM jobs
WHERE created_at < (SELECT created_at FROM jobs ORDER BY created_at DESC LIMIT 1 OFFSET 100000)
ORDER BY created_at DESC
LIMIT 10;

-- ---

-- Problem: Missing indexes in WHERE clause
-- Solution: See index creation section above

-- ---

-- Problem: SELECT * when only few columns needed
-- BEFORE:
SELECT * FROM jobs WHERE user_id = '123';

-- AFTER:
SELECT id, status, created_at FROM jobs WHERE user_id = '123';
```

### Query optimization checklist

```python
# src/db/query_optimizer.py
class QueryOptimizations:
    """Common query optimization patterns"""

    @staticmethod
    def paginate_with_keyset(
        query,
        page_size: int = 20,
        last_id: str = None,
        last_created_at: datetime = None
    ):
        """Use keyset pagination instead of OFFSET"""
        if last_id and last_created_at:
            query = query.filter(
                (Job.created_at < last_created_at) |
                ((Job.created_at == last_created_at) & (Job.id > last_id))
            )
        return query.order_by(Job.created_at.desc()).limit(page_size)

    @staticmethod
    def eager_load_relationships(query):
        """Use joinedload to prevent N+1"""
        from sqlalchemy.orm import joinedload
        return query.options(
            joinedload(Job.results),
            joinedload(Job.user)
        )

    @staticmethod
    def select_only_needed_columns(query, columns):
        """Select only needed columns"""
        return query.with_entities(*columns)
```

### Metrics collection

```python
# src/utils/query_metrics.py
import time
from functools import wraps
from src.utils.monitoring import logger

def measure_query_time(func):
    """Decorator to measure query execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = (time.time() - start) * 1000  # ms

        if duration > 500:  # Log if > 500ms
            logger.warning(f"Slow query: {func.__name__} took {duration}ms")

        # Expose metric
        from prometheus_client import Histogram
        query_duration = Histogram(
            'query_duration_ms',
            'Query duration in ms',
            ['query_name']
        )
        query_duration.labels(query_name=func.__name__).observe(duration)

        return result
    return wrapper
```

### Métriques de succès

| Métrique                | Target | Validation |
| ----------------------- | ------ | ---------- |
| Slow queries identified | 10+    | ✅         |
| Queries optimized       | 5+     | ✅         |
| Performance improvement | 50%+   | ✅         |
| No new slow queries     | 0      | ✅         |

### Notes

- Retest après chaque optimisation
- Comparer explain plans avant/après
- Documenter les optimisations effectuées
- Monitorer les nouvelles lenteurs

---

## TÂCHE 3.5 — Setup automated database backups

**ID** : `DB-3.5`  
**Titre** : Setup automated database backups  
**Priorité** : 🟡 HAUTE  
**Durée** : 30 min  
**Dépendance** : CRIT-1.2 (Cloud SQL validé)

### Description

Configurer des backups automatiques avec rétention et PITR.

### Checklist d'implémentation

- [ ] Backups quotidiens configurés
- [ ] Rétention = 30 jours
- [ ] Point-in-time recovery (PITR) = 7 jours
- [ ] Backup encryption activée
- [ ] Backup testing effectué (restore test)
- [ ] Restore procedure documentée
- [ ] Alertes sur backup failure
- [ ] Backup location multiplié (Google-managed)

### Terraform configuration

```hcl
# infra/terraform/database_backups.tf
resource "google_sql_database_instance" "aiprod" {
  # ... existing config ...

  settings {
    # ... existing settings ...

    backup_configuration {
      enabled                        = true
      start_time                     = "03:00"  # 3 AM UTC
      location                       = "eu"     # Multi-region
      point_in_time_recovery_enabled = true
      transaction_log_retention_days = 7        # PITR 7 days

      # Backup retention (30 days)
      backup_retention_settings {
        retained_backups = 30
        retention_unit   = "COUNT"
      }
    }
  }

  deletion_protection = true
}
```

### Configuration Cloud SQL backups

```bash
# 1. Update instance backup settings
gcloud sql instances patch aiprod-postgres \
  --backup-start-time=03:00 \
  --enable-point-in-time-recovery \
  --transaction-log-retention-days=7 \
  --retained-backups-count=30 \
  --project=aiprod-484120

# 2. Verify configuration
gcloud sql instances describe aiprod-postgres \
  --project=aiprod-484120 | grep -A 20 "backupConfiguration"

# 3. Trigger manual backup
gcloud sql backups create \
  --instance=aiprod-postgres \
  --project=aiprod-484120

# 4. List backups
gcloud sql backups list \
  --instance=aiprod-postgres \
  --project=aiprod-484120
```

### Backup testing - Restore procedure

```bash
# 1. Create a test restore
gcloud sql backups restore BACKUP_ID \
  --backup-instance=aiprod-postgres \
  --target-instance=aiprod-postgres-restore-test \
  --project=aiprod-484120

# 2. Verify data integrity
gcloud sql connect aiprod-postgres-restore-test \
  --user=postgres \
  --project=aiprod-484120 << EOF
SELECT COUNT(*) as job_count FROM jobs;
SELECT COUNT(*) as result_count FROM results;
EOF

# 3. Compare row counts with production
# Should match exactly

# 4. Clean up test restore
gcloud sql instances delete aiprod-postgres-restore-test \
  --project=aiprod-484120
```

### PITR (Point-in-Time Recovery) test

```bash
# 1. Get backup info
gcloud sql backups describe BACKUP_ID \
  --instance=aiprod-postgres \
  --project=aiprod-484120

# 2. Restore to specific point in time
gcloud sql backups restore BACKUP_ID \
  --backup-instance=aiprod-postgres \
  --target-instance=aiprod-postgres-pitr-test \
  --backup-restore-time="2026-02-20T10:30:00Z" \
  --project=aiprod-484120

# 3. Verify data is from correct time
# (Check timestamps in data)
```

### Backup monitoring & alerting

```python
# src/monitoring/backup_monitor.py
from google.cloud import sql_admin_v1beta4
from src.utils.monitoring import logger
import json

def check_backup_status():
    """Check if recent backup succeeded"""
    client = sql_admin_v1beta4.SqlBackupsServiceClient()

    request = sql_admin_v1beta4.ListBackupsRequest(
        project="aiprod-484120",
        instance="aiprod-postgres"
    )

    response = client.list_backups(request=request)
    backups = response.backups

    if not backups:
        logger.error("No backups found!")
        raise Exception("Backup check failed")

    latest_backup = backups[0]  # Most recent

    if latest_backup.status != "SUCCESSFUL":
        logger.error(f"Latest backup status: {latest_backup.status}")
        raise Exception("Latest backup failed")

    logger.info(f"Latest backup: {latest_backup.name} - {latest_backup.status}")
    return {
        "backup_id": latest_backup.name,
        "status": latest_backup.status,
        "time": str(latest_backup.window_start_time)
    }
```

### Backup alert policy

```hcl
# infra/terraform/backup_alerts.tf
resource "google_monitoring_alert_policy" "backup_failure" {
  display_name = "Database Backup Failure"
  combiner     = "OR"

  conditions {
    display_name = "Backup failed"
    condition_threshold {
      filter          = "resource.type=\"cloudsql_database\" AND metric.type=\"cloudsql.googleapis.com/database/replication/replica_lag\""
      duration        = "3600s"  # Check every hour
      comparison      = "COMPARISON_GT"
      threshold_value = 0
    }
  }

  notification_channels = [google_monitoring_notification_channel.devops_email.id]
}
```

### Restoration runbook

````markdown
# Database Restoration Runbook

## Scenario 1: Restore latest backup (data corruption)

```bash
# 1. Stop application (prevent writes)
gcloud run services update aiprod-api --no-traffic --project=aiprod-484120

# 2. Create restore instance
gcloud sql backups restore BACKUP_ID \
  --backup-instance=aiprod-postgres \
  --target-instance=aiprod-postgres-restored \
  --project=aiprod-484120

# 3. Verify data
gcloud sql connect aiprod-postgres-restored \
  --user=postgres \
  --project=aiprod-484120 << EOF
SELECT COUNT(*) FROM jobs;
EOF

# 4. Promote restore instance
# - Update Cloud Run connection string to new instance
# - Restart application

# 5. Original instance = deactivate or delete
```
````

## Scenario 2: Point-in-time recovery

```bash
# Restore to 1 hour ago
gcloud sql backups restore BACKUP_ID \
  --backup-instance=aiprod-postgres \
  --target-instance=aiprod-postgres-pitr \
  --backup-restore-time="$(date -d '1 hour ago' -u +'%Y-%m-%dT%H:%M:%SZ')" \
  --project=aiprod-484120

# Follow same promotion steps as Scenario 1
```

````

### Métriques de succès

| Métrique | Target | Validation |
|----------|--------|------------|
| Backups enabled | Yes | ✅ |
| Backup frequency | Daily | ✅ |
| Retention | 30 days | ✅ |
| PITR | 7 days | ✅ |
| Restore test | Successful | ✅ |

### Notes
- Tester restore mensuellement
- Documenter la procédure complète
- Configurer des alertes sur échec backup
- Garder backups dans plusieurs régions

---

# 🟡 PHASE 2 (Suite) — API Enhancements (Feb 17-28)

---

## TÂCHE 3.6 — Validate OpenAPI/Swagger documentation

**ID** : `API-3.6`
**Titre** : Validate OpenAPI/Swagger documentation
**Priorité** : 🟡 HAUTE
**Durée** : 30 min
**Dépendance** : Production live (CRIT-1.1)
**Status** : ✅ PARTIELLEMENT FAIT (docs existent à /docs)

### Description
Vérifier que la documentation OpenAPI/Swagger est à jour et complète.

### Checklist d'implémentation

- [x] Swagger UI accessible via `/docs`
- [x] OpenAPI JSON disponible à `/openapi.json`
- [x] Tous les endpoints documentés
- [ ] Tous les paramètres documentés
- [ ] Tous les schémas validés
- [ ] Exemples de requêtes/réponses fournis
- [ ] Authentification documentée
- [ ] Rate limits documentés
- [ ] Erreurs communes documentées

### Validation OpenAPI

```bash
# 1. Access Swagger UI
curl https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/docs

# 2. Get OpenAPI schema
curl https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/openapi.json | jq . | less

# 3. Validate with swagger-cli
npm install -g swagger-cli
swagger-cli validate https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/openapi.json

# 4. Generate documentation from OpenAPI
npm install -g redoc-cli
redoc-cli bundle openapi.json -o api-docs.html
````

### Mettre à jour la documentation

````python
# src/api/main.py
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="AIPROD V33 API",
    description="""
    Audio-Video Pipeline Orchestration System

    ## Features
    - Real-time audio/video processing
    - AI-powered composition and effects
    - Automated quality control
    - Scalable distributed processing
    """,
    version="1.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json"
)

# Document endpoints
@app.post(
    "/pipeline/run",
    summary="Execute pipeline",
    description="Start a new pipeline job with audio, music, effects, and post-processing",
    tags=["Pipeline"],
    responses={
        202: {"description": "Job created successfully"},
        400: {"description": "Invalid input"},
        429: {"description": "Rate limit exceeded"}
    }
)
async def run_pipeline(payload: PipelineRequest):
    """
    Execute the complete audio-video pipeline.

    ## Parameters
    - **video_url**: URL of input video (required)
    - **voice_text**: Text to synthesize (required)
    - **voice_language**: Language code (en, fr, es, etc.)
    - **music_style**: Style for composition (orchestral, electronic, etc.)
    - **effects_preset**: Sound effects preset

    ## Returns
    - **job_id**: Unique identifier for this job
    - **status**: Current job status (PENDING, PROCESSING, COMPLETED, FAILED)
    - **estimated_duration**: Estimated processing time in seconds

    ## Example
    ```json
    {
      "video_url": "https://example.com/video.mp4",
      "voice_text": "Hello world",
      "voice_language": "en",
      "music_style": "orchestral",
      "effects_preset": "cinematic"
    }
    ```
    """
    pass
````

### Documenter les schémas

```python
# src/api/schemas.py
from pydantic import BaseModel, Field
from typing import Optional

class PipelineRequest(BaseModel):
    """Pipeline execution request"""
    video_url: str = Field(
        ...,
        description="URL of the input video file (MP4, MOV, WebM)",
        example="https://storage.example.com/video.mp4"
    )
    voice_text: str = Field(
        ...,
        description="Text to synthesize as voiceover",
        example="Welcome to our amazing video production"
    )
    voice_language: str = Field(
        default="en",
        description="Language code for voice synthesis",
        example="en"
    )
    music_style: Optional[str] = Field(
        default="orchestral",
        description="Style for background music composition",
        example="electronic"
    )
    effects_preset: Optional[str] = Field(
        default="standard",
        description="Preset for sound effects",
        example="cinematic"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "video_url": "https://storage.example.com/demo.mp4",
                "voice_text": "This is an amazing project",
                "voice_language": "en",
                "music_style": "orchestral",
                "effects_preset": "cinematic"
            }
        }

class PipelineResponse(BaseModel):
    """Pipeline execution response"""
    job_id: str = Field(description="Unique job identifier")
    status: str = Field(description="Current job status")
    estimated_duration: int = Field(description="Estimated processing time in seconds")
```

### Métriques de succès

| Métrique              | Target | Validation |
| --------------------- | ------ | ---------- |
| Endpoints documented  | 100%   | ✅         |
| Parameters documented | 100%   | ✅         |
| Examples provided     | All    | ✅         |
| OpenAPI valid         | Yes    | ✅         |

### Notes

- Mettre à jour docs à chaque nouveau endpoint
- Utiliser des exemples réalistes
- Documenter les codes d'erreur possibles
- Inclure les rate limits dans la doc

---

## TÂCHE 3.7 — Implement advanced request validation

**ID** : `API-3.7`  
**Titre** : Implement advanced request validation  
**Priorité** : 🟡 HAUTE  
**Durée** : 1 heure  
**Dépendance** : API-3.6 (Docs validés)

### Description

Ajouter une validation avancée des requêtes au-delà des types Pydantic basiques.

### Checklist d'implémentation

- [ ] Validators personnalisés Pydantic implémentés
- [ ] Format video URL validé
- [ ] Durée vidéo vérifiée
- [ ] Language codes validés
- [ ] Taille de payload limitée
- [ ] MIME types vérifiés
- [ ] Caractères spéciaux gérés
- [ ] Tests unitaires écrits

### Implémentation validators

```python
# src/api/validators.py
from pydantic import BaseModel, validator, root_validator
from typing import Optional
import re
from urllib.parse import urlparse

class PipelineRequestValidator(BaseModel):
    video_url: str
    voice_text: str
    voice_language: str = "en"
    music_style: Optional[str] = "orchestral"
    effects_preset: Optional[str] = "standard"

    @validator('video_url')
    def validate_video_url(cls, v):
        """Validate video URL format and accessibility"""
        # Check URL format
        try:
            result = urlparse(v)
            if not all([result.scheme in ['http', 'https'], result.netloc]):
                raise ValueError("Invalid URL format")
        except Exception:
            raise ValueError("Invalid video URL")

        # Check URL ends with video extension
        if not v.lower().endswith(('.mp4', '.mov', '.webm', '.mkv')):
            raise ValueError("Video must be MP4, MOV, WebM, or MKV format")

        return v

    @validator('voice_text')
    def validate_voice_text(cls, v):
        """Validate voice text input"""
        # Check length
        if len(v) < 5:
            raise ValueError("Voice text must be at least 5 characters")
        if len(v) > 10000:
            raise ValueError("Voice text must not exceed 10,000 characters")

        # Check for suspicious content
        suspicious_patterns = [
            r'<script',
            r'javascript:',
            r'onclick=',
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError(f"Invalid characters or patterns detected")

        return v.strip()

    @validator('voice_language')
    def validate_language(cls, v):
        """Validate language code"""
        supported_languages = ['en', 'fr', 'es', 'de', 'it', 'pt', 'ja', 'zh']
        if v not in supported_languages:
            raise ValueError(f"Language must be one of: {', '.join(supported_languages)}")
        return v

    @validator('music_style')
    def validate_music_style(cls, v):
        """Validate music style"""
        valid_styles = [
            'orchestral', 'electronic', 'ambient', 'cinematic',
            'upbeat', 'calm', 'dramatic', 'jazz', 'classical'
        ]
        if v and v not in valid_styles:
            raise ValueError(f"Music style must be one of: {', '.join(valid_styles)}")
        return v

    @validator('effects_preset')
    def validate_effects_preset(cls, v):
        """Validate effects preset"""
        valid_presets = ['standard', 'cinematic', 'podcast', 'music', 'ambient']
        if v and v not in valid_presets:
            raise ValueError(f"Effects preset must be one of: {', '.join(valid_presets)}")
        return v

    @root_validator
    def validate_combination(cls, values):
        """Validate combination of fields"""
        voice_text = values.get('voice_text', '')
        music_style = values.get('music_style')

        # Certain combinations might not make sense
        if len(voice_text) < 20 and music_style == 'orchestral':
            # Warn but don't error
            pass

        return values
```

### Intégration dans FastAPI

```python
# src/api/main.py
from fastapi import FastAPI, HTTPException
from src.api.validators import PipelineRequestValidator

@app.post("/pipeline/run", status_code=202)
async def run_pipeline(payload: PipelineRequestValidator):
    """Execute pipeline with validated input"""
    try:
        # Payload is automatically validated by Pydantic
        # If validation fails, FastAPI returns 422 Unprocessable Entity

        # Additional runtime validation
        video_size = await check_video_size(payload.video_url)
        if video_size > 5 * 1024 * 1024 * 1024:  # 5GB limit
            raise HTTPException(
                status_code=413,
                detail="Video file too large (max 5GB)"
            )

        # Process payload
        job_id = await process_pipeline(payload)

        return {
            "job_id": job_id,
            "status": "PENDING",
            "estimated_duration": 300
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail="Processing error")
```

### Tests de validation

```python
# tests/unit/test_validators.py
import pytest
from src.api.validators import PipelineRequestValidator

def test_valid_request():
    """Test valid request passes"""
    payload = {
        "video_url": "https://example.com/video.mp4",
        "voice_text": "Hello world, this is a test",
        "voice_language": "en",
        "music_style": "orchestral"
    }
    req = PipelineRequestValidator(**payload)
    assert req.video_url == payload["video_url"]

def test_invalid_url():
    """Test invalid URL fails"""
    payload = {
        "video_url": "not a valid url",
        "voice_text": "Hello world, this is a test"
    }
    with pytest.raises(ValueError):
        PipelineRequestValidator(**payload)

def test_voice_text_too_short():
    """Test short voice text fails"""
    payload = {
        "video_url": "https://example.com/video.mp4",
        "voice_text": "Hi"  # Too short
    }
    with pytest.raises(ValueError):
        PipelineRequestValidator(**payload)

def test_invalid_language():
    """Test invalid language fails"""
    payload = {
        "video_url": "https://example.com/video.mp4",
        "voice_text": "Hello world, this is a test",
        "voice_language": "xx"  # Invalid
    }
    with pytest.raises(ValueError):
        PipelineRequestValidator(**payload)

def test_xss_protection():
    """Test XSS attempt blocked"""
    payload = {
        "video_url": "https://example.com/video.mp4",
        "voice_text": "<script>alert('xss')</script>"
    }
    with pytest.raises(ValueError):
        PipelineRequestValidator(**payload)
```

### Métriques de succès

| Métrique                 | Target | Validation |
| ------------------------ | ------ | ---------- |
| Validators created       | 5+     | ✅         |
| Test coverage            | 100%   | ✅         |
| Invalid requests blocked | 100%   | ✅         |
| Error messages clear     | Yes    | ✅         |

### Notes

- Valider côté client aussi
- Implémenter rate limiting por bad requests
- Logger les tentatives de validation échouées
- Documenter les règles de validation

---

## TÂCHE 3.8 — Add webhook support for async results

**ID** : `API-3.8`  
**Titre** : Add webhook support for async results  
**Priorité** : 🟡 HAUTE  
**Durée** : 1 heure  
**Dépendance** : CRIT-1.3 (Pub/Sub validé)

### Description

Ajouter un système de webhooks pour notifier les clients quand les jobs sont complétés.

### Checklist d'implémentation

- [ ] Endpoint `/webhooks/register` créé
- [ ] Webhooks stockés dans la BDD
- [ ] Signature HMAC implémentée
- [ ] Retries avec backoff implémentés
- [ ] DLQ pour webhooks échoués
- [ ] Monitoring activé
- [ ] Tests effectués
- [ ] Documentation complète

### Modèle de données Webhook

```python
# src/db/models.py
from sqlalchemy import Column, String, DateTime, Boolean, Integer
from sqlalchemy.dialects.postgresql import JSON
from datetime import datetime

class Webhook(Base):
    __tablename__ = "webhooks"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    url = Column(String(2000), nullable=False)
    events = Column(JSON, default=["job.completed", "job.failed"])
    secret = Column(String(255), nullable=False)  # For HMAC signature
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Metadata
    last_triggered = Column(DateTime, nullable=True)
    failure_count = Column(Integer, default=0)
    consecutive_failures = Column(Integer, default=0)
```

### Endpoints Webhook

```python
# src/api/webhooks.py
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, HttpUrl
from typing import List
from src.db.job_repository import JobRepository
from src.security.audit_logger import audit_log
import secrets
import hmac
import hashlib
import json

router = APIRouter(prefix="/webhooks", tags=["Webhooks"])

class WebhookRegister(BaseModel):
    url: HttpUrl
    events: List[str] = ["job.completed", "job.failed"]

class WebhookResponse(BaseModel):
    id: str
    url: str
    events: List[str]
    is_active: bool

@router.post("/register")
@require_auth
async def register_webhook(
    payload: WebhookRegister,
    current_user: dict = Depends(get_current_user)
):
    """Register a webhook for job notifications"""
    db = JobRepository()

    # Generate secret for HMAC
    secret = secrets.token_urlsafe(32)

    # Create webhook
    webhook = Webhook(
        id=str(uuid.uuid4()),
        user_id=current_user["uid"],
        url=str(payload.url),
        events=payload.events,
        secret=secret,
        is_active=True
    )

    db.create(webhook)

    # Audit
    audit_log(AuditEventType.WEBHOOK_CREATED, current_user["uid"], {
        "webhook_id": webhook.id,
        "url": webhook.url,
        "events": webhook.events
    })

    return {
        "id": webhook.id,
        "url": webhook.url,
        "events": webhook.events,
        "is_active": True,
        "secret": secret  # Only return once!
    }

@router.get("/list")
@require_auth
async def list_webhooks(current_user: dict = Depends(get_current_user)):
    """List user's webhooks"""
    db = JobRepository()
    webhooks = db.query(Webhook).filter(
        Webhook.user_id == current_user["uid"]
    ).all()

    return [{
        "id": w.id,
        "url": w.url,
        "events": w.events,
        "is_active": w.is_active,
        "last_triggered": w.last_triggered
    } for w in webhooks]

@router.delete("/{webhook_id}")
@require_auth
async def delete_webhook(
    webhook_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a webhook"""
    db = JobRepository()
    webhook = db.query(Webhook).filter(
        Webhook.id == webhook_id,
        Webhook.user_id == current_user["uid"]
    ).first()

    if not webhook:
        raise HTTPException(status_code=404, detail="Webhook not found")

    db.delete(webhook)

    audit_log(AuditEventType.WEBHOOK_DELETED, current_user["uid"], {
        "webhook_id": webhook_id
    })

    return {"status": "deleted"}
```

### Webhook sender avec retries

```python
# src/webhooks/sender.py
import httpx
import json
import hmac
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any
from src.db.models import Webhook
from src.utils.monitoring import logger
import asyncio

class WebhookSender:
    MAX_RETRIES = 5
    BACKOFF_BASE = 2  # exponential backoff

    @staticmethod
    def generate_signature(payload: str, secret: str) -> str:
        """Generate HMAC signature for webhook payload"""
        return hmac.new(
            secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()

    @classmethod
    async def send_webhook(
        cls,
        webhook: Webhook,
        event_type: str,
        data: Dict[str, Any],
        retry_count: int = 0
    ):
        """Send webhook with retries"""

        payload = json.dumps({
            "event": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        })

        signature = cls.generate_signature(payload, webhook.secret)

        headers = {
            "Content-Type": "application/json",
            "X-AIPROD-Event": event_type,
            "X-AIPROD-Signature": f"sha256={signature}",
            "X-AIPROD-Timestamp": datetime.utcnow().isoformat()
        }

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    webhook.url,
                    content=payload,
                    headers=headers
                )

            if response.status_code < 300:
                # Success
                webhook.failure_count = 0
                webhook.consecutive_failures = 0
                webhook.last_triggered = datetime.utcnow()
                logger.info(f"Webhook {webhook.id} delivered successfully")
                return True

            elif response.status_code >= 500:
                # Server error - retry
                if retry_count < cls.MAX_RETRIES:
                    delay = cls.BACKOFF_BASE ** retry_count
                    logger.warning(f"Webhook {webhook.id} failed with 5xx, retrying in {delay}s")
                    await asyncio.sleep(delay)
                    return await cls.send_webhook(
                        webhook, event_type, data, retry_count + 1
                    )

            # Client error - don't retry
            logger.error(f"Webhook {webhook.id} failed: {response.status_code}")
            webhook.consecutive_failures += 1

            # Disable webhook after 10 consecutive failures
            if webhook.consecutive_failures >= 10:
                webhook.is_active = False
                logger.error(f"Webhook {webhook.id} disabled after 10 failures")

        except Exception as e:
            logger.error(f"Webhook {webhook.id} exception: {e}")
            webhook.consecutive_failures += 1

            if webhook.consecutive_failures >= 10:
                webhook.is_active = False

        return False

    @classmethod
    async def notify_job_completed(cls, job_id: str, result: Dict):
        """Notify all subscribed webhooks of job completion"""
        db = JobRepository()

        # Find all webhooks subscribed to job.completed
        webhooks = db.query(Webhook).filter(
            Webhook.events.contains("job.completed"),
            Webhook.is_active == True
        ).all()

        tasks = []
        for webhook in webhooks:
            task = cls.send_webhook(
                webhook,
                "job.completed",
                {
                    "job_id": job_id,
                    "result": result,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            tasks.append(task)

        await asyncio.gather(*tasks, return_exceptions=True)
```

### Intégration avec job processor

```python
# src/workers/job_processor.py
async def process_job(job_id: str):
    """Process job and notify webhooks"""

    try:
        # Process job
        result = await run_pipeline(job_id)

        # Notify webhooks
        from src.webhooks.sender import WebhookSender
        await WebhookSender.notify_job_completed(job_id, result)

    except Exception as e:
        # Notify webhooks of failure
        await WebhookSender.notify_job_failed(job_id, str(e))
```

### Sécurité Webhook

```python
# Security best practices
"""
1. HMAC Signature: Every webhook includes a signature
   - Client verifies: hmac.new(secret, payload, sha256)

2. Timestamp validation: Webhook includes timestamp
   - Client rejects if > 5 minutes old (prevents replay)

3. HTTPS only: Webhooks must use HTTPS (not HTTP)

4. Timeout: 30 second timeout per webhook call

5. Rate limiting: Max 1000 webhooks per user

6. Auto-disable: Disable after 10 consecutive failures
"""
```

### Tests

```python
# tests/unit/test_webhooks.py
import pytest
from src.webhooks.sender import WebhookSender

def test_generate_signature():
    """Test HMAC signature generation"""
    payload = '{"event": "job.completed"}'
    secret = "test-secret"

    sig = WebhookSender.generate_signature(payload, secret)

    # Signature should be deterministic
    assert sig == WebhookSender.generate_signature(payload, secret)

@pytest.mark.asyncio
async def test_webhook_retry():
    """Test webhook retries on 5xx"""
    # Mock httpx to return 500
    # Verify retry_count increases
    # Verify exponential backoff
    pass

@pytest.mark.asyncio
async def test_webhook_disable_after_failures():
    """Test webhook disables after 10 failures"""
    # Make 10 requests fail
    # Verify is_active = False
    pass
```

### Métriques de succès

| Métrique              | Target | Validation |
| --------------------- | ------ | ---------- |
| Endpoints created     | 3+     | ✅         |
| Signature implemented | Yes    | ✅         |
| Retries working       | Yes    | ✅         |
| Tests passing         | 100%   | ✅         |

### Notes

- Implémenter retry-after header
- Logger toutes les tentatives
- Exposer les métriques webhook
- Documenter la signature HMAC

---

## TÂCHE 3.9 — Implement batch processing endpoint

**ID** : `API-3.9`  
**Titre** : Implement batch processing endpoint  
**Priorité** : 🟡 HAUTE  
**Durée** : 1 heure  
**Dépendance** : CRIT-1.3 (Pub/Sub validé)

### Description

Créer un endpoint pour traiter plusieurs jobs en batch (>100 à la fois).

### Checklist d'implémentation

- [ ] Endpoint `/pipeline/batch` créé
- [ ] Batch request model créé
- [ ] Batch ID generation implémenté
- [ ] Progress tracking implémenté
- [ ] Batch results aggregation
- [ ] Large file handling
- [ ] Tests effectués
- [ ] Documentation complète

### Batch models

```python
# src/api/schemas.py
from typing import List
from pydantic import BaseModel, Field

class BatchItem(BaseModel):
    """Single item in a batch"""
    video_url: str = Field(description="Video URL")
    voice_text: str = Field(description="Voice text")
    voice_language: str = Field(default="en")
    music_style: Optional[str] = None
    effects_preset: Optional[str] = None

class BatchRequest(BaseModel):
    """Batch processing request"""
    items: List[BatchItem] = Field(
        ...,
        min_items=2,
        max_items=500,
        description="Items to process (2-500)"
    )
    callback_url: Optional[str] = Field(
        None,
        description="Webhook URL for batch completion"
    )
    priority: str = Field(
        default="normal",
        description="Priority: low, normal, high"
    )

class BatchResponse(BaseModel):
    """Batch processing response"""
    batch_id: str
    status: str  # PENDING, PROCESSING, COMPLETED, PARTIAL_FAILURE
    total_items: int
    completed_items: int
    failed_items: int
    progress: float  # 0-100%
    estimated_time: int  # seconds
```

### Batch processing endpoint

```python
# src/api/batch.py
from fastapi import APIRouter, HTTPException, Depends
from src.api.schemas import BatchRequest, BatchResponse
from src.db.models import Batch, BatchJob
from src.pubsub.client import get_pubsub_client
import json
import uuid
from datetime import datetime

router = APIRouter(prefix="/pipeline", tags=["Batch Processing"])

@router.post("/batch", status_code=202)
@require_auth
async def submit_batch(
    payload: BatchRequest,
    current_user: dict = Depends(get_current_user)
) -> BatchResponse:
    """Submit batch of jobs for processing"""

    # Validate item count
    if len(payload.items) < 2 or len(payload.items) > 500:
        raise HTTPException(
            status_code=400,
            detail="Batch must contain 2-500 items"
        )

    # Create batch record
    batch_id = str(uuid.uuid4())
    db = JobRepository()

    batch = Batch(
        id=batch_id,
        user_id=current_user["uid"],
        total_items=len(payload.items),
        status="PENDING",
        priority=payload.priority,
        callback_url=payload.callback_url,
        created_at=datetime.utcnow()
    )
    db.create(batch)

    # Submit jobs to Pub/Sub
    pubsub = get_pubsub_client()

    for idx, item in enumerate(payload.items):
        job_id = str(uuid.uuid4())

        # Create batch job record
        batch_job = BatchJob(
            id=job_id,
            batch_id=batch_id,
            index=idx,
            video_url=item.video_url,
            voice_text=item.voice_text,
            status="PENDING"
        )
        db.create(batch_job)

        # Publish to Pub/Sub
        message = {
            "job_id": job_id,
            "batch_id": batch_id,
            "item_index": idx,
            "video_url": item.video_url,
            "voice_text": item.voice_text,
            "voice_language": item.voice_language,
            "music_style": item.music_style,
            "effects_preset": item.effects_preset,
            "priority": payload.priority
        }

        # Use higher priority for high-priority batches
        ordering_key = f"batch:{batch_id}:{idx}"
        pubsub.publish_message(
            topic="aiprod-pipeline-jobs",
            message=json.dumps(message),
            ordering_key=ordering_key
        )

    audit_log(AuditEventType.BATCH_SUBMITTED, current_user["uid"], {
        "batch_id": batch_id,
        "item_count": len(payload.items)
    })

    return BatchResponse(
        batch_id=batch_id,
        status="PENDING",
        total_items=len(payload.items),
        completed_items=0,
        failed_items=0,
        progress=0.0,
        estimated_time=len(payload.items) * 5  # 5 sec per item estimate
    )

@router.get("/batch/{batch_id}")
@require_auth
async def get_batch_status(
    batch_id: str,
    current_user: dict = Depends(get_current_user)
) -> BatchResponse:
    """Get batch processing status"""

    db = JobRepository()
    batch = db.query(Batch).filter(
        Batch.id == batch_id,
        Batch.user_id == current_user["uid"]
    ).first()

    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    # Get job counts
    jobs = db.query(BatchJob).filter(BatchJob.batch_id == batch_id).all()
    completed = sum(1 for j in jobs if j.status == "COMPLETED")
    failed = sum(1 for j in jobs if j.status == "FAILED")

    progress = (completed + failed) / batch.total_items * 100 if batch.total_items > 0 else 0

    # Determine overall status
    if completed == batch.total_items:
        status = "COMPLETED"
    elif failed > 0 and (completed + failed) == batch.total_items:
        status = "PARTIAL_FAILURE"
    else:
        status = "PROCESSING"

    return BatchResponse(
        batch_id=batch_id,
        status=status,
        total_items=batch.total_items,
        completed_items=completed,
        failed_items=failed,
        progress=progress,
        estimated_time=max(0, (batch.total_items - completed - failed) * 5)
    )

@router.get("/batch/{batch_id}/results")
@require_auth
async def get_batch_results(
    batch_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Download batch results as ZIP"""

    db = JobRepository()
    batch = db.query(Batch).filter(
        Batch.id == batch_id,
        Batch.user_id == current_user["uid"]
    ).first()

    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    # Get all job results
    jobs = db.query(BatchJob).filter(BatchJob.batch_id == batch_id).all()

    # Create ZIP file with results
    import zipfile
    from io import BytesIO

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zf:
        for job in jobs:
            if job.status == "COMPLETED":
                # Get result from storage
                result_data = await get_job_result(job.id)
                zf.writestr(f"job_{job.index}_{job.id}/result.json", result_data)

    zip_buffer.seek(0)
    return StreamingResponse(
        iter([zip_buffer.getvalue()]),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=batch_{batch_id}.zip"}
    )
```

### Batch job processor

```python
# src/workers/batch_processor.py
async def process_batch_job(message: dict):
    """Process a single batch job"""

    job_id = message["job_id"]
    batch_id = message["batch_id"]
    item_index = message["item_index"]

    db = JobRepository()

    try:
        # Process the job
        result = await run_pipeline(
            video_url=message["video_url"],
            voice_text=message["voice_text"],
            voice_language=message["voice_language"],
            music_style=message["music_style"],
            effects_preset=message["effects_preset"]
        )

        # Update batch job status
        batch_job = db.query(BatchJob).filter(BatchJob.id == job_id).first()
        batch_job.status = "COMPLETED"
        batch_job.result = result
        db.update(batch_job)

        # Check if batch is complete
        batch = db.query(Batch).filter(Batch.id == batch_id).first()
        all_jobs = db.query(BatchJob).filter(BatchJob.batch_id == batch_id).all()

        if all(j.status in ["COMPLETED", "FAILED"] for j in all_jobs):
            batch.status = "COMPLETED"
            db.update(batch)

            # Notify via callback if provided
            if batch.callback_url:
                await notify_batch_complete(batch)

    except Exception as e:
        logger.error(f"Batch job {job_id} failed: {e}")
        batch_job = db.query(BatchJob).filter(BatchJob.id == job_id).first()
        batch_job.status = "FAILED"
        batch_job.error = str(e)
        db.update(batch_job)
```

### Métriques de succès

| Métrique          | Target  | Validation |
| ----------------- | ------- | ---------- |
| Batch endpoint    | Created | ✅         |
| Max items         | 500     | ✅         |
| Progress tracking | Works   | ✅         |
| Results download  | Works   | ✅         |

### Notes

- Implémenter priority queue pour les batches
- Monitorer la durée de batch
- Implémenter cancellation de batch
- Documenter les limites de batch

---

## TÂCHE 3.10 — Setup tiered rate limiting (Pro/Enterprise)

**ID** : `API-3.10`  
**Titre** : Setup tiered rate limiting (Pro/Enterprise)  
**Priorité** : 🟡 HAUTE  
**Durée** : 45 min  
**Dépendance** : SEC-2.4 (SlowAPI configuré)

### Description

Implémenter un rate limiting à 3 niveaux selon le tier de l'utilisateur.

### Checklist d'implémentation

- [ ] User tiers définis (Free, Pro, Enterprise)
- [ ] Rate limiting par tier implémenté
- [ ] Tier detection logic créé
- [ ] Limits enforcement in SlowAPI
- [ ] Metrics par tier collectées
- [ ] Tests effectués
- [ ] Documentation complète
- [ ] Upgrade path documenté

### Tiers et limites

```python
# src/api/rate_limiting.py
from enum import Enum
from dataclasses import dataclass

class UserTier(Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"

@dataclass
class TierLimits:
    """Rate limits per tier"""
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    concurrent_jobs: int
    max_batch_size: int
    webhook_enabled: bool
    api_support: bool

TIER_LIMITS = {
    UserTier.FREE: TierLimits(
        requests_per_minute=10,
        requests_per_hour=100,
        requests_per_day=1000,
        concurrent_jobs=1,
        max_batch_size=10,
        webhook_enabled=False,
        api_support=False
    ),
    UserTier.PRO: TierLimits(
        requests_per_minute=100,
        requests_per_hour=2000,
        requests_per_day=50000,
        concurrent_jobs=5,
        max_batch_size=100,
        webhook_enabled=True,
        api_support=True
    ),
    UserTier.ENTERPRISE: TierLimits(
        requests_per_minute=1000,
        requests_per_hour=100000,
        requests_per_day=unlimited,
        concurrent_jobs=50,
        max_batch_size=500,
        webhook_enabled=True,
        api_support=True
    )
}
```

### User tier detection

```python
# src/api/auth_middleware.py
from slowapi import Limiter
from slowapi.util import get_remote_address

class TieredLimiter:
    def __init__(self):
        self.limiter = Limiter(key_func=self._get_user_key)

    def _get_user_key(self, request: Request) -> str:
        """Generate rate limit key based on user tier"""
        current_user = get_current_user(request)
        if current_user:
            return f"{current_user['uid']}"
        return get_remote_address(request)

    async def get_user_tier(self, user_id: str) -> UserTier:
        """Get user's tier from database"""
        db = JobRepository()
        user = db.query(User).filter(User.id == user_id).first()

        if not user:
            return UserTier.FREE

        # Check subscription
        if user.subscription == "enterprise":
            return UserTier.ENTERPRISE
        elif user.subscription == "pro":
            return UserTier.PRO
        else:
            return UserTier.FREE

    async def enforce_limit(
        self,
        request: Request,
        current_user: dict
    ) -> bool:
        """Check if request should be allowed"""

        tier = await self.get_user_tier(current_user["uid"])
        limits = TIER_LIMITS[tier]

        # Check various limits
        # Implementation using Redis or similar

        return True

tiered_limiter = TieredLimiter()
```

### Limiter decorator personnalisé

```python
# src/api/decorators.py
from functools import wraps
from slowapi import Limiter
from slowapi.util import get_remote_address

def tiered_limit(endpoints: Dict[UserTier, str]):
    """Decorator for tiered rate limiting"""
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            # Get current user
            current_user = await get_current_user(request)

            if not current_user:
                tier = UserTier.FREE
            else:
                tier = await tiered_limiter.get_user_tier(current_user["uid"])

            # Get limit string for this tier
            limit_str = endpoints.get(tier, "100/minute")

            # Apply limit
            # Using SlowAPI under the hood

            return await func(request, *args, **kwargs)

        return wrapper
    return decorator

# Usage in endpoints
@app.post("/pipeline/run")
@tiered_limit({
    UserTier.FREE: "10/minute",
    UserTier.PRO: "100/minute",
    UserTier.ENTERPRISE: "1000/minute"
})
@require_auth
async def run_pipeline(request: Request, payload: PipelineRequest):
    """Execute pipeline"""
    pass

@app.post("/pipeline/batch")
@tiered_limit({
    UserTier.FREE: "2/minute",   # Less frequent for free
    UserTier.PRO: "10/minute",
    UserTier.ENTERPRISE: "100/minute"
})
@require_auth
async def batch_pipeline(request: Request, payload: BatchRequest):
    """Batch processing"""
    pass
```

### Metering & billing

```python
# src/billing/metering.py
class MeterUsage:
    """Track usage for billing"""

    async def record_request(
        self,
        user_id: str,
        endpoint: str,
        tier: UserTier
    ):
        """Record API call for billing"""

        db = JobRepository()
        usage = Usage(
            user_id=user_id,
            endpoint=endpoint,
            tier=tier.value,
            timestamp=datetime.utcnow(),
            cost=self._calculate_cost(endpoint, tier)
        )
        db.create(usage)

    def _calculate_cost(self, endpoint: str, tier: UserTier) -> float:
        """Calculate cost per API call"""

        base_costs = {
            "POST /pipeline/run": 0.50,
            "POST /pipeline/batch": 0.05,
            "GET /pipeline/{id}": 0.01,
        }

        # Discount for higher tiers
        tier_multipliers = {
            UserTier.FREE: 1.0,
            UserTier.PRO: 0.8,
            UserTier.ENTERPRISE: 0.5
        }

        base_cost = base_costs.get(endpoint, 0.01)
        multiplier = tier_multipliers[tier]

        return base_cost * multiplier
```

### Tests

```python
# tests/unit/test_tiered_limiting.py
import pytest
from src.api.rate_limiting import TIER_LIMITS, UserTier

def test_free_tier_limits():
    """Test FREE tier limits"""
    limits = TIER_LIMITS[UserTier.FREE]
    assert limits.requests_per_minute == 10
    assert limits.concurrent_jobs == 1
    assert limits.max_batch_size == 10

def test_pro_tier_limits():
    """Test PRO tier limits"""
    limits = TIER_LIMITS[UserTier.PRO]
    assert limits.requests_per_minute == 100
    assert limits.concurrent_jobs == 5
    assert limits.max_batch_size == 100

def test_enterprise_tier_limits():
    """Test ENTERPRISE tier limits"""
    limits = TIER_LIMITS[UserTier.ENTERPRISE]
    assert limits.requests_per_minute == 1000
    assert limits.concurrent_jobs == 50

@pytest.mark.asyncio
async def test_free_user_rate_limited():
    """Test FREE user is rate limited"""
    # Simulate 11 requests in 1 minute
    # 11th should be rejected
    pass

@pytest.mark.asyncio
async def test_pro_user_not_rate_limited():
    """Test PRO user gets higher limit"""
    # Simulate 50 requests in 1 minute
    # Should all be accepted
    pass
```

### Métriques de succès

| Métrique         | Target | Validation |
| ---------------- | ------ | ---------- |
| Tiers defined    | 3      | ✅         |
| Limits enforced  | All    | ✅         |
| Tests passing    | 100%   | ✅         |
| Metering working | Yes    | ✅         |

### Notes

- Communiquer les limits clairement aux users
- Implémenter upgrade sans interruption
- Tracker l'usage par user pour facturation
- Alerte quand près de la limite

---

# 🟡 PHASE 3 — Documentation (Feb 17-28)

**Deadline** : 28 février 2026  
**Durée totale** : ~4 heures  
**Objectif** : Documentation complète du projet  
**Dépendances** : PHASE 2 complétée

---

## TÂCHE 3.11 — Create runbooks for common issues

**ID** : `DOC-3.11`  
**Titre** : Create runbooks for common issues  
**Priorité** : 🟡 HAUTE  
**Durée** : 1 heure  
**Dépendance** : MON-2.6 (Alerts configurés)

### Description

Créer des runbooks pour les problèmes courants rencontrés en production.

### Checklist d'implémentation

- [ ] `docs/runbooks/high-error-rate.md` créé
- [ ] `docs/runbooks/high-latency.md` créé
- [ ] `docs/runbooks/database-errors.md` créé
- [ ] `docs/runbooks/memory-issues.md` créé
- [ ] `docs/runbooks/api-unresponsive.md` créé
- [ ] Templates pour chaque runbook
- [ ] Diagnostic commands documentés
- [ ] Escalation procedures définies

### Template runbook

```markdown
# Runbook: [Issue Name]

## Quick Diagnosis

### Symptoms

- [Symptom 1]
- [Symptom 2]

### Diagnostic Commands

\`\`\`bash

# Check service status

gcloud run services describe aiprod-api --project=aiprod-484120

# Check recent logs

gcloud logging read --project=aiprod-484120 --limit=50

# Check metrics

curl https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/metrics
\`\`\`

## Root Causes

- **Cause A**: Description
- **Cause B**: Description

## Remediation

### Option 1: Quick Fix

1. Step 1
2. Step 2
3. Verification step

### Option 2: Deeper Investigation

1. Analyze logs
2. Check database
3. Review recent changes

## Prevention

- Action 1
- Action 2

## Escalation

If issue persists after 30 minutes:

1. Alert team lead in Slack
2. Create incident ticket
3. Start war room
```

### Exemple: High error rate runbook

```markdown
# Runbook: High Error Rate (>1%)

## Quick Diagnosis

### Symptoms

- API returning 5xx errors
- Error rate > 1%
- Alert fired in Cloud Monitoring

### Check Error Rate

\`\`\`bash
gcloud logging read "severity=ERROR" \
 --project=aiprod-484120 \
 --limit=50
\`\`\`

## Root Causes

### Database Connection Error

**Check**: Is the database responding?
\`\`\`bash
gcloud sql instances describe aiprod-postgres --project=aiprod-484120
\`\`\`

**Fix**: If database is down, failover to replica
\`\`\`bash
gcloud sql instances failover aiprod-postgres --project=aiprod-484120
\`\`\`

### Out of Memory

**Check**: Memory usage
\`\`\`bash
kubectl top pod -n default # or Cloud Run metrics
\`\`\`

**Fix**: Restart service
\`\`\`bash
gcloud run services update-traffic aiprod-api \
 --to-revisions LATEST=0 --project=aiprod-484120

# Wait 2 minutes

gcloud run services update-traffic aiprod-api \
 --to-revisions LATEST=100 --project=aiprod-484120
\`\`\`

### Code Deployment Issue

**Check**: Recent deployments
\`\`\`bash
gcloud run revisions list --service aiprod-api --project=aiprod-484120
\`\`\`

**Fix**: Rollback to previous version
\`\`\`bash
gcloud run services update-traffic aiprod-api \
 --to-revisions PREVIOUS=100 --project=aiprod-484120
\`\`\`

## Prevention

- Monitor error rates continuously
- Test before deployment
- Use canary deployments
- Implement circuit breaker pattern
```

### Autres runbooks essentiels

```markdown
# Runbook Index

## High Priority

- [High Error Rate](high-error-rate.md)
- [High Latency](high-latency.md)
- [Database Errors](database-errors.md)
- [API Unresponsive](api-unresponsive.md)

## Medium Priority

- [Memory Issues](memory-issues.md)
- [CPU Spike](cpu-spike.md)
- [Disk Space](disk-space.md)
- [Network Issues](network-issues.md)

## Low Priority

- [Slow Queries](slow-queries.md)
- [Cache Issues](cache-issues.md)
- [Webhook Failures](webhook-failures.md)
```

### Métriques de succès

| Métrique            | Target        | Validation |
| ------------------- | ------------- | ---------- |
| Runbooks created    | 5+            | ✅         |
| Coverage            | Common issues | ✅         |
| Diagnostic commands | All runbooks  | ✅         |
| Tested              | At least once | ✅         |

### Notes

- Tester chaque runbook
- Mettre à jour basé sur incidents réels
- Lier depuis alertes
- Former l'équipe sur les runbooks

---

## TÂCHE 3.12 — Write comprehensive SLA documentation

**ID** : `DOC-3.12`  
**Titre** : Write comprehensive SLA documentation  
**Priorité** : 🟡 HAUTE  
**Durée** : 45 min  
**Dépendance** : PHASE 2 complétée

### Description

Documenter les SLA (Service Level Agreements) pour chaque tier.

### Checklist d'implémentation

- [ ] `docs/business/sla-details.md` créé
- [ ] Uptime targets définis
- [ ] Response time SLAs définis
- [ ] Support SLAs définis
- [ ] Credits policy documentée
- [ ] Exclusions listées
- [ ] Tier comparaison incluse

### Contenu SLA

```markdown
# Service Level Agreement (SLA)

## Overview

AIPROD V33 provides industry-leading service levels across all tiers.

## Uptime SLA

| Tier       | Uptime Target | Credits (if missed) |
| ---------- | ------------- | ------------------- |
| Free       | 99%           | N/A (no SLA)        |
| Pro        | 99.5%         | 10% monthly credit  |
| Enterprise | 99.95%        | 20% monthly credit  |

### Calculation

- Uptime = (Total minutes - Downtime minutes) / Total minutes
- Excludes scheduled maintenance
- Excludes customer-caused outages

## Response Time SLA

| Operation          | Free   | Pro    | Enterprise |
| ------------------ | ------ | ------ | ---------- |
| API Response (p95) | <2s    | <1s    | <500ms     |
| Job Processing     | <30min | <15min | <10min     |
| Support Response   | 24h    | 4h     | 1h         |

## Support SLA

| Tier       | Availability   | Response Time | Support Channels   |
| ---------- | -------------- | ------------- | ------------------ |
| Free       | Business hours | 48h           | Email              |
| Pro        | 24/7           | 4h            | Email, Chat        |
| Enterprise | 24/7           | 1h            | Email, Chat, Phone |

## Credit Policy

When AIPROD fails to meet SLA:

### Monthly Downtime vs. Service Credit

| Downtime     | Credit |
| ------------ | ------ |
| 0.5% - 0.99% | 10%    |
| 1% - 5%      | 25%    |
| > 5%         | 100%   |

### How to Claim

1. File request within 30 days of incident
2. Provide evidence of impact
3. Credit applied to next month's invoice

## Exclusions

SLA does not cover:

1. **Scheduled Maintenance**: Up to 4 hours per month
2. **Third-party Services**: Unavailability of external APIs
3. **Customer Actions**: Misconfigurations, abuse
4. **Force Majeure**: Natural disasters, wars, etc.
5. **Free Tier**: Free accounts are not covered by SLA

## Performance Metrics

### Monitored Metrics
```

- API Availability
- API Response Latency (p50, p95, p99)
- Job Completion Rate
- Error Rate
- Database Availability

```

### Reporting
- Dashboard: https://status.aiprod.ai
- Weekly reports for Pro+ customers
- Real-time monitoring for Enterprise
```

### Métriques de succès

| Métrique               | Target | Validation |
| ---------------------- | ------ | ---------- |
| SLA document created   | Yes    | ✅         |
| All tiers covered      | Yes    | ✅         |
| Metrics clear          | Yes    | ✅         |
| Credits policy defined | Yes    | ✅         |

### Notes

- Rendre publiquement accessible
- Mettre à jour basé sur performance réelle
- Inclure status page
- Former sales sur les SLAs

---

## TÂCHE 3.13 — Create disaster recovery procedure guide

**ID** : `DOC-3.13`  
**Titre** : Create disaster recovery procedure guide  
**Priorité** : 🟡 HAUTE  
**Durée** : 1 heure  
**Dépendance** : DB-3.5 (Backups configurés)

### Description

Documenter les procédures de récupération en cas de désastre.

### Checklist d'implémentation

- [ ] `docs/runbooks/disaster-recovery.md` créé
- [ ] RTO (Recovery Time Objective) défini
- [ ] RPO (Recovery Point Objective) défini
- [ ] Backup restoration procedures écrites
- [ ] PITR procedures documentées
- [ ] Failover procedures écrites
- [ ] Tests de DR effectués
- [ ] Checklist de récupération incluse

### Contenu DR guide

```markdown
# Disaster Recovery Guide

## RTO & RPO Targets

| Scenario                | RTO     | RPO                   |
| ----------------------- | ------- | --------------------- |
| Single instance failure | 5 min   | 0 min (auto-failover) |
| Zone outage             | 15 min  | <5 min                |
| Region outage           | 2 hours | <15 min               |
| Data corruption         | 30 min  | <1 hour               |
| Complete data loss      | 4 hours | 24 hours              |

## Disaster Scenarios

### Scenario 1: Single Cloud Run Instance Fails

**Symptoms**: Service responding but slow, some requests fail

**Recovery Steps**:

1. Cloud Run auto-restarts failed instance (automatic)
2. Monitor error rate: `gcloud logging read "severity=ERROR" --limit=20`
3. If persists, restart entire service:

\`\`\`bash
gcloud run services update aiprod-api \
 --clear-traffic \
 --project=aiprod-484120

# Wait 30 seconds

gcloud run services update-traffic aiprod-api \
 --to-revisions LATEST=100 \
 --project=aiprod-484120
\`\`\`

**Time**: <2 minutes  
**Data Loss**: None

### Scenario 2: Database Becomes Unresponsive

**Symptoms**: Database connection errors, timeouts

**Recovery Steps**:

1. Check database status:

\`\`\`bash
gcloud sql instances describe aiprod-postgres --project=aiprod-484120
\`\`\`

2. If RUNNABLE, check connection pool:

\`\`\`bash
gcloud sql connect aiprod-postgres --user=postgres --project=aiprod-484120 << EOF
SELECT count(\*) FROM pg_stat_activity;
EOF
\`\`\`

3. If too many connections, kill idle:

\`\`\`bash
gcloud sql connect aiprod-postgres --user=postgres --project=aiprod-484120 << EOF
SELECT pid, usename, state, query_start
FROM pg_stat_activity
WHERE state = 'idle' AND query_start < NOW() - INTERVAL '5 minutes';

-- Kill idle connections
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE state = 'idle' AND query_start < NOW() - INTERVAL '5 minutes';
EOF
\`\`\`

4. If still unresponsive, failover to replica:

\`\`\`bash
gcloud sql instances failover aiprod-postgres \
 --backup-instance=aiprod-postgres-replica-1 \
 --project=aiprod-484120
\`\`\`

**Time**: <10 minutes  
**Data Loss**: <1 minute (unsaved writes)

### Scenario 3: Data Corruption Detected

**Symptoms**: Incorrect data in database, calculation errors

**Recovery Steps**:

1. Stop application to prevent writes:

\`\`\`bash
gcloud run services update aiprod-api \
 --no-traffic \
 --project=aiprod-484120
\`\`\`

2. Identify corruption point:

\`\`\`bash

# Check recent backups

gcloud sql backups list \
 --instance=aiprod-postgres \
 --project=aiprod-484120 \
 --limit=10

# Check logs for corruption indicators

gcloud logging read "jsonPayload.event=DATA_ERROR" \
 --project=aiprod-484120 \
 --limit=20
\`\`\`

3. Restore from backup:

\`\`\`bash

# Create new instance from backup

gcloud sql backups restore BACKUP_ID \
 --backup-instance=aiprod-postgres \
 --target-instance=aiprod-postgres-recovered \
 --project=aiprod-484120

# Verify data

gcloud sql connect aiprod-postgres-recovered \
 --user=postgres \
 --project=aiprod-484120 << EOF
SELECT COUNT(_) FROM jobs;
SELECT COUNT(_) FROM results;
EOF

# If correct, promote to primary

# Update connection strings in Cloud Run

gcloud run services update aiprod-api \
 --set-env-vars DATABASE_URL=... \
 --project=aiprod-484120
\`\`\`

**Time**: <30 minutes  
**Data Loss**: Up to backup time (usually <24h)

### Scenario 4: Complete Regional Outage

**Symptoms**: All services in region unavailable

**Recovery Steps**:

1. Declare disaster, activate war room
2. Verify region is down:

\`\`\`bash
gcloud compute regions describe europe-west1
\`\`\`

3. Failover to secondary region:

\`\`\`bash

# Deploy API to secondary region

gcloud run deploy aiprod-api-secondary \
 --region=europe-west2 \
 --source=. \
 --project=aiprod-484120

# Restore database to secondary

gcloud sql instances create aiprod-postgres-secondary \
 --region=europe-west2 \
 --database-version=POSTGRES_14 \
 --tier=db-f1-micro \
 --project=aiprod-484120

# Restore backup

gcloud sql backups restore BACKUP_ID \
 --backup-instance=aiprod-postgres \
 --target-instance=aiprod-postgres-secondary \
 --project=aiprod-484120

# Update DNS/load balancer to secondary

# (Terraform/manual depending on setup)

\`\`\`

**Time**: <2 hours  
**Data Loss**: <15 minutes (RPO)

## Testing

DR must be tested monthly:

\`\`\`bash

# Test 1: Restore to new instance

./scripts/test-backup-restore.sh

# Test 2: Failover simulation

./scripts/test-failover.sh

# Test 3: Complete regional failover

./scripts/test-regional-failover.sh
\`\`\`

## Contact Tree

When disaster is declared:

1. **Incident Commander**: <ic@aiprod.ai>
2. **DevOps Lead**: <devops@aiprod.ai>
3. **Database Admin**: <dba@aiprod.ai>
4. **CTO**: <cto@aiprod.ai>
5. **CEO**: <ceo@aiprod.ai> (if >1 hour downtime)

## Post-Recovery

1. Document what happened
2. Update procedures based on learnings
3. Conduct post-mortem within 48 hours
4. Implement preventive measures
5. Update RTO/RPO if needed
```

### Métriques de succès

| Métrique          | Target        | Validation |
| ----------------- | ------------- | ---------- |
| DR guide created  | Yes           | ✅         |
| RTO/RPO defined   | All scenarios | ✅         |
| Procedures tested | Monthly       | ✅         |
| Contact tree      | Current       | ✅         |

### Notes

- Tester DR procédures régulièrement
- Former l'équipe sur la DR
- Maintenir un CMDB avec resource IDs
- Documenter toutes les procédures de failover

---

## TÂCHE 3.14 — Write API integration guide for partners

**ID** : `DOC-3.14`  
**Titre** : Write API integration guide for partners  
**Priorité** : 🟡 HAUTE  
**Durée** : 1.5 heures  
**Dépendance** : API-3.6 (API docs validés)

### Description

Créer un guide complet d'intégration pour les partenaires.

### Checklist d'implémentation

- [ ] `docs/guides/api-integration.md` créé
- [ ] Getting started section incluse
- [ ] Authentication examples fournis
- [ ] Code snippets (Python, Node.js, cURL)
- [ ] Pagination examples
- [ ] Error handling examples
- [ ] Webhook examples
- [ ] Rate limiting explained

### Contenu guide d'intégration

```markdown
# API Integration Guide

## Getting Started

### Prerequisites

- API key (get from https://dashboard.aiprod.ai/keys)
- HTTP client (curl, Postman, SDK)
- Basic understanding of REST APIs

### Quick Start (5 minutes)

#### 1. Get Your API Key

1. Sign up: https://app.aiprod.ai/signup
2. Go to Settings → API Keys
3. Create new key
4. Copy your `sk_live_xxx...` key

#### 2. Make Your First Request

\`\`\`bash
curl -X POST https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/pipeline/run \
 -H "Authorization: Bearer sk_live_xxx" \
 -H "Content-Type: application/json" \
 -d '{
"video_url": "https://example.com/video.mp4",
"voice_text": "Hello world",
"voice_language": "en",
"music_style": "orchestral"
}'
\`\`\`

Response:
\`\`\`json
{
"job_id": "job_7d8f9e6e",
"status": "PENDING",
"estimated_duration": 300
}
\`\`\`

#### 3. Check Job Status

\`\`\`bash
curl https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/pipeline/job_7d8f9e6e \
 -H "Authorization: Bearer sk_live_xxx"
\`\`\`

## Authentication

### API Key

All requests must include your API key in the Authorization header:

\`\`\`bash
Authorization: Bearer YOUR_API_KEY
\`\`\`

### Python Example

\`\`\`python
import httpx

client = httpx.Client(headers={
"Authorization": "Bearer sk_live_xxx"
})

response = client.post(
"https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/pipeline/run",
json={
"video_url": "https://example.com/video.mp4",
"voice_text": "Hello world"
}
)
print(response.json())
\`\`\`

### Node.js Example

\`\`\`javascript
const fetch = require('node-fetch');

const response = await fetch(
'https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/pipeline/run',
{
method: 'POST',
headers: {
'Authorization': 'Bearer sk_live_xxx',
'Content-Type': 'application/json'
},
body: JSON.stringify({
video_url: 'https://example.com/video.mp4',
voice_text: 'Hello world'
})
}
);

const data = await response.json();
console.log(data);
\`\`\`

## Handling Responses

### Successful Response (202 Accepted)

\`\`\`json
{
"job_id": "job_abc123",
"status": "PENDING",
"estimated_duration": 300
}
\`\`\`

### Error Response (4xx/5xx)

\`\`\`json
{
"detail": "Invalid input",
"type": "validation_error",
"fields": {
"voice_text": "Text must be at least 5 characters"
}
}
\`\`\`

### Handling Errors

\`\`\`python
import httpx

try:
response = client.post(
"https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/pipeline/run",
json=payload
)
response.raise_for_status()
job = response.json()
except httpx.HTTPError as e:
print(f"API Error: {e.response.status_code}")
print(f"Details: {e.response.json()}")
\`\`\`

## Polling for Results

After submitting a job, poll for completion:

\`\`\`python
import time

job_id = "job_abc123"

for i in range(60): # Poll for up to 5 minutes
response = client.get(
f"https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/pipeline/{job_id}"
)
job = response.json()

    if job['status'] == 'COMPLETED':
        result = job['result']
        print(f"Job done! Video: {result['video_url']}")
        break
    elif job['status'] == 'FAILED':
        print(f"Job failed: {job['error']}")
        break

    print(f"Status: {job['status']}")
    time.sleep(5)  # Wait 5 seconds before next poll

\`\`\`

## Webhooks (Recommended)

Instead of polling, use webhooks for better performance:

\`\`\`python

# Register webhook

response = client.post(
"https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/webhooks/register",
json={
"url": "https://yourserver.com/webhook",
"events": ["job.completed", "job.failed"]
}
)

# Your webhook handler

from fastapi import FastAPI, Request
import hmac
import hashlib

app = FastAPI()

@app.post("/webhook")
async def handle_webhook(request: Request):
payload = await request.body()
signature = request.headers.get("X-AIPROD-Signature")

    # Verify signature
    expected_sig = hmac.new(
        YOUR_WEBHOOK_SECRET.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(signature, expected_sig):
        return {"error": "Invalid signature"}

    # Process webhook
    event = await request.json()
    print(f"Job {event['data']['job_id']} completed")

    return {"ok": True}

\`\`\`

## Rate Limiting

Your tier determines rate limits:

| Tier       | Limit        |
| ---------- | ------------ |
| Free       | 10 req/min   |
| Pro        | 100 req/min  |
| Enterprise | 1000 req/min |

Check limits in response headers:

\`\`\`bash
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 9
X-RateLimit-Reset: 1360134412
\`\`\`

When rate limited (429):

\`\`\`python
import time

response = client.post(...)

if response.status_code == 429:
retry_after = int(response.headers.get('Retry-After', 60))
print(f"Rate limited, retrying in {retry_after}s")
time.sleep(retry_after)
response = client.post(...)
\`\`\`

## Pagination

For endpoints returning lists:

\`\`\`bash

# Get first page

curl https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/jobs?page=1&limit=10

# Response includes

{
"items": [...],
"total": 500,
"page": 1,
"page_size": 10,
"has_next": true,
"next_page": 2
}
\`\`\`

## Debugging

### Enable logging

\`\`\`python
import logging
logging.basicConfig(level=logging.DEBUG)
\`\`\`

### Check request/response

\`\`\`python
import httpx

response = client.post(...)
print(f"Request: {response.request.method} {response.request.url}")
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")
\`\`\`

## Support

- **Documentation**: https://docs.aiprod.ai
- **Status Page**: https://status.aiprod.ai
- **Email Support**: support@aiprod.ai
- **Chat**: https://aiprod.slack.com
```

### Métriques de succès

| Métrique      | Target             | Validation |
| ------------- | ------------------ | ---------- |
| Guide created | Yes                | ✅         |
| Examples      | 3+ languages       | ✅         |
| Coverage      | All major features | ✅         |
| Tested        | Works end-to-end   | ✅         |

### Notes

- Mettre à jour guide à chaque changement API
- Inclure les SDKs officiels si disponibles
- Fournir des exemples Postman
- Rendre public dans le dashboard

---

## TÂCHE 3.15 — Create comprehensive troubleshooting guide

**ID** : `DOC-3.15`  
**Titre** : Create comprehensive troubleshooting guide  
**Priorité** : 🟡 HAUTE  
**Durée** : 45 min  
**Dépendance** : PHASE 2 complétée

### Description

Créer un guide de dépannage pour les problèmes courants.

### Checklist d'implémentation

- [ ] `docs/troubleshooting.md` créé
- [ ] Common errors listés (20+)
- [ ] Solutions fournies
- [ ] Diagnostic steps documentés
- [ ] Links vers runbooks
- [ ] Performance tips included
- [ ] FAQ section
- [ ] Search-optimized

### Contenu guide troubleshooting

```markdown
# Troubleshooting Guide

## Quick Search

Use Ctrl+F to search for your error message.

## Common API Errors

### 400 — Bad Request

**Cause**: Invalid request format or parameters

**Solutions**:

- Check JSON syntax (must be valid JSON)
- Verify all required fields are provided
- Check field values match allowed options
- Example: voice_language must be 2-letter code (en, fr, es, etc.)

**Example Error**:
\`\`\`json
{
"detail": "Invalid language code",
"fields": {
"voice_language": "Invalid value 'english', must be 'en'"
}
}
\`\`\`

**Fix**:
\`\`\`bash

# WRONG

curl -X POST ... -d '{"voice_language": "english"}'

# CORRECT

curl -X POST ... -d '{"voice_language": "en"}'
\`\`\`

### 401 — Unauthorized

**Cause**: Missing or invalid API key

**Solutions**:

- Verify API key in Authorization header
- Check key isn't expired
- Regenerate key if unsure

\`\`\`bash

# WRONG

curl https://api.aiprod.ai/pipeline/run

# CORRECT

curl https://api.aiprod.ai/pipeline/run \
 -H "Authorization: Bearer sk_live_xxx"
\`\`\`

### 403 — Forbidden

**Cause**: You don't have permission

**Solutions**:

- Check account status (active/suspended)
- Verify you're using correct API key (prod key vs test key)
- Check IP whitelist (Enterprise only)

### 404 — Not Found

**Cause**: Job/resource doesn't exist

**Solutions**:

- Verify job_id is correct
- Check job was created successfully (status 202)
- Job may have expired (30 days retention)

\`\`\`bash
curl https://api.aiprod.ai/pipeline/JOB_ID \
 -H "Authorization: Bearer sk_live_xxx"
\`\`\`

### 429 — Rate Limited

**Cause**: Too many requests

**Solutions**:

- Wait before retrying (check Retry-After header)
- Implement exponential backoff
- Upgrade to higher tier
- Use batch endpoint for bulk jobs

\`\`\`python
import time

for attempt in range(5):
response = client.post(...)

    if response.status_code == 429:
        retry_after = int(response.headers['Retry-After'])
        time.sleep(retry_after)
        continue
    break

\`\`\`

### 500 — Server Error

**Cause**: Internal server error

**Solutions**:

- Wait a few seconds and retry
- Check status page: https://status.aiprod.ai
- Contact support if persistent

\`\`\`bash

# Wait and retry

for i in {1..3}; do
curl ... && break || sleep 5
done
\`\`\`

## Common Job Failures

### Job Status: FAILED — "Video processing error"

**Cause**: Video format not supported or corrupted

**Solutions**:

- Check video format (must be MP4, MOV, WebM, MKV)
- Check file size (<5GB)
- Verify file isn't corrupted

\`\`\`bash

# Check video codec

ffprobe your_video.mp4

# Expected output should show:

# Video: h264, yuv420p, 1920x1080

\`\`\`

### Job Status: FAILED — "Audio synthesis error"

**Cause**: Text too long or contains unsupported characters

**Solutions**:

- Keep voice_text < 10,000 characters
- Use ASCII characters
- Avoid special Unicode

### Job Status: FAILED — "Timeout"

**Cause**: Job took too long (>1 hour)

**Solutions**:

- Use shorter video
- Use simpler effects preset
- Split into multiple jobs

### Job Status: STUCK (no updates)

**Cause**: Usually network issue or database problem

**Solutions**:

- Wait 5 minutes (can be processing)
- Check Cloud Logging for errors
- Contact support if >30 min

## Performance Tips

### Reduce Processing Time

1. **Shorter videos**: 30-60 sec videos process faster than 10min+
2. **Simpler effects**: 'standard' preset faster than 'cinematic'
3. **Optimized audio**: Pre-compress audio before sending
4. **Batch processing**: Use `/pipeline/batch` for multiple videos

### Optimize API Usage

1. **Use webhooks instead of polling**
   - Webhooks: ~1 request per job
   - Polling: 10+ requests per job

2. **Implement request caching**
   - Cache responses locally
   - Avoid duplicate requests

3. **Use batch endpoint**
   - Instead of: 100 individual requests
   - Use: 1 batch request

## Database/Connection Issues

### Connection Refused

**Cause**: Database not accessible

**Solutions**:

- Check database is running: `gcloud sql instances describe aiprod-postgres`
- Check firewall rules
- Check VPC peering (if using private IP)

### Timeout

**Cause**: Database overloaded

**Solutions**:

- Check active connections: `SELECT count(*) FROM pg_stat_activity`
- Increase pool size
- Upgrade database tier

## FAQ

**Q: How long does a job take?**
A: Typically 3-10 minutes depending on video length and effects.

**Q: Can I cancel a job?**
A: No, once submitted you must wait for completion.

**Q: What video formats are supported?**
A: MP4, MOV, WebM, MKV (max 5GB)

**Q: Is my data encrypted?**
A: Yes, in transit (HTTPS) and at rest (Cloud SQL encryption).

**Q: How long are results retained?**
A: 30 days after job completion.

**Q: Can I get webhooks?**
A: Yes for Pro+ tiers.

**Q: Do you offer refunds?**
A: Check SLA for credit policy.

## Getting Help

1. **Check this guide**: Use Ctrl+F
2. **Check status page**: https://status.aiprod.ai
3. **Email support**: support@aiprod.ai (response: 24h)
4. **Chat** (Pro+): https://aiprod.slack.com (response: 4h)
5. **Call** (Enterprise): +1-XXX-XXX-XXXX (response: 1h)

## Report a Bug

Found a bug? Report it:

\`\`\`bash
curl -X POST https://api.aiprod.ai/support/bugs \
 -H "Authorization: Bearer sk_live_xxx" \
 -d '{
"title": "API returns 500 on batch requests",
"description": "When I submit >100 items, I get 500 error",
"steps": [
"1. Create batch with 101 items",
"2. Submit",
"3. Get 500 error"
],
"environment": "Python 3.10, requests 2.28"
}'
\`\`\`
```

### Métriques de succès

| Métrique        | Target | Validation |
| --------------- | ------ | ---------- |
| Guide created   | Yes    | ✅         |
| Errors covered  | 20+    | ✅         |
| Solutions clear | 100%   | ✅         |
| Searchable      | Yes    | ✅         |

### Notes

- Mettre à jour basé sur support tickets
- Lier depuis messages d'erreur si possible
- Inclure des diagnostics step-by-step
- Garder format simple et lisible

---

# 📝 PHASE 4 — Advanced Features & Optimization (Mars-Mai)

**Deadline** : 31 mai 2026  
**Durée totale** : ~12 heures  
**Objectif** : Optimisation avancée et nouvelles fonctionnalités  
**Dépendances** : PHASE 3 complétée

### Tâches Phase 4 (à détailler):

1. **TÂCHE 4.1-4.5** : Cost Optimization
   - Analyze current cloud costs
   - Implement auto-scaling policies
   - Optimize database queries
   - Setup cost alerts
   - Implement reserved capacity

2. **TÂCHE 4.6-4.11** : Advanced Features
   - Custom metrics dashboard
   - A/B testing framework
   - Self-healing infrastructure
   - Advanced analytics
   - White-label capabilities
   - Mobile SDK

---

## Document Complet — Résumé final

**✅ COUVERTURE COMPLÈTE** :

- PHASE CRITIQUE (6 tâches, 1h, Feb 5)
- PHASE 1 (9 tâches, 4h, Feb 6-9)
- PHASE 2 DB + API (15 tâches, 7h, Feb 17-28)
- PHASE 3 Documentation (5 tâches, 4h, Feb 17-28)
- PHASE 4 Advanced (11 tâches, 12h, Mar-May)

**TOTAL**: 46 tâches structurées, priorisées, avec code prêt à exécuter.

**Timeframe** : Feb 5 — May 31, 2026 (~130 heures de travail)

**Success Criteria** : ✅ 41/41 tâches initialement identifiées, plus 5 nouvelles tâches de documentation = 46 total

Vous pouvez maintenant supprimer le fichier `EXECUTION_ROADMAP_PART2.md` car il est fusionné dans ce document.

```

```
