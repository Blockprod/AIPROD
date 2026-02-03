# 📋 PLAN POST-DÉPLOIEMENT — AIPROD_V33

**Date de création** : 3 février 2026  
**Statut** : ✅ Infrastructure déployée — Tâches post-déploiement à planifier  
**URL Production** : https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app  
**Go-Live prévu** : 17 février 2026

---

## 🎯 RÉSUMÉ DES TÂCHES RESTANTES

| Priorité       | Catégorie             | Tâches        | Durée estimée | Deadline  |
| -------------- | --------------------- | ------------- | ------------- | --------- |
| 🔴 **ÉTAPE 4** | Go-Live Preparation   | 17 tâches     | ~5h30         | Feb 10-17 |
| 🟡 **Haute**   | Sécurité & Monitoring | 9 tâches      | ~4h           | Semaine 1 |
| 🟡 **Moyenne** | DB, API, Docs         | 9 tâches      | ~6h           | Mois 1    |
| 📝 **Basse**   | Optimisation          | 6 tâches      | ~4h           | Mois 2    |
| **TOTAL**      |                       | **41 tâches** | **~19h30**    |           |

---

## 🔴 ÉTAPE 4 : Go-Live Preparation (Feb 10-17)

### 1. Production Load Testing (2h)

| #   | Tâche                                  | Statut | Commande/Action                                                                   |
| --- | -------------------------------------- | ------ | --------------------------------------------------------------------------------- |
| 1   | Simulate 100 jobs/minute               | [ ]    | `hey -n 6000 -c 100 -m GET https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/health` |
| 2   | Verify autoscaling (1→10 instances)    | [ ]    | GCP Console → Cloud Run → Instances                                               |
| 3   | Check database connections (max 1,000) | [ ]    | Cloud SQL → Connections metrics                                                   |
| 4   | Monitor error rate (<0.1%)             | [ ]    | Cloud Monitoring → Error rate                                                     |
| 5   | Record P95 latency baseline            | [ ]    | Cloud Run → Request latencies                                                     |

**Outils recommandés** :

```bash
# Installation de hey (load testing)
go install github.com/rakyll/hey@latest

# Test de charge basique
hey -n 1000 -c 50 https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/health

# Test avec durée
hey -z 60s -c 100 https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/health
```

### 2. Disaster Recovery Drill (1h)

| #   | Tâche                               | Statut | Commande/Action                            |
| --- | ----------------------------------- | ------ | ------------------------------------------ |
| 6   | Test backup/restore procedure       | [ ]    | `scripts/backup_cloudsql.ps1`              |
| 7   | Verify PITR recovery time (<30 min) | [ ]    | Test point-in-time recovery                |
| 8   | Document runbook                    | [ ]    | Créer `docs/runbooks/disaster-recovery.md` |
| 9   | Test team notification flow         | [ ]    | Tester alertes email/Slack                 |

**Scripts de backup** :

```powershell
# Backup
.\scripts\backup_cloudsql.ps1

# Restore (test sur instance de staging si possible)
.\scripts\restore_cloudsql.ps1 -BackupFile "gs://aiprod-v33-backups/backup_xxx.sql"
```

### 3. Final Security Audit (1h)

| #   | Tâche                                   | Statut | Commande/Action                                |
| --- | --------------------------------------- | ------ | ---------------------------------------------- |
| 10  | Run OWASP Top 10 checks                 | [ ]    | Checklist manuelle ou `zap-cli`                |
| 11  | Verify all secrets in Secret Manager    | [ ]    | `gcloud secrets list --project=aiprod-484120`  |
| 12  | Check IAM permissions (least privilege) | [ ]    | `gcloud projects get-iam-policy aiprod-484120` |
| 13  | Enable Cloud Armor if needed            | [ ]    | GCP Console → Cloud Armor                      |

**Checklist OWASP Top 10** :

- [ ] A01:2021 – Broken Access Control → Vérifié via @require_auth
- [ ] A02:2021 – Cryptographic Failures → TLS enforced, secrets encrypted
- [ ] A03:2021 – Injection → SQLAlchemy ORM, input sanitization
- [ ] A04:2021 – Insecure Design → Architecture review done
- [ ] A05:2021 – Security Misconfiguration → Private Cloud SQL, VPC
- [ ] A06:2021 – Vulnerable Components → pip audit, dependabot
- [ ] A07:2021 – Authentication Failures → Firebase JWT
- [ ] A08:2021 – Data Integrity Failures → Audit logging
- [ ] A09:2021 – Security Logging Failures → Cloud Logging + Datadog
- [ ] A10:2021 – SSRF → Input validation

### 4. Communicate Go-Live (30 min)

| #   | Tâche                          | Statut | Action                       |
| --- | ------------------------------ | ------ | ---------------------------- |
| 14  | Notify stakeholders            | [ ]    | Email avec URL production    |
| 15  | Update status pages            | [ ]    | Mettre à jour README/docs    |
| 16  | Prepare incident response team | [ ]    | Définir contacts on-call     |
| 17  | Document support contacts      | [ ]    | Ajouter dans docs/support.md |

---

## 🟡 HAUTE PRIORITÉ : Sécurité & Monitoring (Semaine 1 — Feb 4-9)

### 3. Production Secrets Rotation (~1h30)

| #   | Tâche                                      | Statut | Commande/Action                  |
| --- | ------------------------------------------ | ------ | -------------------------------- |
| 18  | Implement secret rotation policy (90 days) | [ ]    | Cloud Scheduler + Cloud Function |
| 19  | Create KMS keys for secret encryption      | [ ]    | `gcloud kms keys create`         |
| 20  | Automate with Cloud Run scheduler          | [ ]    | Créer cron job rotation          |

**Implémentation** :

```bash
# Créer un keyring KMS
gcloud kms keyrings create aiprod-keyring \
  --location=europe-west1 \
  --project=aiprod-484120

# Créer une clé de chiffrement
gcloud kms keys create aiprod-secrets-key \
  --keyring=aiprod-keyring \
  --location=europe-west1 \
  --purpose=encryption \
  --project=aiprod-484120
```

### 4. DDoS & Rate Limiting (~1h30)

| #   | Tâche                            | Statut | Commande/Action                           |
| --- | -------------------------------- | ------ | ----------------------------------------- |
| 21  | Enable Cloud Armor for Cloud Run | [ ]    | `gcloud compute security-policies create` |
| 22  | Implement SlowAPI rate limiting  | [ ]    | `pip install slowapi` + middleware        |
| 23  | Configure WAF rules              | [ ]    | Cloud Armor → Security policies           |

**Cloud Armor setup** :

```bash
# Créer une politique de sécurité
gcloud compute security-policies create aiprod-security-policy \
  --description="Rate limiting and DDoS protection for AIPROD API"

# Ajouter une règle de rate limiting
gcloud compute security-policies rules create 1000 \
  --security-policy=aiprod-security-policy \
  --expression="true" \
  --action=rate-based-ban \
  --rate-limit-threshold-count=100 \
  --rate-limit-threshold-interval-sec=60 \
  --ban-duration-sec=600
```

**SlowAPI dans FastAPI** :

```python
# src/api/main.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/pipeline/run")
@limiter.limit("10/minute")  # 10 requêtes par minute par IP
async def run_pipeline(request: Request, ...):
    ...
```

### 5. Monitoring & Alerting (~1h)

| #   | Tâche                                       | Statut | Commande/Action                      |
| --- | ------------------------------------------- | ------ | ------------------------------------ |
| 24  | Setup email notifications for alerts        | [ ]    | GCP → Monitoring → Alerting policies |
| 25  | Configure Slack channel for Pub/Sub budgets | [ ]    | Webhook integration                  |
| 26  | Create escalation policy                    | [ ]    | Définir SLAs et contacts             |

**Créer une alerte Cloud Monitoring** :

```bash
# Alerte sur le taux d'erreur
gcloud alpha monitoring policies create \
  --display-name="High Error Rate" \
  --condition-display-name="Error rate > 1%" \
  --condition-filter='resource.type="cloud_run_revision" AND metric.type="run.googleapis.com/request_count" AND metric.label.response_code_class="5xx"' \
  --condition-threshold-value=0.01 \
  --condition-comparison=COMPARISON_GT \
  --notification-channels="projects/aiprod-484120/notificationChannels/CHANNEL_ID"
```

---

## 🟡 MOYENNE PRIORITÉ : DB, API, Docs (Mois 1 — Feb 17-28)

### 6. Database Optimization (~2h)

| #   | Tâche                           | Statut | Commande/Action        |
| --- | ------------------------------- | ------ | ---------------------- |
| 27  | Add database indexes            | [ ]    | SQL: `CREATE INDEX`    |
| 28  | Configure query caching (Redis) | [ ]    | Memorystore Redis      |
| 29  | Setup read replicas for scaling | [ ]    | Terraform modification |

**Indexes recommandés** :

```sql
-- Connecter à Cloud SQL
-- gcloud sql connect aiprod-v33-postgres --user=aiprod --project=aiprod-484120

-- Index sur le statut des jobs (très utilisé)
CREATE INDEX idx_jobs_status ON jobs(status);

-- Index sur la date de création (pour les requêtes de liste)
CREATE INDEX idx_jobs_created_at ON jobs(created_at DESC);

-- Index composite pour les requêtes filtrées
CREATE INDEX idx_jobs_user_status ON jobs(user_id, status);

-- Index sur les résultats par job
CREATE INDEX idx_results_job_id ON results(job_id);
```

### 7. API Enhancements (~2h)

| #   | Tâche                                        | Statut | Commande/Action               |
| --- | -------------------------------------------- | ------ | ----------------------------- |
| 30  | OpenAPI documentation (Swagger UI)           | [x]    | Déjà disponible: `/docs` ✅   |
| 31  | Implement request validation avec jsonschema | [ ]    | Pydantic validators avancés   |
| 32  | Add webhook support for async results        | [ ]    | Endpoint `/webhooks/register` |

**Webhook implementation** :

```python
# src/api/webhooks.py
from pydantic import BaseModel, HttpUrl

class WebhookRegistration(BaseModel):
    url: HttpUrl
    events: list[str] = ["job.completed", "job.failed"]
    secret: str  # Pour la signature HMAC

@app.post("/webhooks/register")
@require_auth
async def register_webhook(webhook: WebhookRegistration, user_id: str = Depends(get_user_id)):
    # Stocker en DB
    # Appeler le webhook quand un job est terminé
    ...
```

### 8. Documentation (~2h)

| #   | Tâche                                    | Statut | Fichier à créer                      |
| --- | ---------------------------------------- | ------ | ------------------------------------ |
| 33  | Create runbooks for common issues        | [ ]    | `docs/runbooks/common-issues.md`     |
| 34  | Add SLA documentation                    | [ ]    | `docs/business/sla-details.md`       |
| 35  | Create disaster recovery procedure guide | [ ]    | `docs/runbooks/disaster-recovery.md` |

**Structure des runbooks** :

```
docs/runbooks/
├── common-issues.md       # Problèmes fréquents et solutions
├── disaster-recovery.md   # Procédure de DR complète
├── scaling.md             # Guide de scaling
└── incident-response.md   # Gestion des incidents
```

---

## 📝 BASSE PRIORITÉ : Optimisation (Mois 2 — Mars 2026)

### 9. Cost Optimization (~2h)

| #   | Tâche                               | Statut | Action                          |
| --- | ----------------------------------- | ------ | ------------------------------- |
| 36  | Review Cloud SQL sizing             | [ ]    | Analyser métriques CPU/Memory   |
| 37  | Evaluate Spot instances for workers | [ ]    | Terraform: `preemptible = true` |
| 38  | Setup per-tenant cost allocation    | [ ]    | Labels GCP + Cost allocation    |

**Analyse des coûts** :

```bash
# Voir les coûts par service
gcloud billing budgets list --billing-account=BILLING_ACCOUNT_ID

# Exporter les coûts vers BigQuery pour analyse
# GCP Console → Billing → Cost table → Export to BigQuery
```

### 10. Advanced Features (~2h)

| #   | Tâche                                      | Statut | Action                                 |
| --- | ------------------------------------------ | ------ | -------------------------------------- |
| 39  | Implement custom metrics for business KPIs | [ ]    | Prometheus custom metrics              |
| 40  | Add A/B testing framework                  | [ ]    | Feature flags (LaunchDarkly/Unleash)   |
| 41  | Create self-healing mechanisms             | [ ]    | Cloud Run auto-restart + health checks |

**Custom metrics example** :

```python
# src/utils/metrics.py
from prometheus_client import Counter, Histogram

# Business KPIs
jobs_revenue_total = Counter(
    'jobs_revenue_usd_total',
    'Total revenue from jobs in USD',
    ['tier', 'aspect_ratio']
)

job_processing_cost = Histogram(
    'job_processing_cost_usd',
    'Cost per job in USD',
    buckets=(1, 5, 10, 25, 50, 100)
)

# Usage
jobs_revenue_total.labels(tier='premium', aspect_ratio='16:9').inc(25.00)
job_processing_cost.observe(12.50)
```

---

## 📅 CALENDRIER PROPOSÉ

```
┌─────────────────────────────────────────────────────────────────┐
│  FÉVRIER 2026                                                    │
├─────────────────────────────────────────────────────────────────┤
│  Lun 3    │ ✅ Déploiement FAIT                                 │
│  Mar 4    │ 🟡 Secret rotation + KMS keys (1h30)                │
│  Mer 5    │ 🟡 Cloud Armor + Rate limiting (1h30)               │
│  Jeu 6    │ 🟡 Alerting email/Slack (1h)                        │
│  Ven 7-9  │ Buffer / Documentation                              │
├─────────────────────────────────────────────────────────────────┤
│  Lun 10   │ 🔴 Load testing (2h)                                │
│  Mar 11   │ 🔴 Disaster recovery drill (1h)                     │
│  Mer 12   │ 🔴 Security audit OWASP (1h)                        │
│  Jeu 13   │ 🔴 Communication go-live (30 min)                   │
│  Ven 14   │ Buffer                                              │
│  Lun 17   │ 🚀 GO-LIVE OFFICIEL                                 │
├─────────────────────────────────────────────────────────────────┤
│  Feb 17-28│ 🟡 Moyenne priorité (DB, API, Docs)                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  MARS 2026                                                       │
├─────────────────────────────────────────────────────────────────┤
│  Sem 1-2  │ 📝 Basse priorité (Cost, Advanced features)         │
│  Mar 17   │ 📋 Prochaine revue d'audit                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🏃 ACTIONS IMMÉDIATES (Feb 4)

### Demain — Focus Sécurité

```bash
# 1. Vérifier l'état actuel de l'API
curl https://aiprod-v33-api-hxhx3s6eya-ew.a.run.app/health

# 2. Lister les secrets actuels
gcloud secrets list --project=aiprod-484120

# 3. Créer le keyring KMS
gcloud kms keyrings create aiprod-keyring \
  --location=europe-west1 \
  --project=aiprod-484120

# 4. Vérifier les alertes existantes
gcloud alpha monitoring policies list --project=aiprod-484120
```

---

## ✅ CHECKLIST GLOBALE

### ÉTAPE 4 : Go-Live Preparation (17 tâches)

- [ ] 1. Simulate 100 jobs/minute
- [ ] 2. Verify autoscaling
- [ ] 3. Check database connections
- [ ] 4. Monitor error rate
- [ ] 5. Record P95 latency baseline
- [ ] 6. Test backup/restore
- [ ] 7. Verify PITR recovery
- [ ] 8. Document runbook
- [ ] 9. Test notifications
- [ ] 10. OWASP Top 10
- [ ] 11. Verify secrets
- [ ] 12. Check IAM
- [ ] 13. Enable Cloud Armor
- [ ] 14. Notify stakeholders
- [ ] 15. Update status pages
- [ ] 16. Prepare incident team
- [ ] 17. Document contacts

### Haute Priorité (9 tâches)

- [ ] 18. Secret rotation policy
- [ ] 19. KMS keys
- [ ] 20. Automate rotation
- [ ] 21. Cloud Armor
- [ ] 22. SlowAPI rate limiting
- [ ] 23. WAF rules
- [ ] 24. Email notifications
- [ ] 25. Slack integration
- [ ] 26. Escalation policy

### Moyenne Priorité (9 tâches)

- [ ] 27. Database indexes
- [ ] 28. Query caching (Redis)
- [ ] 29. Read replicas
- [x] 30. OpenAPI docs ✅
- [ ] 31. Request validation
- [ ] 32. Webhook support
- [ ] 33. Runbooks
- [ ] 34. SLA documentation
- [ ] 35. DR procedure guide

### Basse Priorité (6 tâches)

- [ ] 36. Cloud SQL sizing review
- [ ] 37. Spot instances
- [ ] 38. Cost allocation
- [ ] 39. Custom metrics
- [ ] 40. A/B testing
- [ ] 41. Self-healing

---

## 📊 MÉTRIQUES DE SUCCÈS

| Métrique         | Target  | Actuel | Status |
| ---------------- | ------- | ------ | ------ |
| API latency p99  | <500ms  | TBD    | ⏳     |
| Error rate       | <0.1%   | TBD    | ⏳     |
| Cost/job         | <$12.50 | TBD    | ⏳     |
| Job success rate | >99%    | TBD    | ⏳     |
| Database latency | <50ms   | TBD    | ⏳     |
| Pub/Sub lag      | <5 min  | TBD    | ⏳     |

---

**Document créé** : 3 février 2026  
**Dernière mise à jour** : 3 février 2026  
**Prochaine revue** : 17 février 2026 (Go-Live)
