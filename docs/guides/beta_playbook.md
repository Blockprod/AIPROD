# 📚 AIPROD V33 - Beta Program Playbook

**Guide complet pour réussir le programme beta AIPROD V33**

---

## 🎯 Vue d'Ensemble

Le **Beta Program AIPROD V33** est conçu pour 10 agences innovantes (10-50 employés).

**Période**: 3 mois gratuits  
**Tier**: PLATINUM (full features)  
**Support**: Dedicated success manager  
**Goal**: 5 jobs/semaine, quality score > 0.75

---

## 📋 Phase 1 : Onboarding (Jour 1)

### 1.1 - Réception des Credentials

Vous recevrez 3 fichiers:

```
📧 Email avec:
  ├─ API Key (aiprod_beta_xxx_yyy)
  ├─ Credentials JSON (aiprod_credentials.json)
  └─ GCS Folder (gs://aiprod-484120-aiprod-beta/clients/{client_id}/)
```

### 1.2 - Appel Onboarding (30 min)

**Objectif**: Comprendre vos cas d'usage

**Agenda**:

1. Présentation AIPROD V33 (5 min)
   - Architecture multi-backend
   - Garanties de qualité SLA
   - Économies potentielles
2. Vos besoins (10 min)
   - Types de vidéos
   - Fréquence de production
   - Contraintes qualité/coût
3. Démo live (10 min)
   - Appel API
   - ICC approval workflow
   - Cost estimation
4. Q&A + Next Steps (5 min)

**Préparation requise**:

- [ ] Installer Python + `pip install google-cloud-storage`
- [ ] Configurer Google Cloud SDK
- [ ] Préparer 2-3 briefs vidéo existants
- [ ] Identifier 2-3 personnes clés de l'équipe

---

## 🚀 Phase 2 : First Job (Jour 1-2)

### 2.1 - Setup Environnement

```bash
# 1. Installer les dépendances
pip install -r aiprod_requirements.txt

# 2. Configurer les credentials
export AIPROD_API_KEY="aiprod_beta_xxx_yyy"
export AIPROD_GCS_BUCKET="aiprod-484120-aiprod-beta"
export AIPROD_CLIENT_ID="client_xxx"

# 3. Tester la connexion
python -c "
import aiprod
client = aiprod.Client(api_key=os.getenv('AIPROD_API_KEY'))
print('✅ Connection OK')
"
```

### 2.2 - Premier Job - Quick Social

**Cas d'usage**: Social media content (30 sec, budget-friendly)

**Code**:

```python
import aiprod

client = aiprod.Client(api_key="your_api_key")

# Quick Social preset = fast turnaround + low cost
response = client.pipeline.run(
    content="A majestic golden eagle soaring over mountain peaks at sunset",
    preset="quick_social",  # 30s, quality 0.7+, ~$0.30
    callbacks={
        "on_complete": lambda job: print(f"✅ Job done: {job.id}"),
        "on_cost": lambda estimate: print(f"💰 Estimated: ${estimate}")
    }
)

print(f"Job ID: {response.job_id}")
print(f"Status: {response.status}")
print(f"Cost Estimate: ${response.cost_estimate}")
```

**Résultats attendus**:

- ✅ Génération: 45-60 secondes
- ✅ Qualité: 0.7-0.8
- ✅ Coût: $0.25-0.35
- ✅ Livraison: MP4 1080p

### 2.3 - Vérifier le Résultat

```bash
# Vérifier le job
curl -X GET https://api.aiprod.app/api/v1/job/{job_id} \
  -H "Authorization: Bearer aiprod_beta_xxx_yyy"

# Résultat:
{
  "job_id": "job_abc123",
  "status": "COMPLETED",
  "duration_sec": 54,
  "quality_score": 0.82,
  "cost_actual": 0.32,
  "cost_estimated": 0.30,
  "video_url": "gs://aiprod-484120-aiprod-beta/clients/xxx/output/job_abc123.mp4",
  "backend_used": "runway_gen3"
}
```

---

## 📊 Phase 3 : Weekly Engagement (Semaines 1-4)

### 3.1 - Weekly Targets

**Objectif**: 5 jobs minimum par semaine

| Semaine | Volume  | Type Vidéo     | Quality Target |
| ------- | ------- | -------------- | -------------- |
| 1       | 5 jobs  | Social + Quick | 0.70+          |
| 2       | 5 jobs  | Mixed          | 0.72+          |
| 3       | 7 jobs  | Brand Campaign | 0.75+          |
| 4       | 10 jobs | Variety        | 0.73+          |

### 3.2 - Weekly Feedback Loop

**Chaque jeudi à 2pm PT**:

1. **Check-in call** (15 min)
   - Succès de la semaine
   - Problèmes rencontrés
   - Questions techniques
2. **Submit feedback** (async)
   - Form: https://aiprod.typeform.com/beta-feedback
   - Topics:
     - Quels presets avez-vous utilisés?
     - Satisfaction qualité (1-10)
     - Améliorations suggérées
     - Cas d'usage découverts
3. **Review metrics** (dashboard)
   - Jobs completed
   - Avg quality score
   - Cost per job
   - API response times

### 3.3 - Success Metrics Tracking

```json
{
  "week_1": {
    "jobs_target": 5,
    "jobs_actual": 5,
    "quality_target": 0.7,
    "quality_actual": 0.72,
    "satisfaction": 8,
    "status": "✅ ON TRACK"
  },
  "week_2": {
    "jobs_target": 5,
    "jobs_actual": 6,
    "quality_target": 0.72,
    "quality_actual": 0.75,
    "satisfaction": 8.5,
    "status": "✅ ON TRACK"
  }
}
```

---

## 🎯 Phase 4 : Case Study Documentation (Week 2-3)

### 4.1 - Documenter 2-3 Cas d'Utilisation

**Template**:

```markdown
## Case Study: [Your Agency Name]

### Challenge

- Besoin: [description]
- Volume: [X jobs/mois avant]
- Budget: [$Y spent before]
- Timeline: [Z days to produce]

### Solution with AIPROD V33

- Preset utilisé: [quick_social|brand_campaign|premium_spot]
- Pipeline: [Fast Track|Full|Premium]
- Time saved: [before vs after]

### Results

- Videos produced: [X]
- Quality score: [0.8+]
- Cost per video: [$]
- Cost savings: [% vs Runway direct]
- Client satisfaction: [rating/10]

### Code Example

\`\`\`python

# Your implementation

\`\`\`

### Testimonial

"[Quote from team member]" - [Name], [Title]
```

### 4.2 - Soumettre pour Publication

Une fois documenté:

```bash
# Email à: beta-support@aiprod.app
# Sujet: Case Study Submission - [Your Agency]

# Inclure:
# - Markdown file
# - 1-2 screenshots/video thumbnails
# - Permission d'utiliser votre nom (ou anonyme)
```

---

## 💡 Tips & Best Practices

### Preset Selection

**🟢 quick_social** (Fast + Budget)

- Utilisé pour: Social media, quick turnaround
- Duration: 30 secondes
- Quality: 0.7 +
- Cost: $0.30-0.35/video
- Turnaround: 45-60s

**🟠 brand_campaign** (Balanced)

- Utilisé pour: Brand videos, ads
- Duration: Jusqu'à 2 min
- Quality: 0.7+ garanti
- Cost: $0.90-1.50/video
- Turnaround: 90-120s

**🔴 premium_spot** (Quality First)

- Utilisé pour: Premium content, broadcasts
- Duration: Jusqu'à 5 min
- Quality: 0.9+ garantie
- Cost: $1.50-3.00/video
- Turnaround: 120-180s

### Cost Optimization

```python
# ❌ DON'T: Ask for 0.95 quality if 0.75 is fine
response = client.pipeline.run(
    content="...",
    quality_target=0.95  # Unnecessary cost increase
)

# ✅ DO: Match quality to use case
response = client.pipeline.run(
    content="...",
    preset="quick_social"  # 0.7 quality = lower cost
)

# ✅ DO: Use cache for brand consistency
response = client.pipeline.run(
    content="...",
    brand_id="your_brand",  # Cache hit = faster + cheaper
    consistency_markers={...}
)
```

### ICC Workflow Best Practice

```python
# 1. Generate initial manifest
job = client.pipeline.run(
    content="...",
    preset="brand_campaign"
)

# 2. User reviews shots (manually or via ICC UI)
manifest = client.job(job.id).manifest()
print(manifest.shot_list)  # Can be edited

# 3. Approve and proceed to render
client.job(job.id).approve(
    manifest=manifest  # Updated manifest
)

# 4. Monitor quality
result = client.job(job.id).result()
print(f"Quality: {result.quality_score}")
```

---

## 📞 Support & Resources

### Communication Channels

| Channel                            | Purpose          | Response Time |
| ---------------------------------- | ---------------- | ------------- |
| **Slack** (invite link)            | Quick questions  | < 2 hours     |
| **Email**: beta-support@aiprod.app | Technical issues | < 4 hours     |
| **Weekly Call**                    | Progress review  | Scheduled     |
| **GitHub Issues**                  | Bug reports      | < 24 hours    |

### Documentation

- 📖 [Full API Docs](https://docs.aiprod.app/api)
- 🎨 [Presets Guide](https://docs.aiprod.app/presets)
- 💰 [Cost Calculator](https://aiprod.app/pricing)
- 🔐 [Security & Compliance](https://docs.aiprod.app/security)

### Common Questions

**Q: Can I use multiple API keys?**  
A: Yes, you can request up to 3 API keys for different teams.

**Q: What happens after 3 months of free tier?**  
A: We'll discuss pricing options. Early adopters typically get special rates.

**Q: Can I integrate with our existing tools?**  
A: Yes! We support webhooks, Zapier, and custom integrations.

**Q: What data is stored?**  
A: Only videos in your GCS folder. No data shared with other clients.

---

## 🎁 Beta Benefits Summary

```
✅ FREE PLATINUM TIER for 3 months
  ├─ Full features (ICC, consistency cache, etc.)
  ├─ Multi-user collaboration
  ├─ White-label delivery
  └─ Dedicated account manager

✅ 500 FREE JOBS during beta
  ├─ Full cost covered by AIPROD
  ├─ Unlimited quality revisions
  └─ Priority queue

✅ SPECIAL PRICING POST-BETA
  ├─ 30% discount vs published rates
  ├─ Annual commitment option
  └─ Custom SLA available

✅ OPPORTUNITY TO SHAPE V34
  ├─ Your feedback drives roadmap
  ├─ Early access to new features
  └─ Co-marketing opportunities
```

---

## 📈 Success Criteria

### Week 1-2

- [ ] 5+ jobs completed
- [ ] Quality score 0.70+
- [ ] Zero critical issues
- [ ] Team trained on API

### Week 3-4

- [ ] 7+ jobs/week
- [ ] Quality score 0.73+
- [ ] 1 case study documented
- [ ] Feedback provided

### Month 2

- [ ] 10+ jobs/week
- [ ] Consistent 0.75+ quality
- [ ] 2-3 case studies ready
- [ ] Production integration

### Month 3

- [ ] 15+ jobs/week
- [ ] Strong quality consistency
- [ ] Case studies published
- [ ] Ready for paid tier

---

## 🎬 Next Steps

1. **Confirm receipt** of credentials (reply to email)
2. **Schedule onboarding call** (link in credentials email)
3. **Review documentation** (see Resources above)
4. **Make first API call** (detailed guide in Phase 2)
5. **Log in dashboard** (https://dashboard.aiprod.app)

---

## 🙌 Questions?

**Email**: beta-support@aiprod.app  
**Slack**: #aiprod-beta channel  
**Phone**: +1 (415) 555-0123 (during business hours)

---

**Welcome to AIPROD V33 Beta Program! Let's create amazing videos together.** 🚀🎬

**Last updated**: January 15, 2026  
**Version**: 1.0 Beta
