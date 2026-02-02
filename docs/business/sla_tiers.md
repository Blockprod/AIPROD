# AIPROD V33 - SLA & Pricing Tiers

## Vue d'Ensemble des Tiers

| Tier         | Prix Base | Prix Usage  | SLA Latence | Qualité Garantie | Features              |
| ------------ | --------- | ----------- | ----------- | ---------------- | --------------------- |
| **BRONZE**   | $99/mois  | $0.35/vidéo | 5 min       | 0.6+             | Fast Track only       |
| **GOLD**     | $299/mois | $0.95/min   | 15 min      | 0.7+             | Full pipeline + ICC   |
| **PLATINUM** | $999/mois | $1.50/min   | 5 min       | 0.9+             | Premium + White-label |

---

## 🥉 BRONZE - Starter

**Prix:** $99/mois + $0.35/vidéo

### Inclus

- ✅ Fast Track pipeline (vidéos 30s max)
- ✅ Preset `quick_social` uniquement
- ✅ 100 vidéos incluses/mois
- ✅ Support email (48h réponse)
- ✅ API REST standard
- ✅ Stockage GCS 10GB

### SLA

- **Latence:** 5 minutes max par vidéo
- **Qualité:** Score 0.6+ garanti
- **Uptime:** 99%
- **Dépassement coût:** N/A (fixe par vidéo)

### Limites

- ❌ Pas d'ICC (Interactive Creative Control)
- ❌ Pas de cache de cohérence marque
- ❌ Pas de multi-review
- ❌ Maximum 30 secondes par vidéo
- ❌ Pas de priorité haute

### Cas d'usage idéal

- Tests et prototypage
- Contenu social media rapide
- Petites équipes marketing
- Volume faible (<100 vidéos/mois)

---

## 🥇 GOLD - Professional ⭐ Recommandé

**Prix:** $299/mois + $0.95/minute de vidéo

### Inclus

- ✅ Pipeline complet avec 11 agents
- ✅ Tous les presets (`quick_social`, `brand_campaign`, `premium_spot`)
- ✅ **Interactive Creative Control (ICC)**
- ✅ **Cache de cohérence marque (7 jours)**
- ✅ WebSocket temps réel
- ✅ Support prioritaire (24h réponse)
- ✅ API REST + Webhooks
- ✅ Stockage GCS 100GB
- ✅ Dashboard analytics

### SLA

- **Latence:** 15 minutes max par vidéo standard
- **Qualité:** Score 0.7+ garanti
- **Uptime:** 99.5%
- **Dépassement coût:** ±20% max

### Features Exclusives

- 🎯 **ICC Manifest Editor:** Modifiez le script avant rendu
- 🎨 **Brand Consistency:** Réutilisation style entre projets
- 📊 **Cost Estimation:** Estimation avant génération
- 🔔 **Real-time Updates:** WebSocket pour suivi live

### Limites

- ❌ Pas de multi-review (approval chain)
- ❌ Pas de white-label
- ❌ Maximum 120 secondes par vidéo

### Cas d'usage idéal

- Agences marketing (10-50 employés)
- Campagnes de marque
- Production régulière (50-500 vidéos/mois)
- Cohérence visuelle importante

---

## 💎 PLATINUM - Enterprise

**Prix:** $999/mois + $1.50/minute de vidéo

### Inclus

- ✅ **Tout le tier GOLD**
- ✅ Multi-review avec approval chain
- ✅ White-label delivery
- ✅ Account manager dédié
- ✅ SLA premium
- ✅ Stockage GCS illimité
- ✅ Custom API endpoints
- ✅ Priority queue
- ✅ Multi-tenant support

### SLA Premium

- **Latence:** 5 minutes max (priorité haute)
- **Qualité:** Score 0.9+ garanti
- **Uptime:** 99.9%
- **Dépassement coût:** ±10% max
- **Support:** 4h réponse, téléphone direct

### Features Exclusives

- 👥 **Multi-Review:** Chaîne d'approbation multi-utilisateurs
- 🏷️ **White-Label:** Assets livrés sans branding AIPROD
- 🎖️ **Priority Queue:** Vos jobs passent en premier
- 📞 **Account Manager:** Contact dédié
- 🔧 **Custom Integrations:** Endpoints sur mesure
- 📈 **Advanced Analytics:** Métriques détaillées par projet/client

### Cas d'usage idéal

- Grandes agences (50+ employés)
- Marques Fortune 500
- Production haute qualité
- Spots publicitaires TV/Web
- Volume élevé (500+ vidéos/mois)

---

## 📊 Comparaison Détaillée

| Feature               | BRONZE     | GOLD           | PLATINUM        |
| --------------------- | ---------- | -------------- | --------------- |
| **Pipeline**          | Fast Track | Full 11 agents | Full + Priority |
| **Max Duration**      | 30s        | 120s           | 180s            |
| **ICC**               | ❌         | ✅             | ✅              |
| **Manifest Edit**     | ❌         | ✅             | ✅              |
| **Consistency Cache** | ❌         | ✅ 7 jours     | ✅ 30 jours     |
| **WebSocket**         | ❌         | ✅             | ✅              |
| **Multi-Review**      | ❌         | ❌             | ✅              |
| **White-Label**       | ❌         | ❌             | ✅              |
| **Priority Queue**    | ❌         | ❌             | ✅              |
| **Account Manager**   | ❌         | ❌             | ✅              |
| **Support**           | Email 48h  | Email 24h      | Phone 4h        |
| **Uptime SLA**        | 99%        | 99.5%          | 99.9%           |
| **Quality SLA**       | 0.6+       | 0.7+           | 0.9+            |
| **Cost Variance**     | N/A        | ±20%           | ±10%            |

---

## 💰 Exemples de Coûts

### Scénario 1: Startup Social Media (BRONZE)

```
- 80 vidéos/mois × 30s
- Coût: $99 + (80 × $0.35) = $127/mois
```

### Scénario 2: Agence Marketing (GOLD)

```
- 50 vidéos/mois × 60s moyenne
- Coût: $299 + (50 × $0.95) = $346.50/mois
```

### Scénario 3: Grande Marque (PLATINUM)

```
- 100 vidéos/mois × 90s moyenne
- Coût: $999 + (100 × 1.5 × $1.50) = $1,224/mois
```

---

## 🔄 Upgrade Path

```
BRONZE → GOLD
- Gain: ICC, Consistency Cache, Quality SLA
- Coût additionnel: ~$200/mois base

GOLD → PLATINUM
- Gain: Multi-Review, White-Label, Priority, Account Manager
- Coût additionnel: ~$700/mois base
```

---

## 📋 SLA Détaillés

### Latence

| Tier     | Fast Track | Standard | Premium |
| -------- | ---------- | -------- | ------- |
| BRONZE   | 5 min      | N/A      | N/A     |
| GOLD     | 5 min      | 15 min   | N/A     |
| PLATINUM | 3 min      | 5 min    | 5 min   |

### Qualité (Quality Score)

| Tier     | Minimum | Cible | Compensation si < minimum          |
| -------- | ------- | ----- | ---------------------------------- |
| BRONZE   | 0.6     | 0.7   | Régénération gratuite              |
| GOLD     | 0.7     | 0.8   | Régénération gratuite + 10% crédit |
| PLATINUM | 0.9     | 0.95  | Régénération gratuite + 20% crédit |

### Uptime

| Tier     | SLA   | Compensation/heure down |
| -------- | ----- | ----------------------- |
| BRONZE   | 99%   | N/A                     |
| GOLD     | 99.5% | 5% crédit mensuel       |
| PLATINUM | 99.9% | 10% crédit mensuel      |

---

## 🚀 Comment Démarrer

1. **Choisir votre tier** selon volume et besoins
2. **Créer un compte** sur aiprod.io
3. **Configurer votre API key**
4. **Tester avec preset `quick_social`**
5. **Upgrader selon utilisation**

---

_Contact: sales@aiprod.io | Beta Access: beta@aiprod.io_
