# 🎯 Plan d'Amélioration AIPROD - "Cost Intelligence First"

**Date**: Février 6, 2026  
**Vision**: Transformer AIPROD en **plateforme d'orchestration vidéo intelligente basée sur les coûts**, où chaque utilisateur devient **stratégiste économique** avant de générer.

---

## 🔥 Le Insight Central

**Le vrai problème utilisateur**: Pas la résolution, c'est l'incertitude économique.

```
❌ Avant (utilisateur type):
   "Je veux une vidéo"
   ↓
   Appelle API
   ↓
   "Euh... ça m'a coûté $5?"
   ↓
   😡 Churn + mauvaise review

✅ Après AIPROD (avec plan):
   "Je veux une vidéo"
   ↓
   Appelle /video/plan
   ↓
   "Voici 3 options: $0.04, $0.08, $0.50"
   ↓
   Choisit conscient → Génère serein
   ↓
   😊 "Excellent, c'était exactement $0.08!"
   ↓
   😍 Retention + Premium upsell
```

**Positioning Shift**:

- ❌ De: "Premium video generator with better backends"
- ✅ À: **"Cost-aware strategic video platform"** ← Defensible, unique, scalable

---

## 🎯 Objectif Général

Construire une **couche de Cost Intelligence** qui:

1. **Planifie** avant de générer (transparence)
2. **Optimise** automatiquement (IA recommande)
3. **Justifie** les choix (reçu réel vs estimé)
4. **Fidélise** par clarté économique

**Différenciation**: Autres platforms généralisent. AIPROD "stratégise".

---

## 📊 État Actuel vs Vision

| Aspect             | Actuel                | Vision 2026                         |
| ------------------ | --------------------- | ----------------------------------- |
| **Approche**       | Tunnel → Gen → Hope   | Plan → Choose → Execute → Receipt   |
| **Utilisateur**    | Passif                | **Stratégiste informé**             |
| **Transparence**   | "Ça a coûté combien?" | "C'était exactement le coût prédit" |
| **Value**          | Qualité vidéo         | **Confiance économique**            |
| **Positionnement** | Commodity builder     | **Cost oracle**                     |
| **Revenue Upside** | Margin fixe           | **Premium tier** (guided = +30%)    |

---

## 🚀 Plan par Priorité (4 semaines)

### 🔴 P0 - Semaine 1: Cost Intelligence Core

#### 1.1 Implémenter CostEstimator (Effort: 6h)

**Le nouveau cœur d'AIPROD**: Système de recommandation économique.

```python
# src/agents/cost_estimator.py

from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class GenerationTier(Enum):
    """Trois stratégies de génération"""
    PREMIUM = "premium"      # Budget illimité, meilleure qualité
    BALANCED = "balanced"    # Sweet spot: qualité/coût
    ECONOMY = "economy"      # Minimal, ultra-economique

@dataclass
class GenerationPlan:
    """Blueprint pour une génération vidéo"""
    tier: GenerationTier
    backend: str             # "runway_gen4_turbo", "veo-3.0", "veo-2.0"
    estimated_cost_usd: float
    estimated_time_sec: int
    quality_tier: str        # "ULTRA", "HIGH", "GOOD"
    resolution: str          # "4K", "1080p", "720p"
    recommended: bool = False
    reason: str = ""         # Pourquoi ce tier pour ce user

    def to_dict(self) -> dict:
        return {
            "tier": self.tier.value,
            "backend": self.backend,
            "estimated_cost": self.estimated_cost_usd,
            "time_seconds": self.estimated_time_sec,
            "quality": self.quality_tier,
            "resolution": self.resolution,
            "recommended": self.recommended,
            "reason": self.reason
        }

class CostEstimator:
    """Moteur d'estimation intelligente des coûts vidéo"""

    def __init__(self):
        self.runway_balance = None
        self.user_history = {}

    def estimate_plans(self,
                      prompt: str,
                      user_id: str = None,
                      constraints: dict = None) -> list[GenerationPlan]:
        """
        Retourne 3 plans (premium, balanced, economy) + recommendation

        Args:
            prompt: Description de la vidéo
            user_id: ID utilisateur (pour historique)
            constraints: {"max_cost": 0.10, "min_quality": "HIGH", etc}

        Returns:
            Liste de 3 GenerationPlan triés par qualité
        """

        # Analyser prompt
        prompt_tokens = len(prompt.split())
        complexity = self._assess_complexity(prompt)

        # Vérifier équilibre Runway
        runway_credits = self._check_runway_balance()

        # Charger historique utilisateur
        user_prefs = self._load_user_preferences(user_id) if user_id else {}

        plans = []

        # === TIER 1: PREMIUM ===
        # Condition: Runway avec assez de credits
        if runway_credits >= 35:
            premium_plan = GenerationPlan(
                tier=GenerationTier.PREMIUM,
                backend="runway_gen4_turbo",
                estimated_cost_usd=0.50,
                estimated_time_sec=30,
                quality_tier="ULTRA ⭐⭐⭐⭐⭐",
                resolution="2K-4K (depending on model)",
                recommended=False,
                reason="Maximum quality, Runway's flagship model"
            )
            plans.append(premium_plan)

        # === TIER 2: BALANCED (Always available) ===
        # C'est le "default" intelligent
        balanced_reason = "Best quality/cost ratio"
        if user_prefs.get("prefers_fast"):
            balanced_reason += " + Fast execution"

        balanced_plan = GenerationPlan(
            tier=GenerationTier.BALANCED,
            backend="veo-3.0",
            estimated_cost_usd=0.08,
            estimated_time_sec=45,
            quality_tier="HIGH ⭐⭐⭐⭐",
            resolution="1080p native",
            recommended=True,  # ← AI wisdom
            reason=balanced_reason
        )
        plans.append(balanced_plan)

        # === TIER 3: ECONOMY ===
        # Parfait pour social media, prototypes
        economy_plan = GenerationPlan(
            tier=GenerationTier.ECONOMY,
            backend="veo-2.0",
            estimated_cost_usd=0.04,
            estimated_time_sec=48,
            quality_tier="GOOD ⭐⭐⭐",
            resolution="720p (social-optimized)",
            recommended=False,
            reason="Perfect for TikTok/Instagram Reels"
        )
        plans.append(economy_plan)

        # === APPLIQUER CONSTRAINTS ===
        if constraints:
            plans = self._filter_by_constraints(plans, constraints)

        # === JUSTIFIER RECOMMENDATION ===
        weighted_recommendation = self._compute_recommendation(
            plans,
            user_prefs,
            runway_credits
        )
        for plan in plans:
            if plan.tier == weighted_recommendation:
                plan.recommended = True
            else:
                plan.recommended = False

        logger.info(
            f"Generated {len(plans)} plans for '{prompt[:30]}...' "
            f"(Runway: {runway_credits} credits, User: {user_id})"
        )

        return plans

    def _assess_complexity(self, prompt: str) -> str:
        """Analyser complexité du prompt"""
        tokens = len(prompt.split())
        if tokens > 150:
            return "high"
        elif tokens > 50:
            return "medium"
        return "low"

    def _check_runway_balance(self) -> int:
        """Vérifier credits Runway (avec cache 5min)"""
        try:
            import runway
            account = runway.Account.get()
            credits = account.balance.credits
            logger.info(f"Runway balance: {credits} credits")
            return credits
        except Exception as e:
            logger.warning(f"Could not fetch Runway balance: {e}")
            return 0

    def _load_user_preferences(self, user_id: str) -> dict:
        """Charger historique utilisateur pour perso"""
        # TODO: Implémenter cache DB
        return {
            "avg_spend": 0.06,
            "prefers_fast": False,
            "preferred_backend": "veo3"
        }

    def _filter_by_constraints(self,
                               plans: list[GenerationPlan],
                               constraints: dict) -> list[GenerationPlan]:
        """Filtrer plans selon constraints utilisateur"""
        filtered = []

        for plan in plans:
            # Budget constraint
            if "max_cost" in constraints:
                if plan.estimated_cost_usd > constraints["max_cost"]:
                    continue

            # Quality constraint
            if "min_quality" in constraints:
                quality_levels = ["GOOD", "HIGH", "ULTRA"]
                plan_qual = plan.quality_tier.split()[0]
                required_qual = constraints["min_quality"]
                if quality_levels.index(plan_qual) < quality_levels.index(required_qual):
                    continue

            # Time constraint
            if "max_time_sec" in constraints:
                if plan.estimated_time_sec > constraints["max_time_sec"]:
                    continue

            filtered.append(plan)

        return filtered if filtered else plans

    def _compute_recommendation(self,
                               plans: list[GenerationPlan],
                               user_prefs: dict,
                               runway_credits: int) -> GenerationTier:
        """Calculer tier recommandé via scoring"""

        scores = {tier: 0 for tier in GenerationTier}

        # Runway premium si disponible et user n'a pas budget limité
        if runway_credits >= 35 and user_prefs.get("avg_spend", 0) > 0.30:
            scores[GenerationTier.PREMIUM] += 50

        # Balanced = default safe choice for most
        scores[GenerationTier.BALANCED] += 100

        # Economy si user budget-conscious
        if user_prefs.get("avg_spend", 0) < 0.05:
            scores[GenerationTier.ECONOMY] += 40

        return max(scores, key=scores.get)


# === API ENDPOINTS ===

from fastapi import FastAPI, HTTPException
from typing import Optional

app = FastAPI()
cost_estimator = CostEstimator()

@app.post("/video/plan")
async def plan_video(
    prompt: str,
    user_id: Optional[str] = None,
    max_cost: Optional[float] = None,
    min_quality: Optional[str] = None
):
    """
    Endpoint utilisateur: voir les 3 options AVANT committing

    Usage:
        curl -X POST http://localhost:8000/video/plan \
             -d "prompt=futuristic AI dashboard" \
             -d "max_cost=0.10" \
             -d "min_quality=HIGH"

    Response:
        {
            "prompt_summary": "futuristic AI dashboard",
            "plans": [
                {
                    "tier": "premium",
                    "cost": 0.50,
                    "time": 30,
                    "quality": "ULTRA ⭐⭐⭐⭐⭐",
                    "recommended": false
                },
                {
                    "tier": "balanced",
                    "cost": 0.08,
                    "time": 45,
                    "quality": "HIGH ⭐⭐⭐⭐",
                    "recommended": true,  ← Pick this!
                    "reason": "Best quality/cost ratio"
                },
                {
                    "tier": "economy",
                    "cost": 0.04,
                    "time": 48,
                    "quality": "GOOD ⭐⭐⭐",
                    "recommended": false
                }
            ],
            "ai_wisdom": "I recommend BALANCED for your budget & use case"
        }
    """

    constraints = {}
    if max_cost:
        constraints["max_cost"] = max_cost
    if min_quality:
        constraints["min_quality"] = min_quality

    plans = cost_estimator.estimate_plans(prompt, user_id, constraints)

    if not plans:
        raise HTTPException(status_code=400, detail="No plans match constraints")

    return {
        "prompt_summary": prompt[:60] + "..." if len(prompt) > 60 else prompt,
        "plans": [p.to_dict() for p in plans],
        "ai_wisdom": f"I recommend {[p for p in plans if p.recommended][0].tier.value.upper()} for this request",
        "message": "Choose your strategy ⬇️"
    }

@app.post("/video/generate")
async def generate_video_with_plan(
    prompt: str,
    tier: str = "balanced",  # "premium", "balanced", or "economy"
    user_id: Optional[str] = None
):
    """
    Endpoint principal: Générer avec le tier choisi

    Usage:
        curl -X POST http://localhost:8000/video/generate \
             -d "prompt=futuristic AI dashboard" \
             -d "tier=balanced" \
             -d "user_id=user123"

    Response:
        {
            "video_url": "gs://aiprod.../video_abc123.mp4",
            "video_metadata": {
                "resolution": "1920x1080",
                "duration_sec": 5,
                "codec": "h264"
            },
            "cost_receipt": {
                "tier": "balanced",
                "backend": "veo-3.0",
                "estimated_cost": 0.08,
                "actual_cost": 0.076,
                "savings": 0.004,
                "message": "Actually cheaper than expected! 🎉"
            }
        }
    """

    # Récupérer le plan
    gen_tier = GenerationTier[tier.upper()]

    # Trouver le plan demandé
    plans = cost_estimator.estimate_plans(prompt, user_id)
    selected_plan = next((p for p in plans if p.tier == gen_tier), plans[1])  # Default: balanced

    # Générer vidéo
    from src.agents.render_executor import RenderExecutor
    executor = RenderExecutor(
        backend=selected_plan.backend,
        quality_level=selected_plan.quality_tier.split()[0]
    )

    result = executor.run(prompt)

    # Retourner avec reçu économique
    actual_cost = result.get("cost", selected_plan.estimated_cost_usd)
    savings = selected_plan.estimated_cost_usd - actual_cost

    return {
        "video_url": result["path"],
        "video_metadata": result["metadata"],
        "cost_receipt": {
            "tier": selected_plan.tier.value,
            "backend": selected_plan.backend,
            "estimated_cost": round(selected_plan.estimated_cost_usd, 4),
            "actual_cost": round(actual_cost, 4),
            "savings": round(savings, 4),
            "message": f"✓ Cost exactly as predicted!" if abs(savings) < 0.001 else f"💰 Saved ${savings:.3f}!"
        },
        "timestamp": datetime.now().isoformat()
    }
```

**Tâches**:

- [ ] Créer `src/agents/cost_estimator.py` (copier code ci-dessus)
- [ ] Ajouter endpoints `/video/plan` et `/video/generate` à API
- [ ] Tester avec 3 prompts différents
- [ ] Vérifier Runway integration (credentials)
- [ ] Documenter dans README

**Critères de Succès**:

- `/video/plan` retourne 3 options en < 2s
- Recommendation engine fonctionne
- `/video/generate` exécute et retourne coût réel
- Coût réel vs estimé écart < 5%

---

#### 1.2 Construire Video Passport UI (Effort: 4h)

**Interface frontend simple mais puissante**: Utilisateur voit les 3 options, choisit conscient.

```jsx
// dashboard/src/components/VideoPlanner.jsx

import React, { useState } from "react";
import "./VideoPlanner.css";

export function VideoPlanner() {
  const [prompt, setPrompt] = useState("");
  const [plans, setPlans] = useState([]);
  const [selectedTier, setSelectedTier] = useState("balanced");
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [result, setResult] = useState(null);

  const handlePlanRequest = async () => {
    if (!prompt.trim()) return;

    setLoading(true);
    try {
      const response = await fetch("/video/plan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });
      const data = await response.json();
      setPlans(data.plans);
      setSelectedTier("balanced");
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleGenerate = async () => {
    if (!selectedTier || !plans.length) return;

    setGenerating(true);
    try {
      const response = await fetch("/video/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, tier: selectedTier }),
      });
      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error(err);
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div className="video-planner">
      <h1>🎬 AIPROD Video Planner</h1>

      {/* === STEP 1: INPUT === */}
      {!plans.length && (
        <div className="step step-input">
          <h2>Describe Your Video</h2>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="E.g., 'Futuristic AI dashboard with glowing metrics, dark theme, sci-fi ambient..'"
            rows={4}
          />
          <button
            onClick={handlePlanRequest}
            disabled={loading || !prompt.trim()}
            className="btn btn-primary"
          >
            {loading ? "Analyzing..." : "See Options →"}
          </button>
        </div>
      )}

      {/* === STEP 2: PLAN SELECTION === */}
      {plans.length > 0 && !result && (
        <div className="step step-plans">
          <h2>Choose Your Strategy</h2>
          <div className="plans-grid">
            {plans.map((plan) => (
              <div
                key={plan.tier}
                className={`plan-card ${selectedTier === plan.tier ? "selected" : ""} ${plan.recommended ? "recommended" : ""}`}
                onClick={() => setSelectedTier(plan.tier)}
              >
                <div className="plan-header">
                  <h3 className="plan-tier">{plan.tier.toUpperCase()}</h3>
                  {plan.recommended && (
                    <span className="badge">🤖 Recommended</span>
                  )}
                </div>

                <div className="plan-cost">${plan.estimated_cost}</div>
                <div className="plan-quality">{plan.quality}</div>
                <div className="plan-resolution">{plan.resolution}</div>
                <div className="plan-time">
                  ⏱️ {plan.time_seconds}s generation
                </div>

                <div className="plan-reason">
                  <small>{plan.reason}</small>
                </div>

                {selectedTier === plan.tier && (
                  <div className="checkmark">✓</div>
                )}
              </div>
            ))}
          </div>

          <button
            onClick={handleGenerate}
            disabled={generating}
            className="btn btn-generate btn-primary"
          >
            {generating
              ? "Generating..."
              : `Generate with ${selectedTier.toUpperCase()} ▶️`}
          </button>
        </div>
      )}

      {/* === STEP 3: RESULT & RECEIPT === */}
      {result && (
        <div className="step step-result">
          <h2>✓ Video Generated!</h2>

          <div className="video-preview">
            <video src={result.video_url} controls width="100%" height="400" />
          </div>

          <div className="cost-receipt">
            <h3>💰 Cost Breakdown</h3>
            <div className="receipt-grid">
              <div className="receipt-item">
                <span>Tier Selected</span>
                <strong>{result.cost_receipt.tier.toUpperCase()}</strong>
              </div>
              <div className="receipt-item">
                <span>Backend Used</span>
                <strong>{result.cost_receipt.backend}</strong>
              </div>
              <div className="receipt-item">
                <span>Estimated Cost</span>
                <strong>${result.cost_receipt.estimated_cost}</strong>
              </div>
              <div className="receipt-item">
                <span>Actual Cost</span>
                <strong>${result.cost_receipt.actual_cost}</strong>
              </div>
              <div className="receipt-item savings">
                <span>Savings</span>
                <strong style={{ color: "#4ade80" }}>
                  ${result.cost_receipt.savings}
                </strong>
              </div>
            </div>
            <div className="receipt-message">{result.cost_receipt.message}</div>
          </div>

          <div className="video-metadata">
            <h3>📊 Video Details</h3>
            <pre>{JSON.stringify(result.video_metadata, null, 2)}</pre>
          </div>

          <button
            onClick={() => {
              setPrompt("");
              setPlans([]);
              setResult(null);
            }}
            className="btn btn-secondary"
          >
            Generate Another ↻
          </button>
        </div>
      )}
    </div>
  );
}
```

```css
/* dashboard/src/components/VideoPlanner.css */

.video-planner {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.video-planner h1 {
  text-align: center;
  margin-bottom: 3rem;
  font-size: 2.5rem;
}

.step {
  margin-bottom: 2rem;
}

/* === INPUT STEP === */
.step-input textarea {
  width: 100%;
  padding: 1rem;
  font-size: 1rem;
  border: 2px solid #e5e7eb;
  border-radius: 8px;
  margin-bottom: 1rem;
  font-family: inherit;
}

/* === PLANS GRID === */
.plans-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1.5rem;
  margin-bottom: 2rem;
}

.plan-card {
  border: 2px solid #e5e7eb;
  border-radius: 12px;
  padding: 1.5rem;
  cursor: pointer;
  transition: all 0.3s ease;
  position: relative;
  background: #f9fafb;
}

.plan-card:hover {
  border-color: #3b82f6;
  box-shadow: 0 4px 12px rgba(59, 130, 246, 0.1);
  transform: translateY(-2px);
}

.plan-card.selected {
  border-color: #3b82f6;
  background: #eff6ff;
  box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
}

.plan-card.recommended {
  border-color: #8b5cf6;
  background: #faf5ff;
}

.plan-card .checkmark {
  position: absolute;
  top: 1rem;
  right: 1rem;
  font-size: 1.5rem;
  color: #3b82f6;
}

.plan-tier {
  margin: 0;
  font-size: 1.2rem;
  font-weight: 600;
}

.badge {
  display: inline-block;
  background: #8b5cf6;
  color: white;
  padding: 0.25rem 0.75rem;
  border-radius: 999px;
  font-size: 0.75rem;
  margin-top: 0.5rem;
}

.plan-cost {
  font-size: 2rem;
  font-weight: 700;
  color: #059669;
  margin: 1rem 0;
}

.plan-quality {
  font-size: 1.1rem;
  font-weight: 600;
  margin: 0.5rem 0;
}

.plan-resolution {
  color: #6b7280;
  margin: 0.5rem 0;
}

.plan-time {
  color: #6b7280;
  margin: 0.5rem 0;
}

.plan-reason {
  margin-top: 1rem;
  padding-top: 1rem;
  border-top: 1px solid #e5e7eb;
  color: #6b7280;
  font-style: italic;
}

/* === RESULT === */
.video-preview {
  margin-bottom: 2rem;
  border-radius: 8px;
  overflow: hidden;
  background: #000;
}

.cost-receipt {
  background: #f0fdf4;
  border: 2px solid #86efac;
  border-radius: 12px;
  padding: 2rem;
  margin-bottom: 2rem;
}

.receipt-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1rem;
  margin: 1rem 0;
}

.receipt-item {
  display: flex;
  flex-direction: column;
}

.receipt-item span {
  font-size: 0.875rem;
  color: #6b7280;
  margin-bottom: 0.25rem;
}

.receipt-item strong {
  font-size: 1.25rem;
  color: #059669;
}

.receipt-item.savings strong {
  color: #4ade80;
}

.receipt-message {
  text-align: center;
  font-size: 1.1rem;
  font-weight: 600;
  color: #059669;
  margin-top: 1rem;
}

/* === BUTTONS === */
.btn {
  padding: 0.75rem 1.5rem;
  font-size: 1rem;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 600;
  transition: all 0.2s ease;
  display: inline-block;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-primary {
  background: #3b82f6;
  color: white;
}

.btn-primary:hover:not(:disabled) {
  background: #2563eb;
  transform: scale(1.02);
}

.btn-secondary {
  background: #6b7280;
  color: white;
}

.btn-secondary:hover {
  background: #4b5563;
}

.btn-generate {
  width: 100%;
  padding: 1rem;
  font-size: 1.1rem;
  margin-top: 1rem;
}
```

**Tâches**:

- [ ] Créer dossier `dashboard/` avec React app
- [ ] Ajouter VideoPlanner.jsx et CSS
- [ ] Connecter aux endpoints API
- [ ] Tester avec 3 cas d'usage
- [ ] Valider UX mobile

**Critères de Succès**:

- UI clean, intuitive, pas de jargon technique
- Utilisateur comprend différences des 3 tiers en < 5 secondes
- Sélection facile (couleur/highlighting)
- Reçu final satisfaisant et clair

---

### 🟠 P1 - Semaine 2: Garantir Qualité Vidéo

#### 2.1 Tester Veo 3.0 + Implémenter Quality Validator (Effort: 5h)

**But**: Assurer que toutes vidéos respectent seuils de qualité selon tier.

```python
# src/agents/video_quality_validator.py

import subprocess
import json
from dataclasses import dataclass

@dataclass
class QualitySpec:
    """Seuils de qualité par tier"""
    tier: str
    min_width: int
    min_height: int
    min_bitrate_kbps: int
    expected_codec: str

QUALITY_SPECS = {
    "premium": QualitySpec(
        tier="premium",
        min_width=2560,
        min_height=1440,
        min_bitrate_kbps=8000,
        expected_codec="h264"
    ),
    "balanced": QualitySpec(
        tier="balanced",
        min_width=1920,
        min_height=1080,
        min_bitrate_kbps=3500,
        expected_codec="h264"
    ),
    "economy": QualitySpec(
        tier="economy",
        min_width=1280,
        min_height=720,
        min_bitrate_kbps=1500,
        expected_codec="h264"
    ),
}

class VideoQualityValidator:
    def validate(self, video_path: str, tier: str = "balanced") -> dict:
        """Valider vidéo contre spec du tier"""
        spec = QUALITY_SPECS[tier]

        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_format",
             "-show_streams", "-of", "json", video_path],
            capture_output=True,
            text=True,
            timeout=10
        )

        data = json.loads(result.stdout)
        stream = data["streams"][0]

        checks = {
            "resolution_ok": (stream["width"] >= spec.min_width
                             and stream["height"] >= spec.min_height),
            "bitrate_ok": int(stream.get("bit_rate", 0)) // 1000 >= spec.min_bitrate_kbps,
            "codec_ok": stream.get("codec_name") == spec.expected_codec,
        }

        passed = sum(checks.values())
        total = len(checks)

        return {
            "passed": passed == total,
            "passed_count": passed,
            "total_checks": total,
            "checks": checks,
            "metrics": {
                "resolution": f"{stream['width']}x{stream['height']}",
                "bitrate_kbps": int(stream.get("bit_rate", 0)) // 1000,
                "codec": stream.get("codec_name"),
                "duration_sec": float(data["format"].get("duration", 0))
            }
        }
```

**Tâches**:

- [ ] Tester Veo 3.0 (changer model dans generate_veo_video.py)
- [ ] Si 1080p native: mettre à jour comme default backend
- [ ] Sinon: implémenter Real-ESRGAN upscaling
- [ ] Intégrer VideoQualityValidator dans RenderExecutor
- [ ] Tester avec 5 vidéos réelles

**Critères de Succès**:

- Veo 3.0 retourne au moins 1080p
- Validator rejette vidéos < tier spec
- 100% des vidéos passent validation avant sortie
- Temps validation < 3s

---

#### 2.2 Construire Resolution Profile System (Effort: 3h)

**But**: Adapter paramètres selon contexte utilisation (social, web, broadcast).

(Code identique au plan original, réutiliser)

---

### 🟡 P2 - Semaine 3: Observabilité & Fidélisation

#### 3.1 Dashboard Temps Réel + Webhooks (Effort: 8h)

Voir plan original (adapté pour montrer cost intelligence metrics).

**Metrics clés**:

- Total généré ($ + units)
- Cost distribution par tier (pie chart)
- Runway vs Veo utilisation
- User satisfaction (cost accuracy)

---

## 📈 Roadmap Visuelle

```
SEMAINE 1           │ SEMAINE 2           │ SEMAINE 3        │ SEMAINE 4
────────────────────┼─────────────────────┼──────────────────┼────────────
P0.1: CostEstimator │ P1.1: Veo 3 + Quality│ P2.1: Dashboard  │ P3: Launch
P0.2: UI Planning   │ P1.2: Profiles      │ P2.2: Webhooks   │ Growth
                    │                      │                   │ experiments

🚀 MVP après P0 (plan + UI) ✓
🔥 Production après P1 (quality guarantees) ✓
⭐ Fully Observable après P2 ✓
```

---

## 💰 Positioning & Revenue Impact

### Before (Generic Generator):

```
Feature: "Better backends + 1080p"
Market: Commodity builders (Runway, Pika, RunwayML clones)
Margin: 25-35%
Defensibility: Low (copy in 2 weeks)
```

### After (Cost Intelligence Platform):

```
Feature: "Strategic video planning with cost forecasting"
Market: AI budget-conscious creators (indie creators, agencies)
Margin: 35-50% (premium tier upsell)
Defensibility: High (data moat: user preferences + accuracy feedback loop)

Revenue Tiers:
- Free tier: /video/plan + economy tier ($0.04/video to Veo)
- Pro: $20/mo = unlimited plans + priority queue + webhook
- Enterprise: Custom SLA + dedicated backends + cost guarantees

Example: 1000 users × $20/mo × 30% margin = $6k/mo profit
```

---

## 🎯 Critères de Succès Global

| Métrique          | P0 Done                 | P1 Done                         | P2 Done                      |
| ----------------- | ----------------------- | ------------------------------- | ---------------------------- |
| Cost transparency | ✅ User voit 3 options  | ✅ Reçu réel vs estimé          | ✅ Historique cost dashboard |
| Quality guarantee | ⚠️ Veo 3 tested         | ✅ All videos pass quality gate | ✅ SLA monitoring            |
| User retention    | 📊 TBM                  | ✅ Predict 40% improvement      | ✅ $20/mo cohort formed      |
| Differentiation   | ✅ Cost planning unique | ✅ Profiles differentiate       | ✅ Cost forecasting moat     |

---

## 🎬 Success Story (Expected)

```
WEEK 1:
User: "I'll generate a video"
Cost Estimator: "Here are 3 plans: $0.04, $0.08, $0.50"
User: "Wow, so transparent! I pick balanced."
Generates: CONFIDENT, knowing exact cost

WEEK 2:
Cost Receipt: "$0.076 actual vs $0.08 estimated - we saved you $0.004!"
User: "Wow, even cheaper than promised? Love this platform!"

WEEK 4:
Dashboard shows: "You've generated 45 videos for $3.62 total"
User: "With Runway it would've been $45. Moving to AIPROD!"

WEEK 8:
User upgrades to Pro ($20/mo): "Worth it for the planning tools"
```

---

## ✅ Checklist Implémentation

### P0 - Semaine 1

- [ ] `src/agents/cost_estimator.py` complet
- [ ] Endpoints `/video/plan` fonctionnel
- [ ] Endpoint `/video/generate` fonctionnel
- [ ] Dashboard VideoPlanner.jsx crée et connecté
- [ ] Test avec 5 prompts réels
- [ ] Commit: "🎯 Cost Intelligence Core - planning before generation"

### P1 - Semaine 2

- [ ] Veo 3.0 testé + résolution confirmée
- [ ] VideoQualityValidator implémenté
- [ ] Resolution profiles créés (SOCIAL/WEB/BROADCAST)
- [ ] Quality gates dans pipeline
- [ ] Test complet: 3 profiles × 3 tiers
- [ ] Commit: "✅ Quality guarantees per tier + resolution profiles"

### P2 - Semaine 3

- [ ] Cost metrics dashboard
- [ ] Webhook system
- [ ] Cost accuracy tracking
- [ ] User history DB
- [ ] Commit: "📊 Real-time observability + notifications"

### P3 - Semaine 4

- [ ] Launch announcement
- [ ] User feedback loop
- [ ] Free tier SLA docs
- [ ] Pro tier payment setup
- [ ] Commit: "🚀 AIPROD Cost Intelligence - MVP Launch"

---

## 📞 Questions Clés Avant Démarrage

1. **Runway API Access**: Tu peux bien appeler `/account` pour vérifier balance?
2. **Database**: Besoin d'importer user_preferences et generation_history quelque part?
3. **Payment**: Prêt pour intégrer Stripe pour le tier Pro?
4. **UI Hosting**: React app sur même serveur que FastAPI ou séparé?

---

**Vision Finale**:

AIPROD = "Le Turbo Tax de la vidéo AI" 🚀

_Non pas juste un générateur, mais un PLANIFICATEUR conscient des coûts._
