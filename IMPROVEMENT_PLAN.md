# 📋 Plan d'Amélioration Post-Production AIPROD

**Date**: Février 6, 2026  
**Contexte**: Après la génération réussie du premier promo vidéo avec coûts optimisés ($0.04), identification des gaps de qualité de production et plan de scalabilité.

---

## 🎯 Objectif Général

Transformer AIPROD d'une plateforme de proof-of-concept à **production-ready** en résolvant les lacunes de qualité (résolution vidéo) et en mettant en place une architecture observable et scalable pour la génération multi-backend.

---

## 📊 État Actuel vs Cible

| Aspect                 | Actuel                 | Cible 2026                                    | Gap        |
| ---------------------- | ---------------------- | --------------------------------------------- | ---------- |
| **Résolution Vidéo**   | 1280×720 (HD)          | 1920×1080+ (1080p min) / 3840×2160 (4K idéal) | 1 niveau   |
| **Coûts**              | $0.04/vidéo (optimisé) | Maintenir < $0.05                             | Aucun      |
| **Fiabilité**          | Single backend active  | Multi-backend + fallback                      | Nécessaire |
| **Validation Qualité** | Manuelle               | Automatique (quality gates)                   | Nécessaire |
| **Observabilité**      | Logs basiques          | Dashboard temps réel + métriques              | Nécessaire |
| **Adaptabilité**       | Génération fixe        | Profiles multi-contextes                      | Nécessaire |

---

## 🚀 Plan par Priorité

### 🔴 P0 - Critique (Semaine 1)

#### 1.1 Tester Veo 3.0 pour Native 1080p+ (Effort: 1h)

**Problème**: Veo 2 génère en 720p, résolution non paramétrable via Gemini API.

**Solution**:

```python
# scripts/generate_veo_video.py (ligne ~50)
# Changement:
# model_name = "veo-2.0-generate-001"
# À:
model_name = "veo-3.0-generate-001"  # ou veo-3.1-generate-001
```

**Tâches**:

- [ ] Modifier modèle dans `scripts/generate_veo_video.py`
- [ ] Exécuter avec même prompt
- [ ] Vérifier résolution via `ffprobe output.mp4`
- [ ] Si 1080p+: mettre à jour standard, sinon passer à 1.2

**Critètres de Succès**:

- Vidéo générée avec Veo 3.0
- Résolution confirmée 1080p ou supérieure
- Coût reste < $0.05

**En cas d'échec**: Passer directement à 1.2 (Upscaling)

---

#### 1.2 Implémenter Upscaling Real-ESRGAN (Effort: 4h)

**Problème**: Garantir 1080p même si Veo 3.0 échoue.

**Architecture**:

```
Veo 2/3 (720p ou 1080p)
    ↓
[Real-ESRGAN 4x upscale si < 1080p]
    ↓
Sortie garantie 1080p
    ↓
Métriques qualité (VMAF, SSIM)
```

**Implémentation**:

1. **Dépendance**:

```bash
pip install realesrgan
# Télécharge modèle: RealESRGAN_x4plus.pth (67 MB)
```

2. **Nouveau fichier**: `src/agents/video_upscaler.py`

```python
from realesrgan import RealESRGANer
import cv2

class VideoUpscaler:
    def __init__(self, scale=4):
        self.upsampler = RealESRGANer(
            scale=scale,
            model_name='RealESRGAN_x4plus',
            tile=400,  # Process par tiles pour économiser RAM
            tile_pad=10,
            pre_pad=0,
            half=True  # FP16 pour GPU
        )

    def upscale_video(self, input_path: str, output_path: str) -> dict:
        """Upscale 720p → 1440p (2.25x amélioration)"""
        # Lire vidéo
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Upscale frame par frame
        frames_upscaled = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            output, _ = self.upsampler.enhance(frame, outscale=2)
            frames_upscaled.append(output)

        # Encoder final avec ffmpeg
        # Retourner métadonnées
        return {
            "original_res": (1280, 720),
            "upscaled_res": (2560, 1440),
            "frames": len(frames_upscaled),
            "fps": fps
        }
```

3. **Intégration dans RenderExecutor**:

```python
# src/agents/render_executor.py
def run(self, ...):
    # ... génération existante ...
    video_path = self._generate_with_backend()

    # Vérifier résolution
    resolution = self._get_video_resolution(video_path)
    if resolution[0] < 1920:  # < 1080p
        upscaler = VideoUpscaler(scale=2)  # 2x pour 720→1440
        metadata = upscaler.upscale_video(video_path, output_path)
        logger.info(f"Upscaled to {metadata['upscaled_res']}")

    return video_path
```

**Tâches**:

- [ ] Créer `src/agents/video_upscaler.py`
- [ ] Ajouter dépendance à `requirements.txt`
- [ ] Intégrer dans `RenderExecutor.run()`
- [ ] Tester avec vidéo existante: avant/après comparaison
- [ ] Mesurer temps d'exécution (target: < 10s pour 5s vidéo)
- [ ] Mesurer coût cloud (GPU inference): target < $0.001

**Critètres de Succès**:

- Vidéo 720p → 1440p confirmée via ffprobe
- Qualité visuelle acceptable (pas de mode "blurry")
- Temps: < 15s pour 5s vidéo
- Coût ajout: < $0.001/vidéo

**Dépendance**: Après P0.1 (pour savoir si vraiment nécessaire)

---

### 🟠 P1 - Haute Priorité (Semaine 2)

#### 2.1 Implémenter Quality Validation Gate (Effort: 2h)

**Problème**: Vidéos dégradées peuvent atteindre les utilisateurs.

**Solution**: Automatiser les contrôles qualité avec `ffprobe`:

```python
# src/agents/video_quality_validator.py

from dataclasses import dataclass
import subprocess
import json

@dataclass
class QualitySpec:
    min_width: int = 1920
    min_height: int = 1080
    min_bitrate_kbps: int = 2500
    expected_codec: str = "h264"
    expected_fps: int = 24

class VideoQualityValidator:
    def __init__(self, spec: QualitySpec = None):
        self.spec = spec or QualitySpec()
        self.metrics = {}

    def validate(self, video_path: str) -> tuple[bool, dict]:
        """
        Retourne: (is_valid: bool, metrics: dict)
        """
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_format",
                 "-show_streams", "-of", "json", video_path],
                capture_output=True,
                text=True,
                timeout=10
            )

            data = json.loads(result.stdout)
            stream = data["streams"][0]

            self.metrics = {
                "width": stream.get("width"),
                "height": stream.get("height"),
                "bitrate_kbps": int(stream.get("bit_rate", 0)) // 1000,
                "codec": stream.get("codec_name"),
                "fps": eval(stream.get("r_frame_rate", "0/1")),
                "duration_sec": float(data["format"].get("duration", 0))
            }

            # Validation
            checks = {
                "resolution": self.metrics["width"] >= self.spec.min_width
                           and self.metrics["height"] >= self.spec.min_height,
                "bitrate": self.metrics["bitrate_kbps"] >= self.spec.min_bitrate_kbps,
                "codec": self.metrics["codec"] == self.spec.expected_codec,
                "fps": abs(self.metrics["fps"] - self.spec.expected_fps) < 0.5
            }

            is_valid = all(checks.values())

            return is_valid, {
                "metrics": self.metrics,
                "checks": checks,
                "passed": sum(checks.values()),
                "total": len(checks)
            }

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False, {"error": str(e)}

    def get_report(self) -> str:
        """Rapport lisible"""
        return f"""
VIDEO QUALITY REPORT
====================
Resolution:  {self.metrics['width']}×{self.metrics['height']} ✓
Bitrate:     {self.metrics['bitrate_kbps']} kbps ✓
Codec:       {self.metrics['codec']} ✓
FPS:         {self.metrics['fps']:.1f} ✓
Duration:    {self.metrics['duration_sec']:.1f}s
        """
```

**Intégration**:

```python
# src/agents/render_executor.py

def run(self, ...):
    video_path = self._generate_with_backend()

    # Valider qualité
    validator = VideoQualityValidator()
    is_valid, results = validator.validate(video_path)

    if not is_valid:
        failed_checks = [k for k, v in results["checks"].items() if not v]
        logger.error(f"Quality gate failed: {failed_checks}")

        # Retry logic avec fallback
        self._retry_with_fallback(reason="quality_gate_failed")
        return self.run(...)  # Récursion

    logger.info(validator.get_report())
    return video_path
```

**Tâches**:

- [ ] Créer `src/agents/video_quality_validator.py`
- [ ] Implémenter parse ffprobe + checks
- [ ] Ajouter appel dans `RenderExecutor.run()`
- [ ] Tester avec vidéo défaillante (brute résolution)
- [ ] Documenter seuils acceptables
- [ ] Ajouter retry logic avec fallback backend

**Critères de Succès**:

- Détecte < 1080p et rejette
- Détecte codec incorrect et rejette
- 100% des vidéos passent validation avant sortie
- Temps validation: < 2s par vidéo

---

#### 2.2 Construire Resolution Profile System (Effort: 3h)

**Problème**: Générer toujours en 4K c'est cher et inutile pour social media.

**Solution**: Adapter paramètres selon contexte d'usage.

```python
# src/agents/resolution_profiles.py

from enum import Enum
from dataclasses import dataclass

class ResolutionProfile(Enum):
    """Profils d'utilisation avec specs associées"""

    SOCIAL = "social"          # TikTok, Instagram, YouTube Shorts
    WEB = "web"                # Sites web, blogs, newsletters
    BROADCAST = "broadcast"    # Télévision, cinéma, archivage

@dataclass
class ProfileSpec:
    profile: ResolutionProfile
    min_width: int
    min_height: int
    preferred_fps: int
    target_backend: str
    estimated_cost_usd: float
    use_cases: list

PROFILES = {
    ResolutionProfile.SOCIAL: ProfileSpec(
        profile=ResolutionProfile.SOCIAL,
        min_width=720,
        min_height=720,
        preferred_fps=24,
        target_backend="veo-2.0",  # Assez bon pour petits écrans
        estimated_cost_usd=0.04,
        use_cases=["TikTok", "Instagram Reels", "YouTube Shorts", "Twitter"]
    ),

    ResolutionProfile.WEB: ProfileSpec(
        profile=ResolutionProfile.WEB,
        min_width=1920,
        min_height=1080,
        preferred_fps=24,
        target_backend="veo-3.0",  # 1080p native + upscale
        estimated_cost_usd=0.06,
        use_cases=["Website hero", "Blog", "Portfolio", "Demo"]
    ),

    ResolutionProfile.BROADCAST: ProfileSpec(
        profile=ResolutionProfile.BROADCAST,
        min_width=3840,
        min_height=2160,
        preferred_fps=30,  # 4K @ 30fps standard
        target_backend="veo-3.1",  # Meilleur modèle + 4K upscale
        estimated_cost_usd=0.15,
        use_cases=["TV", "Cinema", "Archive", "Premium print"]
    ),
}

class ResolutionProfileSelector:
    @staticmethod
    def select(use_case: str) -> ResolutionProfile:
        """Auto-detect profile depuis description"""
        use_case_lower = use_case.lower()

        if any(x in use_case_lower for x in ["tiktok", "shorts", "instagram", "reels"]):
            return ResolutionProfile.SOCIAL
        elif any(x in use_case_lower for x in ["website", "web", "blog", "demo"]):
            return ResolutionProfile.WEB
        elif any(x in use_case_lower for x in ["broadcast", "tv", "cinema", "4k", "archive"]):
            return ResolutionProfile.BROADCAST

        return ResolutionProfile.WEB  # Default

    @staticmethod
    def get_spec(profile: ResolutionProfile) -> ProfileSpec:
        return PROFILES[profile]
```

**Intégration API**:

```python
# src/api/main.py

@app.post("/video/generate")
async def generate_video(
    prompt: str,
    profile: str = "web"  # "social", "web", "broadcast"
):
    profile_enum = ResolutionProfile[profile.upper()]
    spec = ResolutionProfileSelector.get_spec(profile_enum)

    executor = RenderExecutor(
        backend=spec.target_backend,
        target_resolution=(spec.min_width, spec.min_height),
        quality_spec=QualitySpec(
            min_width=spec.min_width,
            min_height=spec.min_height
        )
    )

    return {
        "profile": profile,
        "estimated_cost": spec.estimated_cost_usd,
        "video_path": executor.run(prompt)
    }
```

**Tâches**:

- [ ] Créer `src/agents/resolution_profiles.py`
- [ ] Définir 3 profiles avec specs
- [ ] Implémenter selector
- [ ] Ajouter param "profile" à endpoint API
- [ ] Documenter dans README
- [ ] Tester chaque profile

**Critères de Succès**:

- SOCIAL: 720p, $0.04, validation < 2s
- WEB: 1080p, $0.06, validation < 3s
- BROADCAST: 4K, $0.15, validation < 5s (upscale long)
- Auto-selection fonctionne pour 90% des cas

---

### 🟡 P2 - Moyenne Priorité (Semaine 3-4)

#### 3.1 Real-Time Monitoring Dashboard (Effort: 8h)

**Problème**: Pas de visibilité sur santé système, files d'attente, coûts temps réel.

**Architecture**:

```
FastAPI Metrics Endpoint
        ↓
WebSocket (Live updates)
        ↓
React Dashboard (Frontend)
        ↓
Display: Queue / Costs / Backend Health / Errors
```

**Backend Metrics**:

```python
# src/api/metrics.py

from dataclasses import dataclass, field
from datetime import datetime
import asyncio

@dataclass
class GenerationMetrics:
    total_generated: int = 0
    total_cost_usd: float = 0.0
    avg_generation_time_sec: float = 0.0
    success_rate: float = 1.0
    queue_length: int = 0
    active_backends: dict = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)

    errors_last_hour: int = 0
    most_used_profile: str = "web"

class MetricsCollector:
    def __init__(self):
        self.metrics = GenerationMetrics()
        self.generation_times = []
        self.generation_queue = asyncio.Queue()

    async def track_generation(self, prompt: str, profile: str):
        """WebSocket pour tracking temps réel"""
        start = time.time()
        await self.generation_queue.put({
            "status": "started",
            "profile": profile,
            "timestamp": start
        })

        try:
            # Exécuter génération
            result = await generate_video_async(prompt, profile)
            elapsed = time.time() - start

            self.metrics.total_generated += 1
            self.metrics.total_cost_usd += result['cost']
            self.generation_times.append(elapsed)
            self.metrics.avg_generation_time_sec = sum(self.generation_times) / len(self.generation_times)

            await self.generation_queue.put({
                "status": "completed",
                "elapsed_sec": elapsed,
                "cost": result['cost']
            })

        except Exception as e:
            self.metrics.errors_last_hour += 1
            await self.generation_queue.put({
                "status": "failed",
                "error": str(e)
            })

# Endpoint
@app.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket):
    await websocket.accept()

    while True:
        # Envoyer métriques chaque 2s
        await websocket.send_json({
            "total_generated": metrics.metrics.total_generated,
            "total_cost": metrics.metrics.total_cost_usd,
            "queue": metrics.generation_queue.qsize(),
            "avg_time_sec": metrics.metrics.avg_generation_time_sec,
            "timestamp": datetime.now().isoformat()
        })
        await asyncio.sleep(2)
```

**Frontend Dashboard** (React):

```jsx
// dashboard/src/MetricsDash.jsx
import { useEffect, useState } from "react";

export function MetricsDashboard() {
  const [metrics, setMetrics] = useState({});

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws/metrics");
    ws.onmessage = (e) => setMetrics(JSON.parse(e.data));
    return () => ws.close();
  }, []);

  return (
    <div className="dashboard">
      <h1>AIPROD Live Metrics</h1>
      <div className="grid">
        <Card
          title="Total Generated"
          value={metrics.total_generated}
          icon="🎬"
        />
        <Card
          title="Total Cost"
          value={`$${metrics.total_cost?.toFixed(2)}`}
          icon="💰"
        />
        <Card title="Queue Size" value={metrics.queue} icon="⏳" />
        <Card
          title="Avg Time"
          value={`${metrics.avg_time_sec?.toFixed(1)}s`}
          icon="⏱️"
        />
      </div>
    </div>
  );
}
```

**Tâches**:

- [ ] Créer `src/api/metrics.py` avec collecteur
- [ ] Ajouter WebSocket endpoint `/ws/metrics`
- [ ] Créer dossier `dashboard/` avec React app
- [ ] Implémenter composants: Cards, Charts, Queue monitor
- [ ] Ajouter authentification (JWT)
- [ ] Déployer sur port 3000
- [ ] Tester avec génération réelle

**Critères de Succès**:

- Dashboard affiche 5+ métriques temps réel
- Latence < 100ms entre génération et display
- Authentification fonctionne
- Charts historiques des 24h dernières heures
- Mobile responsive

---

#### 3.2 Système de Notification Webhook (Effort: 4h)

**Problème**: Utilisateurs ne savent pas quand vidéos sont prêtes.

```python
# src/agents/webhook_manager.py

class WebhookManager:
    async def notify_completion(self, job_id: str, video_url: str, cost: float):
        """Appeller webhook utilisateur quand vidéo ready"""
        user = await db.get_user_by_job(job_id)

        if user.webhook_url:
            payload = {
                "event": "video_completed",
                "job_id": job_id,
                "video_url": video_url,
                "cost_usd": cost,
                "timestamp": datetime.now().isoformat()
            }

            async with httpx.AsyncClient() as client:
                await client.post(
                    user.webhook_url,
                    json=payload,
                    timeout=10,
                    headers={"X-AIPROD-Signature": self._sign_payload(payload)}
                )

    def _sign_payload(self, payload: dict) -> str:
        """HMAC pour sécurité"""
        import hmac
        import hashlib
        return hmac.new(
            WEBHOOK_SECRET.encode(),
            json.dumps(payload).encode(),
            hashlib.sha256
        ).hexdigest()
```

**Tâches**:

- [ ] Implémenter `WebhookManager`
- [ ] Ajouter signature HMAC
- [ ] Retry logic (3x avec exponential backoff)
- [ ] Documenter payload format
- [ ] Tester avec webhook.site

---

### 🔵 P3 - Nice-to-Have (Semaine 5+)

#### 4.1 Cost Prediction API (Effort: 6h)

Prédire coûts avant génération basé sur prompt, modèle, résolution.

#### 4.2 Batch Generation (Effort: 8h)

Génération parallèle de multiples vidéos avec mise en queue.

#### 4.3 A/B Testing Framework (Effort: 10h)

Tester même prompt sur différents modèles/paramètres, comparer résultats.

#### 4.4 Video Caching & Deduplication (Effort: 6h)

Détecter prompts semblables, servir depuis cache si hit > 90%.

---

## 📈 Roadmap Visuelle

```
SEMAINE 1    │ SEMAINE 2     │ SEMAINE 3-4       │ SEMAINE 5+
─────────────┼───────────────┼──────────────────┼────────────
P0.1: Veo 3  │ P1.1: Quality │ P2.1: Dashboard  │ P3.1: Cost Pred
P0.2: Upscal │ P1.2: Profiles│ P2.2: Webhooks   │ P3.2: Batch
             │               │ P2.3: Monitoring │ P3.3: A/B Test
             │               │                  │ P3.4: Caching

🚀 Production Ready après P1 ✓
🔥 Fully Observable après P2 ✓
⭐ Feature Complete après P3 ✓
```

---

## 💰 Coûts Estimés (Impact Financier)

| Tâche               | Coûts Dev          | Coûts Runtime          | ROI                            |
| ------------------- | ------------------ | ---------------------- | ------------------------------ |
| P0.1 (Veo 3 test)   | Gratuit (1h temps) | $0.00 (test)           | Fort si 1080p native           |
| P0.2 (Upscaling)    | Gratuit (4h temps) | +$0.001/vidéo          | Bon (qualité garantie)         |
| P1.1 (Quality Gate) | Gratuit (2h temps) | $0.00 (ffprobe)        | Excellent (prévient débâcles)  |
| P1.2 (Profiles)     | Gratuit (3h temps) | $0.00 (logique)        | Excellent (économies sociales) |
| P2.1 (Dashboard)    | Gratuit (8h temps) | GCP Compute $50-100/mo | Bon (observabilité)            |
| P2.2 (Webhooks)     | Gratuit (4h temps) | $0.00                  | Excellent (UX)                 |

**Total Time Investment**: ~35 heures  
**ROI**: Production-ready platform + $100/mo observabilité = bien investi

---

## ✅ Checklist d'Implémentation

### P0 - Semaine 1

- [ ] Tester Veo 3.0 (1h)
- [ ] Si nécessaire, implémenter Real-ESRGAN (4h)
- [ ] Valider résolution pipeline
- [ ] Commit: "Résolution 1080p+ garantie (Veo 3 ou upscaling)"

### P1 - Semaine 2

- [ ] Video Quality Validator (2h)
- [ ] Resolution Profiles (3h)
- [ ] Intégrer dans RenderExecutor
- [ ] Tester 3 profiles end-to-end
- [ ] Update README avec profiles
- [ ] Commit: "Quality gates + resolution profiles"

### P2 - Semaine 3-4

- [ ] Dashboard backend WebSocket (6h)
- [ ] Dashboard frontend React (6h)
- [ ] Webhook system (4h)
- [ ] Test bout-à-bout
- [ ] Documenter API
- [ ] Commit: "Real-time monitoring + webhooks"

### P3 - Selon capacité

- [ ] Cost prediction
- [ ] Batch generation
- [ ] A/B testing
- [ ] Caching

---

## 🎯 Critères de Succès Global

|               | Baseline | Target P0         | Target P1          | Target P2               |
| ------------- | -------- | ----------------- | ------------------ | ----------------------- |
| Résolution    | 720p     | 1080p             | 1080p              | 1080p+                  |
| Coûts         | $0.04    | < $0.06           | < $0.06            | < $0.06                 |
| Fiabilité     | 95%      | 98%               | 99%+               | 99.5%                   |
| Observabilité | Logs     | Logs              | Logs + Metrics     | Dashboard temps réel    |
| Adaptabilité  | Single   | Single + Fallback | Profiles multiples | Intelligent auto-select |

**Objectif Principal**: ✅ Être prêt pour production par fin Semaine 2

---

## 📞 Contact & Questions

- Modifications du plan? → Créer issue GitHub
- Blocages techniques? → Session debug interactif
- Priorités changent? → Réajuster roadmap

**Plan créé**: 6 Février 2026  
**Prochaine revue**: 13 Février 2026 (après P0)
