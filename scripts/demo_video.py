#!/usr/bin/env python3
"""
AIPROD V33 - Demo Video Script

Script pour générer une vidéo de démonstration du système.
Montre le workflow complet: brief → estimation → ICC → rendu → QA.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiohttp

# Configuration
API_BASE = "http://localhost:8000"  # ou URL Cloud Run
DEMO_BRIEFS = [
    {
        "name": "Social Media Quick",
        "preset": "quick_social",
        "brief": "Annonce flash d'une nouvelle app mobile, style jeune et dynamique",
        "style": "modern energetic",
        "duration": 15
    },
    {
        "name": "Brand Campaign",
        "preset": "brand_campaign",
        "brief": "Lancement d'une montre connectée premium, style Apple avec transitions fluides",
        "style": "cinematic minimal",
        "duration": 45
    },
    {
        "name": "Premium Spot",
        "preset": "premium_spot",
        "brief": "Film publicitaire pour parfum de luxe, ambiance mystérieuse et sensuelle",
        "style": "luxe cinematic",
        "duration": 60
    }
]

class DemoRunner:
    """Exécute la démonstration AIPROD V33."""
    
    def __init__(self, api_base: str = API_BASE):
        self.api_base = api_base
        self.session: "aiohttp.ClientSession | None" = None
        self.results: list[dict] = []
        
    async def setup(self):
        """Initialise la session HTTP."""
        import aiohttp
        self.session = aiohttp.ClientSession()
        
    async def teardown(self):
        """Ferme la session."""
        if self.session:
            await self.session.close()
            
    async def check_health(self) -> bool:
        """Vérifie que l'API est accessible."""
        print("\n" + "="*60)
        print("🏥 Vérification de l'API AIPROD V33...")
        print("="*60)
        
        if self.session is None:
            return False
        
        try:
            async with self.session.get(f"{self.api_base}/health") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"✅ API Status: {data.get('status', 'unknown')}")
                    print(f"   Version: {data.get('version', 'unknown')}")
                    return True
        except Exception as e:
            print(f"❌ API indisponible: {e}")
            return False
        return False
        
    async def show_presets(self):
        """Affiche les presets disponibles."""
        print("\n" + "="*60)
        print("📋 PRESETS DISPONIBLES")
        print("="*60)
        
        if self.session is None:
            return
        
        try:
            async with self.session.get(f"{self.api_base}/presets") as resp:
                if resp.status == 200:
                    presets = await resp.json()
                    for name, config in presets.items():
                        print(f"\n🎬 {name.upper()}")
                        print(f"   Description: {config.get('description', 'N/A')}")
                        print(f"   Duration: {config.get('max_duration', 0)}s max")
                        print(f"   Quality Target: {config.get('quality_target', 0)}")
                        print(f"   Price: ~${config.get('base_price', 0):.2f}")
        except Exception as e:
            print(f"❌ Erreur: {e}")
            
    async def estimate_cost(self, brief_config: dict) -> dict:
        """Estime le coût d'une production."""
        print(f"\n💰 Estimation coût: {brief_config['name']}...")
        
        if self.session is None:
            return {}
        
        payload = {
            "preset": brief_config["preset"],
            "duration_seconds": brief_config["duration"],
            "style": brief_config["style"]
        }
        
        try:
            async with self.session.post(
                f"{self.api_base}/cost-estimate",
                json=payload
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"   💵 AIPROD: ${data.get('aiprod_cost', 0):.2f}")
                    print(f"   📊 Runway direct: ${data.get('runway_cost', 0):.2f}")
                    print(f"   💰 Économie: {data.get('savings_percent', 0):.0f}%")
                    return data
        except Exception as e:
            print(f"❌ Erreur estimation: {e}")
        return {}
        
    async def create_job(self, brief_config: dict) -> str:
        """Crée un job de production."""
        print(f"\n🎬 Création job: {brief_config['name']}...")
        
        if self.session is None:
            return ""
        
        payload = {
            "preset": brief_config["preset"],
            "brief": brief_config["brief"],
            "style": brief_config["style"],
            "duration_seconds": brief_config["duration"]
        }
        
        try:
            async with self.session.post(
                f"{self.api_base}/job/create",
                json=payload
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    job_id = data.get("job_id", "unknown")
                    print(f"   ✅ Job créé: {job_id}")
                    print(f"   📋 État: {data.get('state', 'unknown')}")
                    return job_id
        except Exception as e:
            print(f"❌ Erreur création: {e}")
        return ""
        
    async def get_manifest(self, job_id: str) -> dict:
        """Récupère le manifest créatif."""
        print(f"\n📄 Récupération manifest pour {job_id}...")
        
        if self.session is None:
            return {}
        
        try:
            async with self.session.get(
                f"{self.api_base}/job/{job_id}/manifest"
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    manifest = data.get("manifest", {})
                    print(f"   📝 Titre: {manifest.get('title', 'N/A')}")
                    print(f"   🎬 Shots: {len(manifest.get('shots', []))}")
                    print(f"   ⏱️ Durée: {manifest.get('duration', 0)}s")
                    return manifest
        except Exception as e:
            print(f"❌ Erreur manifest: {e}")
        return {}
        
    async def update_manifest(self, job_id: str, updates: dict) -> bool:
        """Met à jour le manifest (ICC)."""
        print(f"\n✏️ Mise à jour manifest pour {job_id}...")
        
        if self.session is None:
            return False
        
        try:
            async with self.session.patch(
                f"{self.api_base}/job/{job_id}/manifest",
                json={"updates": updates}
            ) as resp:
                if resp.status == 200:
                    print("   ✅ Manifest mis à jour")
                    return True
        except Exception as e:
            print(f"❌ Erreur update: {e}")
        return False
        
    async def approve_job(self, job_id: str) -> bool:
        """Approuve le job pour rendu."""
        print(f"\n👍 Approbation job {job_id}...")
        
        if self.session is None:
            return False
        
        try:
            async with self.session.post(
                f"{self.api_base}/job/{job_id}/approve"
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"   ✅ Job approuvé")
                    print(f"   📋 Nouvel état: {data.get('state', 'unknown')}")
                    return True
        except Exception as e:
            print(f"❌ Erreur approbation: {e}")
        return False
        
    async def get_qa_report(self, job_id: str) -> dict:
        """Récupère le rapport QA."""
        print(f"\n📊 Rapport QA pour {job_id}...")
        
        if self.session is None:
            return {}
        
        try:
            async with self.session.get(
                f"{self.api_base}/job/{job_id}/qa"
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    qa = data.get("qa_report", {})
                    print(f"   📈 Score global: {qa.get('overall_score', 0):.2f}")
                    print(f"   ✅ Checks passés: {qa.get('passed_checks', 0)}")
                    print(f"   ⚠️ Warnings: {qa.get('warnings', 0)}")
                    return qa
        except Exception as e:
            print(f"❌ Erreur QA: {e}")
        return {}
        
    async def run_demo(self, brief_index: int = 1):
        """
        Exécute la démo complète pour un brief.
        
        Args:
            brief_index: Index du brief (0=quick, 1=brand, 2=premium)
        """
        brief_config = DEMO_BRIEFS[brief_index]
        
        print("\n" + "="*60)
        print(f"🎬 DÉMONSTRATION: {brief_config['name']}")
        print("="*60)
        print(f"Brief: {brief_config['brief']}")
        print(f"Style: {brief_config['style']}")
        print(f"Durée: {brief_config['duration']}s")
        print(f"Preset: {brief_config['preset']}")
        
        # Étape 1: Estimation coût
        cost_data = await self.estimate_cost(brief_config)
        await asyncio.sleep(1)
        
        # Étape 2: Création job
        job_id = await self.create_job(brief_config)
        if not job_id:
            print("❌ Échec création job")
            return
        await asyncio.sleep(1)
        
        # Étape 3: Récupération manifest (ICC)
        manifest = await self.get_manifest(job_id)
        await asyncio.sleep(1)
        
        # Étape 4: Modification manifest (démo ICC)
        if brief_config["preset"] in ["brand_campaign", "premium_spot"]:
            print("\n🎨 INTERACTIVE CREATIVE CONTROL")
            print("   Simulation d'édition du manifest...")
            updates = {
                "color_grade": "cinematic_warm",
                "music_style": "electronic_ambient"
            }
            await self.update_manifest(job_id, updates)
            await asyncio.sleep(1)
        
        # Étape 5: Approbation
        await self.approve_job(job_id)
        await asyncio.sleep(1)
        
        # Étape 6: Rapport QA (simulé)
        qa_report = await self.get_qa_report(job_id)
        
        # Résumé
        self.results.append({
            "name": brief_config["name"],
            "job_id": job_id,
            "cost_estimate": cost_data,
            "qa_report": qa_report,
            "timestamp": datetime.now().isoformat()
        })
        
        print("\n" + "="*60)
        print(f"✅ DÉMO TERMINÉE: {brief_config['name']}")
        print("="*60)
        
    async def run_full_demo(self):
        """Exécute la démo pour tous les presets."""
        start_time = time.time()
        
        print("\n" + "="*60)
        print("🚀 AIPROD V33 - DÉMONSTRATION COMPLÈTE")
        print("="*60)
        print(f"Démarrage: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        await self.setup()
        
        # Health check
        if not await self.check_health():
            print("\n❌ API non disponible - Exécution en mode simulation")
            await self.simulate_demo()
            return
            
        # Affichage presets
        await self.show_presets()
        
        # Exécution des 3 démos
        for i, brief in enumerate(DEMO_BRIEFS):
            await self.run_demo(i)
            if i < len(DEMO_BRIEFS) - 1:
                print("\n⏳ Pause avant prochain brief...\n")
                await asyncio.sleep(2)
        
        # Résumé final
        await self.print_summary()
        
        await self.teardown()
        
        elapsed = time.time() - start_time
        print(f"\n⏱️ Durée totale démo: {elapsed:.1f}s")
        
    async def simulate_demo(self):
        """Mode simulation quand l'API n'est pas disponible."""
        print("\n🔮 MODE SIMULATION")
        print("="*60)
        
        for brief in DEMO_BRIEFS:
            print(f"\n🎬 {brief['name']}")
            print(f"   Brief: {brief['brief']}")
            print(f"   Coût estimé: ~${brief['duration'] * 0.02:.2f}")
            print(f"   Durée rendu: ~{brief['duration'] * 2}s")
            print("   ✅ Simulation terminée")
            await asyncio.sleep(0.5)
            
    async def print_summary(self):
        """Affiche le résumé de la démo."""
        print("\n" + "="*60)
        print("📊 RÉSUMÉ DE LA DÉMONSTRATION")
        print("="*60)
        
        total_cost = 0
        for result in self.results:
            print(f"\n🎬 {result['name']}")
            print(f"   Job ID: {result['job_id']}")
            cost = result.get('cost_estimate', {}).get('aiprod_cost', 0)
            total_cost += cost
            print(f"   Coût: ${cost:.2f}")
            qa_score = result.get('qa_report', {}).get('overall_score', 0)
            print(f"   QA Score: {qa_score:.2f}")
            
        print(f"\n💰 COÛT TOTAL ESTIMÉ: ${total_cost:.2f}")
        print(f"📈 ÉCONOMIE vs CONCURRENTS: ~{total_cost * 0.8:.2f}$")


async def main():
    """Point d'entrée principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AIPROD V33 Demo")
    parser.add_argument(
        "--preset",
        choices=["quick", "brand", "premium", "all"],
        default="all",
        help="Preset à démontrer"
    )
    parser.add_argument(
        "--api",
        default=API_BASE,
        help="URL de l'API"
    )
    
    args = parser.parse_args()
    
    demo = DemoRunner(api_base=args.api)
    
    if args.preset == "all":
        await demo.run_full_demo()
    else:
        preset_map = {"quick": 0, "brand": 1, "premium": 2}
        await demo.setup()
        if await demo.check_health():
            await demo.run_demo(preset_map[args.preset])
        else:
            await demo.simulate_demo()
        await demo.teardown()


if __name__ == "__main__":
    asyncio.run(main())
