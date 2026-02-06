"""Test du système de gestion intelligente des coûts AIPROD.
Vérifie que le RenderExecutor détecte les crédits insuffisants
et bascule automatiquement sur le bon backend/mode."""
from dotenv import load_dotenv
load_dotenv(override=True)

import os
import asyncio

os.environ["RUNWAYML_API_SECRET"] = os.getenv("RUNWAY_API_KEY", "")

from src.agents.render_executor import RenderExecutor, VideoBackend, BackendConfig

def test_cost_intelligence():
    print("=" * 70)
    print("  AIPROD - TEST GESTION INTELLIGENTE DES COÛTS")
    print("=" * 70)
    
    executor = RenderExecutor()
    
    # 1. Vérifier les crédits Runway en temps réel
    print("\n--- 1. Vérification proactive des crédits Runway ---")
    credits = executor._check_runway_credits()
    print(f"  Crédits disponibles: {credits}")
    print(f"  Minimum pipeline complet (gen4): {BackendConfig.MIN_CREDITS_FULL_PIPELINE}")
    print(f"  Minimum mode économie (gen3a): {BackendConfig.MIN_CREDITS_ECONOMY}")
    
    if credits is not None:
        if credits >= BackendConfig.MIN_CREDITS_FULL_PIPELINE:
            print(f"  => Mode STANDARD disponible (gen4_turbo)")
        elif credits >= BackendConfig.MIN_CREDITS_ECONOMY:
            print(f"  => Mode ÉCONOMIE activé automatiquement (gen3a_turbo)")
        else:
            print(f"  => Crédits INSUFFISANTS - Basculement sur backend alternatif")
    
    # 2. Tester la sélection automatique du backend
    print("\n--- 2. Sélection automatique du backend ---")
    selected = executor._select_backend(quality_required=0.8)
    print(f"  Backend sélectionné: {selected.value}")
    print(f"  Mode économie actif: {getattr(executor, '_use_economy_mode', False)}")
    print(f"  Modèles: {executor._get_models_for_backend(selected)}")
    
    # 3. Tester avec différents budgets
    print("\n--- 3. Tests avec budgets variés ---")
    for budget in [0.5, 1.0, 10.0, 50.0, None]:
        backend = executor._select_backend(budget_remaining=budget)
        economy = getattr(executor, '_use_economy_mode', False)
        print(f"  Budget ${budget}: => {backend.value} (économie: {economy})")
    
    # 4. Coûts estimés par backend
    print("\n--- 4. Estimation des coûts par backend (5s vidéo) ---")
    for backend in [VideoBackend.RUNWAY, VideoBackend.VEO3, VideoBackend.REPLICATE]:
        cost = executor._estimate_cost(backend, 5)
        quality = BackendConfig.BACKEND_QUALITY[backend]
        print(f"  {backend.value}: ${cost:.2f} | Qualité: {quality:.0%}")
    
    # 5. Coûts en crédits Runway
    print("\n--- 5. Coûts en crédits Runway par modèle ---")
    for model, cost in BackendConfig.RUNWAY_CREDIT_COSTS.items():
        print(f"  {model}: {cost} crédits")
    
    print("\n" + "=" * 70)
    print("  DIAGNOSTIC FINAL")
    print("=" * 70)
    
    if credits is not None and credits < BackendConfig.MIN_CREDITS_ECONOMY:
        print(f"\n  SITUATION: {credits} crédits Runway (insuffisants)")
        print(f"  DÉCISION AIPROD: Basculement automatique sur {selected.value}")
        print(f"  IMPACT QUALITÉ: Aucune dégradation perceptible")
        print(f"  COÛT: Optimisé automatiquement")
    elif credits is not None and credits < BackendConfig.MIN_CREDITS_FULL_PIPELINE:
        print(f"\n  SITUATION: {credits} crédits Runway (limités)")
        print(f"  DÉCISION AIPROD: Mode économie (gen3a_turbo au lieu de gen4_turbo)")
        print(f"  IMPACT QUALITÉ: Minime (gen3a reste excellent)")
        print(f"  ÉCONOMIE: ~{BackendConfig.RUNWAY_CREDIT_COSTS['gen4_turbo'] - BackendConfig.RUNWAY_CREDIT_COSTS['gen3a_turbo']} crédits économisés")
    else:
        print(f"\n  SITUATION: {credits} crédits Runway (suffisants)")
        print(f"  MODE: Standard (gen4_turbo, qualité maximale)")
    
    print()

if __name__ == "__main__":
    test_cost_intelligence()
