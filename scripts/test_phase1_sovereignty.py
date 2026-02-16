"""Tests de validation de la Phase 1 — Souveraineté."""

print("=== TEST SOUVERAINETE Phase 1 ===")
print()

# Test 1: Import aiprod_core sans google
print("[1] Import aiprod_core...")
try:
    import aiprod_core
    print("   ✅ aiprod_core importé sans erreur")
except Exception as e:
    print(f"   ❌ Erreur: {e}")

# Test 2: Import aiprod_pipelines sans google
print("[2] Import aiprod_pipelines...")
try:
    import aiprod_pipelines
    print("   ✅ aiprod_pipelines importé sans erreur")
except Exception as e:
    print(f"   ❌ Erreur: {e}")

# Test 3: Import aiprod_trainer sans google/wandb
print("[3] Import aiprod_trainer...")
try:
    import aiprod_trainer
    print("   ✅ aiprod_trainer importé sans erreur")
except Exception as e:
    print(f"   ❌ Erreur: {e}")

# Test 4: Import gemini_client (doit fonctionner même sans google-generativeai)
print("[4] Import gemini_client (optionnel)...")
try:
    from aiprod_pipelines.api.integrations.gemini_client import GeminiAPIClient
    client = GeminiAPIClient()
    mode = "mock" if client.model is None else "live"
    print(f"   ✅ GeminiAPIClient en mode: {mode}")
except Exception as e:
    print(f"   ❌ Erreur: {e}")

# Test 5: Import captioning sans GeminiFlashCaptioner
print("[5] Import captioning (QwenOmni par défaut)...")
try:
    from aiprod_trainer.captioning import QwenOmniCaptioner, CaptionerType, create_captioner
    print("   ✅ QwenOmniCaptioner importé")
except Exception as e:
    print(f"   ❌ Erreur: {e}")

# Test 6: Import vae_trainer sans wandb obligatoire
print("[6] Import vae_trainer...")
try:
    from aiprod_trainer.vae_trainer import VideoVAETrainer, VAETrainerConfig
    print(f"   ✅ VideoVAETrainer importé, use_wandb default = {VAETrainerConfig().use_wandb}")
except Exception as e:
    print(f"   ❌ Erreur: {e}")

# Test 7: Vérifier que requirements.txt ne contient pas de packages cloud
print("[7] Vérification requirements.txt...")
with open("requirements.txt") as f:
    content = f.read()
    clean_lines = [l.strip() for l in content.split("\n") if l.strip() and not l.strip().startswith("#")]
    forbidden = ["boto3", "google-cloud-storage", "stripe", "wandb"]
    found = [p for p in forbidden if p in clean_lines]
    if found:
        print(f"   ❌ Packages non-souverains trouvés: {found}")
    else:
        print("   ✅ Aucun package cloud dans requirements.txt")

# Test 8: Vérifier les chemins par défaut
print("[8] Vérification chemins par défaut...")
from aiprod_core.model.text_encoder.bridge import LLMBridgeConfig
cfg = LLMBridgeConfig()
if "models/" in cfg.model_name:
    print(f'   ✅ LLMBridgeConfig.model_name = "{cfg.model_name}" (local)')
else:
    print(f'   ❌ LLMBridgeConfig.model_name = "{cfg.model_name}" (HuggingFace Hub!)')

# Test 9: Vérifier Dockerfiles sans google
print("[9] Vérification Dockerfiles...")
for dfile in ["deploy/docker/Dockerfile", "deploy/docker/Dockerfile.gpu"]:
    with open(dfile) as f:
        content = f.read()
    google_refs = [l.strip() for l in content.split("\n") 
                   if "google" in l.lower() and not l.strip().startswith("#")]
    if google_refs:
        print(f"   ❌ {dfile}: Google refs trouvées: {google_refs}")
    else:
        print(f"   ✅ {dfile}: Aucune référence Google active")

print()
print("=== FIN DES TESTS SOUVERAINETE Phase 1 ===")
