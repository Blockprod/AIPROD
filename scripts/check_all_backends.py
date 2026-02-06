"""Vérifie tous les backends alternatifs disponibles pour la génération vidéo."""
from dotenv import load_dotenv
load_dotenv(override=True)
import os

print("=== VERIFICATION DES BACKENDS ALTERNATIFS ===")
print()

# 1. Replicate
rep_key = os.getenv("REPLICATE_API_TOKEN", "") or os.getenv("REPLICATE_API_KEY", "")
if rep_key and rep_key != "your-replicate-api-key":
    preview = repr(rep_key[:30])
    print(f"1. REPLICATE_API_TOKEN: {preview}...")
else:
    print("1. REPLICATE_API_TOKEN: NON CONFIGURE")

# 2. GCP / Vertex AI
gcp_creds_env = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
gcp_proj = os.getenv("GOOGLE_CLOUD_PROJECT", "") or os.getenv("GCP_PROJECT_ID", "")
print(f"2. GCP Project: {gcp_proj}")
print(f"   GOOGLE_APPLICATION_CREDENTIALS: {gcp_creds_env or 'NON CONFIGURE'}")

# Check credentials file
creds_file = os.path.join("credentials", "terraform-key.json")
if os.path.exists(creds_file):
    import json
    with open(creds_file) as f:
        creds = json.load(f)
    print(f"   credentials/terraform-key.json: EXISTE")
    print(f"     type: {creds.get('type', '?')}")
    print(f"     project_id: {creds.get('project_id', '?')}")
    email = creds.get("client_email", "?")
    print(f"     client_email: {email[:50]}...")
else:
    print("   credentials/terraform-key.json: ABSENT")

# 3. Gemini API
gemini = os.getenv("GEMINI_API_KEY", "")
if gemini:
    print(f"3. GEMINI_API_KEY: {gemini[:20]}...")
else:
    print("3. GEMINI_API_KEY: NON CONFIGURE")

# 4. Check packages
print()
print("=== PACKAGES INSTALLES ===")

try:
    import google.cloud.aiplatform
    print("  google-cloud-aiplatform: INSTALLE")
except ImportError:
    print("  google-cloud-aiplatform: PAS INSTALLE")

try:
    import replicate
    print(f"  replicate: INSTALLE (v{replicate.__version__})")
except ImportError:
    print("  replicate: PAS INSTALLE")

try:
    import google.generativeai as genai
    print("  google-generativeai: INSTALLE")
except ImportError:
    print("  google-generativeai: PAS INSTALLE")

# 5. Test Gemini API connectivity
print()
print("=== TEST CONNECTIVITE GEMINI ===")
if gemini:
    try:
        import google.generativeai as genai
        genai.configure(api_key=gemini)
        models = [m.name for m in genai.list_models() if "video" in m.name.lower() or "veo" in m.name.lower() or "imagen" in m.name.lower()]
        if models:
            print(f"  Modeles video/image disponibles: {models}")
        else:
            all_models = [m.name for m in genai.list_models()]
            print(f"  Pas de modeles video trouves.")
            print(f"  Modeles disponibles ({len(all_models)}):")
            for m in all_models[:15]:
                print(f"    - {m}")
            if len(all_models) > 15:
                print(f"    ... et {len(all_models) - 15} autres")
    except Exception as e:
        print(f"  Erreur Gemini: {e}")
else:
    print("  Gemini non configure, skip")

# 6. Test Vertex AI connectivity
print()
print("=== TEST VERTEX AI ===")
try:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_file
    import google.cloud.aiplatform as aip
    aip.init(project=gcp_proj, location="us-central1")
    print(f"  Vertex AI initialise: project={gcp_proj}")
    print("  Connectivite OK")
except Exception as e:
    print(f"  Erreur Vertex AI: {e}")
