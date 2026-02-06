"""
AIPROD - Génération de vidéo promotionnelle via Google Vertex AI (Imagen + Veo)
Utilise le service account GCP existant pour générer sans crédits Runway.
"""
from dotenv import load_dotenv
load_dotenv(override=True)

import os
import time
import json
import httpx

# Configurer les credentials GCP
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "credentials", "terraform-key.json"
)

GCP_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "aiprod-484120")
LOCATION = "us-central1"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "promo")

print("=" * 70)
print("  AIPROD - GENERATION VIDEO PROMOTIONNELLE")
print("  Backend: Google Vertex AI (Imagen 3 + Veo 2)")
print(f"  Projet GCP: {GCP_PROJECT}")
print("=" * 70)

# Créer le dossier output
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Méthode 1: Gemini API avec generation multimodale ---
print("\n--- Methode 1: Gemini API (google-generativeai) ---")
try:
    import google.generativeai as genai
    
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    genai.configure(api_key=gemini_key)
    
    # Lister les modèles disponibles pour la génération d'images
    print("  Recherche de modeles de generation d'images/video...")
    all_models = []
    for m in genai.list_models():
        name = m.name.lower()
        if any(k in name for k in ["imagen", "veo", "video", "image"]):
            all_models.append(m.name)
            print(f"    - {m.name}")
    
    if not all_models:
        print("  Aucun modele image/video trouve via Gemini API")
        # Lister tout pour debug
        for m in genai.list_models():
            if "generate" in str(getattr(m, 'supported_generation_methods', [])):
                print(f"    [gen] {m.name}")
    
except ImportError:
    print("  google-generativeai pas installe")
except Exception as e:
    print(f"  Erreur Gemini: {e}")

# --- Méthode 2: Vertex AI directement ---
print("\n--- Methode 2: Vertex AI (google-cloud-aiplatform) ---")
try:
    import google.cloud.aiplatform as aip
    from google.cloud import aiplatform_v1
    
    aip.init(project=GCP_PROJECT, location=LOCATION)
    print(f"  Vertex AI initialise OK (projet: {GCP_PROJECT})")
    
    # Essayer Imagen 3 pour l'image de base
    print("\n  [IMAGE] Generation avec Imagen 3...")
    try:
        from vertexai.preview.vision_models import ImageGenerationModel
        
        model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-002")
        
        prompt = (
            "Futuristic AI video production dashboard interface, dark theme, "
            "glowing blue and purple neon accents, holographic AI brain in center, "
            "real-time cost optimization graphs and metrics floating around, "
            "text 'AIPROD' in large futuristic font at top, "
            "professional tech startup, 4K quality, ultra detailed, "
            "cinematic lighting, dark background"
        )
        
        response = model.generate_images(
            prompt=prompt,
            number_of_images=1,
            aspect_ratio="16:9",
            safety_filter_level="block_few",
        )
        
        if response.images:
            img = response.images[0]
            img_path = os.path.join(OUTPUT_DIR, "aiprod_promo_image.png")
            img.save(img_path)
            print(f"  Image generee: {img_path}")
            print(f"  Taille: {os.path.getsize(img_path) / 1024:.0f} KB")
        else:
            print("  Pas d'image generee")
            
    except ImportError as ie:
        print(f"  vertexai.preview.vision_models non disponible: {ie}")
    except Exception as e:
        print(f"  Erreur Imagen: {e}")
    
    # Essayer Veo pour la vidéo 
    print("\n  [VIDEO] Generation avec Veo 2...")
    try:
        from vertexai.preview.vision_models import VideoGenerationModel
        
        video_model = VideoGenerationModel.from_pretrained("veo-002-generate-001")
        
        video_prompt = (
            "Smooth cinematic camera orbit around a futuristic AI dashboard. "
            "Glowing blue and purple neon lights illuminate data streams flowing "
            "through a holographic AI brain. Cost optimization metrics animate on screen "
            "showing prices dropping. Professional tech demo, 4K cinematic quality, "
            "dark background with vibrant accents."
        )
        
        response = video_model.generate_videos(
            prompt=video_prompt,
            number_of_videos=1,
            duration_seconds=4,
        )
        
        print(f"  Video task created!")
        
        # Attendre la completion
        if hasattr(response, 'result'):
            result = response.result()
            print(f"  Video generee!")
        
    except ImportError as ie:
        print(f"  vertexai.preview.vision_models.VideoGenerationModel non disponible: {ie}")
    except Exception as e:
        print(f"  Erreur Veo: {e}")

except ImportError as ie:
    print(f"  Erreur import: {ie}")
except Exception as e:
    print(f"  Erreur Vertex AI: {e}")

# --- Méthode 3: API REST directe vers Vertex AI ---
print("\n--- Methode 3: API REST Vertex AI (Imagen) ---")
try:
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account
    
    creds_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "credentials", "terraform-key.json")
    
    credentials = service_account.Credentials.from_service_account_file(
        creds_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(Request())
    token = credentials.token
    
    print(f"  Token GCP obtenu (service account)")
    
    # Appel API REST Imagen
    url = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{GCP_PROJECT}/locations/{LOCATION}/publishers/google/models/imagen-3.0-generate-002:predict"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "instances": [{
            "prompt": (
                "Futuristic AI video production dashboard, dark theme with glowing "
                "blue and purple neon accents, holographic AI brain processing data, "
                "cost optimization graphs, text 'AIPROD' in large futuristic font, "
                "professional tech startup style, 4K quality, ultra detailed"
            )
        }],
        "parameters": {
            "sampleCount": 1,
            "aspectRatio": "16:9",
            "safetyFilterLevel": "block_few",
        }
    }
    
    print(f"  Appel API Imagen 3 (REST)...")
    resp = httpx.post(url, headers=headers, json=payload, timeout=120)
    print(f"  HTTP {resp.status_code}")
    
    if resp.status_code == 200:
        data = resp.json()
        predictions = data.get("predictions", [])
        if predictions:
            import base64
            img_b64 = predictions[0].get("bytesBase64Encoded", "")
            if img_b64:
                img_bytes = base64.b64decode(img_b64)
                img_path = os.path.join(OUTPUT_DIR, "aiprod_promo_banner.png")
                with open(img_path, "wb") as f:
                    f.write(img_bytes)
                print(f"  Image generee: {img_path}")
                print(f"  Taille: {len(img_bytes) / 1024:.0f} KB")
            else:
                print(f"  Pas de bytesBase64Encoded dans la reponse")
                print(f"  Keys: {list(predictions[0].keys())}")
        else:
            print(f"  Pas de predictions dans la reponse")
            print(f"  Response: {str(data)[:500]}")
    else:
        print(f"  Erreur: {resp.text[:500]}")

except Exception as e:
    print(f"  Erreur REST: {e}")

print("\n" + "=" * 70)
print("  GENERATION TERMINEE")
print("=" * 70)
