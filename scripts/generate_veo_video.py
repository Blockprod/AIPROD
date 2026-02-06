"""
AIPROD - Génération vidéo promotionnelle via Gemini API + Veo
Utilise la clé Gemini API pour accéder aux modèles Veo.
"""
from dotenv import load_dotenv
load_dotenv(override=True)

import os
import time

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "promo")
os.makedirs(OUTPUT_DIR, exist_ok=True)

gemini_key = os.getenv("GEMINI_API_KEY", "")
print(f"Gemini API Key: {gemini_key[:20]}...")

# Prompt promotionnel AIPROD
VIDEO_PROMPT = (
    "A sleek futuristic AI technology dashboard with dark theme. "
    "Glowing blue and purple neon lights illuminate a holographic AI brain "
    "at the center processing data streams into cinematic video. "
    "Cost optimization metrics and graphs float around showing real-time savings. "
    "Smooth cinematic camera orbit around the dashboard. "
    "Professional 4K quality, ultra detailed, cinematic lighting."
)

# --- Essayer google.genai (nouveau SDK recommandé) ---
print("\n=== Generation video avec google.genai + Veo ===")
try:
    from google import genai
    
    client = genai.Client(api_key=gemini_key)
    print("  Client genai cree OK")
    print(f"  Prompt: {VIDEO_PROMPT[:80]}...")
    print(f"  Lancement generation Veo 2...")
    
    operation = client.models.generate_videos(
        model="veo-2.0-generate-001",
        prompt=VIDEO_PROMPT,
        config={
            "aspect_ratio": "16:9",
            "number_of_videos": 1,
            "duration_seconds": 5,
            "person_generation": "dont_allow",
        }
    )
    
    print(f"  Operation lancee! Polling en cours...")
    
    # Polling
    max_wait = 360  # 6 minutes max
    start = time.time()
    
    while not operation.done and (time.time() - start) < max_wait:
        elapsed = time.time() - start
        print(f"  ... generation en cours ({elapsed:.0f}s / {max_wait}s)")
        time.sleep(15)
        try:
            operation = client.operations.get(operation)
        except Exception as poll_err:
            print(f"  Polling error: {poll_err}")
            time.sleep(5)
    
    elapsed = time.time() - start
    
    if operation.done:
        print(f"\n  Operation terminee en {elapsed:.0f}s !")
        result = operation.result
        
        if result and hasattr(result, 'generated_videos') and result.generated_videos:
            for i, gv in enumerate(result.generated_videos):
                video = gv.video
                
                # Sauvegarder via URI ou bytes
                if hasattr(video, 'uri') and video.uri:
                    print(f"  Video URI: {video.uri}")
                    import httpx
                    video_data = httpx.get(video.uri, timeout=60).content
                    video_path = os.path.join(OUTPUT_DIR, "aiprod_promo.mp4")
                    with open(video_path, "wb") as f:
                        f.write(video_data)
                    print(f"  VIDEO SAUVEGARDEE: {video_path}")
                    print(f"  Taille: {len(video_data) / 1024:.0f} KB")
                    
                elif hasattr(video, 'video_bytes') and video.video_bytes:
                    video_path = os.path.join(OUTPUT_DIR, "aiprod_promo.mp4")
                    with open(video_path, "wb") as f:
                        f.write(video.video_bytes)
                    print(f"  VIDEO SAUVEGARDEE: {video_path}")
                    print(f"  Taille: {len(video.video_bytes) / 1024:.0f} KB")
                else:
                    print(f"  Video objet: {dir(video)}")
        else:
            print(f"  Pas de videos generees")
            print(f"  Result type: {type(result)}")
            print(f"  Result: {result}")
    else:
        print(f"\n  Timeout apres {max_wait}s - la generation prend plus longtemps que prevu")
        print(f"  Operation: {operation}")
        
except ImportError:
    print("  Paquet google-genai non installe")
    print("  Installation en cours...")
    import subprocess
    subprocess.run([
        os.path.join(os.path.dirname(os.path.dirname(__file__)), ".venv311", "Scripts", "pip.exe"),
        "install", "google-genai"
    ])
    print("  Installe ! Relancez ce script.")
except Exception as e:
    print(f"  Erreur: {e}")
    import traceback
    traceback.print_exc()

# Résultat final
print("\n" + "=" * 70)
print("  FICHIERS GENERES:")
print("=" * 70)
if os.path.exists(OUTPUT_DIR):
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(fpath)
        print(f"  {f}: {size / 1024:.0f} KB")
