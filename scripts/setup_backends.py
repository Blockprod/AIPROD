#!/usr/bin/env python3
"""
Configuration Assistant pour AIPROD - Activation des backends vidéo réels
Guide interactif pour configurer Runway, Veo-3, et Replicate
"""

import os
import json
from pathlib import Path

# Configuration paths
WORKSPACE = Path(__file__).parent.absolute()
ENV_FILE = WORKSPACE / ".env"
ENV_EXAMPLE = WORKSPACE / ".env.example"

def read_env_file():
    """Lit le fichier .env actuel"""
    if ENV_FILE.exists():
        with open(ENV_FILE, 'r') as f:
            return f.read()
    return ""

def write_env_file(content):
    """Écrit le fichier .env"""
    with open(ENV_FILE, 'w') as f:
        f.write(content)
    print(f"✅ .env updated: {ENV_FILE}")

def configure_backend(backend_name, api_key_var, instructions_url=None):
    """Configure un backend spécifique"""
    print(f"\n{'='*70}")
    print(f"🔑 CONFIGURING: {backend_name}")
    print(f"{'='*70}")
    
    if instructions_url:
        print(f"\n📖 Instructions: {instructions_url}")
    
    print(f"\nEnter your {api_key_var} (or press Enter to skip):")
    api_key = input(">>> ").strip()
    
    if api_key:
        return {api_key_var: api_key}
    else:
        print(f"⏭️  Skipping {backend_name}...")
        return None

def show_status():
    """Affiche l'état actuel des clés API"""
    print("\n" + "="*70)
    print("📊 CURRENT API KEYS STATUS")
    print("="*70)
    
    env_content = read_env_file()
    
    backends = {
        "RUNWAY_API_KEY": "Runway ML (Primary - Best Quality)",
        "REPLICATE_API_TOKEN": "Replicate (Fallback - Budget)",
        "GEMINI_API_KEY": "Google Gemini (LLM)",
        "GCP_PROJECT_ID": "Google Cloud Project (Veo-3)"
    }
    
    for var, desc in backends.items():
        if var in env_content and not f"{var}=<" in env_content:
            # Extract value (masked)
            for line in env_content.split("\n"):
                if line.startswith(var):
                    value = line.split("=", 1)[1].strip()
                    if value and not value.startswith("<"):
                        masked = value[:4] + "..." + value[-4:] if len(value) > 8 else "***"
                        print(f"✅ {var:<25} {masked:<15} ({desc})")
                    break
        else:
            print(f"❌ {var:<25} NOT SET        ({desc})")

def main():
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                  🎬 AIPROD VIDEO RENDERING SETUP                  ║
║              Activate Real Backends for Video Generation           ║
╚════════════════════════════════════════════════════════════════════╝
    """)
    
    # Show current status
    show_status()
    
    print(f"\n\n{'='*70}")
    print("📋 SETUP OPTIONS")
    print("="*70)
    print("""
1. Configure Runway ML (RECOMMENDED - Best Quality)
2. Configure Replicate (Budget-Friendly Fallback)
3. Configure Google Cloud (Veo-3 Premium Backend)
4. View detailed instructions
5. Exit
    """)
    
    choice = input("Select option (1-5): ").strip()
    
    if choice == "1":
        config = configure_backend(
            "Runway ML",
            "RUNWAY_API_KEY",
            "https://app.runwayml.com/api-keys"
        )
        if config:
            update_env(config)
            print("\n✅ Runway configured! API will now use real video generation.")
    
    elif choice == "2":
        config = configure_backend(
            "Replicate",
            "REPLICATE_API_TOKEN",
            "https://replicate.com/account/api-tokens"
        )
        if config:
            update_env(config)
            print("\n✅ Replicate configured! Fallback backend activated.")
    
    elif choice == "3":
        print("\n" + "="*70)
        print("🌐 GOOGLE CLOUD / VEO-3 SETUP")
        print("="*70)
        print("""
GCP Project ID: aiprod-484120
Region: us-central1

Steps:
1. Go to: https://console.cloud.google.com
2. Select Project: "aiprod"
3. Enable Vertex AI API
4. Request Veo-3 access: https://cloud.google.com/vertex-ai/generative-ai/docs/image/veo3-overview
5. Copy your GCP_PROJECT_ID below:
        """)
        gcp_project = input("GCP Project ID: ").strip()
        if gcp_project:
            update_env({"GCP_PROJECT_ID": gcp_project})
            print(f"\n✅ GCP configured! Project: {gcp_project}")
    
    elif choice == "4":
        show_detailed_instructions()
        main()  # Restart menu
    
    elif choice == "5":
        print("\n👋 Goodbye!")
        return
    
    # Ask if want to continue
    print("\n" + "="*70)
    repeat = input("\nConfigure another backend? (y/n): ").strip().lower()
    if repeat == "y":
        main()
    else:
        print("\n" + "="*70)
        print("✨ SETUP COMPLETE!")
        print("="*70)
        print("""
Next steps:
1. Restart the API server:
   .venv311\\Scripts\\python.exe -m uvicorn src.api.main:app --reload

2. Call the video generation endpoint again:
   POST http://localhost:8000/pipeline/run
   
3. Monitor the logs to see real video generation happening!
        """)

def update_env(new_config):
    """Met à jour le fichier .env avec les nouvelles clés"""
    env_content = read_env_file()
    
    for key, value in new_config.items():
        if f"{key}=" in env_content:
            # Remplacer la ligne existante
            lines = env_content.split("\n")
            env_content = "\n".join(
                f"{key}={value}" if line.startswith(f"{key}=") else line
                for line in lines
            )
        else:
            # Ajouter la nouvelle ligne
            env_content += f"\n{key}={value}"
    
    write_env_file(env_content)

def show_detailed_instructions():
    """Affiche les instructions détaillées par backend"""
    print(f"""
{'='*70}
📖 DETAILED BACKEND CONFIGURATION GUIDE
{'='*70}

## 1️⃣ RUNWAY ML (Primary Backend - Recommended)
────────────────────────────────────────────────────────────

Status: ⭐⭐⭐ Best Quality (0.95)
Cost: $30/5-second video
Speed: ~30 seconds

Setup:
  a) Go to: https://app.runwayml.com
  b) Create free account (get $5 credits!)
  c) Navigation: Settings → API Keys
  d) Create new API key
  e) Copy and paste in setup wizard

Test command:
  curl -X POST http://localhost:8000/pipeline/run \\
    -H "Content-Type: application/json" \\
    -d '{{"content": "A beautiful sunset at the beach", "duration_sec": 5}}'

Expected: Real 5-second video generated! 🎬


## 2️⃣ REPLICATE (Fallback - Budget Friendly)
────────────────────────────────────────────────────────────

Status: ⭐ Budget-Friendly Quality (0.75)
Cost: $0.26/5-second video
Speed: ~20 seconds

Setup:
  a) Go to: https://replicate.com
  b) Sign up free
  c) Dashboard → Account → API Tokens
  d) Copy token
  e) Add to .env as REPLICATE_API_TOKEN

Fallback Logic:
  - If Runway fails → Replicate auto-activates
  - If both fail → Mock mode with error logs


## 3️⃣  GOOGLE VEO-3 (Premium via Vertex AI)
────────────────────────────────────────────────────────────

Status: ⭐⭐ High Quality (0.92)
Cost: $2.60/5-second video
Speed: ~40 seconds

Setup:
  a) GCP Project: aiprod-484120
  b) Enable APIs:
     - gcloud services enable aiplatform.googleapis.com
     - gcloud services enable cloudaicompanion.googleapis.com
  c) Request Veo-3 early access:
     https://cloud.google.com/vertex-ai/generative-ai/docs/image/veo3-overview
  d) Add GCP_PROJECT_ID to .env

Note: Requires early access approval (usually 24-48 hours)


## 📊 SELECTION LOGIC
────────────────────────────────────────────────────────────

AIPROD automatically selects the best backend based on:
  - Budget remaining
  - Quality required
  - Backend health (tracks failures)

Decision Tree:
  Budget >= $30 AND Quality >= 0.9  → Runway ML
  Budget >= $2.60 AND Quality >= 0.8 → Veo-3
  Budget >= $0.26 OR Quality Low     → Replicate
  No API keys                         → Mock mode


## 🔍 DIAGNOSTICS
────────────────────────────────────────────────────────────

Check endpoint status:
  curl http://localhost:8000/health

View API documentation:
  http://localhost:8000/docs

Monitor real-time metrics:
  curl http://localhost:8000/metrics | grep pipeline

{'='*70}
    """)

if __name__ == "__main__":
    main()
