#!/usr/bin/env python3
"""
AIPROD API Launcher - Loads .env and starts Uvicorn with proper environment
"""

import os
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

# Get workspace root
workspace = Path(__file__).parent.parent
env_file = workspace / ".env"

print("\n" + "="*70)
print("🚀 AIPROD API SERVER LAUNCHER")
print("="*70)

# Load .env into current process
if env_file.exists():
    print(f"\n📄 Loading environment from: {env_file}")
    load_dotenv(env_file, override=True)
    print("✅ Environment variables loaded into Python process")
    
    # Verify critical variables
    critical = ["RUNWAY_API_KEY", "GEMINI_API_KEY", "GCP_PROJECT_ID"]
    print("\n📊 Environment Status:")
    for var in critical:
        value = os.getenv(var)
        if value:
            masked = value[:20] + "..." + value[-5:] if len(value) > 25 else "***"
            print(f"   ✅ {var:<25} = {masked}")
        else:
            print(f"   ❌ {var:<25} = NOT SET")
else:
    print(f"⚠️  .env not found at {env_file}")
    print("   Continuing with system environment variables...")

print("\n" + "="*70)
print("🎬 Starting FastAPI Server with Uvicorn...")
print("="*70)
print("\n📡 Server will listen on http://0.0.0.0:8000")
print("📚 API Docs: http://localhost:8000/docs")
print("💚 Health Check: http://localhost:8000/health\n")

# Change to workspace directory
os.chdir(workspace)

# IMPORTANT: Pass the environment dict to subprocess so Uvicorn inherits the loaded variables
# NOTE: Removed --reload to avoid issues with module reloading and load_dotenv()
try:
    subprocess.run([
        sys.executable,
        "-m", "uvicorn",
        "src.api.main:app",
        "--host", "0.0.0.0",
        "--port", "8000"
    ], env=os.environ)  # <-- PASS env variables explicitly!
except KeyboardInterrupt:
    print("\n\n👋 API Server stopped (Ctrl+C)")
except Exception as e:
    print(f"\n❌ Error: {e}")
    sys.exit(1)
