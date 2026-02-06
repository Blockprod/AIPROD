#!/usr/bin/env python3
"""
Start AIPROD API with .env loaded into environment
This ensures API keys are available to the RenderExecutor
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
workspace = Path(__file__).parent.parent
env_file = workspace / ".env"

if env_file.exists():
    print(f"📄 Loading environment from: {env_file}")
    load_dotenv(env_file, override=True)
    print("✅ Environment variables loaded!")
    
    # Verify critical variables are loaded
    critical_vars = ["GEMINI_API_KEY", "GCP_PROJECT_ID", "RUNWAY_API_KEY"]
    for var in critical_vars:
        value = os.getenv(var)
        if value:
            masked = value[:10] + "..." + value[-5:] if len(value) > 15 else "****"
            print(f"   ✅ {var:<25} = {masked}")
        else:
            print(f"   ❌ {var:<25} = NOT SET")
else:
    print(f"⚠️  .env not found at: {env_file}")
    print("   Continuing with system environment variables...")

# Now start UV icorn
print("\n🚀 Starting AIPROD API Server...")
print("-" * 70)

os.chdir(workspace)
os.execvp(
    sys.executable,
    [sys.executable, "-m", "uvicorn", "src.api.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
)
