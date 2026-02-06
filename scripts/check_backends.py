#!/usr/bin/env python3
"""
Check which video rendering backends are installed and available
"""

import sys

backends = {
    "runwayml": "Runway ML (Primary - Best Quality)",
    "google.cloud.aiplatform": "Google Vertex AI (Veo-3 - Premium)",
    "replicate": "Replicate (Budget Fallback)"
}

print("🔍 VIDEO RENDERING BACKEND AVAILABILITY\n" + "="*70)

for package, description in backends.items():
    try:
        __import__(package)
        print(f"✅ {package:<30} INSTALLED")
        print(f"   └─ {description}\n")
    except ImportError:
        print(f"❌ {package:<30} NOT INSTALLED")
        print(f"   └─ {description}\n")
        print(f"   Install with: pip install {package.split('.')[0]}\n")

# Also check for RunwayML from a different import
print("\n" + "="*70)
print("Checking alternative Runway imports...")
try:
    from runwayml import RunwayML
    print("✅ from runwayml import RunwayML - SUCCESS")
except ImportError as e:
    print(f"❌ from runwayml import RunwayML - FAILED: {e}")

print("\n" + "="*70)
print("📊 Impact on AIPROD:")
print("   - Without any backend installed → Mock mode (no real videos)")
print("   - With at least 1 backend → Real video generation enabled!")
print("   - Recommended: Install at least 2 for fallback")
