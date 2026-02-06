#!/usr/bin/env python3
"""
Check the REAL status of video generation and backend selection
"""

import sys
import io

# Fix Unicode on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import requests
import json
from pathlib import Path

# Read the promotion prompt
workspace = Path(__file__).parent.parent
workspace_config = {
    "content": """AIPROD: Transform Text to Professional Video in 4K

Opening: [Dark background with subtle motion] 
Narrator: "Script to 4K video in seconds"

[Scene 1 - 3 sec] Fast transitions of text transforming into vibrant video frames
Show: Script document → stunning video scene
Colors: Pink, blue, yellow gradients
Text overlay: "Transform Scripts to Video"

[Scene 2 - 2 sec] Show icons: AI chip, lightning bolt, camera
Emphasize: AI-powered Intelligence
Speed indicator: "10x faster"

[Scene 3 - 2 sec] Professional video clips showing quality
Text: "Enterprise-Grade Quality"
Subtitle: "4K Resolution, 60fps"

[Scene 4 - 3 sec] Finale: 
Logo appears (AIPROD film reel)
Call to action: "Visit GitHub for Code"
Color fade: Rainbow to AIPROD brand colors

Overall tone: Professional, dynamic, innovative
Music: Upbeat, modern, tech-forward
Duration: 10 seconds total""",
    "duration_sec": 10,
    "preset": "brand_campaign",
    "priority": "high",
}

print("=" * 70)
print("🎬 CHECKING VIDEO GENERATION STATUS")
print("=" * 70)

try:
    response = requests.post(
        "http://localhost:8000/pipeline/run",
        json=workspace_config,
        timeout=120
    )
    
    if response.status_code != 200:
        print(f"❌ API Error: {response.status_code}")
        print(response.text)
        exit(1)
    
    data = response.json()
    
    # Extract key information
    status = data.get("status", "unknown")
    render_data = data.get("data", {}).get("render", {})
    cost_data = data.get("data", {}).get("cost_estimate", {})
    
    print(f"\n✅ API Status: {status}")
    print(f"\n📊 RENDER DETAILS:")
    print(f"   Backend Used: {render_data.get('backend', 'NOT FOUND')}")
    print(f"   Status: {render_data.get('status', 'NOT FOUND')}")
    print(f"   Video URL: {render_data.get('video_url', 'NOT FOUND')}")
    print(f"   Image URL: {render_data.get('image_url', 'NOT FOUND')}")
    
    print(f"\n💰 COST OPTIMIZATION:")
    print(f"   Backend Selected: {cost_data.get('backend_selected', 'N/A')}")
    print(f"   AIPROD Optimized Cost: ${cost_data.get('aiprod_optimized', 'N/A')}")
    print(f"   Runway Direct Cost: ${cost_data.get('runway_alone', 'N/A')}")
    print(f"   Savings: ${cost_data.get('savings', 'N/A')} ({cost_data.get('savings_percent', 'N/A')}%)")
    
    # Determine if it's real or mock
    print(f"\n" + "=" * 70)
    if "mock" in render_data.get('backend', '').lower() or "mock" in render_data.get('status', '').lower():
        print("⚠️  RESULT: MOCK MODE (Simulated Video)")
        print("\n   Reason: Video generation is simulated, not real")
        print("   This happens when:")
        print("   - Runway API key is not loaded properly")
        print("   - OR Runway API is not available")
        print("   - OR Backend fell to mock mode for safety")
        print("\n   To enable real videos:")
        print("   1. Ensure RUNWAY_API_KEY is set in environment")
        print("   2. Restart API with proper env vars loaded")
        print("   3. Check Runway API account has credits")
    else:
        print("✨ RESULT: REAL VIDEO GENERATION")
        print(f"\n   ✅ Real video generated via {render_data.get('backend', 'UNKNOWN').upper()}")
        print(f"   📍 Video URL: {render_data.get('video_url', 'N/A')}")
        print(f"   💾 Size: {len(str(render_data.get('assets', {})))} bytes metadata")
    
    print("=" * 70)
    
except requests.exceptions.ConnectionError:
    print("❌ Cannot connect to API at http://localhost:8000")
    print("   Start the API with: .venv311\\Scripts\\python.exe -m uvicorn src.api.main:app")
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")
