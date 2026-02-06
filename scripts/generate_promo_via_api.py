#!/usr/bin/env python3
"""
Generate promotional video via AIPROD API
This script demonstrates AIPROD's capability by using the API itself to generate a promotional video
"""

import requests
import json
import sys
from pathlib import Path

# API endpoint
API_URL = "http://localhost:8000/pipeline/run"

# Promotional script/prompt
PROMO_PROMPT = """AIPROD: Transform Text to Professional Video in 4K

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
Duration: 10 seconds total"""

# Request payload
payload = {
    "content": PROMO_PROMPT,
    "duration_sec": 10,
    "preset": "brand_campaign",
    "priority": "high",
    "lang": "en"
}

headers = {
    "Content-Type": "application/json"
}

print("🎬 Generating promotional video via AIPROD API...")
print(f"📍 Endpoint: {API_URL}")
print(f"📝 Prompt length: {len(PROMO_PROMPT)} characters")
print(f"⏱️  Duration: {payload['duration_sec']} seconds")
print(f"🎨 Preset: {payload['preset']}")
print("-" * 70)

try:
    response = requests.post(API_URL, json=payload, headers=headers, timeout=60)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ SUCCESS! API responded with status 200")
        print("\n📊 Response data:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Extract job_id if available
        if "data" in result and "job_id" in result["data"]:
            job_id = result["data"]["job_id"]
            print(f"\n📋 Job ID: {job_id}")
            print(f"🔍 Check Endpoint: /pipeline/job/{job_id}")
            print(f"📡 Full Check URL: http://localhost:8000/pipeline/job/{job_id}")
    else:
        print(f"❌ API Error: Status {response.status_code}")
        print(f"Response: {response.text}")
        sys.exit(1)
        
except requests.exceptions.ConnectionError:
    print("❌ Connection Error: Could not connect to API at http://localhost:8000")
    print("   Make sure the API server is running!")
    sys.exit(1)
except requests.exceptions.Timeout:
    print("❌ Timeout: API request took too long (>60 seconds)")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {type(e).__name__}: {str(e)}")
    sys.exit(1)
