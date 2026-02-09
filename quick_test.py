#!/usr/bin/env python3
"""
Direct HTTP test of endpoints (avoids import issues)
"""
import json
import sys
try:
    import requests
except:
    sys.exit(1)

BASE = "http://localhost:8000"

# Test 1: Health
r = requests.get(f"{BASE}/health")
if r.status_code == 200:
    print("✓ API Health: PASS")
else:
    print(f"✗ API Health: {r.status_code}")
    sys.exit(1)

# Test 2: Check routes registered
r = requests.get(f"{BASE}/openapi.json")
if r.status_code == 200:
    routes = r.json().get("paths", {})
    video_routes = [p for p in routes.keys() if "video" in p]
    if "/video/plan" in routes and "/video/generate" in routes:
        print(f"✓ Endpoints Registered: {video_routes}")
        print("✓ /video/plan: FOUND")
        print("✓ /video/generate: FOUND")
    else:
        print(f"✗ Video endpoints not found in OpenAPI")
        print(f"  Available: {video_routes}")
        sys.exit(1)
else:
    print(f"✗ OpenAPI check failed: {r.status_code}")
    sys.exit(1)

print("\n✓ ALL TESTS PASSED - Endpoints are live!")
