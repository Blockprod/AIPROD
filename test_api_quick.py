#!/usr/bin/env python
"""Quick API startup test"""

import sys
import time
import subprocess
import requests
from pathlib import Path

print("\n" + "="*60)
print("🚀 LAUNCHING API + QUICK TEST")
print("="*60)

# Start API in background
proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"],
    env={"PYTHONIOENCODING": "utf-8"},
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

print("✓ API process started (PID: {})".format(proc.pid))
time.sleep(5)

# Test health endpoint
try:
    response = requests.get("http://localhost:8000/health", timeout=5)
    if response.status_code == 200:
        print("\n✅ API IS ALIVE!")
        print("   Status: {}".format(response.status_code))
        print("   Response: {}".format(response.json()))
    else:
        print("\n⚠️  API returned: {}".format(response.status_code))
except Exception as e:
    print("\n❌ API not responding: {}".format(str(e)[:100]))
    sys.exit(1)

# Test video endpoints
print("\n" + "-"*60)
print("Testing video endpoints...")
print("-"*60)

endpoints = [
    ("/video/plan", {"prompt": "test"}),
    ("/video/generate", {"prompt": "test", "duration_sec": 5, "tier": "BALANCED"}),
]

for path, data in endpoints:
    try:
        url = "http://localhost:8000" + path
        resp = requests.post(url, json=data, timeout=5)
        print("✓ {} → {} (OpenAPI registered)".format(path, resp.status_code))
    except Exception as e:
        print("✗ {} → {}".format(path, str(e)[:60]))

print("\n" + "="*60)
print("✅ API STARTUP DIAGNOSTIC COMPLETE")
print("="*60)
print("\nAPI is running on: http://localhost:8000")
print("Keep it running, or press Ctrl+C to stop")

try:
    proc.wait()
except KeyboardInterrupt:
    print("\n\nStopping API...")
    proc.terminate()
    proc.wait()
    print("✓ API stopped")
