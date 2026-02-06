"""Check Runway ML API account status and credits."""
from dotenv import load_dotenv
load_dotenv(override=True)

import os
import httpx

key = os.getenv("RUNWAY_API_KEY", "")
print(f"API Key: {key[:20]}...{key[-8:]}")
print(f"Key length: {len(key)} chars")
print()

# Try RunwayML SDK
try:
    from runwayml import RunwayML
    client = RunwayML(api_key=key)
    print("RunwayML SDK loaded OK")
    base = getattr(client, "_base_url", "unknown")
    print(f"Client base URL: {base}")
    print()
except Exception as e:
    print(f"SDK error: {e}")
    print()

# Direct API calls to check account
headers = {
    "Authorization": f"Bearer {key}",
    "Content-Type": "application/json",
    "X-Runway-Version": "2024-11-06",
}

print("=" * 60)
print("RUNWAY API ACCOUNT CHECK")
print("=" * 60)

# 1. List recent tasks (validates API key)
print("\n--- 1. Validating API Key (List Tasks) ---")
try:
    resp = httpx.get("https://api.dev.runwayml.com/v1/tasks", headers=headers, timeout=15)
    print(f"HTTP {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        if isinstance(data, list):
            print(f"  Tasks found: {len(data)}")
            for t in data[:3]:
                print(f"  - ID: {t.get('id', '?')}, Status: {t.get('status', '?')}")
        elif isinstance(data, dict):
            for k, v in list(data.items())[:8]:
                print(f"  {k}: {str(v)[:120]}")
    elif resp.status_code == 401:
        print("  UNAUTHORIZED - API key is invalid or expired!")
    elif resp.status_code == 403:
        print("  FORBIDDEN - API key lacks permissions or account suspended")
    else:
        print(f"  Response: {resp.text[:300]}")
except Exception as e:
    print(f"  Error: {e}")

# 2. Try creating a minimal task to check credits
print("\n--- 2. Credit Check (Create Test Task) ---")
try:
    task_payload = {
        "promptText": "A gentle sunrise over mountains, cinematic, 4K",
        "model": "gen3a_turbo",
        "duration": 5,
    }
    resp = httpx.post(
        "https://api.dev.runwayml.com/v1/image_to_video",
        headers=headers,
        json=task_payload,
        timeout=15,
    )
    print(f"HTTP {resp.status_code}")
    data = resp.json() if resp.text else {}
    
    if resp.status_code == 200:
        print(f"  Task created! ID: {data.get('id', '?')}")
        print(f"  Status: {data.get('status', '?')}")
        print(f"  Credits OK!")
    elif resp.status_code == 401:
        print("  UNAUTHORIZED - Key invalid")
        print(f"  Detail: {data}")
    elif resp.status_code == 402:
        print("  PAYMENT REQUIRED - No credits remaining!")
        print(f"  Detail: {data}")
    elif resp.status_code == 403:
        print("  FORBIDDEN - Account issue")
        print(f"  Detail: {data}")
    elif resp.status_code == 422:
        print(f"  Validation error (key works, request format issue)")
        print(f"  Detail: {str(data)[:300]}")
    elif resp.status_code == 429:
        print("  RATE LIMITED - Too many requests (but key works!)")
    else:
        print(f"  Response: {str(data)[:300]}")
except Exception as e:
    print(f"  Error: {e}")

# 3. Check organizations endpoint
print("\n--- 3. Organization/Account Info ---")
try:
    resp = httpx.get("https://api.dev.runwayml.com/v1/organizations", headers=headers, timeout=15)
    print(f"HTTP {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        if isinstance(data, list):
            for org in data:
                print(f"  Org: {org.get('name', '?')}")
                print(f"  Plan: {org.get('plan', '?')}")
                print(f"  Credits: {org.get('credits', '?')}")
        elif isinstance(data, dict):
            for k, v in list(data.items())[:8]:
                print(f"  {k}: {str(v)[:120]}")
    else:
        print(f"  Response: {resp.text[:300]}")
except Exception as e:
    print(f"  Error: {e}")

# 4. Try the SDK directly
print("\n--- 4. SDK Direct Test ---")
try:
    os.environ["RUNWAYML_API_SECRET"] = key
    from runwayml import RunwayML
    client = RunwayML()
    
    # Try text_to_video
    task = client.image_to_video.create(
        model="gen3a_turbo",
        prompt_text="A gentle sunrise over misty mountains",
        duration=5,
    )
    print(f"  Task created via SDK!")
    print(f"  Task ID: {task.id}")
    print(f"  Status: {task.status}")
    print(f"  Credits: SUFFICIENT")
except Exception as e:
    error_str = str(e)
    print(f"  SDK Error: {error_str[:300]}")
    if "401" in error_str or "unauthorized" in error_str.lower():
        print("  => API KEY IS INVALID")
    elif "402" in error_str or "payment" in error_str.lower() or "credit" in error_str.lower():
        print("  => NO CREDITS - Need to add credits to Runway account")
    elif "insufficient" in error_str.lower():
        print("  => INSUFFICIENT CREDITS")
    elif "image" in error_str.lower() or "required" in error_str.lower():
        print("  => Key works, but image_to_video needs an image input")
        print("  => ACCOUNT IS ACTIVE!")

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)
