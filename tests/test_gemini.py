import httpx

key = "AIzaSyAUdogIIbGavH9gvZi7SvteGKcdfz9tRbw"

# Test different models
models = [
    "gemini-1.5-pro",
    "gemini-2.0-flash",
    "gemini-pro"
]

payload = {
    "contents": [{"parts": [{"text": "Say hello"}]}]
}

for model in models:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    try:
        r = httpx.post(f"{url}?key={key}", json=payload, timeout=10)
        print(f"✅ {model}: Status {r.status_code}")
        if r.status_code == 200:
            print(f"   Working!")
            break
    except Exception as e:
        print(f"❌ {model}: {e}")

