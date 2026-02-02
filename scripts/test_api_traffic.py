import requests
import time

BASE_URL = "http://localhost:8000"


# Liste des endpoints GET à tester
endpoints = [
    "/docs",
    "/openapi.json",
    "/health"
]

# Payload de test pour /pipeline/run
pipeline_payload = {
    "content": "Test API traffic",
    "priority": "low",
    "lang": "en",
    "preset": None,
    "duration_sec": 10
}

def test_endpoints():
    for endpoint in endpoints:
        url = BASE_URL + endpoint
        try:
            response = requests.get(url)
            print(f"GET {url} -> {response.status_code} | {response.text}")
        except Exception as e:
            print(f"Erreur lors de l'appel à {url} : {e}")

# Ajout du test POST sur /pipeline/run
def test_pipeline_run():
    url = BASE_URL + "/pipeline/run"
    try:
        response = requests.post(url, json=pipeline_payload)
        print(f"POST {url} -> {response.status_code} | {response.text}")
    except Exception as e:
        print(f"Erreur lors du POST à {url} : {e}")


if __name__ == "__main__":
    print("Début des tests API...")
    for i in range(10):
        print(f"Itération {i+1}/10")
        test_endpoints()
        test_pipeline_run()
        time.sleep(1)  # Pause entre les requêtes
    print("Tests terminés.")
