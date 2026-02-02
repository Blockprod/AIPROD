"""
Test de validation Phase 2 - Vérifier que Prometheus/Grafana/Jaeger fonctionnent
Run: python tests/phase2_health_check.py
"""

import asyncio
import sys
import subprocess
from datetime import datetime
import time


def check_docker_running():
    """Vérifier que Docker est en cours d'exécution."""
    try:
        result = subprocess.run(["docker", "ps"], capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Docker check failed: {e}")
        return False


def check_container(name: str, port: int) -> bool:
    """Vérifier qu'un container est en cours d'exécution et accessible."""
    try:
        # Vérifier que le container existe
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={name}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if name not in result.stdout:
            print(f"❌ Container {name} not running")
            return False

        # Vérifier que le port répond
        result = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True, timeout=5
        )

        if f":{port}" in result.stdout or f"127.0.0.1:{port}" in result.stdout:
            print(f"✅ {name:20} listening on port {port}")
            return True
        else:
            print(f"⏳ {name:20} not yet listening (port {port})")
            return False

    except Exception as e:
        print(f"⚠️  {name:20} check failed: {e}")
        return False


async def test_prometheus_metrics():
    """Test que Prometheus scrape les métriques."""
    try:
        import httpx

        async with httpx.AsyncClient() as client:
            # Vérifier que Prometheus répond
            response = await client.get(
                "http://localhost:9090/api/v1/query", params={"query": "up"}, timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    print(
                        f"✅ Prometheus metrics available: {len(data.get('data', {}).get('result', []))} series"
                    )
                    return True

            print(f"⚠️  Prometheus: {response.status_code}")
            return False

    except Exception as e:
        print(f"⚠️  Prometheus test failed: {e}")
        return False


async def test_grafana():
    """Test que Grafana est accessible."""
    try:
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:3000/api/health", timeout=10)

            if response.status_code == 200:
                data = response.json()
                print(f"✅ Grafana running: {data.get('commit', 'N/A')}")
                return True

            print(f"⚠️  Grafana: {response.status_code}")
            return False

    except Exception as e:
        print(f"⚠️  Grafana test failed: {e}")
        return False


async def test_fastapi_metrics():
    """Test que FastAPI expose les métriques."""
    try:
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/metrics", timeout=10)

            if response.status_code == 200:
                metrics_text = response.text
                # Chercher au moins une métrique Prometheus
                if "# HELP" in metrics_text or "# TYPE" in metrics_text:
                    count = metrics_text.count("\n")
                    print(f"✅ FastAPI metrics endpoint: {count} lines")
                    return True

            print(f"⚠️  FastAPI metrics: {response.status_code}")
            return False

    except Exception as e:
        print(f"⚠️  FastAPI metrics test failed: {e}")
        return False


def print_header(title: str):
    """Afficher un header de section."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


async def main():
    """Test complet Phase 2."""
    print_header("PHASE 2 - OBSERVABILITÉ HEALTH CHECK")
    print(f"Timestamp: {datetime.now().isoformat()}\n")

    results = {}

    # 1. Vérifier Docker
    print("🐳 Checking Docker...")
    if not check_docker_running():
        print("\n❌ Docker is not running. Please start Docker first:")
        print("   docker-compose -f docker-compose.monitoring.yml up -d\n")
        return False

    # 2. Vérifier les containers
    print_header("CONTAINER STATUS")
    containers = [
        ("prometheus", 9090),
        ("grafana", 3000),
        ("alertmanager", 9093),
        ("jaeger", 16686),
        ("node-exporter", 9100),
    ]

    container_status = {}
    for name, port in containers:
        container_status[name] = check_container(name, port)

    if not any(container_status.values()):
        print("\n⏳ No containers running yet. Starting...")
        print("   docker-compose -f docker-compose.monitoring.yml up -d\n")
        print("   Wait 30 seconds for containers to be healthy.\n")
        return False

    # 3. Tests asynchrones
    print_header("ENDPOINT HEALTH")

    # Attendre que Prometheus soit prêt
    for i in range(5):
        print(f"Attempt {i+1}/5...")
        prometheus_ok = await test_prometheus_metrics()
        grafana_ok = await test_grafana()
        fastapi_ok = await test_fastapi_metrics()

        if prometheus_ok and grafana_ok and fastapi_ok:
            break

        if i < 4:
            print("Waiting 5 seconds before retry...\n")
            time.sleep(5)

    # 4. Summary
    print_header("PHASE 2 READINESS")

    all_checks = [
        ("Docker", check_docker_running()),
        ("Prometheus", prometheus_ok),
        ("Grafana", grafana_ok),
        ("FastAPI Metrics", fastapi_ok),
        ("Jaeger", container_status.get("jaeger", False)),
    ]

    passed = sum(1 for _, result in all_checks if result)
    total = len(all_checks)

    for check_name, result in all_checks:
        status = "✅" if result else "❌"
        print(f"{status} {check_name:25} {'OK' if result else 'FAILED'}")

    print(f"\nScore: {passed}/{total} checks passed")

    if passed >= 4:
        print("\n🚀 PHASE 2 IS READY! Next steps:\n")
        print("  1. Open Grafana: http://localhost:3000 (admin/admin)")
        print("  2. Open Prometheus: http://localhost:9090")
        print("  3. Open Jaeger: http://localhost:16686")
        print("  4. Start FastAPI: python -m uvicorn src.api.main:app --reload")
        print(
            "  5. Execute test pipeline: POST http://localhost:8000/api/pipeline/execute"
        )
        print("\n")
        return True
    else:
        print("\n⚠️  Phase 2 not fully ready. Check the errors above.\n")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
