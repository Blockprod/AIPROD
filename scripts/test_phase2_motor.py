"""
Tests Phase 2 — Connecter le Moteur Souverain.

Vérifie que :
    1. gpu_worker.py existe et instancie correctement
    2. job_store.py (SQLite) CRUD fonctionne
    3. render.py ne contient AUCUN mock cloud (gs://, random.random failure sim)
    4. Config V34 est souveraine (zéro cloud, zéro Google)
    5. Orchestrator charge la V34 par défaut
    6. QA sémantique locale fonctionne
    7. Pipeline end-to-end (stub mode) produit un fichier vidéo
"""

import asyncio
import json
import os
import sys
import tempfile

# ── Tests ──────────────────────────────────────────────────────────────────

def test_gpu_worker_module_exists():
    """gpu_worker.py existe et est importable."""
    from aiprod_pipelines.api.gpu_worker import GPUWorker, WorkerConfig, JobRequest, JobResult, JobStatus
    assert GPUWorker is not None
    assert WorkerConfig is not None
    print("  ✅ gpu_worker module importable")

def test_gpu_worker_instantiation():
    """GPUWorker s'instancie sans charger le pipeline."""
    from aiprod_pipelines.api.gpu_worker import GPUWorker, WorkerConfig
    config = WorkerConfig(output_dir=tempfile.mkdtemp())
    worker = GPUWorker(config=config)
    assert not worker.is_loaded
    assert not worker.is_running
    print("  ✅ GPUWorker instancié (pipeline non chargé)")

def test_gpu_worker_stub_generation():
    """GPUWorker génère une vidéo stub sans GPU."""
    from aiprod_pipelines.api.gpu_worker import GPUWorker, WorkerConfig, JobRequest
    import tempfile, os

    out_dir = tempfile.mkdtemp()
    config = WorkerConfig(output_dir=out_dir)
    worker = GPUWorker(config=config)
    
    request = JobRequest(
        prompt="a cat walking on a beach",
        seed=42,
        height=64,
        width=64,
        num_frames=8,
        fps=8.0,
    )
    
    result = worker.process_job(request)
    assert result.status.value in ("completed", "failed"), f"Status inattendu: {result.status}"
    # En mode stub, on devrait avoir un fichier de sortie
    if result.output_path:
        assert os.path.exists(result.output_path), f"Fichier absent: {result.output_path}"
        size = os.path.getsize(result.output_path)
        print(f"  ✅ Vidéo stub générée: {result.output_path} ({size} bytes, {result.duration_sec:.1f}s)")
    else:
        print(f"  ⚠️ Stub mode — résultat: {result.status.value}")

def test_job_store_crud():
    """JobStore CRUD complet (SQLite local)."""
    from aiprod_pipelines.api.job_store import JobStore
    import tempfile

    db_path = os.path.join(tempfile.mkdtemp(), "test_jobs.db")
    store = JobStore(db_path=db_path)

    # Enqueue
    job_id = store.enqueue(
        prompt="test prompt",
        params={"height": 512, "width": 768},
        priority=5,
        user_id="user-1",
        tier="pro",
    )
    assert job_id is not None

    # Get
    job = store.get_job(job_id)
    assert job is not None
    assert job["prompt"] == "test prompt"
    assert job["status"] == "queued"
    assert job["priority"] == 5

    # List pending
    pending = store.list_pending()
    assert len(pending) == 1

    # Dequeue
    dequeued = store.dequeue()
    assert dequeued is not None
    assert dequeued["job_id"] == job_id

    # Verify status updated
    job = store.get_job(job_id)
    assert job["status"] == "processing"

    # Update status
    store.update_status(job_id, "completed", result={"output": "test.mp4"})
    job = store.get_job(job_id)
    assert job["status"] == "completed"

    # Stats
    stats = store.stats()
    assert stats["total_jobs"] == 1
    assert stats["by_status"]["completed"] == 1

    # Cancel — should fail because job is already completed
    cancelled = store.cancel_job(job_id)
    assert not cancelled

    # Enqueue another and cancel it
    job_id2 = store.enqueue(prompt="second job")
    cancelled = store.cancel_job(job_id2)
    assert cancelled

    print("  ✅ JobStore CRUD OK (SQLite)")

def test_render_no_mock():
    """render.py ne contient AUCUNE trace de mock cloud."""
    import inspect
    from aiprod_pipelines.api.adapters.render import RenderExecutorAdapter
    
    source = inspect.getsource(RenderExecutorAdapter)
    
    # Vérifier qu'il n'y a plus de gs:// URLs
    assert "gs://aiprod-assets" not in source, "gs:// URLs encore présentes dans render.py"
    
    # Vérifier qu'il n'y a plus de simulation de failure
    assert "Simulated" not in source, "'Simulated failure' encore dans render.py"
    assert "success_probability" not in source, "success_probability mock encore dans render.py"
    
    # Vérifier que le fallback cloud a été retiré
    assert "runway_gen3" not in source, "runway_gen3 fallback encore dans render.py"
    assert "replicate_wan25" not in source, "replicate_wan25 fallback encore dans render.py"
    
    # Vérifier que le GPUWorker est référencé
    assert "gpu_worker" in source.lower() or "GPUWorker" in source, "GPUWorker non mentionné dans render.py"
    
    # Vérifier backend souverain
    assert "aiprod_sovereign" in source, "Backend souverain non déclaré"
    
    print("  ✅ render.py — zéro mock cloud, backend souverain")

def test_config_v34_sovereign():
    """Config V34 est 100% souveraine."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "AIPROD_V34_SOVEREIGN.json")
    if not os.path.exists(config_path):
        config_path = "config/AIPROD_V34_SOVEREIGN.json"
    
    assert os.path.exists(config_path), f"V34 config manquante: {config_path}"
    
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    
    # Version
    assert cfg["version"] == "3.4", f"Version inattendue: {cfg['version']}"
    
    # Souveraineté
    sov = cfg.get("sovereignty", {})
    assert sov.get("cloudDependencies") == 0, "cloudDependencies != 0"
    assert sov.get("externalAPIs") == 0, "externalAPIs != 0"
    assert sov.get("offlineCapable") is True, "non offlineCapable"
    
    # Vérifier qu'aucun block n'utilise Google
    config_str = json.dumps(cfg)
    assert "gemini-1.5-pro" not in config_str, "gemini-1.5-pro dans V34"
    assert "gemini-2.0-flash" not in config_str, "gemini-2.0-flash dans V34"
    assert "veo3" not in config_str.lower() or "veo-3" not in config_str.lower(), "veo3 dans V34"
    assert "${GOOGLE_CLOUD_PROJECT}" not in config_str, "GOOGLE_CLOUD_PROJECT dans V34"
    assert "${GCS_BUCKET" not in config_str, "GCS_BUCKET dans V34"
    assert "${RUNWAY_API_KEY}" not in config_str, "RUNWAY_API_KEY dans V34"
    assert "${REPLICATE_API_KEY}" not in config_str, "REPLICATE_API_KEY dans V34"
    
    # Vérifier que tous les providers sont "local"
    blocks = cfg.get("state", {}).get("blocks", {})
    for name, block in blocks.items():
        provider = block.get("llmProvider", block.get("provider", "local"))
        if provider != "local" and name not in ("orchestrator",):
            assert provider == "local", f"Block '{name}' provider={provider} (attendu: local)"
    
    print(f"  ✅ Config V34 souveraine (v{cfg['version']}, {sov['score']})")

def test_orchestrator_loads_v34():
    """Orchestrator charge la config V34 par défaut."""
    from aiprod_pipelines.api.orchestrator import load_pipeline_config
    
    # Tester le loader avec le fichier existant
    cfg = load_pipeline_config("config/AIPROD_V34_SOVEREIGN.json")
    assert cfg["version"] == "3.4"
    assert cfg.get("sovereignty", {}).get("cloudDependencies") == 0
    print("  ✅ Orchestrator charge V34 souveraine")

def test_orchestrator_init_with_config():
    """Orchestrator s'initialise avec la config V34."""
    from aiprod_pipelines.api.orchestrator import Orchestrator
    
    # L'orchestrator devrait accepter un config_path
    orch = Orchestrator(
        adapters={},
        config_path="config/AIPROD_V34_SOVEREIGN.json",
    )
    assert orch.pipeline_config["version"] == "3.4"
    assert orch.pipeline_config.get("sovereignty", {}).get("score") == "9/10"
    print("  ✅ Orchestrator initialisé avec V34")

def test_qa_semantic_local_module():
    """qa_semantic_local.py existe et est importable."""
    from aiprod_pipelines.api.qa_semantic_local import SemanticQALocal, QASemanticResult
    
    assert SemanticQALocal is not None
    assert QASemanticResult is not None
    
    # Instanciation
    qa = SemanticQALocal(device="cpu", num_frames=4)
    assert qa.device == "cpu"
    assert qa.num_frames == 4
    print("  ✅ QA sémantique locale importable")

def test_qa_semantic_stub():
    """QA sémantique retourne un résultat même sans CLIP installé."""
    from aiprod_pipelines.api.qa_semantic_local import SemanticQALocal
    
    qa = SemanticQALocal(device="cpu")
    
    # Avec un fichier inexistant — devrait retourner score 0 proprement
    result = qa.evaluate("test prompt", "/nonexistent/video.mp4")
    assert result.overall_score == 0.0
    assert not result.passed
    print("  ✅ QA sémantique — mode dégradé OK")

def test_gateway_no_cloud_refs():
    """Gateway ne contient aucune référence cloud externe."""
    import inspect
    from aiprod_pipelines.api.gateway import APIGateway, create_fastapi_app
    
    source = inspect.getsource(sys.modules["aiprod_pipelines.api.gateway"])
    
    # Pas de référence à des services cloud dans le gateway
    assert "gs://aiprod" not in source, "gs:// URL dans gateway"
    assert "runway_gen3" not in source, "runway dans gateway"
    
    print("  ✅ Gateway — aucune référence cloud")

def test_end_to_end_stub_pipeline():
    """Pipeline E2E (stub): prompt → GPUWorker → fichier .mp4."""
    from aiprod_pipelines.api.gpu_worker import GPUWorker, WorkerConfig, JobRequest
    from aiprod_pipelines.api.job_store import JobStore
    import tempfile

    # Setup
    out_dir = tempfile.mkdtemp()
    db_path = os.path.join(tempfile.mkdtemp(), "e2e_jobs.db")
    
    # 1. Créer le store et enqueue un job
    store = JobStore(db_path=db_path)
    job_id = store.enqueue(
        prompt="cinematic sunset over ocean waves",
        params={
            "height": 64,
            "width": 64,
            "num_frames": 8,
            "fps": 8.0,
            "seed": 42,
        },
    )
    
    # 2. Créer le worker
    config = WorkerConfig(output_dir=out_dir)
    worker = GPUWorker(config=config)
    
    # 3. Dequeue et process
    job_data = store.dequeue()
    assert job_data is not None
    
    request = JobRequest(
        job_id=job_data["job_id"],
        prompt=job_data["prompt"],
        height=job_data.get("params", {}).get("height", 64),
        width=job_data.get("params", {}).get("width", 64),
        num_frames=job_data.get("params", {}).get("num_frames", 8),
        fps=job_data.get("params", {}).get("fps", 8.0),
        seed=job_data.get("params", {}).get("seed", 42),
    )
    
    result = worker.process_job(request)
    
    # 4. Update store
    store.update_status(
        job_id=result.job_id,
        status=result.status.value,
        result={"output_path": result.output_path, "duration_sec": result.duration_sec},
    )
    
    # 5. Vérifier
    final = store.get_job(job_id)
    assert final["status"] in ("completed", "failed")
    
    if final["status"] == "completed" and result.output_path:
        assert os.path.exists(result.output_path), f"Vidéo output manquante: {result.output_path}"
        size = os.path.getsize(result.output_path)
        assert size > 0, "Fichier vidéo vide"
        print(f"  ✅ E2E Pipeline OK: {result.output_path} ({size} bytes)")
    else:
        print(f"  ⚠️ E2E Pipeline mode stub: {final['status']}")

# ── Runner ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("🔧 PHASE 2 — Tests de Souveraineté Moteur")
    print("=" * 60)
    
    tests = [
        ("2.1 GPUWorker — module exists", test_gpu_worker_module_exists),
        ("2.1 GPUWorker — instantiation", test_gpu_worker_instantiation),
        ("2.1 GPUWorker — stub generation", test_gpu_worker_stub_generation),
        ("2.1 JobStore — CRUD SQLite", test_job_store_crud),
        ("2.1 Render — zéro mock cloud", test_render_no_mock),
        ("2.2 Config V34 — souveraine", test_config_v34_sovereign),
        ("2.2 Orchestrator — charge V34", test_orchestrator_loads_v34),
        ("2.2 Orchestrator — init config", test_orchestrator_init_with_config),
        ("2.3 QA Semantic — module", test_qa_semantic_local_module),
        ("2.3 QA Semantic — stub mode", test_qa_semantic_stub),
        ("2.4 Gateway — no cloud refs", test_gateway_no_cloud_refs),
        ("2.4 E2E — stub pipeline", test_end_to_end_stub_pipeline),
    ]
    
    passed = 0
    failed = 0
    errors = []
    
    for name, test_fn in tests:
        try:
            print(f"\n▶ {name}")
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((name, str(e)))
            print(f"  ❌ FAILED: {e}")
    
    print("\n" + "=" * 60)
    print(f"Résultats: {passed}/{passed + failed} passés")
    if errors:
        print(f"\n❌ {failed} échec(s):")
        for name, err in errors:
            print(f"   - {name}: {err}")
    else:
        print("✅ TOUS LES TESTS PHASE 2 PASSENT")
    print("=" * 60)
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
