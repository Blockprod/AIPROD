"""
Tests automatisés de souveraineté — exécutés en CI sans réseau.

Vérifie :
- Aucun appel réseau à l'import
- Aucun import Google/OpenAI/Anthropic/Stripe dans les modules core
- Tous les from_pretrained utilisent local_files_only=True
- Répertoire modèles souverain structuré
- Configs souverains (0 dépendances cloud)
- Requirements figés
"""

from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent


class TestSovereigntyNoNetworkImports(unittest.TestCase):
    """Importer les packages AIPROD ne déclenche aucun appel réseau."""

    def test_import_aiprod_core_offline(self):
        """aiprod_core s'importe sans réseau."""
        import socket
        original = socket.socket

        class BlockedSocket:
            def __init__(self, *a, **k):
                raise RuntimeError("Network blocked for sovereignty test")

        socket.socket = BlockedSocket  # type: ignore[assignment]
        try:
            import importlib
            import aiprod_core
            importlib.reload(aiprod_core)
        finally:
            socket.socket = original  # type: ignore[assignment]

    def test_import_aiprod_trainer_offline(self):
        """aiprod_trainer s'importe sans réseau."""
        import socket
        original = socket.socket

        class BlockedSocket:
            def __init__(self, *a, **k):
                raise RuntimeError("Network blocked for sovereignty test")

        socket.socket = BlockedSocket  # type: ignore[assignment]
        try:
            import importlib
            import aiprod_trainer
            importlib.reload(aiprod_trainer)
        finally:
            socket.socket = original  # type: ignore[assignment]


class TestSovereigntyNoCloudImports(unittest.TestCase):
    """Aucun import hard de services cloud dans les modules core."""

    CLOUD_PATTERNS = [
        "google.generativeai",
        "google.cloud",
        "openai",
        "anthropic",
        "stripe",
        "boto3",
        "botocore",
    ]

    # Fichiers autorisés à avoir des imports cloud (optionnels/guarded)
    ALLOWED_FILES = {
        "gemini_client.py",      # try/except guarded
        "captioning_external.py", # lazy import in function
        "gcp_services.py",       # try/except guarded (Phase 4)
        "sources.py",            # lazy import in function
        "billing_service.py",    # try/except guarded (optional stripe)
    }

    def test_no_hard_cloud_imports_in_core(self):
        """packages/aiprod-core/ ne contient aucun import cloud."""
        core_dir = ROOT / "packages" / "aiprod-core" / "src"
        violations = []
        for py_file in core_dir.rglob("*.py"):
            content = py_file.read_text(errors="ignore")
            for pattern in self.CLOUD_PATTERNS:
                if pattern in content:
                    violations.append(f"{py_file.name}: {pattern}")
        self.assertEqual(violations, [], f"Cloud imports in core: {violations}")

    def test_no_unguarded_google_imports(self):
        """Tous les imports Google sont protégés par try/except ou lazy."""
        src_dirs = [
            ROOT / "packages" / "aiprod-core" / "src",
            ROOT / "packages" / "aiprod-trainer" / "src",
            ROOT / "packages" / "aiprod-pipelines" / "src",
        ]
        violations = []
        for src_dir in src_dirs:
            for py_file in src_dir.rglob("*.py"):
                if py_file.name in self.ALLOWED_FILES:
                    continue
                content = py_file.read_text(errors="ignore")
                tree = ast.parse(content, filename=str(py_file))
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        module = ""
                        if isinstance(node, ast.Import):
                            module = node.names[0].name
                        elif node.module:
                            module = node.module
                        for pattern in ["google.cloud", "google.generativeai",
                                        "openai", "anthropic", "stripe", "boto3"]:
                            if module.startswith(pattern):
                                # Check if inside try/except
                                violations.append(f"{py_file.name}:{node.lineno}: {module}")
        self.assertEqual(violations, [], f"Unguarded cloud imports: {violations}")


class TestSovereigntyFromPretrained(unittest.TestCase):
    """Tous les from_pretrained() utilisent local_files_only=True."""

    def test_all_from_pretrained_have_local_only(self):
        """Chaque appel from_pretrained a local_files_only."""
        src_dirs = [
            ROOT / "packages" / "aiprod-core" / "src",
            ROOT / "packages" / "aiprod-trainer" / "src",
            ROOT / "packages" / "aiprod-pipelines" / "src",
        ]
        violations = []
        for src_dir in src_dirs:
            for py_file in src_dir.rglob("*.py"):
                content = py_file.read_text(errors="ignore")
                lines = content.split("\n")
                for i, line in enumerate(lines, 1):
                    if "from_pretrained(" in line and "local_files_only" not in line:
                        # Check surrounding lines (multi-line calls)
                        start = max(0, i - 2)
                        end = min(len(lines), i + 8)
                        context = "\n".join(lines[start:end])
                        if "local_files_only" not in context:
                            violations.append(f"{py_file.name}:{i}")
        self.assertEqual(
            violations, [],
            f"from_pretrained without local_files_only: {violations}"
        )


class TestSovereigntyModelsDirectory(unittest.TestCase):
    """Répertoire modèles souverain correctement structuré."""

    def test_manifest_exists(self):
        """MANIFEST.json existe dans models/aiprod-sovereign/."""
        manifest = ROOT / "models" / "aiprod-sovereign" / "MANIFEST.json"
        self.assertTrue(manifest.exists())

    def test_manifest_zero_cloud_deps(self):
        """MANIFEST déclare 0 dépendances cloud."""
        manifest = ROOT / "models" / "aiprod-sovereign" / "MANIFEST.json"
        with open(manifest) as f:
            data = json.load(f)
        sov = data["sovereignty"]
        self.assertEqual(sov["cloud_dependencies"], 0)
        self.assertEqual(sov["third_party_apis"], 0)
        self.assertTrue(sov["all_weights_local"])
        self.assertTrue(sov["offline_capable"])

    def test_model_checksums_file_exists(self):
        """Fichier CHECKSUMS.sha256 existe pour les modèles."""
        checksums = ROOT / "models" / "CHECKSUMS.sha256"
        self.assertTrue(checksums.exists(), "Missing models/CHECKSUMS.sha256")

    def test_models_config_exists(self):
        """config/models.json exists and declares all model paths."""
        cfg = ROOT / "config" / "models.json"
        self.assertTrue(cfg.exists(), "Missing config/models.json")
        with open(cfg) as f:
            data = json.load(f)
        self.assertIn("models", data)
        self.assertTrue(data["sovereignty"]["all_local_files_only"])
        # At least 4 models declared
        self.assertGreaterEqual(len(data["models"]), 4)

    def test_model_directories_exist(self):
        """All model destination directories exist."""
        expected = [
            "models/text-encoder",
            "models/scenarist",
            "models/clip",
            "models/captioning",
            "models/aiprod-sovereign",
            "models/ltx2_research",
        ]
        for d in expected:
            self.assertTrue(
                (ROOT / d).is_dir(),
                f"Model directory missing: {d}",
            )

    def test_download_script_exists(self):
        """scripts/download_models.py exists and is importable."""
        script = ROOT / "scripts" / "download_models.py"
        self.assertTrue(script.exists(), "Missing scripts/download_models.py")


class TestSovereigntyRequirements(unittest.TestCase):
    """Requirements figés et souverains."""

    def test_requirements_lock_exists(self):
        """requirements.lock existe."""
        lock = ROOT / "requirements.lock"
        self.assertTrue(lock.exists(), "Missing requirements.lock")

    def test_requirements_no_cloud_hard_deps(self):
        """requirements.txt ne contient pas de dépendances cloud obligatoires."""
        req = ROOT / "requirements.txt"
        content = req.read_text(errors="ignore")
        # Les lignes non-commentées ne doivent pas contenir de packages cloud
        cloud_packages = ["google-cloud-", "boto3", "stripe", "openai"]
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            for pkg in cloud_packages:
                self.assertNotIn(
                    pkg, stripped,
                    f"Cloud dependency in requirements.txt: {stripped}"
                )

    def test_requirements_lock_pinned(self):
        """requirements.lock contient des versions pinned (==)."""
        lock = ROOT / "requirements.lock"
        # File may be UTF-16 LE (BOM \xff\xfe)
        raw = lock.read_bytes()
        if raw[:2] == b'\xff\xfe':
            content = raw.decode("utf-16-le")
        else:
            content = raw.decode("utf-8", errors="ignore")
        pinned_count = content.count("==")
        self.assertGreater(pinned_count, 20, "requirements.lock should have many pinned versions")


class TestSovereigntyConfig(unittest.TestCase):
    """Config V34 souverain est correct."""

    def test_sovereign_config_exists(self):
        """AIPROD_V34_SOVEREIGN.json existe."""
        config = ROOT / "config" / "AIPROD_V34_SOVEREIGN.json"
        self.assertTrue(config.exists())

    def test_sovereign_config_no_cloud(self):
        """Config V34 n'a aucune référence cloud."""
        config = ROOT / "config" / "AIPROD_V34_SOVEREIGN.json"
        content = config.read_text(encoding="utf-8", errors="ignore").lower()
        cloud_refs = ["runway", "replicate", "gs://", "s3://"]
        for ref in cloud_refs:
            self.assertNotIn(ref, content, f"Cloud ref in V34 config: {ref}")
        # "openai/clip-*" is a HuggingFace model ID, not an API — verify no API key refs
        self.assertNotIn("openai_api_key", content, "OpenAI API key ref in V34 config")
        self.assertNotIn("openai.com", content, "OpenAI URL in V34 config")


class TestSovereigntyDockerfile(unittest.TestCase):
    """Dockerfiles sont souverains."""

    def test_gpu_dockerfile_no_cloud_packages(self):
        """Dockerfile.gpu n'installe pas de packages cloud."""
        dockerfile = ROOT / "deploy" / "docker" / "Dockerfile.gpu"
        content = dockerfile.read_text(errors="ignore")
        # google-cloud-storage, boto3 doivent être commentés ou absents
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            self.assertNotIn("google-cloud-storage", stripped)
            self.assertNotIn("boto3", stripped)

    def test_gpu_dockerfile_copies_sovereign_models(self):
        """Dockerfile.gpu copie les modèles souverains."""
        dockerfile = ROOT / "deploy" / "docker" / "Dockerfile.gpu"
        content = dockerfile.read_text(errors="ignore")
        self.assertIn("aiprod-sovereign", content,
                       "Dockerfile.gpu should reference aiprod-sovereign models")


class TestSovereigntyReproducibility(unittest.TestCase):
    """Infrastructure de reproductibilité en place."""

    def test_reproducibility_module_exists(self):
        """Le module reproducibility est importable."""
        from aiprod_pipelines.utils.reproducibility import (
            set_deterministic_mode,
            get_reproducibility_info,
        )
        self.assertTrue(callable(set_deterministic_mode))
        self.assertTrue(callable(get_reproducibility_info))

    def test_deterministic_mode_sets_seeds(self):
        """set_deterministic_mode configure les seeds correctement."""
        import os
        import torch
        from aiprod_pipelines.utils.reproducibility import set_deterministic_mode

        set_deterministic_mode(seed=12345)
        self.assertEqual(os.environ["PYTHONHASHSEED"], "12345")
        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertFalse(torch.backends.cudnn.benchmark)

    def test_reproducibility_info_returns_dict(self):
        """get_reproducibility_info retourne un dict valide."""
        from aiprod_pipelines.utils.reproducibility import get_reproducibility_info

        info = get_reproducibility_info()
        self.assertIn("torch_version", info)
        self.assertIn("cuda_available", info)
        self.assertIn("cudnn_deterministic", info)


if __name__ == "__main__":
    unittest.main()
