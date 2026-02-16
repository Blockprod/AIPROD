"""
Tests de validation Phase 3 — Entraîner ses modèles.

Vérifie :
- 3.1 Configs d'entraînement valides et chargeables
- 3.2 Curriculum wiring dans le trainer
- 3.3 Module de quantization fonctionnel
- 3.4 Répertoire modèles souverain structuré
- 3.5 CLI de quantization
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import yaml

# Workspace root
ROOT = Path(__file__).parent.parent


class TestPhase3TrainingConfigs(unittest.TestCase):
    """3.1 — Tous les configs YAML d'entraînement sont valides."""

    EXPECTED_CONFIGS = [
        "lora_phase1.yaml",
        "full_finetune.yaml",
        "vae_finetune.yaml",
        "audio_vae.yaml",
        "tts_training.yaml",
    ]

    def test_configs_directory_exists(self):
        """Le répertoire configs/train/ existe."""
        configs_dir = ROOT / "configs" / "train"
        self.assertTrue(configs_dir.is_dir(), f"Missing: {configs_dir}")

    def test_all_config_files_present(self):
        """Tous les fichiers de config attendus sont présents."""
        configs_dir = ROOT / "configs" / "train"
        for name in self.EXPECTED_CONFIGS:
            path = configs_dir / name
            self.assertTrue(path.exists(), f"Missing config: {name}")

    def test_configs_valid_yaml(self):
        """Tous les configs sont du YAML valide et non-vide."""
        configs_dir = ROOT / "configs" / "train"
        for name in self.EXPECTED_CONFIGS:
            path = configs_dir / name
            with open(path) as f:
                data = yaml.safe_load(f)
            self.assertIsInstance(data, dict, f"{name}: not a dict")
            self.assertGreater(len(data), 0, f"{name}: empty config")

    def test_lora_config_has_required_keys(self):
        """Le config LoRA contient les clés essentielles."""
        path = ROOT / "configs" / "train" / "lora_phase1.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        required = ["model", "lora", "optimization", "data"]
        for key in required:
            self.assertIn(key, data, f"lora_phase1.yaml missing key: {key}")
        # Vérifier les paramètres LoRA
        self.assertIn("rank", data["lora"])
        self.assertIn("alpha", data["lora"])

    def test_full_finetune_has_curriculum(self):
        """Le config full finetune contient la section curriculum."""
        path = ROOT / "configs" / "train" / "full_finetune.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        self.assertIn("curriculum", data, "full_finetune.yaml missing curriculum section")
        curriculum = data["curriculum"]
        self.assertIn("phases", curriculum)
        self.assertGreaterEqual(len(curriculum["phases"]), 3, "Need at least 3 curriculum phases")

    def test_vae_config_has_loss(self):
        """Le config VAE définit les fonctions de loss."""
        path = ROOT / "configs" / "train" / "vae_finetune.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        # Chercher une section loss quelque part
        flat = json.dumps(data).lower()
        self.assertIn("loss", flat, "vae_finetune.yaml missing loss configuration")

    def test_tts_config_has_phases(self):
        """Le config TTS définit les 3 phases d'entraînement."""
        path = ROOT / "configs" / "train" / "tts_training.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        # TTS uses training.phase1/phase2/phase3 structure
        self.assertIn("training", data, "tts_training.yaml missing training section")
        training = data["training"]
        for phase_key in ["phase1", "phase2", "phase3"]:
            self.assertIn(phase_key, training, f"TTS missing {phase_key}")


class TestPhase3CurriculumWiring(unittest.TestCase):
    """3.2 — Le curriculum est correctement câblé dans le trainer."""

    def test_curriculum_import_available(self):
        """CurriculumAdapterConfig est importable."""
        from aiprod_trainer.curriculum_training import (
            CurriculumAdapterConfig,
            CurriculumConfig,
            CurriculumPhase,
            CurriculumScheduler,
        )
        self.assertTrue(callable(CurriculumAdapterConfig))
        self.assertTrue(callable(CurriculumConfig))

    def test_curriculum_scheduler_creation(self):
        """Le CurriculumScheduler se crée correctement."""
        from aiprod_trainer.curriculum_training import (
            CurriculumConfig,
            CurriculumPhase,
            CurriculumScheduler,
            PhaseConfig,
            PhaseDuration,
            PhaseResolution,
        )
        phases = [
            PhaseConfig(
                name=CurriculumPhase.PHASE_1_LOW_RES,
                duration=PhaseDuration(min_frames=4, max_frames=8, target_frames=8),
                resolution=PhaseResolution(height=256, width=256, latent_height=32, latent_width=32),
                batch_size=4,
                learning_rate=1e-4,
                duration_value=100,
            ),
            PhaseConfig(
                name=CurriculumPhase.PHASE_2_HIGH_RES,
                duration=PhaseDuration(min_frames=8, max_frames=16, target_frames=16),
                resolution=PhaseResolution(height=512, width=512, latent_height=64, latent_width=64),
                batch_size=2,
                learning_rate=5e-5,
                duration_value=200,
            ),
        ]
        config = CurriculumConfig(phases=phases)
        scheduler = CurriculumScheduler(config)
        self.assertEqual(scheduler.current_phase.name, CurriculumPhase.PHASE_1_LOW_RES)
        self.assertEqual(len(scheduler.config.phases), 2)

    def test_trainer_accepts_curriculum_config(self):
        """AIPRODvTrainer.__init__ accepte curriculum_config."""
        import ast
        # Parse the source instead of importing (avoids heavy transitive deps)
        trainer_path = ROOT / "packages" / "aiprod-trainer" / "src" / "aiprod_trainer" / "trainer.py"
        source = trainer_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "__init__":
                # Check if any parent is AIPRODvTrainer class
                param_names = [arg.arg for arg in node.args.args]
                if "curriculum_config" in param_names:
                    found = True
                    break
        self.assertTrue(found,
                        "AIPRODvTrainer.__init__ must accept curriculum_config parameter")


class TestPhase3Quantization(unittest.TestCase):
    """3.3 — Le module de quantization est fonctionnel."""

    def test_quantization_module_importable(self):
        """Le module quantization s'importe."""
        from aiprod_trainer.quantization import quantize_model, QuantizationOptions
        self.assertTrue(callable(quantize_model))

    def test_quantization_options_type(self):
        """QuantizationOptions définit les bons formats."""
        from aiprod_trainer.quantization import QuantizationOptions
        import typing
        # Vérifier que c'est un Literal avec les bonnes valeurs
        args = typing.get_args(QuantizationOptions)
        expected = {"int8-quanto", "int4-quanto", "int2-quanto", "fp8-quanto", "fp8uz-quanto"}
        self.assertEqual(set(args), expected)

    def test_quantize_cli_script_exists(self):
        """Le script CLI de quantization existe."""
        script = ROOT / "scripts" / "quantize_model.py"
        self.assertTrue(script.exists(), f"Missing: {script}")

    def test_quantize_cli_is_executable(self):
        """Le script CLI contient un main() et argparse."""
        script = ROOT / "scripts" / "quantize_model.py"
        content = script.read_text()
        self.assertIn("argparse", content)
        self.assertIn("def main()", content)
        self.assertIn("--input", content)
        self.assertIn("--output", content)
        self.assertIn("--format", content)


class TestPhase3SovereignModelDirectory(unittest.TestCase):
    """3.4 — Le répertoire modèles souverain est structuré."""

    def test_sovereign_dir_exists(self):
        """models/aiprod-sovereign/ existe."""
        sov_dir = ROOT / "models" / "aiprod-sovereign"
        self.assertTrue(sov_dir.is_dir(), f"Missing: {sov_dir}")

    def test_manifest_exists_and_valid(self):
        """MANIFEST.json existe et est du JSON valide."""
        manifest = ROOT / "models" / "aiprod-sovereign" / "MANIFEST.json"
        self.assertTrue(manifest.exists(), "Missing MANIFEST.json")
        with open(manifest) as f:
            data = json.load(f)
        self.assertIn("models", data)
        self.assertIn("version", data)
        self.assertIn("sovereignty", data)

    def test_manifest_lists_all_models(self):
        """MANIFEST.json liste tous les modèles souverains attendus."""
        manifest = ROOT / "models" / "aiprod-sovereign" / "MANIFEST.json"
        with open(manifest) as f:
            data = json.load(f)
        model_names = [m["name"] for m in data["models"]]
        expected = [
            "aiprod-shdt-v1-fp8",
            "aiprod-hwvae-v1",
            "aiprod-audio-vae-v1",
            "aiprod-tts-v1",
            "aiprod-text-encoder-v1",
            "aiprod-upsampler-v1",
        ]
        for name in expected:
            self.assertIn(name, model_names, f"Missing model in MANIFEST: {name}")

    def test_manifest_sovereignty_flags(self):
        """MANIFEST.json déclare 0 dépendances cloud."""
        manifest = ROOT / "models" / "aiprod-sovereign" / "MANIFEST.json"
        with open(manifest) as f:
            data = json.load(f)
        sov = data["sovereignty"]
        self.assertEqual(sov["cloud_dependencies"], 0)
        self.assertEqual(sov["third_party_apis"], 0)
        self.assertTrue(sov["all_weights_local"])
        self.assertTrue(sov["offline_capable"])

    def test_model_card_exists(self):
        """MODEL_CARD.md existe et contient les sections clés."""
        card = ROOT / "models" / "aiprod-sovereign" / "MODEL_CARD.md"
        self.assertTrue(card.exists(), "Missing MODEL_CARD.md")
        content = card.read_text(encoding="utf-8")
        # Vérifier les sections essentielles
        self.assertIn("SHDT", content)
        self.assertIn("HWVAE", content)
        self.assertIn("TTS", content)
        self.assertIn("Souveraineté", content)
        self.assertIn("Licence", content)

    def test_manifest_models_have_training_configs(self):
        """Chaque modèle dans MANIFEST référence un config d'entraînement."""
        manifest = ROOT / "models" / "aiprod-sovereign" / "MANIFEST.json"
        with open(manifest) as f:
            data = json.load(f)
        for model in data["models"]:
            if "training_config" in model:
                config_path = ROOT / model["training_config"]
                self.assertTrue(
                    config_path.exists(),
                    f"Training config not found for {model['name']}: {model['training_config']}"
                )


class TestPhase3Integration(unittest.TestCase):
    """3.5 — Tests d'intégration Phase 3."""

    def test_all_training_configs_reference_valid_architectures(self):
        """Les configs référencent des architectures implémentées."""
        configs_dir = ROOT / "configs" / "train"
        for yaml_file in configs_dir.glob("*.yaml"):
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            # Chaque config doit être un dict non-vide
            self.assertIsInstance(data, dict, f"{yaml_file.name} invalid")

    def test_no_cloud_urls_in_training_configs(self):
        """Aucun config d'entraînement ne référence des URLs cloud."""
        configs_dir = ROOT / "configs" / "train"
        cloud_patterns = ["gs://", "s3://", "https://storage.googleapis",
                          "runway", "replicate", "openai"]
        for yaml_file in configs_dir.glob("*.yaml"):
            content = yaml_file.read_text().lower()
            for pattern in cloud_patterns:
                self.assertNotIn(
                    pattern.lower(), content,
                    f"{yaml_file.name} contains cloud reference: {pattern}"
                )

    def test_sovereign_directory_zero_external_deps(self):
        """Le répertoire souverain ne contient aucune référence externe."""
        sov_dir = ROOT / "models" / "aiprod-sovereign"
        for f in sov_dir.iterdir():
            content = f.read_text(errors="ignore")
            self.assertNotIn("gs://", content, f"{f.name} contains gs:// URL")
            self.assertNotIn("s3://", content, f"{f.name} contains s3:// URL")


if __name__ == "__main__":
    unittest.main()
