"""
Phase 5 — Excellence & Differentiation — Unit Tests
=====================================================

Tests for:
  5.1  RLHF / DPO trainer + feedback store + model promoter
  5.2  AI Scenarist (rule-based decomposer + data model)
  5.3  Camera control (Bézier, templates, shake, conditioning)
  5.4  AIPROD v3 architecture config
  5.5  Desktop engine, plugins, on-prem server
"""

from __future__ import annotations

import json
import math
import unittest
from unittest.mock import MagicMock


# ===================================================================
# 5.1  RLHF / DPO
# ===================================================================


class TestPreferencePair(unittest.TestCase):
    def test_auto_id(self):
        from aiprod_pipelines.inference.reward_modeling.rlhf_trainer import PreferencePair

        pair = PreferencePair(
            prompt="a cat", chosen_video_id="v1", rejected_video_id="v2"
        )
        self.assertTrue(pair.pair_id)
        self.assertGreater(pair.timestamp, 0)

    def test_confidence(self):
        from aiprod_pipelines.inference.reward_modeling.rlhf_trainer import PreferencePair

        pair = PreferencePair(confidence=0.95)
        self.assertAlmostEqual(pair.confidence, 0.95)


class TestFeedbackRecord(unittest.TestCase):
    def test_auto_fields(self):
        from aiprod_pipelines.inference.reward_modeling.rlhf_trainer import FeedbackRecord

        fb = FeedbackRecord(tenant_id="t1", job_id="j1", rating=4.5)
        self.assertTrue(fb.record_id)
        self.assertEqual(fb.rating, 4.5)


class TestFeedbackStore(unittest.TestCase):
    def test_add_and_count(self):
        from aiprod_pipelines.inference.reward_modeling.rlhf_trainer import (
            FeedbackStore, FeedbackRecord, PreferencePair,
        )

        store = FeedbackStore()
        store.add_feedback(FeedbackRecord(job_id="j1", rating=5.0))
        store.add_feedback(FeedbackRecord(job_id="j2", rating=2.0))
        self.assertEqual(store.num_feedback, 2)

    def test_add_pair(self):
        from aiprod_pipelines.inference.reward_modeling.rlhf_trainer import (
            FeedbackStore, PreferencePair,
        )

        store = FeedbackStore()
        store.add_pair(PreferencePair(chosen_video_id="v1", rejected_video_id="v2"))
        self.assertEqual(store.num_pairs, 1)

    def test_filter_pairs(self):
        from aiprod_pipelines.inference.reward_modeling.rlhf_trainer import (
            FeedbackStore, PreferencePair, FeedbackSource,
        )

        store = FeedbackStore()
        store.add_pair(PreferencePair(confidence=0.9, source=FeedbackSource.HUMAN))
        store.add_pair(PreferencePair(confidence=0.3, source=FeedbackSource.AUTOMATED))

        human_only = store.get_pairs(source=FeedbackSource.HUMAN)
        self.assertEqual(len(human_only), 1)

        high_conf = store.get_pairs(min_confidence=0.5)
        self.assertEqual(len(high_conf), 1)

    def test_auto_generate_pairs(self):
        from aiprod_pipelines.inference.reward_modeling.rlhf_trainer import (
            FeedbackStore, FeedbackRecord,
        )

        store = FeedbackStore()
        store.add_feedback(FeedbackRecord(job_id="aaaa_j1", rating=5.0))
        store.add_feedback(FeedbackRecord(job_id="aaaa_j2", rating=1.0))
        count = store.auto_generate_pairs(rating_gap=1.5)
        self.assertGreater(count, 0)
        self.assertGreater(store.num_pairs, 0)


class TestDPOConfig(unittest.TestCase):
    def test_defaults(self):
        from aiprod_pipelines.inference.reward_modeling.rlhf_trainer import DPOConfig

        cfg = DPOConfig()
        self.assertEqual(cfg.beta, 0.1)
        self.assertEqual(cfg.max_steps, 1000)


class TestDPOTrainer(unittest.TestCase):
    def test_init_no_torch(self):
        from aiprod_pipelines.inference.reward_modeling.rlhf_trainer import DPOTrainer

        trainer = DPOTrainer(policy_model=MagicMock(), reference_model=MagicMock())
        self.assertEqual(trainer._step_count, 0)

    def test_metrics(self):
        from aiprod_pipelines.inference.reward_modeling.rlhf_trainer import DPOTrainer

        trainer = DPOTrainer(policy_model=MagicMock())
        m = trainer.metrics
        self.assertEqual(m["step"], 0)


class TestPPOConfig(unittest.TestCase):
    def test_defaults(self):
        from aiprod_pipelines.inference.reward_modeling.rlhf_trainer import PPOConfig

        cfg = PPOConfig()
        self.assertEqual(cfg.clip_epsilon, 0.2)
        self.assertEqual(cfg.kl_penalty, 0.1)


class TestPPOTrainer(unittest.TestCase):
    def test_init(self):
        from aiprod_pipelines.inference.reward_modeling.rlhf_trainer import PPOTrainer

        trainer = PPOTrainer(
            policy_model=MagicMock(),
            reward_model=MagicMock(),
        )
        self.assertEqual(trainer._step_count, 0)

    def test_metrics_empty(self):
        from aiprod_pipelines.inference.reward_modeling.rlhf_trainer import PPOTrainer

        trainer = PPOTrainer(policy_model=MagicMock(), reward_model=MagicMock())
        self.assertEqual(trainer.metrics["step"], 0)


class TestModelCandidate(unittest.TestCase):
    def test_auto_id(self):
        from aiprod_pipelines.inference.reward_modeling.rlhf_trainer import ModelCandidate

        c = ModelCandidate(name="v3-dpo", reward_score=0.85)
        self.assertTrue(c.model_id)


class TestModelPromoter(unittest.TestCase):
    def test_register_and_evaluate(self):
        from aiprod_pipelines.inference.reward_modeling.rlhf_trainer import (
            ModelPromoter, ModelCandidate,
        )

        promoter = ModelPromoter(min_evaluations=10, min_preference_rate=0.5)
        c1 = ModelCandidate(
            name="v1", reward_score=0.8, human_preference_rate=0.6, num_evaluations=50
        )
        promoter.register(c1)
        winner = promoter.evaluate()
        self.assertEqual(winner, c1.model_id)

    def test_not_enough_evaluations(self):
        from aiprod_pipelines.inference.reward_modeling.rlhf_trainer import (
            ModelPromoter, ModelCandidate,
        )

        promoter = ModelPromoter(min_evaluations=100)
        c = ModelCandidate(num_evaluations=5, human_preference_rate=0.9, reward_score=0.9)
        promoter.register(c)
        self.assertIsNone(promoter.evaluate())

    def test_promote(self):
        from aiprod_pipelines.inference.reward_modeling.rlhf_trainer import (
            ModelPromoter, ModelCandidate,
        )

        promoter = ModelPromoter()
        c = ModelCandidate(name="best")
        promoter.register(c)
        ok = promoter.promote(c.model_id)
        self.assertTrue(ok)
        self.assertEqual(promoter.production_model_id, c.model_id)

    def test_promotion_history(self):
        from aiprod_pipelines.inference.reward_modeling.rlhf_trainer import (
            ModelPromoter, ModelCandidate,
        )

        promoter = ModelPromoter()
        c = ModelCandidate()
        promoter.register(c)
        promoter.promote(c.model_id)
        self.assertEqual(len(promoter.promotion_history), 1)


# ===================================================================
# 5.2  AI Scenarist
# ===================================================================


class TestSceneShot(unittest.TestCase):
    def test_to_dict(self):
        from aiprod_pipelines.inference.scenarist import SceneShot, ShotType, CameraMove

        shot = SceneShot(
            description="A wide shot of a sunset over the ocean",
            shot_type=ShotType.WIDE,
            camera_move=CameraMove.PAN_RIGHT,
            duration_sec=5.0,
        )
        d = shot.to_dict()
        self.assertEqual(d["shot_type"], "wide")
        self.assertEqual(d["camera_move"], "pan_right")
        self.assertEqual(d["duration_sec"], 5.0)


class TestScene(unittest.TestCase):
    def test_total_duration(self):
        from aiprod_pipelines.inference.scenarist import Scene, SceneShot

        scene = Scene(
            title="Test",
            shots=[
                SceneShot(duration_sec=3.0),
                SceneShot(duration_sec=5.0),
            ],
        )
        self.assertEqual(scene.total_duration, 8.0)


class TestStoryboard(unittest.TestCase):
    def test_to_json(self):
        from aiprod_pipelines.inference.scenarist import Storyboard, Scene, SceneShot

        sb = Storyboard(
            title="Test Film",
            genre="cinematic",
            scenes=[
                Scene(title="S1", shots=[SceneShot(description="shot 1", duration_sec=3.0)]),
            ],
        )
        j = sb.to_json()
        data = json.loads(j)
        self.assertEqual(data["title"], "Test Film")
        self.assertEqual(len(data["scenes"]), 1)

    def test_recalculate_duration(self):
        from aiprod_pipelines.inference.scenarist import Storyboard, Scene, SceneShot

        sb = Storyboard(scenes=[
            Scene(shots=[SceneShot(duration_sec=4.0), SceneShot(duration_sec=6.0)]),
        ])
        dur = sb.recalculate_duration()
        self.assertEqual(dur, 10.0)


class TestRuleBasedDecomposer(unittest.TestCase):
    def test_simple_decomposition(self):
        from aiprod_pipelines.inference.scenarist import RuleBasedDecomposer, CreativeConfig

        decomp = RuleBasedDecomposer()
        sb = decomp.decompose(
            "A vast landscape at sunset. A person walking towards the horizon. "
            "Close up on their face showing determination.",
            CreativeConfig(target_duration_sec=15.0),
        )
        self.assertGreater(len(sb.scenes), 0)
        total_shots = sum(len(s.shots) for s in sb.scenes)
        self.assertGreaterEqual(total_shots, 3)

    def test_shot_type_detection(self):
        from aiprod_pipelines.inference.scenarist import RuleBasedDecomposer, ShotType

        decomp = RuleBasedDecomposer()
        self.assertEqual(decomp._detect_shot_type("A vast landscape"), ShotType.WIDE)
        self.assertEqual(decomp._detect_shot_type("Close up on face"), ShotType.CLOSE_UP)

    def test_camera_detection(self):
        from aiprod_pipelines.inference.scenarist import RuleBasedDecomposer, CameraMove

        decomp = RuleBasedDecomposer()
        self.assertEqual(decomp._detect_camera("Camera follows the character"), CameraMove.TRACKING)

    def test_empty_prompt(self):
        from aiprod_pipelines.inference.scenarist import RuleBasedDecomposer

        decomp = RuleBasedDecomposer()
        sb = decomp.decompose("")
        self.assertGreater(len(sb.scenes), 0)


class TestLLMScenarist(unittest.TestCase):
    def test_fallback_no_llm(self):
        from aiprod_pipelines.inference.scenarist import LLMScenarist, CreativeConfig

        scenarist = LLMScenarist()
        # Without a loaded model, should use fallback
        sb = scenarist.generate_storyboard(
            "A robot explores an abandoned city at night",
            CreativeConfig(target_duration_sec=10.0),
        )
        self.assertGreater(len(sb.scenes), 0)

    def test_unload_model(self):
        from aiprod_pipelines.inference.scenarist import LLMScenarist

        scenarist = LLMScenarist()
        scenarist.unload_model()  # should not raise


class TestCreativeConfig(unittest.TestCase):
    def test_defaults(self):
        from aiprod_pipelines.inference.scenarist import CreativeConfig

        cfg = CreativeConfig()
        self.assertEqual(cfg.genre, "cinematic")
        self.assertEqual(cfg.target_duration_sec, 30.0)


# ===================================================================
# 5.3  Camera Control
# ===================================================================


class TestCameraState(unittest.TestCase):
    def test_to_vector(self):
        from aiprod_core.camera_control import CameraState

        state = CameraState(x=1, y=2, z=3, yaw=10, pitch=5, roll=0, fov=50, focus_distance=5)
        v = state.to_vector()
        self.assertEqual(len(v), 8)
        self.assertEqual(v[0], 1)

    def test_lerp(self):
        from aiprod_core.camera_control import CameraState

        a = CameraState(x=0, fov=30)
        b = CameraState(x=10, fov=60)
        mid = CameraState.lerp(a, b, 0.5)
        self.assertAlmostEqual(mid.x, 5.0)
        self.assertAlmostEqual(mid.fov, 45.0)


class TestBezierTrajectory(unittest.TestCase):
    def test_single_point(self):
        from aiprod_core.camera_control import BezierTrajectory, CameraState

        traj = BezierTrajectory()
        traj.add_point(CameraState(x=5))
        state = traj.evaluate(0.5)
        self.assertAlmostEqual(state.x, 5.0)

    def test_linear(self):
        from aiprod_core.camera_control import BezierTrajectory, CameraState

        traj = BezierTrajectory()
        traj.add_point(CameraState(x=0))
        traj.add_point(CameraState(x=10))
        state = traj.evaluate(0.5)
        self.assertAlmostEqual(state.x, 5.0, places=2)

    def test_cubic(self):
        from aiprod_core.camera_control import BezierTrajectory, CameraState

        traj = BezierTrajectory()
        traj.add_point(CameraState(x=0))
        traj.add_point(CameraState(x=2))
        traj.add_point(CameraState(x=8))
        traj.add_point(CameraState(x=10))
        start = traj.evaluate(0.0)
        end = traj.evaluate(1.0)
        self.assertAlmostEqual(start.x, 0.0, places=2)
        self.assertAlmostEqual(end.x, 10.0, places=2)

    def test_sample(self):
        from aiprod_core.camera_control import BezierTrajectory, CameraState

        traj = BezierTrajectory()
        traj.add_point(CameraState(x=0))
        traj.add_point(CameraState(x=10))
        frames = traj.sample(5)
        self.assertEqual(len(frames), 5)

    def test_order(self):
        from aiprod_core.camera_control import BezierTrajectory, CameraState

        traj = BezierTrajectory()
        traj.add_point(CameraState())
        traj.add_point(CameraState())
        traj.add_point(CameraState())
        self.assertEqual(traj.order, 2)


class TestCameraShake(unittest.TestCase):
    def test_no_shake(self):
        from aiprod_core.camera_control import CameraShake, ShakeConfig, ShakeMode, CameraState

        shake = CameraShake(ShakeConfig(mode=ShakeMode.NONE))
        states = [CameraState(yaw=0) for _ in range(10)]
        result = shake.apply(states)
        for s in result:
            self.assertAlmostEqual(s.yaw, 0.0)

    def test_handheld_adds_noise(self):
        from aiprod_core.camera_control import CameraShake, ShakeConfig, ShakeMode, CameraState

        shake = CameraShake(ShakeConfig(mode=ShakeMode.HANDHELD, intensity=1.0))
        states = [CameraState(yaw=0) for _ in range(24)]
        result = shake.apply(states, fps=24.0)
        # At least some frames should have non-zero yaw
        any_nonzero = any(abs(s.yaw) > 0.001 for s in result)
        self.assertTrue(any_nonzero)

    def test_stabilised_small_noise(self):
        from aiprod_core.camera_control import CameraShake, ShakeConfig, ShakeMode, CameraState

        shake = CameraShake(ShakeConfig(mode=ShakeMode.STABILISED, intensity=1.0))
        states = [CameraState(yaw=0) for _ in range(24)]
        result = shake.apply(states, fps=24.0)
        max_yaw = max(abs(s.yaw) for s in result)
        self.assertLess(max_yaw, 1.0)  # small perturbation


class TestCinematicTemplates(unittest.TestCase):
    def test_all_templates(self):
        from aiprod_core.camera_control import build_template, CinematicTemplate

        for tmpl in CinematicTemplate:
            states = build_template(tmpl, num_frames=24, duration_sec=2.0)
            self.assertEqual(len(states), 24, f"Template {tmpl.value} failed")

    def test_dolly_zoom_fov_increases(self):
        from aiprod_core.camera_control import build_template, CinematicTemplate

        states = build_template(CinematicTemplate.DOLLY_ZOOM, num_frames=24)
        self.assertGreater(states[-1].fov, states[0].fov)


class TestCameraConditioningSignal(unittest.TestCase):
    def test_to_matrix(self):
        from aiprod_core.camera_control import CameraConditioningSignal, CameraState

        signal = CameraConditioningSignal(
            states=[CameraState(x=i) for i in range(5)]
        )
        mat = signal.to_matrix()
        self.assertEqual(len(mat), 5)
        self.assertEqual(len(mat[0]), 8)

    def test_to_tensor(self):
        from aiprod_core.camera_control import CameraConditioningSignal, CameraState

        signal = CameraConditioningSignal(
            states=[CameraState() for _ in range(10)]
        )
        tensor = signal.to_tensor()
        self.assertIsNotNone(tensor)


class TestCameraController(unittest.TestCase):
    def test_from_template(self):
        from aiprod_core.camera_control import CameraController, CinematicTemplate

        ctrl = CameraController()
        signal = ctrl.from_template(CinematicTemplate.PUSH_IN, num_frames=48)
        self.assertEqual(signal.num_frames, 48)

    def test_from_keyframes(self):
        from aiprod_core.camera_control import CameraController, CameraState

        ctrl = CameraController()
        keyframes = [CameraState(x=0), CameraState(x=5), CameraState(x=10)]
        signal = ctrl.from_keyframes(keyframes, num_frames=30)
        self.assertEqual(signal.num_frames, 30)

    def test_from_scenarist_shot(self):
        from aiprod_core.camera_control import CameraController

        ctrl = CameraController()
        signal = ctrl.from_scenarist_shot("medium", "dolly_in", duration_sec=4.0, fps=24.0)
        self.assertEqual(signal.num_frames, 96)  # 4s × 24fps

    def test_list_templates(self):
        from aiprod_core.camera_control import CameraController

        templates = CameraController.list_templates()
        self.assertIn("orbit", templates)
        self.assertIn("dolly_zoom", templates)


# ===================================================================
# 5.4  AIPROD v3 Architecture
# ===================================================================


class TestAIPRODv3Config(unittest.TestCase):
    def test_default_config(self):
        from aiprod_core.model.transformer.aiprod_v3 import AIPRODv3Config, AIPROD_V3_BASE

        self.assertEqual(AIPROD_V3_BASE.hidden_dim, 1536)
        self.assertEqual(AIPROD_V3_BASE.num_blocks, 28)

    def test_param_estimate(self):
        from aiprod_core.model.transformer.aiprod_v3 import AIPROD_V3_BASE

        est = AIPROD_V3_BASE.total_params_estimate
        self.assertIn("B", est)  # should be in billions

    def test_small_config(self):
        from aiprod_core.model.transformer.aiprod_v3 import AIPROD_V3_SMALL

        self.assertEqual(AIPROD_V3_SMALL.hidden_dim, 768)
        self.assertEqual(AIPROD_V3_SMALL.num_blocks, 12)

    def test_large_config(self):
        from aiprod_core.model.transformer.aiprod_v3 import AIPROD_V3_LARGE

        self.assertEqual(AIPROD_V3_LARGE.hidden_dim, 2048)


class TestFlowMatchingSampler(unittest.TestCase):
    def test_schedule(self):
        from aiprod_core.model.transformer.aiprod_v3 import FlowMatchingSampler, FlowMatchingConfig

        sampler = FlowMatchingSampler(FlowMatchingConfig(num_steps=10))
        schedule = sampler.get_schedule()
        self.assertEqual(len(schedule), 11)  # steps + 1
        # Should be decreasing
        self.assertGreater(schedule[0], schedule[-1])


class TestAIPRODv3ModelStub(unittest.TestCase):
    def test_init(self):
        from aiprod_core.model.transformer.aiprod_v3 import AIPRODv3Model, AIPRODv3Config

        model = AIPRODv3Model(AIPRODv3Config(hidden_dim=256, num_blocks=2, num_heads=4))
        self.assertIsNotNone(model.config)


# ===================================================================
# 5.5  Desktop & Plugins
# ===================================================================


class TestGPUProfiles(unittest.TestCase):
    def test_all_profiles_exist(self):
        from aiprod_pipelines.inference.desktop_plugins import GPU_PROFILES, GPUProfile

        for p in GPUProfile:
            self.assertIn(p, GPU_PROFILES)

    def test_rtx_5090(self):
        from aiprod_pipelines.inference.desktop_plugins import GPU_PROFILES, GPUProfile

        cfg = GPU_PROFILES[GPUProfile.RTX_5090]
        self.assertEqual(cfg.vram_gb, 32.0)
        self.assertTrue(cfg.bf16)
        self.assertEqual(cfg.max_resolution, 3840)

    def test_generic_conservative(self):
        from aiprod_pipelines.inference.desktop_plugins import GPU_PROFILES, GPUProfile

        cfg = GPU_PROFILES[GPUProfile.GENERIC_CUDA]
        self.assertEqual(cfg.vram_gb, 8.0)
        self.assertFalse(cfg.enable_tensorrt)


class TestDesktopInferenceEngine(unittest.TestCase):
    def test_init(self):
        from aiprod_pipelines.inference.desktop_plugins import DesktopInferenceEngine, GPUProfile

        engine = DesktopInferenceEngine(GPUProfile.RTX_4090)
        self.assertEqual(engine.profile.profile, GPUProfile.RTX_4090)

    def test_load(self):
        from aiprod_pipelines.inference.desktop_plugins import DesktopInferenceEngine

        engine = DesktopInferenceEngine()
        self.assertTrue(engine.load_model())
        self.assertTrue(engine.is_loaded)

    def test_generate(self):
        from aiprod_pipelines.inference.desktop_plugins import DesktopInferenceEngine

        engine = DesktopInferenceEngine()
        engine.load_model()
        result = engine.generate("A sunset", width=1920, height=1080)
        self.assertEqual(result["status"], "completed")


class TestOFXPluginSpec(unittest.TestCase):
    def test_manifest(self):
        from aiprod_pipelines.inference.desktop_plugins import OFXPluginSpec

        spec = OFXPluginSpec()
        manifest = spec.to_ofx_manifest()
        self.assertEqual(manifest["OfxPluginIdentifier"], "ai.aiprod.videogen")
        self.assertIn("parameters", manifest)


class TestDaVinciResolvePlugin(unittest.TestCase):
    def test_submit_render(self):
        from aiprod_pipelines.inference.desktop_plugins import (
            DaVinciResolvePlugin, DesktopInferenceEngine,
        )

        engine = DesktopInferenceEngine()
        engine.load_model()
        plugin = DaVinciResolvePlugin(engine=engine)
        job_id = plugin.submit_render("A forest scene")
        status = plugin.get_job_status(job_id)
        self.assertEqual(status["status"], "completed")


class TestPremierePluginSpec(unittest.TestCase):
    def test_manifest(self):
        from aiprod_pipelines.inference.desktop_plugins import PremierePluginSpec

        spec = PremierePluginSpec()
        manifest = spec.to_manifest()
        self.assertEqual(manifest["id"], "ai.aiprod.premiere")
        self.assertIn("panels", manifest)


class TestPremiereProPlugin(unittest.TestCase):
    def test_queue_render(self):
        from aiprod_pipelines.inference.desktop_plugins import PremiereProPlugin

        plugin = PremiereProPlugin()
        job_id = plugin.queue_render("A drone shot", duration_sec=10.0, resolution="4K")
        self.assertTrue(job_id)
        self.assertEqual(plugin.queue_length, 1)


class TestOnPremConfig(unittest.TestCase):
    def test_defaults(self):
        from aiprod_pipelines.inference.desktop_plugins import OnPremConfig

        cfg = OnPremConfig()
        self.assertEqual(cfg.port, 9100)
        self.assertEqual(cfg.auth_mode, "api_key")


class TestOnPremiseServer(unittest.TestCase):
    def test_init_gpus(self):
        from aiprod_pipelines.inference.desktop_plugins import OnPremiseServer, OnPremConfig

        server = OnPremiseServer(OnPremConfig(gpu_ids=[0]))
        count = server.initialize_gpus()
        self.assertEqual(count, 1)

    def test_submit_job(self):
        from aiprod_pipelines.inference.desktop_plugins import OnPremiseServer, OnPremConfig

        server = OnPremiseServer(OnPremConfig(gpu_ids=[0]))
        server.initialize_gpus()
        server.register_api_key("key1", "tenant1")

        self.assertEqual(server.validate_key("key1"), "tenant1")

        job_id = server.submit_job("tenant1", "A cinematic shot")
        job = server.get_job(job_id)
        self.assertEqual(job["status"], "completed")

    def test_health(self):
        from aiprod_pipelines.inference.desktop_plugins import OnPremiseServer

        server = OnPremiseServer()
        h = server.health()
        self.assertEqual(h["status"], "healthy")


# ===================================================================
# Cross-module integration
# ===================================================================


class TestScenaristToCameraIntegration(unittest.TestCase):
    """Verify scenarist output can drive camera controller."""

    def test_scenarist_shots_to_camera(self):
        from aiprod_pipelines.inference.scenarist import RuleBasedDecomposer, CreativeConfig
        from aiprod_core.camera_control import CameraController

        decomp = RuleBasedDecomposer()
        sb = decomp.decompose(
            "A drone aerial shot of a city. Follow a person walking.",
            CreativeConfig(target_duration_sec=10.0),
        )

        ctrl = CameraController()
        for scene in sb.scenes:
            for shot in scene.shots:
                signal = ctrl.from_scenarist_shot(
                    shot.shot_type.value,
                    shot.camera_move.value,
                    duration_sec=shot.duration_sec,
                )
                self.assertGreater(signal.num_frames, 0)


class TestCameraToV3Integration(unittest.TestCase):
    """Verify camera conditioning signal shape is compatible with v3 model."""

    def test_signal_dimensions(self):
        from aiprod_core.camera_control import CameraController, CinematicTemplate
        from aiprod_core.model.transformer.aiprod_v3 import AIPRODv3Config

        ctrl = CameraController()
        signal = ctrl.from_template(CinematicTemplate.ORBIT, num_frames=48)
        mat = signal.to_matrix()

        config = AIPRODv3Config()
        # Each frame should have camera_embed_dim values
        self.assertEqual(len(mat[0]), config.camera_embed_dim)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
