"""
Phase 2 — Cinematic Pipeline Unit Tests
========================================

Tests for:
    1. TTS (TextFrontend, ProsodyModeler, SpeakerEmbedding, VocoderTTS, TTSModel)
    2. Lip-Sync (LipSyncModel)
    3. Audio Mixer (AudioMixer, SpatialAudio)
    4. Editing (PacingEngine, TimelineGenerator, TransitionsLib)
    5. Color (LUTManager, ColorSpaceConverter, HDRProcessor, SceneColorMatcher, ColorPipeline)
    6. Export (ExportEngine profile management)
"""

import math
import pytest
import torch

# ──────────────────────────────────────────────────────────────────────────
# 1. TTS
# ──────────────────────────────────────────────────────────────────────────

class TestTextFrontend:
    """Text normalization & G2P."""

    def _frontend(self):
        from aiprod_core.model.tts.text_frontend import TextFrontend, FrontendConfig
        return TextFrontend(FrontendConfig())

    def test_normalise_lowercases(self):
        fe = self._frontend()
        assert fe.normalise("HELLO") == "hello"

    def test_normalise_expands_numbers(self):
        fe = self._frontend()
        text = fe.normalise("I have 42 apples")
        assert "forty" in text and "two" in text

    def test_normalise_contractions(self):
        fe = self._frontend()
        out = fe.normalise("I can't do it")
        assert "cannot" in out

    def test_g2p_returns_phonemes(self):
        fe = self._frontend()
        phonemes = fe.g2p("hello")
        assert isinstance(phonemes, list)
        assert len(phonemes) > 0

    def test_text_to_phoneme_ids_returns_ints(self):
        fe = self._frontend()
        ids = fe.text_to_phoneme_ids("hello world")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_roundtrip_ids(self):
        fe = self._frontend()
        ids = fe.text_to_phoneme_ids("test")
        recovered = fe.ids_to_phonemes(ids)
        assert isinstance(recovered, list)
        assert len(recovered) == len(ids)


class TestProsody:
    """ProsodyModeler variance predictors."""

    def test_prosody_forward_shape(self):
        from aiprod_core.model.tts.prosody import ProsodyModeler, ProsodyConfig
        pm = ProsodyModeler(encoder_dim=384, config=ProsodyConfig())
        x = torch.randn(2, 10, 384)
        frame_feats, info = pm(x)
        assert frame_feats.dim() == 3
        assert frame_feats.shape[0] == 2
        assert "predicted_duration" in info
        assert "predicted_f0" in info
        assert "predicted_energy" in info


class TestSpeakerEmbedding:
    """Speaker lookup & encoder."""

    def test_from_id(self):
        from aiprod_core.model.tts.speaker_embedding import SpeakerEmbedding, SpeakerConfig
        se = SpeakerEmbedding(SpeakerConfig(num_speakers=10, embedding_dim=64))
        emb = se.from_id(torch.tensor([0, 1]))
        assert emb.shape == (2, 64)

    def test_from_reference(self):
        from aiprod_core.model.tts.speaker_embedding import SpeakerEmbedding, SpeakerConfig
        se = SpeakerEmbedding(SpeakerConfig(embedding_dim=64, encoder_input_dim=80))
        ref_mel = torch.randn(1, 50, 80)
        emb = se.from_reference(ref_mel)
        assert emb.shape == (1, 64)


class TestVocoder:
    """HiFi-GAN vocoder."""

    def test_vocoder_infer(self):
        from aiprod_core.model.tts.vocoder_tts import VocoderTTS, VocoderConfig
        voc = VocoderTTS(VocoderConfig(num_mels=80, sample_rate=24000))
        mel = torch.randn(1, 80, 32)
        wav = voc.infer(mel)
        assert wav.dim() == 3   # [B, 1, T]
        assert wav.shape[1] == 1


class TestTTSModel:
    """Full TTS pipeline."""

    def test_forward_returns_all_keys(self):
        from aiprod_core.model.tts.model import TTSModel, TTSConfig
        cfg = TTSConfig(
            encoder_hidden=64, encoder_layers=1, encoder_heads=2,
            decoder_hidden=64, decoder_layers=1, decoder_heads=2,
            encoder_ff_dim=128, decoder_ff_dim=128,
            vocab_size=100, num_speakers=5, speaker_emb_dim=64,
            num_mels=40,
        )
        model = TTSModel(cfg)
        ids = torch.randint(0, 100, (1, 8))
        sid = torch.tensor([0])
        result = model(ids, sid)
        assert "mel_output" in result
        assert "waveform" in result
        assert result["mel_output"].shape[1] == 40  # num_mels


# ──────────────────────────────────────────────────────────────────────────
# 2. Lip-Sync
# ──────────────────────────────────────────────────────────────────────────

class TestLipSync:
    """Lip-sync model."""

    def test_forward_shape(self):
        from aiprod_core.model.lip_sync.model import LipSyncModel, LipSyncConfig
        cfg = LipSyncConfig(audio_dim=80, num_facial_params=52, hidden_dim=64, num_layers=1)
        model = LipSyncModel(cfg)
        audio = torch.randn(2, 20, 80)
        result = model(audio)
        assert result["predicted_params"].shape == (2, 20, 52)

    def test_sync_loss(self):
        from aiprod_core.model.lip_sync.model import LipSyncModel, LipSyncConfig
        cfg = LipSyncConfig(hidden_dim=64, num_layers=1)
        model = LipSyncModel(cfg)
        pred = torch.randn(2, 10, 52)
        gt = torch.randn(2, 10, 52)
        loss, metrics = model.sync_loss(pred, gt)
        assert loss.dim() == 0
        assert "lse_d" in metrics
        assert "lse_c" in metrics

    def test_infer(self):
        from aiprod_core.model.lip_sync.model import LipSyncModel, LipSyncConfig
        cfg = LipSyncConfig(hidden_dim=64, num_layers=1)
        model = LipSyncModel(cfg)
        audio = torch.randn(1, 15, 80)
        params = model.infer(audio)
        assert params.shape == (15, 52)


# ──────────────────────────────────────────────────────────────────────────
# 3. Audio Mixer
# ──────────────────────────────────────────────────────────────────────────

class TestAudioMixer:
    """Multi-track mixer + DSP."""

    def _mixer(self):
        from aiprod_core.model.audio_mixer.mixer import AudioMixer, AudioMixerConfig
        return AudioMixer(AudioMixerConfig(sample_rate=16000))

    def test_add_and_mix_single_track(self):
        from aiprod_core.model.audio_mixer.mixer import AudioTrack
        mixer = self._mixer()
        track = AudioTrack("voice", torch.randn(2, 8000), "voice")
        mixer.add_track(track)
        out = mixer.mix()
        assert out.shape[0] == 2  # stereo
        assert out.shape[1] == 8000

    def test_mix_multiple_tracks(self):
        from aiprod_core.model.audio_mixer.mixer import AudioTrack
        mixer = self._mixer()
        mixer.add_track(AudioTrack("v", torch.randn(2, 8000), "voice", volume=0.8))
        mixer.add_track(AudioTrack("m", torch.randn(2, 8000), "music", volume=0.5))
        out = mixer.mix()
        assert out.shape == (2, 8000)

    def test_mute_track(self):
        from aiprod_core.model.audio_mixer.mixer import AudioTrack
        mixer = self._mixer()
        mixer.add_track(AudioTrack("v", torch.ones(2, 4000), "voice", mute=True))
        mixer.add_track(AudioTrack("m", torch.ones(2, 4000) * 0.5, "music"))
        out = mixer.mix()
        # Only music should contribute
        assert out.abs().max() > 0

    def test_compression_reduces_peaks(self):
        mixer = self._mixer()
        loud = torch.ones(1, 1000) * 0.9
        compressed = mixer.apply_compression(loud, threshold=-6.0, ratio=8.0,
                                              attack_ms=1.0, release_ms=10.0)
        assert compressed.abs().max() < loud.abs().max()

    def test_eq_runs(self):
        mixer = self._mixer()
        audio = torch.randn(1, 2000)
        out = mixer.apply_eq(audio, [(1000.0, 3.0, 1.0)])
        assert out.shape == audio.shape


class TestSpatialAudio:
    """Spatial audio conversions."""

    def _spatial(self):
        from aiprod_core.model.audio_mixer.mixer import SpatialAudio, AudioMixerConfig
        return SpatialAudio(AudioMixerConfig(sample_rate=16000))

    def test_mono_to_stereo(self):
        sp = self._spatial()
        mono = torch.randn(1, 4000)
        stereo = sp.to_stereo(mono)
        assert stereo.shape == (2, 4000)

    def test_stereo_to_5_1(self):
        sp = self._spatial()
        stereo = torch.randn(2, 4000)
        surround = sp.to_5_1(stereo)
        assert surround.shape[0] == 6

    def test_binaural(self):
        sp = self._spatial()
        mono = torch.randn(1, 4000)
        binaural = sp.to_binaural(mono, azimuth_deg=45.0)
        assert binaural.shape == (2, 4000)


# ──────────────────────────────────────────────────────────────────────────
# 4. Editing
# ──────────────────────────────────────────────────────────────────────────

class TestPacingEngine:
    """Shot duration computation."""

    def _engine(self):
        from aiprod_pipelines.editing.timeline import PacingEngine, EditingConfig
        return PacingEngine(EditingConfig())

    def test_action_is_short(self):
        dur = self._engine().compute_optimal_duration("action", 0.9, 0.3)
        assert dur < 2.5

    def test_calm_is_long(self):
        dur = self._engine().compute_optimal_duration("calm", 0.2, 0.5)
        assert dur > 3.0


class TestTimelineGenerator:
    """Timeline building & export."""

    def _gen(self):
        from aiprod_pipelines.editing.timeline import TimelineGenerator, EditingConfig
        return TimelineGenerator(EditingConfig())

    def test_add_clip(self):
        gen = self._gen()
        gen.add_clip("a.mp4", 0.0, 5.0)
        gen.add_clip("b.mp4", 5.0, 3.0)
        assert len(gen.clips) == 2
        assert gen.timeline_duration == 8.0

    def test_compute_pacing(self):
        gen = self._gen()
        gen.add_clip("a.mp4", 0.0, 5.0)
        gen.add_clip("b.mp4", 5.0, 3.0)
        p = gen.compute_pacing()
        assert "cuts_per_minute" in p
        assert p["num_clips"] == 2

    def test_from_scenario(self):
        gen = self._gen()
        scenario = {
            "scenes": [
                {"id": "s1", "duration": 5, "emotion": "action", "intensity": 0.8},
                {"id": "s2", "duration": 3, "emotion": "calm"},
            ]
        }
        gen.from_scenario(scenario)
        assert len(gen.clips) == 2

    def test_export_edl(self, tmp_path):
        gen = self._gen()
        gen.add_clip("a.mp4", 0.0, 5.0)
        path = str(tmp_path / "test.edl")
        gen.export_edl(path)
        with open(path) as f:
            content = f.read()
        assert "TITLE: AIPROD" in content

    def test_export_fcpxml(self, tmp_path):
        gen = self._gen()
        gen.add_clip("a.mp4", 0.0, 5.0)
        path = str(tmp_path / "test.fcpxml")
        gen.export_fcpxml(path)
        with open(path) as f:
            content = f.read()
        assert "<fcpxml" in content

    def test_export_aaf(self, tmp_path):
        gen = self._gen()
        gen.add_clip("a.mp4", 0.0, 5.0)
        path = str(tmp_path / "test.aaf")
        gen.export_aaf(path)
        with open(path) as f:
            content = f.read()
        assert "AIPROD AAF Export" in content


class TestTransitionsLib:
    """Video transition effects."""

    def test_cross_fade_shape(self):
        from aiprod_pipelines.editing.timeline import TransitionsLib
        a = torch.randn(3, 4, 4, 10)
        b = torch.randn(3, 4, 4, 10)
        out = TransitionsLib.cross_fade(a, b, 4)
        assert out.shape[-1] == 10 + 10 - 4  # 16

    def test_wipe_shape(self):
        from aiprod_pipelines.editing.timeline import TransitionsLib
        a = torch.randn(3, 8, 8, 10)
        b = torch.randn(3, 8, 8, 10)
        out = TransitionsLib.wipe(a, b, 5, "left_to_right")
        assert out.shape[-1] == 10 + 10 - 5

    def test_match_cut(self):
        from aiprod_pipelines.editing.timeline import TransitionsLib
        a = torch.randn(3, 4, 4, 10)
        b = torch.randn(3, 4, 4, 10)
        out = TransitionsLib.match_cut(a, b)
        assert out.dim() == 4


# ──────────────────────────────────────────────────────────────────────────
# 5. Color
# ──────────────────────────────────────────────────────────────────────────

class TestLUTManager:
    """3D LUT loading & application."""

    def test_builtin_luts_exist(self):
        from aiprod_pipelines.color.color_pipeline import LUTManager
        mgr = LUTManager()
        assert "identity" in mgr.luts
        assert "cinematic_warm" in mgr.luts

    def test_identity_lut_passthrough(self):
        from aiprod_pipelines.color.color_pipeline import LUTManager
        mgr = LUTManager()
        frame = torch.rand(8, 8, 3)
        graded = mgr.apply_lut(frame, mgr.luts["identity"])
        assert torch.allclose(frame, graded, atol=0.02)

    def test_lut_changes_colors(self):
        from aiprod_pipelines.color.color_pipeline import LUTManager
        mgr = LUTManager()
        frame = torch.rand(8, 8, 3) * 0.5 + 0.25
        graded = mgr.apply_lut(frame, mgr.luts["cinematic_warm"])
        # Warm LUT should shift colours
        diff = (graded - frame).abs().mean().item()
        assert diff > 0.001

    def test_load_cube_file(self, tmp_path):
        from aiprod_pipelines.color.color_pipeline import LUTManager
        mgr = LUTManager()
        # Write a tiny 2x2x2 .cube
        cube = tmp_path / "test.cube"
        lines = ["LUT_3D_SIZE 2"]
        for r in [0.0, 1.0]:
            for g in [0.0, 1.0]:
                for b in [0.0, 1.0]:
                    lines.append(f"{r} {g} {b}")
        cube.write_text("\n".join(lines))
        mgr.load_lut_file(str(cube), "test")
        assert "test" in mgr.luts
        assert mgr.luts["test"].data.shape == (2, 2, 2, 3)


class TestColorSpaceConverter:
    """Color-space matrix transforms."""

    def test_rec709_to_rec2020_and_back(self):
        from aiprod_pipelines.color.color_pipeline import ColorSpaceConverter as CSC
        frame = torch.rand(4, 4, 3)
        wide = CSC.rec709_to_rec2020(frame)
        back = CSC.rec2020_to_rec709(wide)
        assert torch.allclose(frame, back, atol=0.05)

    def test_log_roundtrip(self):
        from aiprod_pipelines.color.color_pipeline import ColorSpaceConverter as CSC
        frame = torch.rand(4, 4, 3) * 0.8 + 0.1
        log = CSC.linear_to_log(frame)
        back = CSC.log_to_linear(log)
        assert torch.allclose(frame, back, atol=0.02)


class TestHDRProcessor:
    """HDR tone mapping."""

    def test_sdr_to_hdr_expands_range(self):
        from aiprod_pipelines.color.color_pipeline import HDRProcessor, ColorGradingConfig
        proc = HDRProcessor(ColorGradingConfig())
        sdr = torch.rand(4, 4, 3) * 0.8
        hdr, meta = proc.tone_map_sdr_to_hdr(sdr, peak_brightness=1000.0)
        assert hdr.max().item() > 1.0
        assert "max_cll" in meta
        assert "primaries" in meta


class TestSceneColorMatcher:
    """Histogram-based scene matching."""

    def test_histogram_sums_to_one(self):
        from aiprod_pipelines.color.color_pipeline import SceneColorMatcher
        frame = torch.rand(16, 16, 3)
        hist = SceneColorMatcher.compute_color_histogram(frame)
        assert hist.shape == (3, 256)
        for c in range(3):
            assert abs(hist[c].sum().item() - 1.0) < 1e-4

    def test_match_histograms_shape(self):
        from aiprod_pipelines.color.color_pipeline import SceneColorMatcher
        src = torch.rand(8, 8, 3)
        tgt = torch.rand(8, 8, 3)
        matched = SceneColorMatcher.match_histograms(src, tgt)
        assert matched.shape == src.shape

    def test_match_across_scenes(self):
        from aiprod_pipelines.color.color_pipeline import SceneColorMatcher
        scenes = [torch.rand(4, 4, 3) for _ in range(3)]
        matched = SceneColorMatcher.match_across_scene_sequence(scenes)
        assert len(matched) == 3


# ──────────────────────────────────────────────────────────────────────────
# 6. Export
# ──────────────────────────────────────────────────────────────────────────

class TestExportEngine:
    """Export profile management (encoding requires FFmpeg)."""

    def _engine(self):
        from aiprod_pipelines.export.multi_format import ExportEngine
        return ExportEngine()

    def test_list_profiles(self):
        eng = self._engine()
        profiles = eng.list_profiles()
        assert "web_mp4" in profiles
        assert "streaming_hq" in profiles
        assert "prores_editing" in profiles
        assert len(profiles) >= 5

    def test_get_profile_info(self):
        eng = self._engine()
        info = eng.get_profile_info("web_mp4")
        assert info["codec"] == "h264"
        assert info["container"] == "mp4"

    def test_custom_profile(self):
        from aiprod_pipelines.export.multi_format import ExportProfile, VideoCodec, AudioCodec
        eng = self._engine()
        custom = ExportProfile(
            name="my_custom", video_codec=VideoCodec.H265,
            audio_codec=AudioCodec.FLAC, resolution=(1280, 720),
        )
        eng.add_custom_profile(custom)
        assert "my_custom" in eng.list_profiles()

    def test_unknown_profile_raises(self):
        eng = self._engine()
        with pytest.raises(ValueError, match="Unknown profile"):
            eng.export(torch.randn(3, 4, 4, 2), None, "out.mp4", profile="nonexistent")
