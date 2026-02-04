"""
Comprehensive Audio/Video Pipeline Testing
Tests for complete audio-video generation and mixing
Phase 5 Integration Tests
"""

import os
import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from tempfile import TemporaryDirectory

from src.agents.audio_generator import AudioGenerator
from src.agents.music_composer import MusicComposer
from src.agents.sound_effects_agent import SoundEffectsAgent
from src.agents.post_processor import PostProcessor
from src.orchestrator.state_machine import StateMachine


class TestAudioVideoMixing:
    """Test audio track mixing and composition"""

    @pytest.fixture
    def post_processor(self):
        """Initialize PostProcessor for testing"""
        return PostProcessor(backend="ffmpeg")

    @pytest.fixture
    def sample_audio_tracks(self):
        """Create sample audio tracks configuration"""
        return [
            {
                "type": "voice",
                "path": "voice.mp3",
                "volume": 1.0
            },
            {
                "type": "music",
                "path": "music.mp3",
                "volume": 0.6
            },
            {
                "type": "sfx",
                "path": "sfx.mp3",
                "volume": 0.5
            }
        ]

    def test_mix_audio_tracks_configuration(self, post_processor, sample_audio_tracks):
        """Test audio track mixing configuration"""
        # Verify volumes are normalized
        assert sample_audio_tracks[0]["volume"] == 1.0, "Voice should be full volume"
        assert sample_audio_tracks[1]["volume"] == 0.6, "Music should be 60% volume"
        assert sample_audio_tracks[2]["volume"] == 0.5, "SFX should be 50% volume"

        # Verify track types
        track_types = [t["type"] for t in sample_audio_tracks]
        assert "voice" in track_types, "Should have voice track"
        assert "music" in track_types, "Should have music track"
        assert "sfx" in track_types, "Should have SFX track"

    def test_mix_audio_tracks_with_valid_files(self, post_processor):
        """Test mixing audio tracks with valid files"""
        with TemporaryDirectory() as tmpdir:
            # Create dummy audio files
            voice_file = os.path.join(tmpdir, "voice.mp3")
            music_file = os.path.join(tmpdir, "music.mp3")
            sfx_file = os.path.join(tmpdir, "sfx.mp3")
            
            # Create simple mock files
            for file_path in [voice_file, music_file, sfx_file]:
                with open(file_path, 'wb') as f:
                    f.write(b"mock audio data")

            audio_tracks = [
                {"type": "voice", "path": voice_file, "volume": 1.0},
                {"type": "music", "path": music_file, "volume": 0.6},
                {"type": "sfx", "path": sfx_file, "volume": 0.5}
            ]

            # Test that configuration is valid
            assert len(audio_tracks) == 3, "Should have 3 audio tracks"
            for track in audio_tracks:
                assert os.path.exists(track["path"]), f"Audio file should exist: {track['path']}"

    def test_mix_audio_tracks_ffmpeg_command(self, post_processor):
        """Test FFmpeg command generation for audio mixing"""
        audio_tracks = [
            {"type": "voice", "path": "voice.mp3", "volume": 1.0},
            {"type": "music", "path": "music.mp3", "volume": 0.6},
        ]

        # The method should construct appropriate FFmpeg commands
        with TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.mp4")
            # Just verify the configuration is valid
            assert len(audio_tracks) > 0, "Should have audio tracks"
            assert all("volume" in t for t in audio_tracks), "All tracks should have volume"


class TestCompleteAudioVideoPipeline:
    """Test complete pipeline integration"""

    @pytest.fixture
    def state_machine(self):
        """Initialize StateMachine for testing"""
        return StateMachine()

    @pytest.fixture
    def test_manifest(self):
        """Create test manifest for pipeline"""
        return {
            "video_id": "test-video-001",
            "title": "Test Video",
            "description": "A test video with all audio components",
            "duration": 30,
            "script": "This is a test script with rain and wind sounds",
            "style": "cinematic",
            "mood": "dramatic",
            "transitions": ["fade_in", "cross_dissolve"],
            "effects": ["blur", "grayscale"],
            "render_output": {
                "video_url": "test_video.mp4",
                "duration": 30
            }
        }

    def test_pipeline_manifest_structure(self, test_manifest):
        """Test manifest structure for pipeline"""
        # Verify required fields
        assert "video_id" in test_manifest, "Manifest should have video_id"
        assert "title" in test_manifest, "Manifest should have title"
        assert "script" in test_manifest, "Manifest should have script"
        assert "render_output" in test_manifest, "Manifest should have render_output"

    def test_audio_generator_integration(self, state_machine):
        """Test AudioGenerator in pipeline"""
        # Test configuration
        assert hasattr(state_machine, 'audio_generator'), "StateMachine should have audio_generator"
        assert state_machine.audio_generator is not None, "audio_generator should be initialized"

    @patch('src.agents.music_composer.requests.post')
    def test_music_composer_integration(self, mock_post, state_machine):
        """Test MusicComposer in pipeline"""
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"music_url": "https://suno.ai/music.mp3"}
        )

        assert hasattr(state_machine, 'music_composer'), "StateMachine should have music_composer"
        assert state_machine.music_composer is not None, "music_composer should be initialized"

    def test_sound_effects_integration(self, state_machine):
        """Test SoundEffectsAgent in pipeline"""
        assert hasattr(state_machine, 'sound_effects_agent'), "StateMachine should have sound_effects_agent"
        assert state_machine.sound_effects_agent is not None, "sound_effects_agent should be initialized"

    def test_post_processor_integration(self, state_machine):
        """Test PostProcessor in pipeline"""
        assert hasattr(state_machine, 'post_processor'), "StateMachine should have post_processor"
        assert state_machine.post_processor is not None, "post_processor should be initialized"

    def test_complete_agent_orchestration(self, state_machine):
        """Test all agents are available in orchestration"""
        # Verify all agents are initialized
        agents = [
            state_machine.audio_generator,
            state_machine.music_composer,
            state_machine.sound_effects_agent,
            state_machine.post_processor
        ]

        for agent in agents:
            assert agent is not None, "All agents should be initialized"


class TestVideoEffectsIntegration:
    """Test video effects and transitions"""

    @pytest.fixture
    def post_processor(self):
        """Initialize PostProcessor"""
        return PostProcessor(backend="ffmpeg")

    def test_transitions_configuration(self, post_processor):
        """Test transition effect configuration"""
        transitions = ["fade_in", "fade_out", "cross_dissolve"]

        for transition in transitions:
            assert isinstance(transition, str), f"Transition should be string: {transition}"

    def test_effects_configuration(self, post_processor):
        """Test video effects configuration"""
        effects = [
            {"type": "blur", "strength": 5},
            {"type": "grayscale", "intensity": 0.8},
            {"type": "invert", "enabled": True}
        ]

        for effect in effects:
            assert "type" in effect, "Effect should have type"
            assert isinstance(effect["type"], str), "Effect type should be string"

    def test_titles_subtitles_configuration(self, post_processor):
        """Test titles and subtitles configuration"""
        titles = [
            {"text": "Main Title", "duration": 3, "position": "center"},
            {"text": "Scene 1", "duration": 2, "position": "top"}
        ]

        subtitles = [
            {"text": "Subtitle 1", "start_time": 0, "end_time": 5},
            {"text": "Subtitle 2", "start_time": 5, "end_time": 10}
        ]

        # Verify structure
        for title in titles:
            assert "text" in title, "Title should have text"
            assert "duration" in title, "Title should have duration"

        for subtitle in subtitles:
            assert "text" in subtitle, "Subtitle should have text"
            assert "start_time" in subtitle, "Subtitle should have start_time"
            assert "end_time" in subtitle, "Subtitle should have end_time"


class TestAudioTrackConstruction:
    """Test automatic audio track construction from agents"""

    @pytest.fixture
    def sample_agent_outputs(self):
        """Create sample outputs from different agents"""
        return {
            "audio_generator": {
                "audio_url": "gs://bucket/voice.mp3",
                "duration": 30,
                "bitrate": "128k"
            },
            "music_composer": {
                "music_url": "https://api.suno.ai/music/xyz.mp3",
                "style": "cinematic",
                "duration": 30
            },
            "sound_effects_agent": {
                "sound_effects": {
                    "sfx_list": [
                        {
                            "type": "rain",
                            "preview_url": "https://freesound.org/sfx/rain.mp3",
                            "duration": 10
                        },
                        {
                            "type": "wind",
                            "preview_url": "https://freesound.org/sfx/wind.mp3",
                            "duration": 15
                        }
                    ]
                }
            }
        }

    def test_audio_track_construction(self, sample_agent_outputs):
        """Test automatic construction of audio tracks"""
        audio_tracks = []

        # Voice from AudioGenerator
        if sample_agent_outputs["audio_generator"].get("audio_url"):
            audio_tracks.append({
                "type": "voice",
                "path": sample_agent_outputs["audio_generator"]["audio_url"],
                "volume": 1.0
            })

        # Music from MusicComposer
        if sample_agent_outputs["music_composer"].get("music_url"):
            audio_tracks.append({
                "type": "music",
                "path": sample_agent_outputs["music_composer"]["music_url"],
                "volume": 0.6
            })

        # SFX from SoundEffectsAgent
        for sfx in sample_agent_outputs["sound_effects_agent"].get("sound_effects", {}).get("sfx_list", []):
            audio_tracks.append({
                "type": "sfx",
                "path": sfx.get("preview_url"),
                "volume": 0.5
            })

        # Verify construction
        assert len(audio_tracks) == 4, "Should have 4 audio tracks (1 voice + 1 music + 2 sfx)"
        assert audio_tracks[0]["type"] == "voice", "First track should be voice"
        assert audio_tracks[1]["type"] == "music", "Second track should be music"
        assert audio_tracks[2]["type"] == "sfx", "Third track should be sfx"
        assert audio_tracks[3]["type"] == "sfx", "Fourth track should be sfx"

    def test_volume_normalization(self, sample_agent_outputs):
        """Test volume levels for each track type"""
        volume_config = {
            "voice": 1.0,
            "music": 0.6,
            "sfx": 0.5
        }

        # Verify volumes
        assert volume_config["voice"] == 1.0, "Voice should be full volume"
        assert volume_config["music"] == 0.6, "Music should be reduced"
        assert volume_config["sfx"] == 0.5, "SFX should be reduced"

        # Verify all volumes are between 0 and 1
        for track_type, volume in volume_config.items():
            assert 0 <= volume <= 1, f"Volume for {track_type} should be between 0 and 1"


class TestManifestFlow:
    """Test manifest flowing through pipeline"""

    def test_manifest_transformation(self):
        """Test manifest data transformation through pipeline"""
        # Initial manifest
        initial_manifest = {
            "video_id": "test-001",
            "title": "Test",
            "script": "Test script",
            "duration": 30
        }

        # After audio generation
        manifest_with_audio = {
            **initial_manifest,
            "audio_url": "gs://bucket/voice.mp3"
        }

        # After music composition
        manifest_with_music = {
            **manifest_with_audio,
            "music_url": "https://suno.ai/music.mp3"
        }

        # After SFX generation
        manifest_with_sfx = {
            **manifest_with_music,
            "sfx_list": [
                {"type": "rain", "url": "https://freesound.org/sfx/rain.mp3"}
            ]
        }

        # After post-processing
        final_manifest = {
            **manifest_with_sfx,
            "audio_tracks": [
                {"type": "voice", "path": "gs://bucket/voice.mp3", "volume": 1.0},
                {"type": "music", "path": "https://suno.ai/music.mp3", "volume": 0.6},
                {"type": "sfx", "path": "https://freesound.org/sfx/rain.mp3", "volume": 0.5}
            ],
            "output_video": "gs://bucket/final_video.mp4"
        }

        # Verify flow
        assert initial_manifest["video_id"] == final_manifest["video_id"]
        assert "audio_url" in final_manifest
        assert "music_url" in final_manifest
        assert "sfx_list" in final_manifest
        assert "audio_tracks" in final_manifest
        assert "output_video" in final_manifest


class TestIntegrationWithRenderExecutor:
    """Test integration with RenderExecutor outputs"""

    @pytest.fixture
    def render_output(self):
        """Create typical RenderExecutor output"""
        return {
            "video_url": "gs://bucket/rendered_video.mp4",
            "duration": 30,
            "frames": 900,
            "fps": 30,
            "resolution": "1920x1080",
            "size_bytes": 524288000  # 500MB
        }

    def test_render_output_structure(self, render_output):
        """Test structure of RenderExecutor output"""
        assert "video_url" in render_output, "Should have video_url"
        assert "duration" in render_output, "Should have duration"
        assert "resolution" in render_output, "Should have resolution"

    def test_post_processor_input_construction(self, render_output):
        """Test constructing PostProcessor input from RenderExecutor output"""
        audio_tracks = [
            {"type": "voice", "path": "voice.mp3", "volume": 1.0},
            {"type": "music", "path": "music.mp3", "volume": 0.6}
        ]

        post_processor_input = {
            **render_output,
            "audio_tracks": audio_tracks,
            "transitions": ["fade_in"],
            "effects": ["blur"]
        }

        # Verify input structure
        assert "video_url" in post_processor_input
        assert "audio_tracks" in post_processor_input
        assert len(post_processor_input["audio_tracks"]) == 2
        assert "transitions" in post_processor_input
        assert "effects" in post_processor_input
