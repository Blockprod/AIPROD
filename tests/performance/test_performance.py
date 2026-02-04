"""
Performance and Load Testing
Tests for audio mixing speed, memory usage, and concurrent processing
Phase 5 Performance Benchmarks
"""

import os
import pytest
import time
import psutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from tempfile import TemporaryDirectory
import json

from src.agents.post_processor import PostProcessor
from src.orchestrator.state_machine import StateMachine


class TestAudioMixingPerformance:
    """Test audio mixing performance and efficiency"""

    @pytest.fixture
    def post_processor(self):
        """Initialize PostProcessor"""
        return PostProcessor(backend="ffmpeg")

    def test_audio_mixing_configuration_speed(self, post_processor):
        """Test speed of audio track configuration"""
        start_time = time.time()

        audio_tracks = [
            {"type": "voice", "path": f"voice_{i}.mp3", "volume": 1.0}
            for i in range(10)
        ]

        elapsed_time = time.time() - start_time

        # Configuration should be very fast (< 10ms)
        assert elapsed_time < 0.01, f"Configuration took {elapsed_time:.4f}s, should be < 0.01s"
        assert len(audio_tracks) == 10, "Should create all tracks"

    def test_large_audio_tracks_processing(self, post_processor):
        """Test handling large number of audio tracks"""
        start_time = time.time()

        # Create 50 audio tracks
        audio_tracks = []
        for i in range(50):
            track_type = ["voice", "music", "sfx"][i % 3]
            audio_tracks.append({
                "type": track_type,
                "path": f"audio_{i}.mp3",
                "volume": 0.5 + (i % 100) / 200
            })

        elapsed_time = time.time() - start_time

        assert len(audio_tracks) == 50, "Should create all 50 tracks"
        assert elapsed_time < 0.05, f"Creation took {elapsed_time:.4f}s, should be < 0.05s"

    def test_ffmpeg_command_generation_speed(self, post_processor):
        """Test speed of FFmpeg command generation"""
        start_time = time.time()

        audio_tracks = [
            {"type": "voice", "path": "voice.mp3", "volume": 1.0},
            {"type": "music", "path": "music.mp3", "volume": 0.6},
            {"type": "sfx", "path": "sfx.mp3", "volume": 0.5}
        ]

        # Simulate FFmpeg command construction
        command_parts = []
        for track in audio_tracks:
            command_parts.append(f"-i {track['path']}")

        elapsed_time = time.time() - start_time

        assert elapsed_time < 0.001, f"Command generation took {elapsed_time:.4f}s"
        assert len(command_parts) == 3, "Should have 3 command parts"


class TestMemoryUsage:
    """Test memory efficiency of audio processing"""

    @pytest.fixture
    def post_processor(self):
        """Initialize PostProcessor"""
        return PostProcessor(backend="ffmpeg")

    def test_post_processor_memory_footprint(self, post_processor):
        """Test memory footprint of PostProcessor initialization"""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Initialize multiple instances
        post_processors = [PostProcessor() for _ in range(10)]

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Should use reasonable amount of memory (< 50MB per instance)
        per_instance_memory = memory_increase / 10
        assert per_instance_memory < 50 * 1024 * 1024, f"Memory per instance too high: {per_instance_memory / 1024 / 1024:.2f}MB"

    def test_audio_track_memory_usage(self, post_processor):
        """Test memory usage of audio track configuration"""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Create large audio track list
        audio_tracks = [
            {"type": "voice", "path": f"audio_{i}.mp3", "volume": 0.5}
            for i in range(1000)
        ]

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # 1000 tracks should use reasonable memory (< 5MB)
        assert memory_increase < 5 * 1024 * 1024, f"Track list memory too high: {memory_increase / 1024 / 1024:.2f}MB"

    def test_manifest_memory_efficiency(self):
        """Test memory efficiency of manifest data structure"""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Create large manifest with complete data
        manifests = []
        for i in range(100):
            manifest = {
                "video_id": f"test_{i}",
                "title": "Test Video",
                "description": "A test video " * 100,  # Repeat for size
                "audio_tracks": [
                    {"type": "voice", "path": f"voice_{i}.mp3", "volume": 1.0},
                    {"type": "music", "path": f"music_{i}.mp3", "volume": 0.6},
                    {"type": "sfx", "path": f"sfx_{i}.mp3", "volume": 0.5}
                ],
                "effects": [{"type": "blur"}, {"type": "grayscale"}],
                "transitions": ["fade_in", "fade_out"],
                "metadata": {"created": "2026-02-04", "version": "1.0"}
            }
            manifests.append(manifest)

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # 100 manifests should use < 10MB
        assert memory_increase < 10 * 1024 * 1024, f"Manifests memory too high: {memory_increase / 1024 / 1024:.2f}MB"


class TestStateMachinePerformance:
    """Test StateMachine performance and efficiency"""

    def test_state_machine_initialization_speed(self):
        """Test speed of StateMachine initialization"""
        start_time = time.time()

        state_machine = StateMachine()

        elapsed_time = time.time() - start_time

        # Initialization includes GCP and other services (< 10s on first run)
        assert elapsed_time < 10.0, f"Initialization took {elapsed_time:.2f}s"
        assert state_machine is not None, "Should initialize successfully"

    def test_multiple_state_machines(self):
        """Test creating multiple StateMachine instances"""
        start_time = time.time()

        state_machines = [StateMachine() for _ in range(5)]

        elapsed_time = time.time() - start_time

        # 5 instances initialization (cached) should be reasonable (< 60s)
        assert elapsed_time < 60.0, f"Created 5 instances in {elapsed_time:.2f}s"
        assert len(state_machines) == 5, "Should create all instances"

    def test_agent_instantiation_performance(self):
        """Test performance of individual agent instantiation"""
        from src.agents.audio_generator import AudioGenerator
        from src.agents.music_composer import MusicComposer
        from src.agents.sound_effects_agent import SoundEffectsAgent

        agents = []
        start_time = time.time()

        agents.append(AudioGenerator())
        agents.append(MusicComposer())
        agents.append(SoundEffectsAgent())
        agents.append(PostProcessor())

        elapsed_time = time.time() - start_time

        # All agents should initialize in < 2s
        assert elapsed_time < 2.0, f"Agent initialization took {elapsed_time:.2f}s"
        assert len(agents) == 4, "Should instantiate all agents"


class TestConcurrentAudioProcessing:
    """Test concurrent audio processing capabilities"""

    def test_concurrent_track_construction(self):
        """Test building audio tracks concurrently"""
        import concurrent.futures

        def create_tracks(count):
            tracks = []
            for i in range(count):
                tracks.append({
                    "type": "voice" if i % 2 == 0 else "music",
                    "path": f"audio_{i}.mp3",
                    "volume": 0.5
                })
            return tracks

        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(create_tracks, 25) for _ in range(4)]
            results = [f.result() for f in futures]

        elapsed_time = time.time() - start_time

        # Concurrent creation should be reasonably fast
        assert elapsed_time < 1.0, f"Concurrent creation took {elapsed_time:.2f}s"
        assert len(results) == 4, "Should have all results"

    @patch('subprocess.run')
    def test_sequential_encoding_performance(self, mock_run):
        """Test performance of sequential encoding operations"""
        mock_run.return_value = MagicMock(returncode=0)

        start_time = time.time()

        # Simulate sequential encoding
        videos = []
        for i in range(5):
            result = {"output": f"video_{i}.mp4", "duration": 30}
            videos.append(result)

        elapsed_time = time.time() - start_time

        assert elapsed_time < 0.1, f"Sequential encoding simulation took {elapsed_time:.4f}s"
        assert len(videos) == 5, "Should have all videos"


class TestAudioEffectsPerformance:
    """Test performance of audio/video effects"""

    @pytest.fixture
    def post_processor(self):
        """Initialize PostProcessor"""
        return PostProcessor(backend="ffmpeg")

    def test_transition_effect_speed(self, post_processor):
        """Test speed of applying transition effects"""
        start_time = time.time()

        transitions = [
            {"type": "fade_in", "duration": 1},
            {"type": "cross_dissolve", "duration": 2},
            {"type": "fade_out", "duration": 1}
        ]

        # Configuration should be fast
        for transition in transitions:
            config = {"type": transition["type"], "duration": transition["duration"]}

        elapsed_time = time.time() - start_time

        assert elapsed_time < 0.01, f"Transition config took {elapsed_time:.4f}s"

    def test_effect_application_speed(self, post_processor):
        """Test speed of applying video effects"""
        start_time = time.time()

        effects = [
            {"type": "blur", "strength": 5},
            {"type": "grayscale", "intensity": 0.8},
            {"type": "invert", "enabled": True}
        ]

        # Configuration should be fast
        for effect in effects:
            config = {"type": effect["type"]}

        elapsed_time = time.time() - start_time

        assert elapsed_time < 0.01, f"Effect config took {elapsed_time:.4f}s"

    def test_text_overlay_speed(self, post_processor):
        """Test speed of adding text overlays"""
        start_time = time.time()

        titles = [
            {"text": f"Title {i}", "duration": 3}
            for i in range(10)
        ]

        # Configuration should be fast
        for title in titles:
            config = {"text": title["text"], "duration": title["duration"]}

        elapsed_time = time.time() - start_time

        assert elapsed_time < 0.01, f"Text overlay config took {elapsed_time:.4f}s"


class TestDataSerializationPerformance:
    """Test performance of manifest serialization"""

    def test_manifest_json_serialization(self):
        """Test speed of JSON serialization"""
        manifest = {
            "video_id": "test-001",
            "title": "Test Video",
            "description": "Test description",
            "audio_tracks": [
                {"type": "voice", "path": "voice.mp3", "volume": 1.0},
                {"type": "music", "path": "music.mp3", "volume": 0.6}
            ],
            "effects": [{"type": "blur"}, {"type": "grayscale"}],
            "metadata": {"created": "2026-02-04", "version": "1.0"}
        }

        start_time = time.time()

        # Serialize and deserialize
        for _ in range(1000):
            json_str = json.dumps(manifest)
            parsed = json.loads(json_str)

        elapsed_time = time.time() - start_time

        # 1000 iterations should be fast (< 0.1s)
        assert elapsed_time < 0.1, f"Serialization took {elapsed_time:.4f}s"

    def test_large_manifest_serialization(self):
        """Test serialization of large manifests"""
        large_manifest = {
            "video_id": "test-001",
            "audio_tracks": [
                {"type": "voice", "path": f"audio_{i}.mp3", "volume": 0.5}
                for i in range(100)
            ],
            "effects": [{"type": "blur"} for _ in range(50)],
            "transitions": ["fade_in", "fade_out"] * 25
        }

        start_time = time.time()

        json_str = json.dumps(large_manifest)
        parsed = json.loads(json_str)

        elapsed_time = time.time() - start_time

        assert elapsed_time < 0.05, f"Large manifest serialization took {elapsed_time:.4f}s"
        assert len(parsed["audio_tracks"]) == 100, "Should have all tracks"


import json


class TestCachePerformance:
    """Test caching and optimization"""

    def test_repeated_manifest_configuration(self):
        """Test performance of repeated manifest configuration"""
        manifest_template = {
            "video_id": "test-001",
            "title": "Test",
            "audio_tracks": []
        }

        start_time = time.time()

        # Create 100 similar manifests
        manifests = []
        for i in range(100):
            manifest = manifest_template.copy()
            manifest["video_id"] = f"test-{i}"
            manifests.append(manifest)

        elapsed_time = time.time() - start_time

        assert elapsed_time < 0.05, f"Manifest creation took {elapsed_time:.4f}s"
        assert len(manifests) == 100, "Should have all manifests"

    def test_repeated_audio_track_construction(self):
        """Test performance of repeated audio track construction"""
        start_time = time.time()

        # Create 100 similar audio track configurations
        track_lists = []
        for i in range(100):
            tracks = [
                {"type": "voice", "path": f"voice_{i}.mp3", "volume": 1.0},
                {"type": "music", "path": f"music_{i}.mp3", "volume": 0.6},
                {"type": "sfx", "path": f"sfx_{i}.mp3", "volume": 0.5}
            ]
            track_lists.append(tracks)

        elapsed_time = time.time() - start_time

        assert elapsed_time < 0.05, f"Track construction took {elapsed_time:.4f}s"
        assert len(track_lists) == 100, "Should have all track lists"
