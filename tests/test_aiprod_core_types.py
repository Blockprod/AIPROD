"""Tests for aiprod_core.types module."""

import pytest
import torch
from aiprod_core.types import (
    ModalityType, VideoPixelShape, AudioShape, LatentShape,
    VideoLatentShape, AudioLatentShape, LatentState, GenerationConfig,
    PrecisionMode, DeviceType
)


class TestModalityType:
    """Test ModalityType enum."""
    
    def test_modality_type_has_required_values(self):
        """Test that ModalityType has all required values."""
        assert hasattr(ModalityType, "VIDEO")
        assert hasattr(ModalityType, "AUDIO")
        assert hasattr(ModalityType, "TEXT")
        assert hasattr(ModalityType, "IMAGE")
    
    def test_modality_type_comparison(self):
        """Test modality type comparison operations."""
        video = ModalityType.VIDEO
        assert video == ModalityType.VIDEO
        assert video != ModalityType.AUDIO


class TestVideoPixelShape:
    """Test VideoPixelShape dataclass."""
    
    def test_video_pixel_shape_defaults(self):
        """Test VideoPixelShape default values."""
        shape = VideoPixelShape()
        assert shape.height == 512
        assert shape.width == 768
        assert shape.num_frames == 49
        assert shape.fps == 24.0
    
    def test_video_pixel_shape_custom_values(self):
        """Test creating VideoPixelShape with custom values."""
        shape = VideoPixelShape(height=256, width=512, num_frames=24, fps=30.0)
        assert shape.height == 256
        assert shape.width == 512
        assert shape.num_frames == 24
        assert shape.fps == 30.0
    
    def test_video_pixel_shape_legacy_frames_kwarg(self):
        """Test VideoPixelShape accepts legacy frames= kwarg."""
        # frames kwarg is converted to num_frames
        shape = VideoPixelShape(height=512, width=768, frames=16)
        assert shape.num_frames == 16
    
    def test_video_pixel_shape_aspect_ratio(self):
        """Test aspect ratio calculation."""
        shape = VideoPixelShape(height=512, width=1024)
        assert shape.aspect_ratio == 2.0
    
    def test_video_pixel_shape_duration(self):
        """Test duration calculation."""
        shape = VideoPixelShape(num_frames=24, fps=24.0)
        assert shape.duration_seconds == 1.0


class TestAudioShape:
    """Test AudioShape dataclass."""
    
    def test_audio_shape_creation(self):
        """Test creating AudioShape."""
        shape = AudioShape(batch=2, channels=1, samples=48000, sample_rate=48000)
        assert shape.batch == 2
        assert shape.channels == 1
        assert shape.samples == 48000
        assert shape.sample_rate == 48000
    
    def test_audio_shape_defaults(self):
        """Test default parameters."""
        shape = AudioShape(batch=1)
        assert shape.batch == 1
        assert shape.channels == 1
        assert shape.samples == 48000
        assert shape.sample_rate == 48000
    
    def test_audio_shape_custom_sample_rate(self):
        """Test audio shape with different sample rates."""
        shape = AudioShape(batch=1, samples=16000, sample_rate=16000)
        assert shape.samples == 16000
        assert shape.sample_rate == 16000


class TestLatentShape:
    """Test LatentShape base class."""
    
    def test_latent_shape_creation(self):
        """Test LatentShape instantiation."""
        shape = LatentShape(batch=2, channels=64, frames=7, height=64, width=96)
        assert shape.batch == 2
        assert shape.channels == 64
        assert shape.frames == 7
        assert shape.height == 64
        assert shape.width == 96
    
    def test_latent_shape_defaults(self):
        """Test default parameters."""
        shape = LatentShape(batch=1)
        assert shape.batch == 1
        assert shape.channels == 64
        assert shape.frames == 7
        assert shape.height == 64
        assert shape.width == 96
    
    def test_latent_shape_compression_factors(self):
        """Test compression factor properties."""
        shape = LatentShape(batch=1)
        assert shape.spatial_factor == 8
        assert shape.temporal_factor == 7


class TestVideoLatentShape:
    """Test VideoLatentShape class."""
    
    def test_video_latent_shape_defaults(self):
        """Test VideoLatentShape defaults."""
        shape = VideoLatentShape()
        assert shape.batch_size == 1
        assert shape.channels == 64
        assert shape.num_frames == 7
        assert shape.height == 64
        assert shape.width == 96
    
    def test_video_latent_shape_custom_values(self):
        """Test VideoLatentShape with custom values."""
        shape = VideoLatentShape(
            batch_size=2,
            channels=32,
            num_frames=8,
            height=32,
            width=48
        )
        assert shape.batch_size == 2
        assert shape.channels == 32
        assert shape.num_frames == 8
        assert shape.height == 32
        assert shape.width == 48
    
    def test_video_latent_shape_seq_len(self):
        """Test sequence length calculation."""
        shape = VideoLatentShape(batch_size=1, channels=64, num_frames=7, height=64, width=96)
        expected_seq_len = 7 * 64 * 96
        assert shape.seq_len == expected_seq_len
    
    def test_video_latent_shape_from_pixel_shape(self):
        """Test creating from pixel shape."""
        pixel_shape = VideoPixelShape(height=512, width=768, num_frames=49)
        latent_shape = VideoLatentShape.from_pixel_shape(
            pixel_shape,
            latent_channels=64,
            batch_size=2
        )
        assert latent_shape.batch_size == 2
        assert latent_shape.channels == 64
        # 512 // 8 = 64 (spatial compression)
        assert latent_shape.height == 64
        assert latent_shape.width == 96


class TestAudioLatentShape:
    """Test AudioLatentShape class."""
    
    def test_audio_latent_shape_defaults(self):
        """Test AudioLatentShape defaults."""
        shape = AudioLatentShape()
        assert shape.batch_size == 1
        assert shape.channels == 64
        assert shape.length == 50
    
    def test_audio_latent_shape_custom_values(self):
        """Test AudioLatentShape with custom values."""
        shape = AudioLatentShape(
            batch_size=2,
            channels=32,
            length=25
        )
        assert shape.batch_size == 2
        assert shape.channels == 32
        assert shape.length == 25
    
    def test_audio_latent_shape_seq_len(self):
        """Test sequence length calculation."""
        shape = AudioLatentShape(batch_size=1, channels=64, length=50)
        assert shape.seq_len == 50
    
    def test_audio_latent_shape_from_video_shape(self):
        """Test creating from video pixel shape."""
        video_shape = VideoPixelShape(height=512, width=768, num_frames=24, fps=24.0)
        audio_latent = AudioLatentShape.from_video_pixel_shape(
            video_shape,
            latent_channels=64,
            audio_sample_rate=24000,
            audio_compression=480,
            batch_size=1
        )
        assert audio_latent.batch_size == 1
        assert audio_latent.channels == 64
        # 1 second @ 24kHz = 24000 samples / 480 = 50 frames
        assert audio_latent.length == 50


class TestLatentState:
    """Test LatentState container."""
    
    def test_latent_state_fields_exist(self):
        """Test that LatentState is a dataclass with expected structure."""
        assert hasattr(LatentState, '__dataclass_fields__')
    
    def test_latent_state_creation(self):
        """Test creating LatentState."""
        latent_tensor = torch.randn(2, 4, 64 * 96)
        
        state = LatentState(
            latent=latent_tensor,
            timestep=torch.tensor([500]),
        )
        assert state.latent.shape == (2, 4, 64 * 96)
        assert state.timestep.shape == (1,)


class TestGenerationConfig:
    """Test GenerationConfig dataclass."""
    
    def test_generation_config_fields_exist(self):
        """Test that GenerationConfig has expected structure."""
        assert hasattr(GenerationConfig, '__dataclass_fields__')
    
    def test_generation_config_creation(self):
        """Test creating GenerationConfig."""
        config = GenerationConfig(
            prompt="A video of a person dancing",
            negative_prompt="blurry, distorted",
            height=512,
            width=768,
            num_frames=49,
            num_inference_steps=50,
            guidance_scale=7.5,
        )
        assert config.prompt == "A video of a person dancing"
        assert config.height == 512
        assert config.width == 768
        assert config.num_frames == 49
    
    def test_generation_config_defaults(self):
        """Test default values."""
        config = GenerationConfig(
            prompt="Test prompt",
            negative_prompt=""
        )
        assert config.num_inference_steps == 50
        assert config.guidance_scale == 7.5
        assert config.precision == PrecisionMode.BF16
        assert config.device == DeviceType.CUDA


class TestShapeConsistency:
    """Test consistency between shape classes."""
    
    def test_video_pixel_shape_properties(self):
        """Test VideoPixelShape calculated properties."""
        # 1 second at 24fps
        shape = VideoPixelShape(height=512, width=1024, num_frames=24, fps=24.0)
        assert shape.aspect_ratio == 1024 / 512
        assert shape.duration_seconds == 1.0
    
    def test_latent_shape_has_compression_info(self):
        """Test LatentShape compression calculations."""
        latent = LatentShape(batch=1, channels=64, frames=7, height=64, width=96)
        assert latent.spatial_factor == 8
        assert latent.temporal_factor == 7
    
    def test_video_latent_from_pixel_consistency(self):
        """Test creating video latent from pixel shape maintains consistency."""
        pixel_shape = VideoPixelShape(height=512, width=768, num_frames=49)
        
        # Create latent shape from pixel shape using class method
        latent_shape = VideoLatentShape.from_pixel_shape(
            pixel_shape,
            latent_channels=64,
            batch_size=1
        )
        
        assert latent_shape.batch_size == 1
        assert latent_shape.channels == 64
        # 512 / 8 = 64, 768 / 8 = 96
        assert latent_shape.height == 64
        assert latent_shape.width == 96
    
    def test_audio_latent_from_video_shape(self):
        """Test creating audio latent from video shape."""
        video_shape = VideoPixelShape(height=512, width=768, num_frames=24, fps=24.0)
        
        audio_latent = AudioLatentShape.from_video_pixel_shape(
            video_shape,
            latent_channels=64,
            audio_sample_rate=24000,
            audio_compression=480,
            batch_size=1
        )
        
        assert audio_latent.batch_size == 1
        assert audio_latent.channels == 64
        # Duration = 24/24 = 1 second
        # Samples = 1 * 24000 = 24000
        # Latent length = 24000 / 480 = 50
        assert audio_latent.length == 50


class TestShapeEdgeCases:
    """Test edge cases for shape classes."""
    
    def test_video_pixel_shape_minimal_resolution(self):
        """Test with minimal resolution."""
        shape = VideoPixelShape(height=64, width=64, num_frames=1, fps=1.0)
        assert shape.duration_seconds == 1.0
    
    def test_audio_shape_zero_batch(self):
        """Test audio shape construction with batch dimension."""
        shape = AudioShape(batch=0, samples=0)
        assert shape.batch == 0
        assert shape.samples == 0
    
    def test_video_latent_shape_single_frame(self):
        """Test latent shape with single frame."""
        shape = VideoLatentShape(batch_size=1, num_frames=1, height=1, width=1)
        assert shape.seq_len == 1
    
    def test_audio_latent_shape_minimal_length(self):
        """Test audio latent with minimal length."""
        shape = AudioLatentShape(batch_size=1, length=1)
        assert shape.seq_len == 1
    
    def test_latent_shape_large_batch(self):
        """Test latent shape with large batch size."""
        shape = LatentShape(batch=256, channels=64, frames=7, height=64, width=96)
        assert shape.batch == 256

