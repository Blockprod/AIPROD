"""
Tests for concrete inference nodes.
"""

import pytest
import torch
from aiprod_pipelines.inference import (
    TextEncodeNode,
    DenoiseNode,
    UpsampleNode,
    DecodeVideoNode,
    AudioEncodeNode,
    CleanupNode,
    GraphContext,
)


class TestTextEncodeNode:
    """Tests for TextEncodeNode."""
    
    def test_initialization(self, mock_text_encoder):
        """Test node initialization."""
        node = TextEncodeNode(mock_text_encoder)
        assert node.text_encoder is not None
        assert node.max_length == 1024
    
    def test_input_keys(self, mock_text_encoder):
        """Test input keys specification."""
        node = TextEncodeNode(mock_text_encoder)
        assert "prompt" in node.input_keys
    
    def test_output_keys(self, mock_text_encoder):
        """Test output keys specification."""
        node = TextEncodeNode(mock_text_encoder)
        assert "embeddings" in node.output_keys
        assert "embeddings_pooled" in node.output_keys
    
    def test_single_string_prompt(self, mock_text_encoder):
        """Test encoding single string prompt."""
        node = TextEncodeNode(mock_text_encoder)
        context = GraphContext()
        context.inputs["prompt"] = "A cat walking"
        
        result = node.execute(context)
        
        assert "embeddings" in result
        assert "embeddings_pooled" in result
        assert result["embeddings"].ndim == 3  # [batch, seq, hidden]
        assert result["embeddings_pooled"].ndim == 2  # [batch, hidden]
    
    def test_list_prompts(self, mock_text_encoder):
        """Test encoding list of prompts."""
        node = TextEncodeNode(mock_text_encoder)
        context = GraphContext()
        context.inputs["prompt"] = ["A cat", "A dog"]
        
        result = node.execute(context)
        
        # With neg prompts: batch=4 (2 pos + 2 neg)
        assert result["embeddings"].shape[0] == 4
    
    def test_negative_prompt(self, mock_text_encoder):
        """Test with negative prompts."""
        node = TextEncodeNode(mock_text_encoder)
        context = GraphContext()
        context.inputs["prompt"] = "A cat"
        context.inputs["negative_prompt"] = "A dog"
        
        result = node.execute(context)
        
        # 2 embeddings (pos + neg)
        assert result["embeddings"].shape[0] == 2
    
    def test_missing_prompt_raises_error(self, mock_text_encoder):
        """Test that missing prompt raises error."""
        node = TextEncodeNode(mock_text_encoder)
        context = GraphContext()
        
        with pytest.raises(ValueError):
            node.execute(context)


class TestDenoiseNode:
    """Tests for DenoiseNode."""
    
    def test_initialization(self, mock_denoising_model, mock_scheduler):
        """Test node initialization."""
        node = DenoiseNode(mock_denoising_model, mock_scheduler)
        assert node.model is not None
        assert node.scheduler is not None
        assert node.num_steps == 20
        assert node.guidance_scale == 7.5
    
    def test_input_keys(self, mock_denoising_model, mock_scheduler):
        """Test input keys."""
        node = DenoiseNode(mock_denoising_model, mock_scheduler)
        assert "latents" in node.input_keys
        assert "embeddings" in node.input_keys
    
    def test_output_keys(self, mock_denoising_model, mock_scheduler):
        """Test output keys."""
        node = DenoiseNode(mock_denoising_model, mock_scheduler)
        assert "latents_denoised" in node.output_keys
    
    def test_denoise_execution(self, mock_denoising_model, mock_scheduler, sample_latents, sample_embeddings):
        """Test denoise execution."""
        node = DenoiseNode(mock_denoising_model, mock_scheduler, num_inference_steps=5)
        context = GraphContext()
        context.inputs["latents"] = sample_latents
        context.inputs["embeddings"] = sample_embeddings
        
        result = node.execute(context)
        
        assert "latents_denoised" in result
        assert result["latents_denoised"].shape == sample_latents.shape
    
    def test_custom_num_steps(self, mock_denoising_model, mock_scheduler):
        """Test custom number of steps."""
        node = DenoiseNode(mock_denoising_model, mock_scheduler, num_inference_steps=10)
        assert node.num_steps == 10
    
    def test_custom_guidance_scale(self, mock_denoising_model, mock_scheduler):
        """Test custom guidance scale."""
        node = DenoiseNode(mock_denoising_model, mock_scheduler, guidance_scale=5.0)
        assert node.guidance_scale == 5.0
    
    def test_missing_embeddings_raises_error(self, mock_denoising_model, mock_scheduler, sample_latents):
        """Test that missing embeddings raises error."""
        node = DenoiseNode(mock_denoising_model, mock_scheduler)
        context = GraphContext()
        context.inputs["latents"] = sample_latents
        
        with pytest.raises(ValueError):
            node.execute(context)


class TestUpsampleNode:
    """Tests for UpsampleNode."""
    
    def test_initialization(self, mock_upsampler):
        """Test node initialization."""
        node = UpsampleNode(mock_upsampler)
        assert node.upsampler is not None
        assert node.scale_factor == 2
    
    def test_input_keys(self, mock_upsampler):
        """Test input keys."""
        node = UpsampleNode(mock_upsampler)
        assert "latents" in node.input_keys
    
    def test_output_keys(self, mock_upsampler):
        """Test output keys."""
        node = UpsampleNode(mock_upsampler)
        assert "latents_upsampled" in node.output_keys
    
    def test_upsample_execution(self, mock_upsampler, sample_latents):
        """Test upsampling execution."""
        node = UpsampleNode(mock_upsampler, scale_factor=2)
        context = GraphContext()
        context.inputs["latents"] = sample_latents
        
        result = node.execute(context)
        
        assert "latents_upsampled" in result
        b, c, f, h, w = sample_latents.shape
        assert result["latents_upsampled"].shape == (b, c, f, h * 2, w * 2)
    
    def test_custom_scale_factor(self, mock_upsampler):
        """Test custom scale factor."""
        node = UpsampleNode(mock_upsampler, scale_factor=4)
        assert node.scale_factor == 4


class TestDecodeVideoNode:
    """Tests for DecodeVideoNode."""
    
    def test_initialization(self, mock_vae_decoder):
        """Test node initialization."""
        node = DecodeVideoNode(mock_vae_decoder)
        assert node.vae_decoder is not None
        assert node.scaling_factor == 0.18215
    
    def test_input_keys(self, mock_vae_decoder):
        """Test input keys."""
        node = DecodeVideoNode(mock_vae_decoder)
        assert "latents_denoised" in node.input_keys
    
    def test_output_keys(self, mock_vae_decoder):
        """Test output keys."""
        node = DecodeVideoNode(mock_vae_decoder)
        assert "video_frames" in node.output_keys
    
    def test_decode_execution(self, mock_vae_decoder, sample_latents):
        """Test decoding execution."""
        node = DecodeVideoNode(mock_vae_decoder)
        context = GraphContext()
        context.inputs["latents_denoised"] = sample_latents
        
        result = node.execute(context)
        
        assert "video_frames" in result
        # Check output shape (assuming 8x upsampling)
        b, c, f, h, w = sample_latents.shape
        frames = result["video_frames"]
        assert frames.shape[0] == b  # batch
        assert frames.shape[1] == f  # frames
    
    def test_custom_scaling_factor(self, mock_vae_decoder):
        """Test custom VAE scaling factor."""
        node = DecodeVideoNode(mock_vae_decoder, vae_scaling_factor=0.5)
        assert node.scaling_factor == 0.5


class TestAudioEncodeNode:
    """Tests for AudioEncodeNode."""
    
    def test_initialization(self):
        """Test node initialization."""
        class MockAudioEncoder:
            pass
        
        audio_encoder = MockAudioEncoder()
        node = AudioEncodeNode(audio_encoder)
        assert node.audio_encoder is not None
    
    def test_input_keys(self):
        """Test input keys."""
        class MockAudioEncoder:
            pass
        
        node = AudioEncodeNode(MockAudioEncoder())
        assert "audio_prompt" in node.input_keys
    
    def test_output_keys(self):
        """Test output keys."""
        class MockAudioEncoder:
            pass
        
        node = AudioEncodeNode(MockAudioEncoder())
        assert "audio_embeddings" in node.output_keys
    
    def test_audio_encoding(self):
        """Test audio encoding execution."""
        class MockAudioEncoder:
            pass
        
        node = AudioEncodeNode(MockAudioEncoder())
        context = GraphContext()
        context.inputs["audio_prompt"] = "Sound of rain"
        
        result = node.execute(context)
        
        assert "audio_embeddings" in result
        assert result["audio_embeddings"].shape[1] == 512  # embedding dim
    
    def test_empty_audio_prompt(self):
        """Test with empty audio prompt."""
        class MockAudioEncoder:
            pass
        
        node = AudioEncodeNode(MockAudioEncoder())
        context = GraphContext()
        
        result = node.execute(context)
        
        assert "audio_embeddings" in result
        # Should return silent embeddings


class TestCleanupNode:
    """Tests for CleanupNode."""
    
    def test_initialization(self):
        """Test node initialization."""
        node = CleanupNode()
        assert node is not None
    
    def test_input_keys(self):
        """Test input keys (should be empty)."""
        node = CleanupNode()
        assert len(node.input_keys) == 0
    
    def test_output_keys(self):
        """Test output keys."""
        node = CleanupNode()
        assert "memory_freed_mb" in node.output_keys
    
    def test_cleanup_execution(self):
        """Test cleanup execution."""
        node = CleanupNode()
        context = GraphContext()
        
        result = node.execute(context)
        
        assert "memory_freed_mb" in result
        assert isinstance(result["memory_freed_mb"], float)
        assert result["memory_freed_mb"] >= 0
