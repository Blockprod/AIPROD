"""
PHASE 4 Integration Tests - GCP Production Hardening
=====================================================

Tests for PHASE 4 components:
- GCP Services adapter
- Performance optimization layer
- Collaboration features
- Video probe integration
- Gemini API client
- Load testing

PHASE 4 implementation (Weeks 11-13).
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import tempfile
import json

# Import system under test
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from aiprod_pipelines.api.adapters.gcp_services import GoogleCloudServicesAdapter
from aiprod_pipelines.api.optimization.performance import PerformanceOptimizationLayer
from aiprod_pipelines.api.collaboration.websocket import CollaborationRoom, CollaborationRoomManager
from aiprod_pipelines.api.integrations.video_probe import VideoProbe
from aiprod_pipelines.api.integrations.gemini_client import GeminiAPIClient
from aiprod_pipelines.api.schema.schemas import Context


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def base_context():
    """Base context for testing."""
    return Context(
        request_id="test_phase4_123",
        state="RENDER_EXECUTION",
        memory={
            "user_prompt": "Cinematic sunset over mountains",
            "duration_sec": 60,
            "complexity": 0.6,
            "budget_usd": 3.0,
            "generated_assets": [
                {
                    "id": "video_1",
                    "url": "gs://test-bucket/video1.mp4",
                    "duration_sec": 30,
                    "file_size_bytes": 5_000_000,
                    "resolution": "1920x1080",
                    "codec": "h264",
                    "bitrate": 4_000_000
                }
            ]
        },
        metadata={"created_at": datetime.now().isoformat()}
    )


# ============================================================================
# GCP Services Adapter Tests
# ============================================================================

class TestGCPServicesAdapter:
    """Test GCP services integration."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test GCP adapter initialization."""
        config = {
            "project_id": "test-project",
            "bucket_name": "test-bucket",
            "location": "us-central1"
        }
        
        adapter = GoogleCloudServicesAdapter(config)
        
        assert adapter.project_id == "test-project"
        assert adapter.bucket_name == "test-bucket"
        assert adapter.location == "us-central1"
    
    @pytest.mark.asyncio
    async def test_execute_uploads_assets(self, base_context):
        """Test asset upload to GCS."""
        config = {"project_id": "test-project", "bucket_name": "test-bucket"}
        adapter = GoogleCloudServicesAdapter(config)
        
        # Mock GCS operations
        with patch.object(adapter, '_upload_to_gcs') as mock_upload:
            mock_upload.return_value = [
                "gs://test-bucket/test_phase4_123/video_1.mp4"
            ]
            
            with patch.object(adapter, '_write_metrics') as mock_metrics:
                result = await adapter.execute(base_context)
                
                # Assert
                assert "delivery_manifest" in result["memory"]
                assert "download_urls" in result["memory"]["delivery_manifest"]
                assert len(result["memory"]["delivery_manifest"]["download_urls"]) == 1


# ============================================================================
# Performance Optimization Tests
# ============================================================================

class TestPerformanceOptimization:
    """Test performance optimization layer."""
    
    def test_initialization(self):
        """Test performance optimizer initialization."""
        optimizer = PerformanceOptimizationLayer()
        
        assert optimizer.gemini_cache is not None
        assert optimizer.consistency_cache is not None
        assert optimizer.adaptive_batch_cache is not None
    
    @pytest.mark.asyncio
    async def test_lazy_loading_configuration(self, base_context):
        """Test lazy loading for large assets."""
        optimizer = PerformanceOptimizationLayer({
            "lazy_loading_threshold": 1_000_000  # 1MB
        })
        
        # Add large asset
        base_context["memory"]["generated_assets"].append({
            "id": "video_2",
            "file_size_bytes": 50_000_000  # 50MB
        })
        
        result = await optimizer._apply_lazy_loading(base_context)
        
        # Check lazy loading configured
        assets = result["memory"]["generated_assets"]
        assert assets[1]["loading_strategy"] == "lazy"
        assert assets[1]["load_on_demand"] is True
    
    @pytest.mark.asyncio
    async def test_cache_hit_rate(self):
        """Test cache hit rate tracking."""
        optimizer = PerformanceOptimizationLayer()
        
        # Cache some results
        optimizer.cache_gemini_result("prompt1", "result1")
        optimizer.cache_gemini_result("prompt2", "result2")
        
        # Hit cache
        result1 = optimizer.get_cached_gemini_result("prompt1")
        assert result1 == "result1"
        
        # Miss cache
        result3 = optimizer.get_cached_gemini_result("prompt3")
        assert result3 is None
        
        # Check stats
        stats = optimizer.get_cache_stats()
        assert stats["gemini"]["hits"] == 1
        assert stats["gemini"]["misses"] == 1
    
    @pytest.mark.asyncio
    async def test_predictive_chunking(self, base_context):
        """Test predictive chunking at scene boundaries."""
        optimizer = PerformanceOptimizationLayer()
        
        # Add visual translation
        base_context["memory"]["visual_translation"] = {
            "scenes": [
                {"duration_sec": 10},
                {"duration_sec": 15},
                {"duration_sec": 5}
            ]
        }
        
        result = await optimizer._apply_predictive_chunking(base_context)
        
        # Check boundaries calculated
        assert "chunk_boundaries" in result["memory"]
        boundaries = result["memory"]["chunk_boundaries"]
        assert len(boundaries) == 3
        assert boundaries == [300, 750, 900]  # 10s, 25s, 30s at 30fps


# ============================================================================
# Collaboration Tests
# ============================================================================

class TestCollaboration:
    """Test collaboration features."""
    
    @pytest.mark.asyncio
    async def test_room_creation(self):
        """Test collaboration room creation."""
        room = CollaborationRoom("job_123")
        
        assert room.job_id == "job_123"
        assert len(room.participants) == 0
        assert len(room.comments) == 0
    
    @pytest.mark.asyncio
    async def test_comment_broadcast(self):
        """Test comment broadcasting."""
        room = CollaborationRoom("job_123")
        
        # Add comment
        await room.broadcast_comment({
            "user_id": "user1",
            "text": "Great video!",
            "asset_id": "video_1"
        })
        
        assert len(room.comments) == 1
        assert room.comments[0]["user_id"] == "user1"
        assert room.comments[0]["text"] == "Great video!"
    
    @pytest.mark.asyncio
    async def test_approval_tracking(self):
        """Test approval tracking."""
        room = CollaborationRoom("job_123")
        
        # Record approval
        await room.record_approval({
            "user_id": "user1",
            "asset_id": "video_1"
        })
        
        # Record rejection
        await room.record_rejection({
            "user_id": "user2",
            "asset_id": "video_1",
            "reason": "Needs revision"
        })
        
        status = room.get_approval_status()
        assert status["total_approvals"] == 1
        assert status["total_rejections"] == 1
    
    def test_room_manager(self):
        """Test room manager."""
        manager = CollaborationRoomManager()
        
        # Create rooms
        room1 = manager.get_or_create_room("job_1")
        room2 = manager.get_or_create_room("job_2")
        
        assert room1.job_id == "job_1"
        assert room2.job_id == "job_2"
        assert len(manager.rooms) == 2


# ============================================================================
# Video Probe Tests
# ============================================================================

class TestVideoProbe:
    """Test ffprobe integration."""
    
    def test_initialization(self):
        """Test video probe initialization."""
        probe = VideoProbe()
        
        assert probe.ffprobe_path == "ffprobe"
        assert probe.timeout_sec == 30
    
    @pytest.mark.asyncio
    async def test_probe_video_mock(self):
        """Test video probing with mock data."""
        probe = VideoProbe()
        
        # Mock ffprobe output
        mock_probe_data = {
            "format": {
                "filename": "test.mp4",
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "duration": "30.0",
                "size": "5000000",
                "bit_rate": "1333333"
            },
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "r_frame_rate": "30/1",
                    "bit_rate": "4000000",
                    "pix_fmt": "yuv420p",
                    "duration": "30.0"
                },
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "sample_rate": "48000",
                    "channels": 2,
                    "bit_rate": "128000",
                    "duration": "30.0"
                }
            ]
        }
        
        # Parse mock data
        metadata = probe._parse_probe_data(mock_probe_data)
        
        # Assert
        assert metadata["format"]["duration"] == 30.0
        assert metadata["video_stream"]["codec_name"] == "h264"
        assert metadata["video_stream"]["width"] == 1920
        assert metadata["video_stream"]["height"] == 1080
        assert metadata["video_stream"]["fps"] == 30.0
        assert metadata["audio_stream"]["codec_name"] == "aac"


# ============================================================================
# Gemini API Client Tests
# ============================================================================

class TestGeminiAPIClient:
    """Test Gemini API integration."""
    
    def test_initialization(self):
        """Test client initialization."""
        client = GeminiAPIClient({"api_key": None})  # Mock mode
        
        assert client.model_name == "gemini-1.5-pro"
        assert client.temperature == 0.7
        assert client.max_tokens == 8000
    
    @pytest.mark.asyncio
    async def test_text_generation_mock(self):
        """Test text generation in mock mode."""
        client = GeminiAPIClient({"api_key": None})
        
        result = await client.generate_text("Test prompt")
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Creative Direction" in result
    
    @pytest.mark.asyncio
    async def test_video_analysis_mock(self):
        """Test video analysis in mock mode."""
        client = GeminiAPIClient({"api_key": None})
        
        result = await client.analyze_video(
            video_url="gs://test/video.mp4",
            prompt="Analyze video quality"
        )
        
        assert "visual_consistency" in result
        assert "style_coherence" in result
        assert "narrative_flow" in result
        assert "prompt_alignment" in result
        
        # Check scores are in valid range
        for score in result.values():
            if isinstance(score, (int, float)):
                assert 0 <= score <= 10
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test rate limiting enforcement."""
        client = GeminiAPIClient({
            "api_key": None,
            "rate_limit_rpm": 2  # Very low limit for testing
        })
        
        # Make requests
        start_time = asyncio.get_event_loop().time()
        
        for i in range(3):
            await client.generate_text(f"Prompt {i}")
        
        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time
        
        # Third request should be delayed
        assert duration >= 1.0  # Should wait at least 1 second


# ============================================================================
# End-to-End PHASE 4 Tests
# ============================================================================

class TestPhase4EndToEnd:
    """End-to-end tests for PHASE 4 components."""
    
    @pytest.mark.asyncio
    async def test_full_production_pipeline(self, base_context):
        """Test complete production pipeline with PHASE 4 features."""
        # Initialize components
        gcp_adapter = GoogleCloudServicesAdapter({
            "project_id": "test-project",
            "bucket_name": "test-bucket"
        })
        
        optimizer = PerformanceOptimizationLayer()
        
        # Apply optimizations
        ctx = await optimizer.optimize_for_performance(base_context)
        
        # Mock GCP operations
        with patch.object(gcp_adapter, '_upload_to_gcs') as mock_upload:
            mock_upload.return_value = ["gs://test-bucket/video.mp4"]
            
            with patch.object(gcp_adapter, '_write_metrics'):
                # Execute
                result = await gcp_adapter.execute(ctx)
                
                # Assert
                assert "gcp_metadata" in result["memory"]
                assert result["memory"]["lazy_loading_enabled"] is True


# ============================================================================
# Run Configuration
# ============================================================================

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ])
