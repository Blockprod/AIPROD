"""
Tests for P1.3: Real Implementations
Tests SemanticQA, VisualTranslator, and GCP Services Integrator with real APIs (mocked).
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta

# Import agents
from src.agents.semantic_qa import SemanticQA
from src.agents.visual_translator import VisualTranslator
from src.agents.gcp_services_integrator import GoogleCloudServicesIntegrator


class TestSemanticQA:
    """Test suite for SemanticQA with real Gemini integration."""

    @pytest.fixture
    def semantic_qa(self):
        """Create SemanticQA instance."""
        with patch.dict("os.environ", {"GEMINI_API_KEY": ""}):
            return SemanticQA()

    @pytest.mark.asyncio
    async def test_semantic_qa_mock_validation(self, semantic_qa):
        """Test SemanticQA returns valid mock validation when no Gemini API key."""
        outputs = {
            "render": {"video_url": "s3://video.mp4"},
            "content": "Test video content",
        }

        result = await semantic_qa.run(outputs)

        assert result is not None
        assert result["semantic_valid"] in [True, False]
        assert "overall_score" in result
        assert 0 <= result["overall_score"] <= 1
        assert result["provider"] == "mock"

    @pytest.mark.asyncio
    async def test_semantic_qa_with_mock_gemini(self, semantic_qa):
        """Test SemanticQA with mocked Gemini API response."""
        outputs = {
            "render": {"video_url": "s3://video.mp4"},
            "content": "Test video content",
        }

        # Mock Gemini response
        mock_response = Mock()
        mock_response.text = json.dumps(
            {
                "quality_score": 0.85,
                "relevance_score": 0.90,
                "coherence_score": 0.80,
                "completeness_score": 0.75,
                "overall_score": 0.82,
                "artifacts": [],
                "improvements": ["Add more detail"],
                "acceptable": True,
                "verdict": "Good quality",
            }
        )

        # Create instance with mocked Gemini
        semantic_qa.use_real_gemini = True
        semantic_qa.client = Mock()
        with patch.object(semantic_qa.client, "models") as mock_models:
            mock_models.generate_content.return_value = mock_response
            result = await semantic_qa.run(outputs)

            assert result["quality_score"] == 0.85
            assert result["overall_score"] == 0.82
            assert result["semantic_valid"] is True

    @pytest.mark.asyncio
    async def test_semantic_qa_invalid_json_fallback(self, semantic_qa):
        """Test SemanticQA falls back to mock when Gemini returns invalid JSON."""
        outputs = {"render": {"test": "data"}}

        # Mock invalid JSON response
        mock_response = Mock()
        mock_response.text = "Invalid JSON response"

        semantic_qa.use_real_gemini = True
        semantic_qa.client = Mock()
        with patch.object(semantic_qa.client, "models") as mock_models:
            mock_models.generate_content.return_value = mock_response
            result = await semantic_qa.run(outputs)

            # Should fallback to mock
            assert result["provider"] == "mock"
            assert "overall_score" in result

    @pytest.mark.asyncio
    async def test_semantic_qa_error_handling(self, semantic_qa):
        """Test SemanticQA handles Gemini API errors gracefully."""
        outputs = {"render": {"test": "data"}}

        semantic_qa.use_real_gemini = True
        semantic_qa.client = Mock()
        with patch.object(semantic_qa.client, "models") as mock_models:
            mock_models.generate_content.side_effect = Exception("API Error")
            result = await semantic_qa.run(outputs)

            # Should fallback to mock on error
            assert result["provider"] == "mock"
            assert result["overall_score"] == 0.75


class TestVisualTranslator:
    """Test suite for VisualTranslator with real Gemini integration."""

    @pytest.fixture
    def visual_translator(self):
        """Create VisualTranslator instance."""
        with patch.dict("os.environ", {"GEMINI_API_KEY": ""}):
            return VisualTranslator()

    @pytest.mark.asyncio
    async def test_visual_translator_mock(self, visual_translator):
        """Test VisualTranslator returns mock translation when no Gemini API key."""
        assets = {
            "title": "Product Launch",
            "description": "New AI Product",
            "color": "blue",
        }

        result = await visual_translator.run(assets, target_lang="fr")

        assert result["status"] == "adapted"
        assert result["language"] == "fr"
        assert result["provider"] == "mock"
        assert "adapted_assets" in result

    @pytest.mark.asyncio
    async def test_visual_translator_with_mock_gemini(self, visual_translator):
        """Test VisualTranslator with mocked Gemini API response."""
        assets = {"title": "Test", "description": "Description"}

        # Mock Gemini response
        mock_response = Mock()
        mock_response.text = json.dumps(
            {
                "adapted_assets": {
                    "title": {
                        "translated_text": "Teste",
                        "cultural_adaptations": ["Use local color preferences"],
                        "design_instructions": "Adapt typography",
                        "localization_notes": "French language version",
                    }
                },
                "language_code": "fr",
                "cultural_insights": ["Use professional tone"],
                "readiness_score": 0.85,
                "status": "adapted",
            }
        )

        # Set use_real_gemini to True and mock the model
        visual_translator.use_real_gemini = True
        visual_translator.client = Mock()
        with patch.object(visual_translator.client, "models") as mock_models:
            mock_models.generate_content.return_value = mock_response
            result = await visual_translator.run(assets, target_lang="fr")

            assert result["status"] == "adapted"
            assert result["language"] == "fr"
            assert result["readiness_score"] == 0.85

    @pytest.mark.asyncio
    async def test_visual_translator_multiple_languages(self, visual_translator):
        """Test VisualTranslator handles multiple languages."""
        assets = {"title": "Test"}

        for lang in ["en", "fr", "es", "de"]:
            result = await visual_translator.run(assets, target_lang=lang)
            assert result["language"] == lang

    @pytest.mark.asyncio
    async def test_visual_translator_error_handling(self, visual_translator):
        """Test VisualTranslator handles errors gracefully."""
        assets = {"title": "Test"}

        visual_translator.use_real_gemini = True
        visual_translator.client = Mock()
        with patch.object(visual_translator.client, "models") as mock_models:
            mock_models.generate_content.side_effect = Exception("API Error")
            result = await visual_translator.run(assets, target_lang="fr")

            # Should fallback to mock
            assert result["provider"] == "mock"


class TestGCPServicesIntegrator:
    """Test suite for GCP Services Integrator with real Cloud Storage."""

    @pytest.fixture
    def gcp_integrator(self):
        """Create GCP Services Integrator instance."""
        with patch.dict(
            "os.environ",
            {
                "GOOGLE_CLOUD_PROJECT": "aiprod-484120",
                "GCS_BUCKET_NAME": "aiprod-v33-assets",
            },
        ):
            with patch("google.cloud.storage.Client"):
                return GoogleCloudServicesIntegrator()

    @pytest.mark.asyncio
    async def test_gcp_integrator_initialization(self, gcp_integrator):
        """Test GCP Services Integrator initializes correctly."""
        assert gcp_integrator.project_id == "aiprod-484120"
        assert gcp_integrator.bucket_name == "aiprod-v33-assets"

    @pytest.mark.asyncio
    async def test_gcp_integrator_mock_urls(self, gcp_integrator):
        """Test GCP Integrator generates correct mock storage URLs."""
        inputs = {
            "job_id": "test-job-123",
            "delivery_manifest": {"id": "manifest-123"},
            "video_path": "/tmp/video.mp4",
        }

        # Force use of mock
        gcp_integrator.use_real_storage = False

        result = await gcp_integrator.run(inputs)

        assert result["status"] == "success"
        assert "storage_urls" in result
        assert "video_assets" in result["storage_urls"]
        assert "gs://aiprod-v33-assets" in result["storage_urls"]["video_assets"]

    @pytest.mark.asyncio
    async def test_gcp_integrator_metrics_collection(self, gcp_integrator):
        """Test GCP Integrator collects metrics correctly."""
        inputs = {
            "job_id": "test-job-123",
            "pipeline_duration": 45,
            "vertex_ai_calls": 5,
            "storage_calls": 3,
            "total_cost": 2.50,
        }

        gcp_integrator.use_real_storage = False
        result = await gcp_integrator.run(inputs)

        assert "gcp_metrics" in result
        metrics = result["gcp_metrics"]
        assert metrics["pipeline_duration_seconds"] == 45
        assert metrics["costs"]["total"] == 2.50

    @pytest.mark.asyncio
    async def test_gcp_integrator_service_status(self, gcp_integrator):
        """Test GCP Integrator checks service status."""
        inputs = {"job_id": "test-job-123"}
        gcp_integrator.use_real_storage = False

        result = await gcp_integrator.run(inputs)

        assert "service_status" in result
        status = result["service_status"]
        assert "overall" in status
        assert "cloudStorage" in status

    @pytest.mark.asyncio
    async def test_gcp_integrator_error_handling(self, gcp_integrator):
        """Test GCP Integrator handles errors gracefully."""
        inputs = {"job_id": "test-job-123"}

        # Force an error
        gcp_integrator.use_real_storage = True
        gcp_integrator.storage_client = None

        result = await gcp_integrator.run(inputs)

        # Should return valid error response
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_gcp_integrator_timestamp_format(self, gcp_integrator):
        """Test GCP Integrator generates valid ISO timestamps."""
        timestamp = gcp_integrator._get_timestamp()

        # Should be ISO format
        assert timestamp.endswith("+00:00") or "T" in timestamp

        # Should be parseable
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


class TestP13Integration:
    """Integration tests for P1.3 real implementations."""

    @pytest.mark.asyncio
    async def test_semantic_qa_provides_scores(self):
        """Test SemanticQA provides consistent scoring."""
        with patch.dict("os.environ", {"GEMINI_API_KEY": ""}):
            qa = SemanticQA()

            outputs = {"render": {"test": "data"}}
            result = await qa.run(outputs)

            # All scores should be present
            assert "quality_score" in result
            assert "relevance_score" in result
            assert "coherence_score" in result
            assert "overall_score" in result

            # All scores in valid range
            for key in [
                "quality_score",
                "relevance_score",
                "coherence_score",
                "overall_score",
            ]:
                assert 0 <= result[key] <= 1

    @pytest.mark.asyncio
    async def test_visual_translator_supports_major_languages(self):
        """Test VisualTranslator supports major languages."""
        with patch.dict("os.environ", {"GEMINI_API_KEY": ""}):
            translator = VisualTranslator()

            languages = ["en", "fr", "es", "de", "it", "pt", "ja", "zh"]
            for lang in languages:
                result = await translator.run({"test": "asset"}, target_lang=lang)
                assert result["language"] == lang
                assert result["status"] == "adapted"

    @pytest.mark.asyncio
    async def test_gcp_integrator_provides_all_outputs(self):
        """Test GCP Integrator provides complete output structure."""
        with patch.dict(
            "os.environ",
            {
                "GOOGLE_CLOUD_PROJECT": "aiprod-484120",
                "GCS_BUCKET_NAME": "aiprod-v33-assets",
            },
        ):
            with patch("google.cloud.storage.Client"):
                integrator = GoogleCloudServicesIntegrator()
                integrator.use_real_storage = False

                inputs = {
                    "job_id": "test-123",
                    "pipeline_duration": 30,
                    "total_cost": 1.50,
                }

                result = await integrator.run(inputs)

                # Check all required fields
                assert result["status"] in ["success", "error"]
                assert "storage_urls" in result
                assert "gcp_metrics" in result
                assert "service_status" in result
                assert "timestamp" in result
