"""
Tests pour le GCP Services Integrator (Real Implementation with Cloud Storage)
"""

import pytest
import asyncio
from unittest.mock import patch
from src.agents.gcp_services_integrator import GoogleCloudServicesIntegrator


@pytest.mark.asyncio
async def test_gcp_integrator_run():
    """Test l'exécution de l'intégrateur GCP."""
    with patch("google.cloud.storage.Client"):
        integrator = GoogleCloudServicesIntegrator()
        integrator.use_real_storage = False  # Use mock

        inputs = {
            "delivery_manifest": {"id": "test-asset-001", "status": "approved"},
            "pipeline_duration": 120,
            "total_cost": 2.50,
        }

        result = await integrator.run(inputs)

        assert "gcp_metrics" in result
        assert "storage_urls" in result
        assert "service_status" in result
        assert result["status"] == "success"


@pytest.mark.asyncio
async def test_gcp_integrator_storage_urls():
    """Test la génération des URLs de stockage."""
    with patch("google.cloud.storage.Client"):
        integrator = GoogleCloudServicesIntegrator()
        integrator.use_real_storage = False  # Use mock

        inputs = {"job_id": "asset-123", "delivery_manifest": {"id": "asset-123"}}

        urls = await integrator._upload_to_storage(inputs)

        # Check structure matches new implementation
        assert "video_assets" in urls
        assert "manifest" in urls
        assert "asset-123" in urls["video_assets"]
        assert urls["video_assets"].startswith("gs://")


@pytest.mark.asyncio
async def test_gcp_integrator_metrics():
    """Test la collecte des métriques GCP."""
    with patch("google.cloud.storage.Client"):
        integrator = GoogleCloudServicesIntegrator()

        inputs = {"pipeline_duration": 300, "total_cost": 2.50}

        metrics = await integrator._collect_metrics(inputs)

        # Check new structure with pipeline_duration_seconds
        assert "pipeline_duration_seconds" in metrics
        assert "api_calls" in metrics
        assert "costs" in metrics
        assert "resource_usage" in metrics
        assert metrics["costs"]["total"] == 2.50


@pytest.mark.asyncio
async def test_gcp_integrator_service_status():
    """Test la vérification du statut des services."""
    with patch("google.cloud.storage.Client"):
        integrator = GoogleCloudServicesIntegrator()
        integrator.use_real_storage = False

        status = await integrator._check_service_status()

        assert "overall" in status
        assert "cloudStorage" in status


def test_gcp_integrator_initialization():
    """Test l'initialisation de l'intégrateur GCP."""
    with patch("google.cloud.storage.Client"):
        integrator = GoogleCloudServicesIntegrator()
        assert integrator.name == "Google Cloud Services Integrator"
        assert "cloudStorage" in integrator.services
        assert "vertexAI" in integrator.services
