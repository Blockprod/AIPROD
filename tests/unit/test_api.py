import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from src.api.main import app, verify_token


# Mock authentication for tests
def mock_verify_token():
    """Mock authentication that returns a test user."""
    return {"uid": "test-user", "email": "test@example.com"}


# Override the dependency
app.dependency_overrides[verify_token] = mock_verify_token

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_pipeline_status():
    response = client.get("/pipeline/status")
    assert response.status_code == 200
    assert "state" in response.json()


def test_pipeline_run_success():
    """Test async /pipeline/run returns job_id and queued status."""
    # Mock database and pub/sub (authentication already overridden globally)
    with patch("src.api.main.get_db_session") as mock_db, patch(
        "src.api.main.get_pubsub_client"
    ) as mock_pubsub, patch("src.api.main.JobRepository") as MockRepo:

        # Setup mocks
        mock_session = MagicMock()
        mock_db.return_value = mock_session

        mock_job = MagicMock()
        mock_job.id = "job-async-123"

        mock_repo_instance = MagicMock()
        mock_repo_instance.create_job.return_value = mock_job
        MockRepo.return_value = mock_repo_instance

        mock_pubsub_client = MagicMock()
        mock_pubsub_client.publish_job.return_value = "msg-123"
        mock_pubsub.return_value = mock_pubsub_client

        payload = {"content": "Test video content", "priority": "high", "lang": "en"}
        headers = {"Authorization": "Bearer test-token"}
        response = client.post("/pipeline/run", json=payload, headers=headers)

        # With mocks, should succeed
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "queued"
        assert data["job_id"] == "job-async-123"


def test_pipeline_run_missing_content():
    """Test validation error for missing content."""
    payload = {"priority": "high"}
    headers = {"Authorization": "Bearer test-token"}
    response = client.post("/pipeline/run", json=payload, headers=headers)
    assert response.status_code == 422  # Validation error


def test_icc_data():
    response = client.get("/icc/data")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
