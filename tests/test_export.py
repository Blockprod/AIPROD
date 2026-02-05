"""
Tests pour le service et les endpoints d'export.
"""

import pytest
import json
from io import BytesIO
import zipfile
from src.api.functions.export_service import ExportService, ExportFormat
from src.api.functions.export_models import ExportRequest


class TestExportService:
    """Tests pour le ExportService."""

    @pytest.fixture
    def export_service(self):
        """Crée une instance ExportService."""
        return ExportService()

    @pytest.fixture
    def sample_job_data(self):
        """Exemple de données de job."""
        return {
            "id": "job-123",
            "user_id": "user-456",
            "preset": "quick_social",
            "state": "completed",
            "created_at": "2026-02-05T10:00:00",
            "updated_at": "2026-02-05T10:05:30",
            "started_at": "2026-02-05T10:00:30",
            "completed_at": "2026-02-05T10:05:30",
            "content": "Generate a marketing video about our new product",
            "metadata": {
                "cost": 1.25,
                "model": "gemini-pro",
                "processing_time": 300
            }
        }

    def test_export_to_json(self, export_service, sample_job_data):
        """Test l'export en JSON."""
        json_str = export_service.export_to_json(sample_job_data)

        # Vérifier qu'on a une chaîne JSON valide
        assert isinstance(json_str, str)

        # Parser le JSON
        data = json.loads(json_str)

        # Vérifier la structure
        assert "metadata" in data
        assert "job" in data
        assert data["job"]["id"] == "job-123"
        assert data["metadata"]["exported_at"] is not None

    def test_export_to_csv_single(self, export_service, sample_job_data):
        """Test l'export en CSV (un seul job)."""
        csv_str = export_service.export_to_csv([sample_job_data])

        # Vérifier qu'on a du CSV
        assert isinstance(csv_str, str)
        assert "job-123" in csv_str  # Le job ID devrait être présentat
        assert "quick_social" in csv_str  # Le preset
        assert "user-456" in csv_str  # L'user ID

        # Vérifier la structure CSV
        lines = csv_str.strip().split("\n")
        assert len(lines) >= 2  # Header + au moins une ligne

    def test_export_to_csv_multiple(self, export_service):
        """Test l'export en CSV (plusieurs jobs)."""
        jobs = [
            {
                "id": "job-1",
                "user_id": "user-1",
                "preset": "quick_social",
                "state": "completed",
                "metadata": {}
            },
            {
                "id": "job-2",
                "user_id": "user-1",
                "preset": "brand_campaign",
                "state": "completed",
                "metadata": {}
            }
        ]

        csv_str = export_service.export_to_csv(jobs)

        # Vérifier les deux jobs
        assert "job-1" in csv_str
        assert "job-2" in csv_str
        assert "quick_social" in csv_str
        assert "brand_campaign" in csv_str

    def test_export_to_zip(self, export_service, sample_job_data):
        """Test l'export en ZIP."""
        zip_bytes = export_service.export_to_zip(sample_job_data)

        # Vérifier qu'on a des bytes
        assert isinstance(zip_bytes, BytesIO)

        # Extraire et vérifier le contenu du ZIP
        with zipfile.ZipFile(zip_bytes, 'r') as zip_file:
            names = zip_file.namelist()

            # Vérifier les fichiers attendus
            assert "metadata.json" in names
            assert "result.json" in names

            # Vérifier les contenus JSON
            metadata = json.loads(zip_file.read("metadata.json"))
            assert metadata["job_id"] == "job-123"

            result = json.loads(zip_file.read("result.json"))
            assert result["metadata"]["cost"] == 1.25

    def test_export_to_zip_with_file(self, export_service, sample_job_data):
        """Test l'export en ZIP avec fichier binaire."""
        include_file = b"video content here"

        zip_bytes = export_service.export_to_zip(sample_job_data, include_file=include_file)

        # Extraire et vérifier
        with zipfile.ZipFile(zip_bytes, 'r') as zip_file:
            names = zip_file.namelist()

            # Le fichier de sortie devrait être présent
            assert "output_file" in names

            # Vérifier le contenu du fichier
            output_content = zip_file.read("output_file")
            assert output_content == include_file

    def test_get_job_export_summary(self, export_service, sample_job_data):
        """Test la création d'un résumé exportable."""
        summary = export_service.get_job_export_summary(sample_job_data)

        # Vérifier les champs
        assert summary["id"] == "job-123"
        assert summary["user_id"] == "user-456"
        assert "duration_seconds" in summary
        assert summary["duration_seconds"] is not None  # Devrait calculer la durée

    def test_validate_export_size_string(self, export_service):
        """Test la validation de taille pour une chaîne."""
        small_string = "a" * 1000  # 1 KB
        large_string = "a" * (export_service.max_export_size + 1)

        assert export_service.validate_export_size(small_string) is True
        assert export_service.validate_export_size(large_string) is False

    def test_validate_export_size_bytes(self, export_service):
        """Test la validation de taille pour des bytes."""
        small_bytes = b"a" * 1000
        large_bytes = b"a" * (export_service.max_export_size + 1)

        assert export_service.validate_export_size(small_bytes) is True
        assert export_service.validate_export_size(large_bytes) is False

    def test_flatten_dict(self, export_service):
        """Test l'aplatissement de dictionnaire."""
        nested = {
            "id": "123",
            "metadata": {
                "cost": 1.5,
                "details": {
                    "model": "gemini"
                }
            }
        }

        flattened = export_service._flatten_dict(nested)

        # Vérifier les clés aplaties
        assert flattened["id"] == "123"
        assert flattened["metadata_cost"] == 1.5
        assert flattened["metadata_details_model"] == "gemini"

    def test_calculate_duration(self, export_service):
        """Test le calcul de durée."""
        start = "2026-02-05T10:00:00"
        end = "2026-02-05T10:05:30"

        duration = export_service._calculate_duration(start, end)

        # Devrait être 330 secondes (5 minutes 30 secondes)
        assert duration == 330

    def test_calculate_duration_none(self, export_service):
        """Test le calcul de durée avec None."""
        duration = export_service._calculate_duration(None, "2026-02-05T10:05:30")
        assert duration is None

        duration = export_service._calculate_duration("2026-02-05T10:00:00", None)
        assert duration is None

    def test_get_export_formats_info(self, export_service):
        """Test la récupération des infos de formats."""
        formats = export_service.get_export_formats_info()

        # Vérifier que tous les formats sont retournés
        assert "json" in formats
        assert "csv" in formats
        assert "zip" in formats

        # Vérifier la structure
        for format_name, format_info in formats.items():
            assert "name" in format_info
            assert "description" in format_info
            assert "mime_type" in format_info
            assert "extension" in format_info

    def test_export_empty_filename(self, export_service):
        """Test l'export avec pas de données."""
        csv_str = export_service.export_to_csv([])
        assert csv_str == ""


class TestExportModels:
    """Tests pour les modèles d'export."""

    def test_export_request_default(self):
        """Test le modèle ExportRequest par défaut."""
        request = ExportRequest()

        assert request.format == "json"
        assert request.include_logs is False
        assert request.include_metadata is True

    def test_export_request_custom(self):
        """Test le modèle ExportRequest avec valeurs custom."""
        request = ExportRequest(
            format="csv",
            include_logs=True,
            include_metadata=False
        )

        assert request.format == "csv"
        assert request.include_logs is True
        assert request.include_metadata is False


@pytest.mark.asyncio
class TestExportEndpoints:
    """Tests d'intégration pour les endpoints d'export."""

    async def test_export_formats_endpoint_structure(self):
        """Test la structure de l'endpoint export formats."""
        # Ceci est un test de structure d'endpoint
        # À exécuter avec FastAPI TestClient

        # L'endpoint devrait:
        # 1. Retourner une liste des formats supportés
        # 2. Inclure json, csv, zip
        # 3. Avoir des infos sur chaque format

        from src.api.functions.export_service import get_export_service
        service = get_export_service()
        formats_info = service.get_export_formats_info()

        assert "json" in formats_info
        assert "csv" in formats_info
        assert "zip" in formats_info

    async def test_export_pipeline_format_validation(self):
        """Test la validation du format d'export."""
        # Format invalide devrait retourner 400

        invalid_formats = ["xml", "pdf", "yaml", "txt"]

        for invalid_format in invalid_formats:
            # En production, chaque format invalide devrait être rejeté
            assert invalid_format not in [f.value for f in ExportFormat]

    async def test_export_pipeline_security(self):
        """Test la sécurité de l'endpoint export."""
        # L'export doit vérifier l'propriété du job
        # L'utilisateur doit être authentifié

        # Tests à faire avec FastAPI TestClient:
        # 1. Sans auth: 401 Unauthorized
        # 2. Avec mauvais user: 403 Forbidden
        # 3. Avec bon user: 200 OK
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
