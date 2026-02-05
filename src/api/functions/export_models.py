"""
Modèles Pydantic pour les exports.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any, List
from datetime import datetime


class ExportRequest(BaseModel):
    """Requête pour exporter un job."""

    format: str = Field(
        default="json",
        description="Format d'export: json, csv, zip"
    )
    include_logs: bool = Field(
        default=False,
        description="Inclure les logs dans l'export"
    )
    include_metadata: bool = Field(
        default=True,
        description="Inclure les métadonnées"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "format": "json",
                "include_logs": True,
                "include_metadata": True
            }
        }
    )


class ExportResponse(BaseModel):
    """Réponse d'export."""

    job_id: str
    format: str
    size_bytes: int
    export_url: Optional[str] = None
    expires_at: Optional[datetime] = None
    status: str = "success"

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "job-123",
                "format": "json",
                "size_bytes": 2048,
                "export_url": "https://api.example.com/export/download/token123",
                "expires_at": "2026-02-06T12:00:00",
                "status": "success"
            }
        }
    )


class JobExportData(BaseModel):
    """Données exportées d'un job."""

    id: str
    user_id: str
    preset: str
    state: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    content_preview: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(json_schema_extra={"example": {
        "id": "job-123",
        "user_id": "user-456",
        "preset": "quick_social",
        "state": "completed",
        "created_at": "2026-02-05T10:00:00",
        "completed_at": "2026-02-05T10:05:30",
        "duration_seconds": 330,
        "content_preview": "Generate a marketing video about...",
        "metadata": {"cost": 1.25, "model": "gemini-pro"}
    }})


class ExportFormatsResponse(BaseModel):
    """Informations sur les formats d'export disponibles."""

    formats: Dict[str, Dict[str, Any]]

    model_config = ConfigDict(json_schema_extra={"example": {
        "formats": {
            "json": {
                "name": "JSON",
                "description": "Format JSON structuré",
                "mime_type": "application/json",
                "extension": ".json"
            },
            "csv": {
                "name": "CSV",
                "description": "Format tabulaire",
                "mime_type": "text/csv",
                "extension": ".csv"
            },
            "zip": {
                "name": "ZIP",
                "description": "Archive complète",
                "mime_type": "application/zip",
                "extension": ".zip"
            }
        }
    }})


class BulkExportRequest(BaseModel):
    """Requête pour exporter plusieurs jobs."""

    job_ids: List[str] = Field(..., description="IDs des jobs à exporter")
    format: str = Field(
        default="zip",
        description="Format d'export (json, csv, zip)"
    )
    include_logs: bool = Field(default=False)

    model_config = ConfigDict(json_schema_extra={"example": {
        "job_ids": ["job-1", "job-2", "job-3"],
        "format": "zip",
        "include_logs": False
    }})


class BulkExportResponse(BaseModel):
    """Réponse d'export en bulk."""

    batch_id: str
    job_count: int
    format: str
    size_bytes: int
    export_url: str
    expires_at: datetime
    status: str = "pending"

    model_config = ConfigDict(json_schema_extra={"example": {
        "batch_id": "batch-789",
        "job_count": 3,
        "format": "zip",
        "size_bytes": 5242880,
        "export_url": "https://api.example.com/export/download/batch-token",
        "expires_at": "2026-02-06T12:00:00",
        "status": "pending"
    }})
