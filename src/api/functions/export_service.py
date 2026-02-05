"""
Service d'export pour les résultats de jobs.
Supporte JSON, CSV et ZIP avec métadonnées complètes.
"""

import json
import csv
import os
import logging
from io import BytesIO, StringIO
from typing import Any, Dict, Optional, List
from datetime import datetime
import zipfile
from enum import Enum

logger = logging.getLogger(__name__)


class ExportFormat(str, Enum):
    """Formats d'export supportés."""
    JSON = "json"
    CSV = "csv"
    ZIP = "zip"


class ExportService:
    """Service pour exporter les résultats de jobs."""

    def __init__(self):
        """Initialise le service d'export."""
        self.max_export_size = int(os.getenv("MAX_EXPORT_SIZE", "10737418240"))  # 10 GB
        logger.info(f"ExportService initialized (max size: {self.max_export_size} bytes)")

    def export_to_json(self, job_data: Dict[str, Any]) -> str:
        """
        Exporte les données d'un job en JSON.

        Args:
            job_data: Dictionnaire contenant les données du job

        Returns:
            Chaîne JSON formatée
        """
        export_data = {
            "metadata": {
                "exported_at": datetime.utcnow().isoformat(),
                "export_format": ExportFormat.JSON.value,
                "version": "1.0",
            },
            "job": job_data,
        }

        # Sérialiser en JSON avec indentation
        return json.dumps(export_data, indent=2, default=str)

    def export_to_csv(
        self, jobs_data: List[Dict[str, Any]], flatten: bool = True
    ) -> str:
        """
        Exporte les données de jobs en CSV.

        Args:
            jobs_data: Liste de dictionnaires contenant les données des jobs
            flatten: Si True, aplatit les structures JSON imbriquées

        Returns:
            Chaîne CSV
        """
        if not jobs_data:
            return ""

        output = StringIO()

        # Déterminer les colonnes
        if flatten:
            # Aplatir les données pour CSV
            flattened_jobs = []
            for job in jobs_data:
                flattened = self._flatten_dict(job)
                flattened_jobs.append(flattened)

            jobs_to_export = flattened_jobs
        else:
            jobs_to_export = jobs_data

        # Faire un dictionnaire pour obtenir toutes les clés
        all_keys = set()
        for job in jobs_to_export:
            all_keys.update(job.keys())

        fieldnames = sorted(list(all_keys))

        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for job in jobs_to_export:
            writer.writerow({field: job.get(field, "") for field in fieldnames})

        return output.getvalue()

    def export_to_zip(
        self, job_data: Dict[str, Any], include_file: Optional[bytes] = None
    ) -> BytesIO:
        """
        Exporte les données d'un job en ZIP.

        Contient:
        - metadata.json: Métadonnées du job
        - result.json: Résultats détaillés
        - logs.txt: Logs optionnels
        - output_file: Fichier de sortie optionnel (video, etc.)

        Args:
            job_data: Dictionnaire contenant les données du job
            include_file: Fichier binaire optionnel à inclure

        Returns:
            BytesIO contenant le fichier ZIP
        """
        zip_buffer = BytesIO()

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            # Ajouter metadata.json
            metadata = {
                "job_id": job_data.get("id"),
                "user_id": job_data.get("user_id"),
                "preset": job_data.get("preset"),
                "state": job_data.get("state"),
                "created_at": job_data.get("created_at"),
                "completed_at": job_data.get("completed_at"),
                "exported_at": datetime.utcnow().isoformat(),
            }
            zip_file.writestr("metadata.json", json.dumps(metadata, indent=2))

            # Ajouter result.json
            result_data = {
                "content": job_data.get("content"),
                "metadata": job_data.get("metadata", {}),
                "state_history": job_data.get("state_history", []),
                "result": job_data.get("result", {}),
            }
            zip_file.writestr("result.json", json.dumps(result_data, indent=2, default=str))

            # Ajouter logs.txt si disponible
            logs = job_data.get("logs")
            if logs:
                zip_file.writestr("logs.txt", logs)

            # Ajouter le fichier de sortie optionnel
            if include_file:
                zip_file.writestr("output_file", include_file)

        zip_buffer.seek(0)
        return zip_buffer

    def get_job_export_summary(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crée un résumé exportable pour un job.

        Args:
            job_data: Dictionnaire du job

        Returns:
            Dictionnaire avec les données format export
        """
        return {
            "id": job_data.get("id"),
            "user_id": job_data.get("user_id"),
            "preset": job_data.get("preset"),
            "state": job_data.get("state"),
            "created_at": job_data.get("created_at"),
            "updated_at": job_data.get("updated_at"),
            "started_at": job_data.get("started_at"),
            "completed_at": job_data.get("completed_at"),
            "duration_seconds": self._calculate_duration(
                job_data.get("started_at"), job_data.get("completed_at")
            ),
            "content_preview": job_data.get("content", "")[:500],
            "metadata": job_data.get("metadata", {}),
        }

    def validate_export_size(self, data: str | bytes) -> bool:
        """
        Valide que la taille de l'export ne dépasse pas la limite.

        Args:
            data: Données à exporter

        Returns:
            True si la taille est valide
        """
        if isinstance(data, str):
            size = len(data.encode("utf-8"))
        else:
            size = len(data)

        if size > self.max_export_size:
            logger.error(f"Export size {size} exceeds limit {self.max_export_size}")
            return False

        return True

    @staticmethod
    def _flatten_dict(d: Dict, parent_key: str = "", sep: str = "_") -> Dict:
        """
        Aplatit un dictionnaire imbriqué pour CSV.

        Args:
            d: Dictionnaire à aplatir
            parent_key: Clé parent (pour la récursion)
            sep: Séparateur pour les clés

        Returns:
            Dictionnaire aplati
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict):
                items.extend(ExportService._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, (list, tuple)):
                items.append((new_key, json.dumps(v)))
            else:
                items.append((new_key, v))

        return dict(items)

    @staticmethod
    def _calculate_duration(start: Optional[str], end: Optional[str]) -> Optional[float]:
        """
        Calcule la durée entre deux timestamps.

        Args:
            start: Timestamp de début (ISO format)
            end: Timestamp de fin (ISO format)

        Returns:
            Durée en secondes, ou None
        """
        if not start or not end:
            return None

        try:
            start_dt = datetime.fromisoformat(start) if isinstance(start, str) else start
            end_dt = datetime.fromisoformat(end) if isinstance(end, str) else end
            return (end_dt - start_dt).total_seconds()
        except Exception as e:
            logger.error(f"Error calculating duration: {e}")
            return None

    def get_export_formats_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Retourne les informations sur les formats d'export disponibles.

        Returns:
            Dictionnaire avec les infos sur chaque format
        """
        return {
            "json": {
                "name": "JSON",
                "description": "Format JSON structuré avec métadonnées complètes",
                "mime_type": "application/json",
                "extension": ".json",
                "best_for": "Integration avec autres systèmes, archivage",
            },
            "csv": {
                "name": "CSV",
                "description": "Format tabulaire compatible avec Excel/Sheets",
                "mime_type": "text/csv",
                "extension": ".csv",
                "best_for": "Analyse de données, import dans BI tools",
            },
            "zip": {
                "name": "ZIP",
                "description": "Archive contenant metadata, results et assets",
                "mime_type": "application/zip",
                "extension": ".zip",
                "best_for": "Export complet avec fichiers binaires",
            },
        }


# Singleton global
_export_service = None


def get_export_service() -> ExportService:
    """Retourne l'instance singleton du ExportService."""
    global _export_service
    if _export_service is None:
        _export_service = ExportService()
    return _export_service
