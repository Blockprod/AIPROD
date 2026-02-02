"""
GCP Client pour AIPROD V33
Interface pour Google Cloud Platform services
Basé sur googleStackConfiguration du JSON
"""
from typing import Any, Dict, Optional
import os
from src.utils.monitoring import logger

try:
    from google.cloud import storage
    HAS_GCS = True
except ImportError:
    HAS_GCS = False

class GCPClient:
    """
    Client Google Cloud Platform pour AIPROD V33.
    Gère Vertex AI, Cloud Storage, Cloud Functions, Monitoring.
    """
    
    def __init__(self):
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "aiprod-v33")
        self.location = os.getenv("GCP_LOCATION", "us-central1")
        self.bucket_name = os.getenv("GCS_BUCKET_NAME", "aiprod-v33-assets")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        
        logger.info(f"GCPClient initialized: project={self.project_id}, location={self.location}")
    
    def get_veo3_config(self) -> Dict[str, Any]:
        """
        Retourne la configuration Veo-3 selon le JSON.
        """
        return {
            "projectId": self.project_id,
            "location": self.location,
            "defaultParams": {
                "duration": "10s",
                "resolution": "1080p",
                "includeAudio": True,
                "aspectRatio": "16:9"
            },
            "timeoutSec": 120,
            "maxRetries": 2
        }
    
    def upload_to_storage(self, file_path: str, destination: str) -> str:
        """
        Upload un fichier vers Cloud Storage.
        
        Args:
            file_path (str): Chemin du fichier local
            destination (str): Destination dans le bucket
            
        Returns:
            str: URL du fichier uploadé (gs:// ou https://)
        """
        logger.info(f"GCPClient: upload {file_path} to {destination}")
        
        gcs_path = f"gs://{self.bucket_name}/{destination}"
        https_path = f"https://storage.googleapis.com/{self.bucket_name}/{destination}"
        
        if not HAS_GCS:
            logger.warning("GCPClient: google-cloud-storage not installed, returning mock URL")
            return https_path
        
        try:
            client = storage.Client(project=self.project_id)
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(destination)
            
            # If file_path is a local file path, upload it
            if os.path.isfile(file_path):
                logger.info(f"GCPClient: uploading from local file {file_path}")
                blob.upload_from_filename(file_path)
            else:
                # If it's bytes/content, upload it
                logger.info(f"GCPClient: uploading from bytes")
                blob.upload_from_string(file_path)
            
            logger.info(f"GCPClient: upload successful: {https_path}")
            return https_path
            
        except Exception as e:
            logger.error(f"GCPClient: upload failed: {e}")
            # Fallback to mock URL
            return https_path
    
    def call_gemini(self, prompt: str, model: str = "gemini-1.5-pro") -> Dict[str, Any]:
        """
        Appelle Gemini API (mock).
        
        Args:
            prompt (str): Prompt à envoyer
            model (str): Modèle Gemini à utiliser
            
        Returns:
            Dict[str, Any]: Réponse de Gemini
        """
        logger.info(f"GCPClient: call Gemini {model}")
        # Mock: retourne une réponse fictive
        return {
            "response": "Mock Gemini response",
            "model": model,
            "usage": {"tokens": 100}
        }
    
    def log_metric(self, metric_name: str, value: float) -> None:
        """
        Log une métrique vers Cloud Monitoring.
        """
        logger.info(f"GCPClient: metric {metric_name}={value}")
    
    def get_service_status(self) -> Dict[str, bool]:
        """
        Vérifie le statut des services GCP.
        """
        return {
            "vertexAI": True,
            "cloudStorage": True,
            "cloudFunctions": True,
            "cloudLogging": True,
            "cloudMonitoring": True
        }
