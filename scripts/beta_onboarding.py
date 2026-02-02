"""
AIPROD V33 - Beta Onboarding Script
Automatise l'onboarding des clients beta avec génération de clés API et configuration GCS
"""

import os
import json
import secrets
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional
from pathlib import Path

from google.cloud import storage
from google.oauth2 import service_account


class BetaOnboardingManager:
    """Gère l'onboarding des clients beta AIPROD V33"""

    def __init__(self, gcp_project: str = "aiprod-484120"):
        """
        Initialise le gestionnaire d'onboarding
        
        Args:
            gcp_project: ID du projet GCP
        """
        self.gcp_project = gcp_project
        self.storage_client = storage.Client(project=gcp_project)
        self.bucket_name = f"{gcp_project}-aiprod-beta"
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self) -> None:
        """Crée le bucket GCS s'il n'existe pas"""
        try:
            bucket = self.storage_client.get_bucket(self.bucket_name)
            print(f"✅ Bucket existant: gs://{self.bucket_name}")
        except Exception:
            print(f"📦 Création du bucket: gs://{self.bucket_name}")
            bucket = self.storage_client.create_bucket(
                self.bucket_name,
                location="US"
            )
            # Configuration des accès
            # Type: ignore pour bucket.iam (API dynamique de Google Cloud)
            bucket.iam.reload()  # type: ignore
            bucket.update()

    def generate_api_key(self, client_name: str, client_id: Optional[str] = None) -> Dict:
        """
        Génère une clé API unique pour un client
        
        Args:
            client_name: Nom du client (agence)
            client_id: ID client (généré si non fourni)
            
        Returns:
            Dict avec les détails de la clé
        """
        if client_id is None:
            client_id = str(uuid.uuid4())[:8]

        api_key = f"aiprod_beta_{client_id}_{secrets.token_urlsafe(32)}"
        
        api_key_data = {
            "client_id": client_id,
            "client_name": client_name,
            "api_key": api_key,
            "created_at": datetime.utcnow().isoformat(),
            "tier": "PLATINUM",
            "status": "ACTIVE",
            "beta_end_date": (datetime.utcnow() + timedelta(days=90)).isoformat(),
            "tier_free_until": (datetime.utcnow() + timedelta(days=90)).isoformat(),
            "job_quota": 500,  # 500 jobs gratuits pendant beta
            "jobs_used": 0,
            "monthly_budget": 3000.0,  # $3000/mois maximum
            "spent_this_month": 0.0,
            "features": {
                "fast_track": True,
                "full_pipeline": True,
                "icc_control": True,
                "consistency_cache": True,
                "multi_user": True,
                "white_label": True,
                "analytics_dashboard": True
            }
        }

        return api_key_data

    def setup_gcs_folders(self, client_id: str, client_name: str) -> Dict:
        """
        Configure les dossiers GCS pour un client
        
        Args:
            client_id: ID du client
            client_name: Nom du client
            
        Returns:
            Dict avec les chemins GCS créés
        """
        bucket = self.storage_client.bucket(self.bucket_name)
        
        folders = {
            "input": f"clients/{client_id}/input/",
            "output": f"clients/{client_id}/output/",
            "cache": f"clients/{client_id}/cache/",
            "analytics": f"clients/{client_id}/analytics/"
        }

        # Crée les fichiers .gitkeep pour les dossiers
        for folder_key, folder_path in folders.items():
            blob = bucket.blob(f"{folder_path}.gitkeep")
            blob.upload_from_string(f"Folder: {folder_key}\nClient: {client_name}")
            print(f"  ✅ Créé: gs://{self.bucket_name}/{folder_path}")

        # Crée un fichier de configuration client
        config = {
            "client_id": client_id,
            "client_name": client_name,
            "created_at": datetime.utcnow().isoformat(),
            "folders": folders,
            "settings": {
                "default_quality_threshold": 0.7,
                "max_concurrent_jobs": 5,
                "default_preset": "brand_campaign"
            }
        }

        config_blob = bucket.blob(f"clients/{client_id}/config.json")
        config_blob.upload_from_string(
            json.dumps(config, indent=2),
            content_type="application/json"
        )
        print(f"  ✅ Configuration créée: gs://{self.bucket_name}/clients/{client_id}/config.json")

        return folders

    def create_api_credentials_file(self, client_id: str, api_key_data: Dict) -> str:
        """
        Crée un fichier de credentials pour le client
        
        Args:
            client_id: ID du client
            api_key_data: Données de la clé API
            
        Returns:
            Chemin du fichier créé
        """
        credentials_dir = Path("credentials") / client_id
        credentials_dir.mkdir(parents=True, exist_ok=True)

        credentials_file = credentials_dir / "aiprod_credentials.json"
        
        credentials_data = {
            "api_key": api_key_data["api_key"],
            "client_id": api_key_data["client_id"],
            "client_name": api_key_data["client_name"],
            "gcs_bucket": self.bucket_name,
            "gcs_folder": f"clients/{client_id}",
            "endpoints": {
                "api_base": "https://api.aiprod.app",
                "pipeline_run": "/api/v1/pipeline/run",
                "cost_estimate": "/api/v1/estimate-cost",
                "job_status": "/api/v1/job/{job_id}",
                "analytics": "/api/v1/analytics"
            },
            "tier": api_key_data["tier"],
            "created_at": api_key_data["created_at"],
            "beta_end_date": api_key_data["beta_end_date"]
        }

        with open(credentials_file, 'w') as f:
            json.dump(credentials_data, f, indent=2)

        print(f"✅ Credentials créées: {credentials_file}")
        return str(credentials_file)

    def register_client(self, client_name: str, contact_email: str) -> Dict:
        """
        Enregistre un client beta complet (clés + GCS + credentials)
        
        Args:
            client_name: Nom de l'agence
            contact_email: Email de contact
            
        Returns:
            Dict avec tous les détails de l'onboarding
        """
        print(f"\n🚀 Onboarding: {client_name} ({contact_email})")
        print("=" * 60)

        # 1. Génère la clé API
        api_key_data = self.generate_api_key(client_name)
        client_id = api_key_data["client_id"]
        print(f"\n✅ Clé API générée: {client_id}")

        # 2. Configure les dossiers GCS
        print(f"\n📦 Configuration GCS pour {client_id}:")
        gcs_folders = self.setup_gcs_folders(client_id, client_name)

        # 3. Crée le fichier credentials
        credentials_file = self.create_api_credentials_file(client_id, api_key_data)

        # 4. Prépare les données d'onboarding
        onboarding_data = {
            "status": "SUCCESS",
            "timestamp": datetime.utcnow().isoformat(),
            "client": {
                "id": client_id,
                "name": client_name,
                "email": contact_email
            },
            "api": {
                "key": api_key_data["api_key"],
                "tier": api_key_data["tier"],
                "quota": api_key_data["job_quota"]
            },
            "gcs": {
                "bucket": self.bucket_name,
                "folders": gcs_folders
            },
            "credentials_file": credentials_file,
            "metrics": {
                "success_threshold": "5 jobs/week",
                "quality_threshold": 0.75,
                "satisfaction_target": "85%+"
            },
            "timeline": {
                "onboarding_call": "30min",
                "first_job": "Day 1",
                "first_week_review": "Day 7",
                "monthly_check_in": "Day 30"
            }
        }

        # 5. Sauvegarde dans GCS
        bucket = self.storage_client.bucket(self.bucket_name)
        onboarding_blob = bucket.blob(f"clients/{client_id}/onboarding.json")
        onboarding_blob.upload_from_string(
            json.dumps(onboarding_data, indent=2),
            content_type="application/json"
        )

        print(f"\n✅ Onboarding complété!")
        print(f"  Clé API: {api_key_data['api_key'][:20]}...")
        print(f"  Tier: {api_key_data['tier']} (3 mois gratuit)")
        print(f"  Quota: {api_key_data['job_quota']} jobs")
        print(f"  Budget: ${api_key_data['monthly_budget']}")

        return onboarding_data

    def generate_onboarding_email(self, onboarding_data: Dict) -> str:
        """
        Génère le template email d'invitation au programme beta
        
        Args:
            onboarding_data: Données d'onboarding
            
        Returns:
            Corps de l'email formaté
        """
        client = onboarding_data["client"]
        api = onboarding_data["api"]
        
        email_template = f"""
Subject: 🎉 Welcome to AIPROD V33 Beta Program!

---

Hi {client['name']} Team,

We're excited to invite you to the AIPROD V33 Beta Program! 🚀

**Your Beta Access Details:**

📌 Client ID: {client['id']}
🔑 API Key: {api['key'][:30]}... (see credentials file)
📊 Tier: {api['tier']} (FREE for 3 months)
💼 Monthly Budget: $3,000 / Month
📦 Free Jobs: {api['quota']} during beta

**Getting Started (30 min):**

1. Download credentials file from your email attachment
2. Review the Beta Playbook: https://docs.aiprod.app/beta-playbook
3. Make your first API call:

   curl -X POST https://api.aiprod.app/api/v1/pipeline/run \\
     -H "Authorization: Bearer {api['key'][:20]}..." \\
     -H "Content-Type: application/json" \\
     -d '{{
       "content": "A majestic golden eagle soaring",
       "preset": "quick_social"
     }}'

**Beta Program Goals:**

✅ Generate 5+ videos per week
✅ Maintain quality score > 0.75
✅ Provide feedback via weekly surveys
✅ Document 2-3 use cases

**Support:**

📧 Beta Support: beta-support@aiprod.app
📅 Weekly Check-ins: Every Thursday 2pm PT
📋 Issue Tracker: https://github.com/aiprod/v33-beta-issues

**Timeline:**

🗓️ Onboarding Call: This week (30 min)
📍 First Video: Day 1
📊 First Review: Day 7
🎯 Monthly Check-in: Day 30

We can't wait to see what you create with AIPROD V33!

Let's build the future of video generation together. 🎬

Best regards,
The AIPROD Team

P.S. Reply to this email to schedule your onboarding call!

---
"""
        return email_template.strip()

    def list_beta_clients(self) -> list:
        """
        Liste tous les clients beta enregistrés
        
        Returns:
            Liste des configurations clients
        """
        bucket = self.storage_client.bucket(self.bucket_name)
        blobs = bucket.list_blobs(prefix="clients/")
        
        clients = {}
        for blob in blobs:
            if "onboarding.json" in blob.name:
                client_id = blob.name.split("/")[1]
                if client_id not in clients:
                    clients[client_id] = {
                        "id": client_id,
                        "onboarding_file": blob.name
                    }
        
        return list(clients.values())


def main():
    """Script principal d'onboarding"""
    import sys
    
    manager = BetaOnboardingManager()

    # Clients beta à onboarder
    beta_clients = [
        {
            "name": "Creative Agency XYZ",
            "email": "contact@creativeagency.com"
        },
        {
            "name": "Digital Studio ABC",
            "email": "hello@digitalstudio.com"
        },
        {
            "name": "Production House 123",
            "email": "ops@productionhouse.com"
        }
    ]

    print("\n" + "="*60)
    print("🚀 AIPROD V33 - BETA ONBOARDING")
    print("="*60)

    for client in beta_clients:
        # Onboard le client
        onboarding = manager.register_client(
            client_name=client["name"],
            contact_email=client["email"]
        )

        # Génère l'email d'invitation
        email = manager.generate_onboarding_email(onboarding)
        
        # Sauvegarde l'email
        email_file = f"emails/beta_invitation_{onboarding['client']['id']}.txt"
        Path("emails").mkdir(exist_ok=True)
        with open(email_file, 'w') as f:
            f.write(email)
        
        print(f"\n📧 Email template sauvegardé: {email_file}")

    # Affiche les clients onboardés
    print("\n" + "="*60)
    print("✅ BETA CLIENTS ONBOARDÉS")
    print("="*60)
    
    clients = manager.list_beta_clients()
    for client in clients:
        print(f"  • {client['id']}")

    print(f"\n✅ Total: {len(clients)} clients beta")
    print("\n📖 Voir docs/beta_playbook.md pour les instructions d'engagement\n")


if __name__ == "__main__":
    main()
