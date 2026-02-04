#!/usr/bin/env python3
"""
Script pour créer/mettre à jour SUNO_API_KEY dans GCP Secret Manager
Usage: python scripts/setup_suno_secret.py <api_key>
"""
import sys
import os
from google.cloud import secretmanager


def create_or_update_secret(project_id: str, secret_id: str, secret_value: str):
    """
    Crée ou met à jour un secret dans GCP Secret Manager.
    """
    client = secretmanager.SecretManagerServiceClient()
    parent = f"projects/{project_id}"

    # Vérifier si le secret existe
    try:
        name = f"{parent}/secrets/{secret_id}"
        client.get_secret(request={"name": name})
        secret_exists = True
        print(f"✅ Secret '{secret_id}' existe déjà")
    except Exception:
        secret_exists = False
        print(f"📝 Secret '{secret_id}' n'existe pas, création...")

    # Créer le secret s'il n'existe pas
    if not secret_exists:
        created_secret = client.create_secret(
            request={
                "parent": parent,
                "secret_id": secret_id,
                "secret": {
                    "replication": {
                        "automatic": {}
                    },
                    "labels": {
                        "env": "production",
                        "service": "music-generator",
                        "provider": "suno"
                    }
                },
            }
        )
        print(f"✅ Secret créé: {created_secret.name}")

    # Ajouter la version du secret
    name = f"{parent}/secrets/{secret_id}"
    response = client.add_secret_version(
        request={
            "parent": name,
            "payload": {"data": secret_value.encode("UTF-8")},
        }
    )
    print(f"✅ Secret version créée: {response.name}")
    print(f"🔐 SUNO_API_KEY est maintenant accessible dans Cloud Run")


def get_secret(project_id: str, secret_id: str, version_id: str = "latest"):
    """
    Récupère un secret depuis GCP Secret Manager.
    """
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")


if __name__ == "__main__":
    # Configuration
    project_id = os.getenv("GCP_PROJECT_ID", "aiprod-484120")
    secret_id = "SUNO_API_KEY"

    if len(sys.argv) < 2:
        print("Usage: python scripts/setup_suno_secret.py <api_key>")
        print("\nExemple:")
        print("  python scripts/setup_suno_secret.py sk-12345...")
        print("\nOu utiliser une variable d'environnement:")
        print("  export SUNO_API_KEY='sk-12345...'")
        print("  python scripts/setup_suno_secret.py")
        sys.exit(1)

    api_key = sys.argv[1]

    if not api_key:
        print("❌ API key vide")
        sys.exit(1)

    print(f"🔐 Créer/mettre à jour SUNO_API_KEY dans {project_id}...")
    create_or_update_secret(project_id, secret_id, api_key)
    print("\n✅ Terminé!")
    print(f"\nPour vérifier:")
    print(f"  gcloud secrets list --project={project_id}")
    print(f"  gcloud secrets versions list {secret_id} --project={project_id}")
