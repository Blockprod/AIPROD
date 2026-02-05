#!/bin/bash
# Script d'initialisation GCP pour AIPROD

set -e

echo "========================================="
echo "AIPROD - GCP Project Setup"
echo "========================================="

# Configuration variables
PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-your-project-id}"
REGION="us-central1"
BUCKET_NAME="aiprod-v33-assets"

echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "Bucket: $BUCKET_NAME"
echo ""

# Vérifier gcloud CLI
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Install it from https://cloud.google.com/sdk/docs/install"
    exit 1
fi

echo "✅ gcloud CLI found"

# Configurer le projet
echo "📦 Setting up project..."
gcloud config set project $PROJECT_ID

# Activer les APIs nécessaires
echo "🔧 Enabling required APIs..."
gcloud services enable \
    run.googleapis.com \
    cloudfunctions.googleapis.com \
    aiplatform.googleapis.com \
    storage.googleapis.com \
    secretmanager.googleapis.com \
    cloudresourcemanager.googleapis.com \
    logging.googleapis.com \
    monitoring.googleapis.com

# Créer le bucket GCS
echo "📦 Creating GCS bucket..."
if gsutil ls -b gs://$BUCKET_NAME &> /dev/null; then
    echo "⚠️  Bucket $BUCKET_NAME already exists"
else
    gsutil mb -l $REGION gs://$BUCKET_NAME
    echo "✅ Bucket created"
fi

# Configurer le versioning
gsutil versioning set on gs://$BUCKET_NAME
echo "✅ Versioning enabled"

# Créer les secrets (placeholders)
echo "🔐 Creating secrets..."
echo "placeholder-key" | gcloud secrets create gemini-api-key \
    --data-file=- \
    --replication-policy="automatic" \
    2>/dev/null || echo "⚠️  Secret gemini-api-key already exists"

# Configurer IAM
echo "👤 Configuring IAM..."
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
SERVICE_ACCOUNT="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/aiplatform.user" \
    --condition=None \
    --quiet

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/storage.objectAdmin" \
    --condition=None \
    --quiet

echo "✅ IAM configured"

echo ""
echo "========================================="
echo "✅ GCP Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Update secrets with real API keys:"
echo "   echo 'YOUR_REAL_KEY' | gcloud secrets versions add gemini-api-key --data-file=-"
echo ""
echo "2. Deploy the application:"
echo "   ./scripts/deploy.sh"
echo ""
