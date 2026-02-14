#!/bin/bash
# AIPROD API - Cloud Run Deployment Script
# PHASE 4 - GCP Production Hardening

set -e

# Configuration
PROJECT_ID="aiprod-production"
REGION="us-central1"
SERVICE_NAME="aiprod-api"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "========================================"
echo "AIPROD API - Cloud Run Deployment"
echo "========================================"
echo ""
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Service: ${SERVICE_NAME}"
echo ""

# Step 1: Configure gcloud
echo "[1/6] Configuring gcloud..."
gcloud config set project ${PROJECT_ID}
gcloud config set run/region ${REGION}

# Step 2: Enable required APIs
echo "[2/6] Enabling required GCP APIs..."
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    containerregistry.googleapis.com \
    storage-api.googleapis.com \
    logging.googleapis.com \
    monitoring.googleapis.com \
    secretmanager.googleapis.com

# Step 3: Build container image
echo "[3/6] Building container image..."
gcloud builds submit --tag ${IMAGE_NAME}:latest .

# Step 4: Create service account (if not exists)
echo "[4/6] Setting up service account..."
SA_NAME="${SERVICE_NAME}-sa"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

if ! gcloud iam service-accounts describe ${SA_EMAIL} &> /dev/null; then
    echo "Creating service account: ${SA_EMAIL}"
    gcloud iam service-accounts create ${SA_NAME} \
        --display-name="AIPROD API Service Account"
fi

# Grant permissions
echo "Granting permissions to service account..."
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/storage.objectAdmin"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/logging.logWriter"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/monitoring.metricWriter"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/secretmanager.secretAccessor"

# Step 5: Create secrets (if not exists)
echo "[5/6] Setting up secrets..."

# Gemini API key
if ! gcloud secrets describe gemini-api-key &> /dev/null; then
    echo "Create gemini-api-key secret manually:"
    echo "  echo -n 'YOUR_API_KEY' | gcloud secrets create gemini-api-key --data-file=-"
else
    echo "Secret gemini-api-key already exists"
fi

# Step 6: Deploy to Cloud Run
echo "[6/6] Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME}:latest \
    --platform managed \
    --region ${REGION} \
    --service-account ${SA_EMAIL} \
    --memory 8Gi \
    --cpu 4 \
    --timeout 300 \
    --min-instances 1 \
    --max-instances 100 \
    --concurrency 80 \
    --port 8080 \
    --allow-unauthenticated \
    --set-env-vars "PROJECT_ID=${PROJECT_ID},BUCKET_NAME=aiprod-generated-assets,LOCATION=${REGION},ENVIRONMENT=production" \
    --set-secrets "GEMINI_API_KEY=gemini-api-key:latest"

# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format='value(status.url)')

echo ""
echo "========================================"
echo "Deployment Complete!"
echo "========================================"
echo ""
echo "Service URL: ${SERVICE_URL}"
echo ""
echo "Test the API:"
echo "  curl ${SERVICE_URL}/health"
echo ""
echo "View logs:"
echo "  gcloud logging read \"resource.type=cloud_run_revision AND resource.labels.service_name=${SERVICE_NAME}\" --limit 50 --format json"
echo ""
echo "Monitor metrics:"
echo "  gcloud monitoring dashboards list"
echo ""
