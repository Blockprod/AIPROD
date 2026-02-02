#!/bin/bash
# Deployment script for AIPROD V33 to Google Cloud
# Usage: ./deploy.sh [environment] [version]

set -e

# Configuration
PROJECT_ID="aiprod-484120"
REGION="us-central1"
GCR_HOSTNAME="gcr.io"
IMAGE_NAME="aiprod-v33"
ENVIRONMENT="${1:-production}"
VERSION="${2:-latest}"

echo "🚀 Deploying AIPROD V33 to $ENVIRONMENT environment"
echo "   Project: $PROJECT_ID"
echo "   Region: $REGION"
echo "   Image: $GCR_HOSTNAME/$PROJECT_ID/$IMAGE_NAME:$VERSION"

# Set GCP project
gcloud config set project $PROJECT_ID

# Build and push Docker image
echo "📦 Building Docker image..."
gcloud builds submit \
    --config=cloudbuild.yaml \
    --substitutions="_IMAGE_NAME=$IMAGE_NAME,_GCR_HOSTNAME=$GCR_HOSTNAME" \
    --async

# Wait for build
echo "⏳ Waiting for build to complete..."
gcloud builds log $(gcloud builds list --limit=1 --format='value(ID)') --stream

echo "✅ Build complete!"

# Deploy API service to Cloud Run
echo "🌐 Deploying API service..."
gcloud run deploy "${IMAGE_NAME}-api" \
    --image="$GCR_HOSTNAME/$PROJECT_ID/$IMAGE_NAME:$VERSION" \
    --platform=managed \
    --region=$REGION \
    --memory=2Gi \
    --cpu=2 \
    --max-instances=10 \
    --min-instances=1 \
    --allow-unauthenticated \
    --set-env-vars="GOOGLE_CLOUD_PROJECT=$PROJECT_ID,GCS_BUCKET_NAME=aiprod-v33-assets,ENVIRONMENT=$ENVIRONMENT" \
    --service-account=aiprod-sa@$PROJECT_ID.iam.gserviceaccount.com

# Deploy worker service to Cloud Run
echo "👷 Deploying worker service..."
gcloud run deploy "${IMAGE_NAME}-worker" \
    --image="$GCR_HOSTNAME/$PROJECT_ID/$IMAGE_NAME:$VERSION" \
    --platform=managed \
    --region=$REGION \
    --memory=4Gi \
    --cpu=4 \
    --max-instances=5 \
    --min-instances=1 \
    --no-allow-unauthenticated \
    --command="python,-m,src.workers.pipeline_worker,--threads,5" \
    --set-env-vars="GOOGLE_CLOUD_PROJECT=$PROJECT_ID,GCS_BUCKET_NAME=aiprod-v33-assets,ENVIRONMENT=$ENVIRONMENT" \
    --service-account=aiprod-sa@$PROJECT_ID.iam.gserviceaccount.com

# Update monitoring and alerts
echo "📊 Updating monitoring configuration..."
gcloud monitoring policies list --format="value(name)" | \
while read policy; do
    gcloud alpha monitoring policies update $policy \
        --notification-channels=$(gcloud alpha monitoring channels list --format="value(name)" --filter="displayName:aiprod") \
        || true
done

echo "✅ Deployment complete!"
echo "📋 Service URLs:"
echo "   API: $(gcloud run services describe ${IMAGE_NAME}-api --region=$REGION --format='value(status.url)')"
echo "   Worker logs: gcloud logging read 'resource.service_name=${IMAGE_NAME}-worker' --limit 50 --region=$REGION"
