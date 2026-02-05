#!/bin/bash
# Script de déploiement AIPROD

set -e

echo "========================================="
echo "AIPROD - Deployment Script"
echo "========================================="

# Configuration
PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-your-project-id}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-aiprod-v33}"
GCS_BUCKET_NAME="${GCS_BUCKET_NAME:-aiprod-v33-assets}"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME:latest"

ENV_FILE=".env"
ENV_VARS_FILE=".env.yaml"

# Mode de déploiement (default: cloudrun)
DEPLOY_MODE="${1:-cloudrun}"

echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Service: $SERVICE_NAME"
echo "Mode: $DEPLOY_MODE"
echo ""

# Charger automatiquement .env si présent (exporte les variables)
if [ -f "$ENV_FILE" ]; then
    echo "🔑 Loading env from $ENV_FILE"
    set -a
    source "$ENV_FILE"
    set +a
fi

# Préparer le flag d'env Cloud Run (priorité au fichier YAML)
if [ -f "$ENV_VARS_FILE" ]; then
    ENV_FLAG=(--set-env-vars-file="$ENV_VARS_FILE")
    echo "📄 Using env vars file: $ENV_VARS_FILE"
else
    ENV_FLAG=(--set-env-vars="GOOGLE_CLOUD_PROJECT=$PROJECT_ID,LOG_LEVEL=INFO,GCS_BUCKET_NAME=$GCS_BUCKET_NAME")
    echo "ℹ️  Using inline env vars"
fi

# Fonction de build Docker
build_image() {
    echo "🔨 Building Docker image..."
    docker build -t $IMAGE_NAME .
    echo "✅ Image built: $IMAGE_NAME"
}

# Fonction de push vers GCR
push_image() {
    echo "📤 Pushing image to GCR..."
    docker push $IMAGE_NAME
    echo "✅ Image pushed"
}

# Déploiement Cloud Run
deploy_cloudrun() {
    echo "🚀 Deploying to Cloud Run..."
    
    gcloud run deploy $SERVICE_NAME \
        --image=$IMAGE_NAME \
        --region=$REGION \
        --platform=managed \
        --allow-unauthenticated \
        --min-instances=1 \
        --max-instances=10 \
        --memory=2Gi \
        --cpu=2 \
        --timeout=300s \
        --port=8000 \
        "${ENV_FLAG[@]}" \
        --set-secrets="GEMINI_API_KEY=gemini-api-key:latest,RUNWAY_API_KEY=runway-api-key:latest"
    
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")
    echo ""
    echo "✅ Cloud Run deployment complete!"
    echo "🌐 Service URL: $SERVICE_URL"
    echo ""
    echo "Test with:"
    echo "curl $SERVICE_URL/health"
}

# Déploiement Cloud Functions
deploy_functions() {
    echo "🚀 Deploying Cloud Functions..."
    
    # Input Sanitizer
    gcloud functions deploy aiprod-sanitizer \
        --gen2 \
        --runtime=python311 \
        --region=$REGION \
        --entry-point=sanitize_input \
        --trigger-http \
        --allow-unauthenticated \
        --memory=512MB \
        --timeout=60s \
        --source=src/api/functions
    
    # Financial Orchestrator
    gcloud functions deploy aiprod-financial \
        --gen2 \
        --runtime=python311 \
        --region=$REGION \
        --entry-point=optimize_costs \
        --trigger-http \
        --allow-unauthenticated \
        --memory=256MB \
        --timeout=30s \
        --source=src/api/functions
    
    # QA Gate
    gcloud functions deploy aiprod-qa-gate \
        --gen2 \
        --runtime=python311 \
        --region=$REGION \
        --entry-point=validate_technical \
        --trigger-http \
        --allow-unauthenticated \
        --memory=512MB \
        --timeout=120s \
        --source=src/api/functions
    
    echo "✅ Cloud Functions deployed!"
}

# Main deployment flow
case "$DEPLOY_MODE" in
    cloudrun)
        build_image
        push_image
        deploy_cloudrun
        ;;
    functions)
        deploy_functions
        ;;
    all)
        build_image
        push_image
        deploy_cloudrun
        deploy_functions
        ;;
    *)
        echo "❌ Unknown deployment mode: $DEPLOY_MODE"
        echo "Usage: $0 [cloudrun|functions|all]"
        exit 1
        ;;
esac

echo ""
echo "========================================="
echo "✅ Deployment Complete!"
echo "========================================="
