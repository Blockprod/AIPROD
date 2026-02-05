#!/bin/bash
# GCP infrastructure setup for AIPROD
# This script sets up all required GCP resources, service accounts, and permissions
# Usage: ./setup-gcp.sh [project_id]

set -e

PROJECT_ID="${1:-aiprod-484120}"
REGION="us-central1"
SA_NAME="aiprod-sa"
BUCKET_NAME="aiprod-v33-assets"
TOPIC_NAME="aiprod-pipeline-events"
SUBSCRIPTION_NAME="aiprod-pipeline-worker"

echo "🔧 Setting up GCP infrastructure for AIPROD"
echo "   Project: $PROJECT_ID"
echo "   Region: $REGION"

# Set default project
gcloud config set project $PROJECT_ID

# 1. Enable required APIs
echo "📡 Enabling required APIs..."
gcloud services enable \
    compute.googleapis.com \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    containerregistry.googleapis.com \
    pubsub.googleapis.com \
    storage-api.googleapis.com \
    cloudlogging.googleapis.com \
    monitoring.googleapis.com \
    artifactregistry.googleapis.com \
    iam.googleapis.com

# 2. Create service account
echo "👤 Creating service account..."
if gcloud iam service-accounts describe ${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com > /dev/null 2>&1; then
    echo "   Service account already exists, skipping creation"
else
    gcloud iam service-accounts create $SA_NAME \
        --display-name="AIPROD Service Account" \
        --description="Service account for AIPROD application"
    echo "   ✓ Service account created"
fi

# 3. Create GCS bucket
echo "📦 Creating Cloud Storage bucket..."
if gsutil ls -b gs://$BUCKET_NAME > /dev/null 2>&1; then
    echo "   Bucket already exists, skipping creation"
else
    gsutil mb -p $PROJECT_ID -l $REGION gs://$BUCKET_NAME
    gsutil versioning set on gs://$BUCKET_NAME
    echo "   ✓ Bucket created with versioning enabled"
fi

# 4. Set bucket CORS for signed URLs
echo "🔐 Configuring bucket CORS..."
cat > /tmp/cors.json << 'EOF'
[
  {
    "origin": ["*"],
    "method": ["GET", "HEAD"],
    "responseHeader": ["Content-Type"],
    "maxAgeSeconds": 3600
  }
]
EOF
gsutil cors set /tmp/cors.json gs://$BUCKET_NAME
echo "   ✓ CORS configured"

# 5. Grant service account permissions
echo "🔑 Setting IAM permissions..."

# Cloud Storage access
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member=serviceAccount:${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com \
    --role=roles/storage.objectAdmin \
    --condition=None

# Cloud Pub/Sub access
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member=serviceAccount:${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com \
    --role=roles/pubsub.editor \
    --condition=None

# Cloud Logging access
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member=serviceAccount:${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com \
    --role=roles/logging.logWriter \
    --condition=None

# Cloud Monitoring access
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member=serviceAccount:${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com \
    --role=roles/monitoring.metricWriter \
    --condition=None

# Artifact Registry access (for Cloud Build)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member=serviceAccount:${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com \
    --role=roles/artifactregistry.writer \
    --condition=None

echo "   ✓ IAM permissions configured"

# 6. Create Pub/Sub topic and subscription
echo "📨 Creating Pub/Sub topic and subscription..."
if gcloud pubsub topics describe $TOPIC_NAME > /dev/null 2>&1; then
    echo "   Topic already exists, skipping creation"
else
    gcloud pubsub topics create $TOPIC_NAME \
        --message-retention-duration=7d
    echo "   ✓ Topic created"
fi

if gcloud pubsub subscriptions describe $SUBSCRIPTION_NAME > /dev/null 2>&1; then
    echo "   Subscription already exists, skipping creation"
else
    gcloud pubsub subscriptions create $SUBSCRIPTION_NAME \
        --topic=$TOPIC_NAME \
        --message-retention-duration=7d \
        --ack-deadline=60 \
        --retention-duration=7d
    echo "   ✓ Subscription created"
fi

# 7. Create Cloud Build service account with necessary permissions
echo "⚙️  Configuring Cloud Build..."
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')
CLOUD_BUILD_SA="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member=serviceAccount:$CLOUD_BUILD_SA \
    --role=roles/run.admin \
    --condition=None

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member=serviceAccount:$CLOUD_BUILD_SA \
    --role=roles/iam.serviceAccountUser \
    --condition=None

echo "   ✓ Cloud Build configured"

# 8. Create secrets in Secret Manager
echo "🔐 Creating secrets in Secret Manager..."
gcloud services enable secretmanager.googleapis.com

# Grant service account access to secrets
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member=serviceAccount:${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com \
    --role=roles/secretmanager.secretAccessor \
    --condition=None

echo "   Note: Set the following secrets manually:"
echo "   - DATABASE_URL (PostgreSQL connection string)"
echo "   - GEMINI_API_KEY (Google Gemini API key)"

# 9. Create Cloud Monitoring dashboard
echo "📊 Creating monitoring dashboard..."
cat > /tmp/dashboard.json << 'EOF'
{
  "displayName": "AIPROD Monitoring",
  "mosaicLayout": {
    "columns": 12,
    "tiles": [
      {
        "width": 6,
        "height": 4,
        "widget": {
          "title": "API Error Rate (5m)",
          "xyChart": {
            "dataSets": [{
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "resource.type=\"cloud_run_revision\" resource.labels.service_name=\"aiprod-v33-api\""
                }
              }
            }]
          }
        }
      },
      {
        "xPos": 6,
        "width": 6,
        "height": 4,
        "widget": {
          "title": "Worker Queue Depth",
          "xyChart": {
            "dataSets": [{
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "resource.type=\"pubsub_subscription\" resource.labels.subscription_id=\"aiprod-pipeline-worker\""
                }
              }
            }]
          }
        }
      }
    ]
  }
}
EOF

gcloud monitoring dashboards create --config-from-file=/tmp/dashboard.json || echo "   Dashboard may already exist"
echo "   ✓ Monitoring dashboard configured"

# 10. Print summary
echo ""
echo "✅ GCP infrastructure setup complete!"
echo ""
echo "📋 Summary:"
echo "   Project: $PROJECT_ID"
echo "   Service Account: ${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
echo "   Bucket: gs://$BUCKET_NAME"
echo "   Pub/Sub Topic: $TOPIC_NAME"
echo "   Region: $REGION"
echo ""
echo "🔐 Next steps:"
echo "   1. Set DATABASE_URL secret:"
echo "      gcloud secrets versions add DATABASE_URL --data-file=- <<< 'postgresql://...'"
echo "   2. Set GEMINI_API_KEY secret:"
echo "      gcloud secrets versions add GEMINI_API_KEY --data-file=- <<< 'YOUR_API_KEY'"
echo "   3. Deploy the application:"
echo "      ./scripts/deploy-gcp.sh production latest"
echo ""
