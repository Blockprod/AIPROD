project_id = "aiprod-484120"
region     = "europe-west1"

service_name    = "aiprod-v33-api"
container_image = "gcr.io/aiprod-484120/aiprod-v33:latest"

cpu         = "2"
memory      = "4Gi"
min_instances = 1
max_instances = 10
concurrency   = 80
timeout_seconds = 3600

pubsub_topic = "aiprod-pipeline-jobs"
pubsub_results_topic = "aiprod-pipeline-results"
pubsub_dlq_topic = "aiprod-pipeline-dlq"
pubsub_worker_subscription = "aiprod-render-worker"
pubsub_results_subscription = "aiprod-results-processor"

env = {
  GOOGLE_CLOUD_PROJECT = "aiprod-484120"
  GCS_BUCKET_NAME      = "aiprod-v33-assets"
}

secret_env = {
  GEMINI_API_KEY  = "GEMINI_API_KEY"
  RUNWAY_API_KEY  = "RUNWAY_API_KEY"
  DATADOG_API_KEY = "DATADOG_API_KEY"
  GCS_BUCKET_NAME = "GCS_BUCKET_NAME"
}

vpc_enabled         = true
vpc_name            = "aiprod-v33-vpc"
vpc_subnet_name     = "aiprod-v33-subnet"
vpc_subnet_cidr     = "10.10.0.0/24"
vpc_connector_name  = "aiprod-v33-connector"
vpc_connector_cidr  = "10.8.0.0/28"
vpc_egress          = "all-traffic"
private_service_cidr = "10.20.0.0/16"

cloudsql_enabled        = true
cloudsql_instance_name  = "aiprod-v33-postgres"
cloudsql_database_name  = "aiprod_v33"
cloudsql_user           = "aiprod"
cloudsql_password       = "CHANGE_ME"

worker_service_name = "aiprod-v33-worker"
worker_container_image = "gcr.io/aiprod-484120/aiprod-v33:latest"
worker_min_instances = 1
worker_max_instances = 5
worker_concurrency = 5
worker_cpu = "4"
worker_memory = "4Gi"
