variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "europe-west1"
}

variable "service_name" {
  description = "Cloud Run service name"
  type        = string
  default     = "aiprod-v33-api"
}

variable "container_image" {
  description = "Container image URI (Artifact Registry or GCR)"
  type        = string
}

variable "cpu" {
  description = "CPU limit for Cloud Run container"
  type        = string
  default     = "2"
}

variable "memory" {
  description = "Memory limit for Cloud Run container"
  type        = string
  default     = "4Gi"
}

variable "min_instances" {
  description = "Minimum number of Cloud Run instances"
  type        = number
  default     = 0
}

variable "max_instances" {
  description = "Maximum number of Cloud Run instances"
  type        = number
  default     = 10
}

variable "concurrency" {
  description = "Max concurrent requests per Cloud Run instance"
  type        = number
  default     = 80
}

variable "timeout_seconds" {
  description = "Request timeout for Cloud Run (seconds)"
  type        = number
  default     = 3600
}

variable "ingress" {
  description = "Ingress settings for Cloud Run"
  type        = string
  default     = "all"
}

variable "env" {
  description = "Environment variables map for Cloud Run"
  type        = map(string)
  default     = {}
}

variable "pubsub_topic" {
  description = "Pub/Sub topic for pipeline jobs"
  type        = string
  default     = "aiprod-pipeline-jobs"
}

variable "pubsub_results_topic" {
  description = "Pub/Sub topic for pipeline results"
  type        = string
  default     = "aiprod-pipeline-results"
}

variable "pubsub_dlq_topic" {
  description = "Pub/Sub dead-letter topic"
  type        = string
  default     = "aiprod-pipeline-dlq"
}

variable "pubsub_worker_subscription" {
  description = "Subscription name for worker pull"
  type        = string
  default     = "aiprod-render-worker"
}

variable "pubsub_results_subscription" {
  description = "Subscription name for results processor"
  type        = string
  default     = "aiprod-results-processor"
}

variable "worker_service_name" {
  description = "Cloud Run worker service name"
  type        = string
  default     = "aiprod-v33-worker"
}

variable "worker_container_image" {
  description = "Worker container image URI (defaults to container_image)"
  type        = string
  default     = ""
}

variable "worker_min_instances" {
  description = "Minimum worker instances"
  type        = number
  default     = 1
}

variable "worker_max_instances" {
  description = "Maximum worker instances"
  type        = number
  default     = 5
}

variable "worker_concurrency" {
  description = "Worker concurrency"
  type        = number
  default     = 5
}

variable "worker_cpu" {
  description = "Worker CPU limit"
  type        = string
  default     = "4"
}

variable "worker_memory" {
  description = "Worker memory limit"
  type        = string
  default     = "4Gi"
}

variable "worker_command" {
  description = "Worker command"
  type        = list(string)
  default     = ["python", "-m", "src.workers.pipeline_worker", "--threads", "5"]
}

variable "artifact_registry_repo" {
  description = "Artifact Registry repository name"
  type        = string
  default     = "aiprod"
}

variable "secrets" {
  description = "Secret names to create in Secret Manager"
  type        = set(string)
  default     = [
    "GEMINI_API_KEY",
    "RUNWAY_API_KEY",
    "DATADOG_API_KEY",
    "GCS_BUCKET_NAME"
  ]
}

variable "secret_values" {
  description = "Optional secret values to create initial secret versions"
  type        = map(string)
  default     = {}
  sensitive   = true
}

variable "secret_env" {
  description = "Map of environment variable name to Secret Manager secret name"
  type        = map(string)
  default     = {}
}

variable "vpc_enabled" {
  description = "Enable VPC connector and private networking"
  type        = bool
  default     = true
}

variable "vpc_name" {
  description = "VPC network name"
  type        = string
  default     = "aiprod-v33-vpc"
}

variable "vpc_subnet_name" {
  description = "VPC subnet name"
  type        = string
  default     = "aiprod-v33-subnet"
}

variable "vpc_subnet_cidr" {
  description = "VPC subnet CIDR range"
  type        = string
  default     = "10.10.0.0/24"
}

variable "vpc_connector_name" {
  description = "Serverless VPC Access connector name"
  type        = string
  default     = "aiprod-v33-connector"
}

variable "vpc_connector_cidr" {
  description = "CIDR range for VPC connector"
  type        = string
  default     = "10.8.0.0/28"
}

variable "vpc_egress" {
  description = "VPC egress setting for Cloud Run"
  type        = string
  default     = "all-traffic"
}

variable "private_service_cidr" {
  description = "CIDR range for private service connection"
  type        = string
  default     = "10.20.0.0/16"
}

variable "cloudsql_enabled" {
  description = "Enable Cloud SQL instance provisioning"
  type        = bool
  default     = false
}

variable "cloudsql_instance_name" {
  description = "Cloud SQL instance name"
  type        = string
  default     = "aiprod-v33-postgres"
}

variable "cloudsql_database_version" {
  description = "Cloud SQL database version"
  type        = string
  default     = "POSTGRES_15"
}

variable "cloudsql_tier" {
  description = "Cloud SQL machine tier"
  type        = string
  default     = "db-custom-2-7680"
}

variable "cloudsql_disk_size_gb" {
  description = "Cloud SQL disk size in GB"
  type        = number
  default     = 50
}

variable "cloudsql_database_name" {
  description = "Cloud SQL database name"
  type        = string
  default     = "AIPROD"
}

variable "cloudsql_user" {
  description = "Cloud SQL database user"
  type        = string
  default     = "aiprod"
}

variable "enable_worker" {
  description = "Enable the Cloud Run worker service"
  type        = bool
  default     = false
}

variable "cloudsql_password" {
  description = "Cloud SQL database password"
  type        = string
  sensitive   = true
  default     = ""
}
