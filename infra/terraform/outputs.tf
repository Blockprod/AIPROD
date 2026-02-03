output "cloud_run_url" {
  description = "Cloud Run service URL"
  value       = google_cloud_run_service.api.status[0].url
}

output "cloud_run_worker_url" {
  description = "Cloud Run worker service URL"
  value       = var.enable_worker ? google_cloud_run_service.worker[0].status[0].url : ""
}

output "service_account_email" {
  description = "Service account email for Cloud Run"
  value       = google_service_account.cloud_run_sa.email
}

output "pubsub_topic" {
  description = "Pub/Sub topic name"
  value       = google_pubsub_topic.pipeline_jobs.name
}

output "pubsub_results_topic" {
  description = "Pub/Sub results topic name"
  value       = google_pubsub_topic.pipeline_results.name
}

output "pubsub_dlq_topic" {
  description = "Pub/Sub dead-letter topic name"
  value       = google_pubsub_topic.pipeline_dlq.name
}

output "cloudsql_connection_name" {
  description = "Cloud SQL instance connection name"
  value       = try(google_sql_database_instance.primary[0].connection_name, null)
}

output "cloudsql_database" {
  description = "Cloud SQL database name"
  value       = try(google_sql_database.primary[0].name, null)
}
