provider "google" {
  project = var.project_id
  region  = var.region
}

locals {
  required_services = [
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "pubsub.googleapis.com",
    "cloudbuild.googleapis.com",
    "artifactregistry.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com",
    "servicenetworking.googleapis.com",
    "vpcaccess.googleapis.com",
    "sqladmin.googleapis.com"
  ]

  cloud_run_roles = [
    "roles/secretmanager.secretAccessor",
    "roles/pubsub.publisher",
    "roles/pubsub.subscriber",
    "roles/cloudsql.client",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/artifactregistry.reader"
  ]

  cloudsql_annotations = var.cloudsql_enabled ? {
    "run.googleapis.com/cloudsql-instances" = google_sql_database_instance.primary[0].connection_name
  } : {}

  vpc_annotations = var.vpc_enabled ? {
    "run.googleapis.com/vpc-access-connector" = google_vpc_access_connector.connector[0].name
    "run.googleapis.com/vpc-access-egress"    = var.vpc_egress
  } : {}

  worker_image = var.worker_container_image != "" ? var.worker_container_image : var.container_image
}

resource "google_project_service" "services" {
  for_each           = toset(local.required_services)
  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

resource "google_service_account" "cloud_run_sa" {
  account_id   = "aiprod-cloud-run"
  display_name = "AIPROD Cloud Run Service Account"
}

resource "google_project_iam_member" "cloud_run_roles" {
  for_each = toset(local.cloud_run_roles)

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

resource "google_compute_network" "vpc" {
  count                   = var.vpc_enabled ? 1 : 0
  name                    = var.vpc_name
  auto_create_subnetworks = false
  project                 = var.project_id

  depends_on = [google_project_service.services]
}

resource "google_compute_subnetwork" "subnet" {
  count         = var.vpc_enabled ? 1 : 0
  name          = var.vpc_subnet_name
  ip_cidr_range = var.vpc_subnet_cidr
  region        = var.region
  network       = google_compute_network.vpc[0].id
  project       = var.project_id
}

resource "google_compute_global_address" "private_service_range" {
  count         = var.vpc_enabled ? 1 : 0
  name          = "${var.vpc_name}-private-service"
  project       = var.project_id
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  address       = cidrhost(var.private_service_cidr, 0)
  prefix_length = tonumber(split("/", var.private_service_cidr)[1])
  network       = google_compute_network.vpc[0].id
}

resource "google_service_networking_connection" "private_vpc_connection" {
  count                   = var.vpc_enabled ? 1 : 0
  network                 = google_compute_network.vpc[0].id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_service_range[0].name]

  depends_on = [google_project_service.services]
}

resource "google_vpc_access_connector" "connector" {
  count        = var.vpc_enabled ? 1 : 0
  name         = var.vpc_connector_name
  region       = var.region
  project      = var.project_id
  ip_cidr_range = var.vpc_connector_cidr
  network      = google_compute_network.vpc[0].name

  depends_on = [google_project_service.services]
}

resource "google_pubsub_topic" "pipeline_jobs" {
  name    = var.pubsub_topic
  project = var.project_id

  depends_on = [google_project_service.services]
}

resource "google_pubsub_topic" "pipeline_results" {
  name    = var.pubsub_results_topic
  project = var.project_id

  depends_on = [google_project_service.services]
}

resource "google_pubsub_topic" "pipeline_dlq" {
  name    = var.pubsub_dlq_topic
  project = var.project_id

  depends_on = [google_project_service.services]
}

resource "google_pubsub_subscription" "worker_subscription" {
  name  = var.pubsub_worker_subscription
  topic = google_pubsub_topic.pipeline_jobs.name

  ack_deadline_seconds = 60
  retain_acked_messages = false
  message_retention_duration = "604800s"

  dead_letter_policy {
    dead_letter_topic     = google_pubsub_topic.pipeline_dlq.id
    max_delivery_attempts = 5
  }
}

resource "google_pubsub_subscription" "results_subscription" {
  name  = var.pubsub_results_subscription
  topic = google_pubsub_topic.pipeline_results.name

  ack_deadline_seconds = 60
  retain_acked_messages = false
  message_retention_duration = "604800s"
}

resource "google_secret_manager_secret" "secrets" {
  for_each = var.secrets

  secret_id = each.value
  project   = var.project_id

  replication {
    automatic = true
  }

  depends_on = [google_project_service.services]
}

resource "google_secret_manager_secret_version" "secret_versions" {
  for_each = { for k, v in var.secret_values : k => v if v != "" }

  secret      = google_secret_manager_secret.secrets[each.key].id
  secret_data = each.value
}

resource "google_sql_database_instance" "primary" {
  count   = var.cloudsql_enabled ? 1 : 0
  name    = var.cloudsql_instance_name
  project = var.project_id
  region  = var.region

  database_version = var.cloudsql_database_version

  settings {
    tier              = var.cloudsql_tier
    disk_size         = var.cloudsql_disk_size_gb
    disk_autoresize   = true
    availability_type = "ZONAL"

    dynamic "ip_configuration" {
      for_each = var.vpc_enabled ? [1] : []
      content {
        ipv4_enabled    = false
        private_network = google_compute_network.vpc[0].id
      }
    }

    backup_configuration {
      enabled            = true
      start_time         = "02:00"
      point_in_time_recovery_enabled = true
    }
  }

  depends_on = [google_project_service.services]
}

resource "google_sql_database" "primary" {
  count   = var.cloudsql_enabled ? 1 : 0
  name    = var.cloudsql_database_name
  project = var.project_id
  instance = google_sql_database_instance.primary[0].name
}

resource "google_sql_user" "primary" {
  count    = var.cloudsql_enabled && var.cloudsql_password != "" ? 1 : 0
  name     = var.cloudsql_user
  project  = var.project_id
  instance = google_sql_database_instance.primary[0].name
  password = var.cloudsql_password
}

resource "google_cloud_run_service" "api" {
  name     = var.service_name
  location = var.region

  metadata {
    annotations = {
      "run.googleapis.com/ingress"         = var.ingress
      "autoscaling.knative.dev/minScale"  = tostring(var.min_instances)
      "autoscaling.knative.dev/maxScale"  = tostring(var.max_instances)
    }
  }

  template {
    metadata {
      annotations = merge(
        {
          "autoscaling.knative.dev/minScale" = tostring(var.min_instances)
          "autoscaling.knative.dev/maxScale" = tostring(var.max_instances)
          "run.googleapis.com/timeoutSeconds" = tostring(var.timeout_seconds)
        },
        local.cloudsql_annotations,
        local.vpc_annotations
      )
    }

    spec {
      service_account_name = google_service_account.cloud_run_sa.email
      container_concurrency = var.concurrency

      containers {
        image = var.container_image

        resources {
          limits = {
            cpu    = var.cpu
            memory = var.memory
          }
        }

        dynamic "env" {
          for_each = var.env
          content {
            name  = env.key
            value = env.value
          }
        }

        dynamic "env" {
          for_each = var.secret_env
          content {
            name = env.key
            value_from {
              secret_key_ref {
                name = env.value
                key  = "latest"
              }
            }
          }
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  depends_on = [google_project_service.services]
}

resource "google_cloud_run_service" "worker" {
  name     = var.worker_service_name
  location = var.region

  metadata {
    annotations = {
      "run.googleapis.com/ingress"         = var.ingress
      "autoscaling.knative.dev/minScale"  = tostring(var.worker_min_instances)
      "autoscaling.knative.dev/maxScale"  = tostring(var.worker_max_instances)
    }
  }

  template {
    metadata {
      annotations = merge(
        {
          "autoscaling.knative.dev/minScale" = tostring(var.worker_min_instances)
          "autoscaling.knative.dev/maxScale" = tostring(var.worker_max_instances)
          "run.googleapis.com/timeoutSeconds" = tostring(var.timeout_seconds)
        },
        local.cloudsql_annotations,
        local.vpc_annotations
      )
    }

    spec {
      service_account_name = google_service_account.cloud_run_sa.email
      container_concurrency = var.worker_concurrency

      containers {
        image   = local.worker_image
        command = var.worker_command

        resources {
          limits = {
            cpu    = var.worker_cpu
            memory = var.worker_memory
          }
        }

        dynamic "env" {
          for_each = var.env
          content {
            name  = env.key
            value = env.value
          }
        }

        dynamic "env" {
          for_each = var.secret_env
          content {
            name = env.key
            value_from {
              secret_key_ref {
                name = env.value
                key  = "latest"
              }
            }
          }
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  depends_on = [google_project_service.services]
}
