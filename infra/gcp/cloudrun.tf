locals {
  opensearch_host = google_compute_instance.opensearch.network_interface[0].network_ip

  api_url      = var.domains.api != "" ? "https://${var.domains.api}" : ""
  frontend_url = var.domains.frontend != "" ? "https://${var.domains.frontend}" : ""

  # Django ALLOWED_HOSTS: wildcard any *.run.app + the api custom domain.
  allowed_hosts = join(",", compact([".run.app", var.domains.api]))

  # Things every service's revision should wait on.
  run_deps = [
    google_secret_manager_secret_version.managed,
    google_project_iam_member.run,
    google_vpc_access_connector.connector,
  ]
}

# --------------------------------------------------------------------------- #
# genos-api (Django)                                                          #
# --------------------------------------------------------------------------- #
resource "google_cloud_run_v2_service" "api" {
  name                = "genos-api"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false # hackathon — allow teardown/recreate

  template {
    service_account = google_service_account.run.email

    vpc_access {
      connector = google_vpc_access_connector.connector.id
      egress    = "PRIVATE_RANGES_ONLY" # public egress stays default; private IPs via connector
    }

    scaling {
      min_instance_count = 1 # keep one warm — cold starts run migrate/collectstatic
      max_instance_count = 4
    }

    containers {
      image = var.images.api

      resources {
        limits            = { cpu = "2", memory = "2Gi" }
        startup_cpu_boost = true # full CPU during the migrate/collectstatic boot burst
      }

      # First boot runs `migrate` + `collectstatic` before gunicorn binds $PORT.
      # Give it a wide window (~7.5 min) so the revision-create doesn't time out
      # and roll back the apply. TODO(part-B): move migrate to a Cloud Run Job
      # and drop it from the serving CMD.
      startup_probe {
        tcp_socket {
          port = 8080
        }
        timeout_seconds   = 10
        period_seconds    = 15
        failure_threshold = 30
      }

      # --- plain env ---
      env {
        name  = "DJANGO_DEBUG"
        value = "false"
      }
      env {
        name  = "ALLOWED_HOSTS"
        value = local.allowed_hosts
      }
      env {
        name  = "OPENSEARCH_HOST"
        value = local.opensearch_host
      }
      env {
        name  = "OPENSEARCH_PORT"
        value = "9200"
      }
      env {
        name  = "LLM_PROVIDER"
        value = "gemini"
      }
      env {
        name  = "GEMINI_USE_VERTEX"
        value = "true"
      }
      env {
        name  = "GEMINI_PROJECT"
        value = var.project_id
      }
      env {
        name  = "GEMINI_LOCATION"
        value = var.vertex_location
      }
      env {
        name  = "EMBEDDING_PROVIDER"
        value = var.embedding_provider
      }
      env {
        name  = "BACKEND_BASE_URL"
        value = local.api_url
      }
      env {
        name  = "FRONTEND_BASE_URL"
        value = local.frontend_url
      }
      # Django CORS/CSRF allow-lists (DEBUG=false → only these origins may make
      # credentialed calls). Without the frontend origin, the browser blocks
      # api calls: "No 'Access-Control-Allow-Origin' header".
      env {
        name  = "CORS_ALLOWED_ORIGINS"
        value = local.frontend_url
      }
      env {
        name  = "CSRF_TRUSTED_ORIGINS"
        value = join(",", compact([local.frontend_url, local.api_url]))
      }

      # --- secret env ---
      env {
        name = "DATABASE_URL"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.managed["genos-database-url"].secret_id
            version = "latest"
          }
        }
      }
      env {
        name = "REDIS_URL"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.managed["genos-redis-url"].secret_id
            version = "latest"
          }
        }
      }
      env {
        name = "JWT_SECRET_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.managed["genos-jwt-secret"].secret_id
            version = "latest"
          }
        }
      }
      env {
        name = "DJANGO_SECRET_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.managed["genos-django-secret"].secret_id
            version = "latest"
          }
        }
      }
    }
  }

  depends_on = [google_secret_manager_secret_version.managed, google_project_iam_member.run]

  lifecycle {
    ignore_changes = [template[0].containers[0].image] # CI owns the image tag
  }
}

# --------------------------------------------------------------------------- #
# genos-sockets (Flask + Socket.IO)                                           #
# --------------------------------------------------------------------------- #
resource "google_cloud_run_v2_service" "sockets" {
  name                = "genos-sockets"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false # hackathon — allow teardown/recreate

  template {
    service_account = google_service_account.run.email

    # Socket.IO fan-out isn't cross-instance without a Redis adapter, so pin to
    # a single instance for the demo (events would otherwise misroute). The
    # agent path (api) is unaffected by this.
    scaling {
      min_instance_count = 1
      max_instance_count = 1
    }

    # Flask/Socket.IO holds long-lived connections; long request timeout so
    # websockets aren't culled mid-stream.
    timeout = "3600s"

    containers {
      image = var.images.sockets
      resources {
        limits = { cpu = "1", memory = "1Gi" }
      }

      env {
        name  = "DJANGO_BASEURL"
        value = local.api_url != "" ? "${local.api_url}/api/v2" : ""
      }
      env {
        name  = "CORS_ORIGINS"
        value = local.frontend_url
      }
      env {
        name = "JWT_SECRET_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.managed["genos-jwt-secret"].secret_id
            version = "latest"
          }
        }
      }
      env {
        name = "FLASK_SECRET_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.managed["genos-flask-secret"].secret_id
            version = "latest"
          }
        }
      }
    }
  }

  depends_on = [google_secret_manager_secret_version.managed, google_project_iam_member.run]

  lifecycle {
    ignore_changes = [template[0].containers[0].image]
  }
}

# --------------------------------------------------------------------------- #
# genos-collab (Node / Hocuspocus) — persists Yjs docs to Cloud SQL           #
# --------------------------------------------------------------------------- #
resource "google_cloud_run_v2_service" "collab" {
  name                = "genos-collab"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false # hackathon — allow teardown/recreate

  template {
    service_account = google_service_account.run.email

    vpc_access {
      connector = google_vpc_access_connector.connector.id
      egress    = "PRIVATE_RANGES_ONLY" # reach Cloud SQL private IP
    }

    scaling {
      min_instance_count = 1
      max_instance_count = 3
    }
    timeout = "3600s"

    containers {
      image = var.images.collab
      resources {
        limits = { cpu = "1", memory = "1Gi" }
      }

      env {
        name  = "DJANGO_BASE_URL"
        value = local.api_url != "" ? "${local.api_url}/api/v2" : ""
      }
      env {
        name  = "PG_HOST"
        value = google_sql_database_instance.pg.private_ip_address
      }
      env {
        name  = "PG_PORT"
        value = "5432"
      }
      env {
        name  = "PG_DATABASE"
        value = var.db_name
      }
      env {
        name  = "PG_USER"
        value = var.db_user
      }
      env {
        name = "PG_PASSWORD"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.managed["genos-db-password"].secret_id
            version = "latest"
          }
        }
      }
      env {
        name = "JWT_SECRET" # same VALUE as the API's JWT_SECRET_KEY, different name
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.managed["genos-jwt-secret"].secret_id
            version = "latest"
          }
        }
      }
    }
  }

  depends_on = [google_secret_manager_secret_version.managed, google_project_iam_member.run]

  lifecycle {
    ignore_changes = [template[0].containers[0].image]
  }
}

# --------------------------------------------------------------------------- #
# genos-frontend (static Vite build) — VITE_* is baked at BUILD time, so this #
# service carries no runtime env; the image must be built with prod URLs.     #
# --------------------------------------------------------------------------- #
resource "google_cloud_run_v2_service" "frontend" {
  name                = "genos-frontend"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false # hackathon — allow teardown/recreate

  template {
    service_account = google_service_account.run.email
    scaling {
      min_instance_count = 0
      max_instance_count = 3
    }
    containers {
      image = var.images.frontend
      resources {
        limits = { cpu = "1", memory = "512Mi" }
      }
    }
  }

  lifecycle {
    ignore_changes = [template[0].containers[0].image]
  }
}

# --------------------------------------------------------------------------- #
# Public access — all four services are internet-facing.                      #
# --------------------------------------------------------------------------- #
resource "google_cloud_run_v2_service_iam_member" "public" {
  for_each = {
    api      = google_cloud_run_v2_service.api.name
    sockets  = google_cloud_run_v2_service.sockets.name
    collab   = google_cloud_run_v2_service.collab.name
    frontend = google_cloud_run_v2_service.frontend.name
  }
  location = var.region
  name     = each.value
  role     = "roles/run.invoker"
  member   = "allUsers"
}
