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
      # LLM-only region override. Empty by default → the app falls back to
      # GEMINI_LOCATION (above). Set var.vertex_llm_location to reach a Gemini
      # model that var.vertex_location doesn't serve (e.g. a *-preview on the
      # `global` endpoint) WITHOUT moving the Vertex embedder / reindex job off
      # var.vertex_location — the region its OpenSearch index was built in.
      # See infra/gcp/README.md "Vertex model + region".
      env {
        name  = "GEMINI_LLM_LOCATION"
        value = var.vertex_llm_location
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

      # --- OAuth login (optional; inert unless a client ID is set in var.oauth).
      # Each provider is wired ONLY when its client ID is non-empty, so by
      # default this references no Secret Manager version — critical, because a
      # secret_key_ref to an unseeded (versionless) secret makes the revision
      # fail to start. Enabling a provider therefore requires, in order:
      # (1) seed genos-<p>-oauth-secret + genos-oauth-token-key versions,
      # (2) set var.oauth.<p>_client_id in tfvars, (3) `terraform apply` — this
      # change does NOT ride the image-only CI deploy. See README + variables.tf.
      dynamic "env" {
        for_each = var.oauth.google_client_id != "" ? [1] : []
        content {
          name  = "GOOGLE_OAUTH_CLIENT_ID"
          value = var.oauth.google_client_id
        }
      }
      dynamic "env" {
        for_each = var.oauth.google_client_id != "" ? [1] : []
        content {
          name = "GOOGLE_OAUTH_CLIENT_SECRET"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.manual["genos-google-oauth-secret"].secret_id
              version = "latest"
            }
          }
        }
      }
      dynamic "env" {
        for_each = var.oauth.github_client_id != "" ? [1] : []
        content {
          name  = "GITHUB_OAUTH_CLIENT_ID"
          value = var.oauth.github_client_id
        }
      }
      dynamic "env" {
        for_each = var.oauth.github_client_id != "" ? [1] : []
        content {
          name = "GITHUB_OAUTH_CLIENT_SECRET"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.manual["genos-github-oauth-secret"].secret_id
              version = "latest"
            }
          }
        }
      }
      # Fernet key for tokens stored at rest. REQUIRED by the login callback
      # (crypto.encrypt runs on every OAuth sign-in), not connect-only — wire it
      # whenever EITHER provider is enabled.
      dynamic "env" {
        for_each = (var.oauth.google_client_id != "" || var.oauth.github_client_id != "") ? [1] : []
        content {
          name = "OAUTH_TOKEN_ENCRYPTION_KEY"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.manual["genos-oauth-token-key"].secret_id
              version = "latest"
            }
          }
        }
      }
      # Tavily API key for the agent's `search_web` tool. Optional — the tool is
      # always registered but no-ops with a "not configured" ToolError when the
      # key is absent. Wired only when var.tavily_enabled; seed the
      # genos-tavily-api-key secret version FIRST (Cloud Run won't start a
      # revision referencing an unversioned secret). See README.
      dynamic "env" {
        for_each = var.tavily_enabled ? [1] : []
        content {
          name = "TAVILY_API_KEY"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.manual["genos-tavily-api-key"].secret_id
              version = "latest"
            }
          }
        }
      }

      # --- Feature-parity secrets mirrored from Railway (each gated on its own
      # var.<feature>_enabled; seed the secret version FIRST). ---

      # Transactional email (Resend). Off ⇒ api uses the console backend (no send).
      dynamic "env" {
        for_each = var.email_enabled ? [1] : []
        content {
          name  = "EMAIL_BACKEND"
          value = "anymail.backends.resend.EmailBackend"
        }
      }
      dynamic "env" {
        for_each = var.email_enabled ? [1] : []
        content {
          name  = "DEFAULT_FROM_EMAIL"
          value = var.email_from
        }
      }
      dynamic "env" {
        for_each = var.email_enabled ? [1] : []
        content {
          name = "RESEND_API_KEY"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.manual["genos-resend-api-key"].secret_id
              version = "latest"
            }
          }
        }
      }

      # Anthropic Claude (user-selectable model). Model + max-tokens use code
      # defaults; only the key is wired.
      dynamic "env" {
        for_each = var.claude_enabled ? [1] : []
        content {
          name = "CLAUDE_API_KEY"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.manual["genos-claude-api-key"].secret_id
              version = "latest"
            }
          }
        }
      }

      # Web Push (VAPID). Public key + admin email are plain; private key is a
      # secret. The public key MUST equal the frontend's VITE_VAPID_PUBLIC_KEY.
      dynamic "env" {
        for_each = var.webpush_enabled ? [1] : []
        content {
          name  = "WEBPUSH_VAPID_PUBLIC_KEY"
          value = var.webpush_vapid_public_key
        }
      }
      dynamic "env" {
        for_each = var.webpush_enabled ? [1] : []
        content {
          name  = "WEBPUSH_VAPID_ADMIN_EMAIL"
          value = var.webpush_vapid_admin_email
        }
      }
      dynamic "env" {
        for_each = var.webpush_enabled ? [1] : []
        content {
          name = "WEBPUSH_VAPID_PRIVATE_KEY"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.manual["genos-webpush-vapid-private-key"].secret_id
              version = "latest"
            }
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
