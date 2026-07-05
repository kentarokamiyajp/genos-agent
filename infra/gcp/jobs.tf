# =========================================================================== #
# Scheduled maintenance jobs — the GCP equivalent of the three Railway crons  #
# (opensearch-reindexer / demo-user-cleanup / agent-judge-sampler).           #
#                                                                             #
# Pattern: one Cloud Run *Job* per cron (the same api image, with the serving #
# CMD overridden to the management command) + one Cloud Scheduler job that    #
# triggers it on the Railway cadence by POSTing to the Cloud Run Admin API    #
# `:run` endpoint.                                                            #
#                                                                             #
# Auth: a dedicated `genos-scheduler` SA holds `roles/run.invoker` on each    #
# job (that role includes `run.jobs.run`, verified against the live project). #
# The jobs themselves run AS the existing `genos-run` runtime SA, so they     #
# inherit Vertex (aiplatform.user), Secret Manager and Cloud SQL access with  #
# no new grants.                                                              #
# =========================================================================== #

locals {
  # Plain env shared by every management-command job. Mirrors the api service.
  # EMBEDDING_PROVIDER especially MUST match the api: the reindexer writes
  # *document* vectors while the api embeds the *query* at search time — if the
  # two use different providers/models the vectors land in different embedding
  # spaces and the k-NN lane returns noise (only BM25 keeps working).
  job_env = {
    DJANGO_DEBUG       = "false"
    ALLOWED_HOSTS      = local.allowed_hosts
    OPENSEARCH_HOST    = local.opensearch_host
    OPENSEARCH_PORT    = "9200"
    LLM_PROVIDER       = "gemini"
    GEMINI_USE_VERTEX  = "true"
    GEMINI_PROJECT     = var.project_id
    GEMINI_LOCATION    = var.vertex_location
    EMBEDDING_PROVIDER = var.embedding_provider
  }

  # Secret env shared by every job (env var name -> Secret Manager secret_id).
  job_secret_env = {
    DATABASE_URL      = "genos-database-url"
    DJANGO_SECRET_KEY = "genos-django-secret"
    JWT_SECRET_KEY    = "genos-jwt-secret"
  }

  # One entry per Railway cron: schedule + the management command + any extra
  # secret env beyond the shared set above.
  cron_jobs = {
    reindexer = {
      job_name = "genos-reindexer"
      schedule = "*/10 * * * *" # every 10 min — incremental OpenSearch reindex
      # Mirror the Railway reindexer entrypoint (minus the Railway SA-key decode
      # — GCP authenticates to Vertex via ADC): make sure the index + mapping
      # exist (opensearch_setup is a near-no-op when the index already exists;
      # it only recreates the mapping / clears stale RagChunk on a *fresh*
      # index) THEN push the last ~11 min of changes (1-min overlap covers the
      # gap between ticks).
      command          = ["sh", "-c", "python manage.py opensearch_setup && python manage.py opensearch_reindex --since-minutes 11"]
      extra_secret_env = {}
    }
    demo_cleanup = {
      job_name = "genos-demo-cleanup"
      schedule = "0 3 * * *" # daily 03:00 UTC — low-traffic window (matches Railway)
      command  = ["python", "manage.py", "cleanup_demo_users"]
      # cleanup fires post_delete cache-invalidation signals on the SAME Redis
      # the api uses, so it needs REDIS_URL (the reindexer/judge jobs don't).
      extra_secret_env = { REDIS_URL = "genos-redis-url" }
    }
    judge_sampler = {
      job_name         = "genos-judge-sampler"
      schedule         = "0 * * * *" # hourly — sample + LLM-judge recent AgentRuns, off the request path
      command          = ["python", "manage.py", "agent_judge_sample"]
      extra_secret_env = {}
    }
  }
}

# --- Cloud Run Jobs -------------------------------------------------------- #
resource "google_cloud_run_v2_job" "cron" {
  for_each = local.cron_jobs

  name                = each.value.job_name
  location            = var.region
  deletion_protection = false # hackathon — allow teardown/recreate

  template {
    template {
      service_account = google_service_account.run.email
      max_retries     = 0 # match Railway restartPolicyType=NEVER; the next tick is the retry
      timeout         = "900s"

      vpc_access {
        connector = google_vpc_access_connector.connector.id
        egress    = "PRIVATE_RANGES_ONLY" # reach Cloud SQL / Redis / OpenSearch private IPs
      }

      containers {
        # Same image as the api service. NOTE: a `:latest` tag is resolved to a
        # digest at apply time and pinned — re-apply (or `gcloud run jobs update
        # <name> --image ...`) to move a job onto a newer api build. No CI owns
        # the job image today, so it is intentionally NOT under ignore_changes:
        # `terraform apply` keeps it synced to var.images.api. See README.
        image   = var.images.api
        command = each.value.command

        resources {
          limits = { cpu = "1", memory = "1Gi" }
        }

        dynamic "env" {
          for_each = local.job_env
          content {
            name  = env.key
            value = env.value
          }
        }

        dynamic "env" {
          for_each = merge(local.job_secret_env, each.value.extra_secret_env)
          content {
            name = env.key
            value_source {
              secret_key_ref {
                secret  = google_secret_manager_secret.managed[env.value].secret_id
                version = "latest"
              }
            }
          }
        }
      }
    }
  }

  depends_on = [google_secret_manager_secret_version.managed, google_project_iam_member.run]
}

# --- Cloud Scheduler SA + permission to execute the jobs ------------------- #
# Cloud Scheduler mints an OAuth token as this SA and calls the Cloud Run
# Admin API. run.invoker on the job includes run.jobs.run — the minimum needed
# to execute it (no project-wide grant).
resource "google_service_account" "scheduler" {
  account_id   = "genos-scheduler"
  display_name = "Genos Cloud Scheduler -> Cloud Run Jobs"
  depends_on   = [google_project_service.enabled]
}

resource "google_cloud_run_v2_job_iam_member" "scheduler_invoke" {
  for_each = google_cloud_run_v2_job.cron

  location = var.region
  name     = each.value.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.scheduler.email}"
}

# --- Cloud Scheduler triggers ---------------------------------------------- #
resource "google_cloud_scheduler_job" "cron" {
  for_each = local.cron_jobs

  name             = each.value.job_name
  region           = var.region
  schedule         = each.value.schedule
  time_zone        = "Etc/UTC" # Railway crons are UTC — keep the same wall-clock
  attempt_deadline = "320s"    # deadline for the *trigger* call, not the job run

  retry_config {
    retry_count = 1 # retry only the trigger HTTP call; task retries are max_retries above
  }

  http_target {
    http_method = "POST"
    # Cloud Run Admin API: execute the job (regional endpoint).
    uri = "https://${var.region}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${var.project_id}/jobs/${each.value.job_name}:run"

    oauth_token {
      service_account_email = google_service_account.scheduler.email
    }
  }

  depends_on = [
    google_cloud_run_v2_job.cron,
    google_cloud_run_v2_job_iam_member.scheduler_invoke,
    google_project_service.enabled,
  ]
}
