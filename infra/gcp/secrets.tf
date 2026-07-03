# Generated secrets + Secret Manager. Cloud Run mounts these as secret env
# (references `latest`). NOTE: generated values land in Terraform state, so
# state is sensitive — keep it local + gitignored (see .gitignore) or use the
# GCS backend with a locked-down bucket.

resource "random_password" "db" {
  length  = 28
  special = false # keep DATABASE_URL free of chars needing URL-encoding
}

resource "random_password" "jwt" {
  length  = 48
  special = false
}

resource "random_password" "django" {
  length  = 50
  special = true
}

resource "random_password" "flask" {
  length  = 48
  special = false
}

locals {
  # Composed at plan time from the private IPs of the managed services.
  database_url = "postgres://${var.db_user}:${random_password.db.result}@${google_sql_database_instance.pg.private_ip_address}:5432/${var.db_name}"
  redis_url    = "redis://${google_redis_instance.cache.host}:6379/1"

  # Generated secrets we create versions for.
  managed_secrets = {
    "genos-database-url"  = local.database_url
    "genos-redis-url"     = local.redis_url
    "genos-db-password"   = random_password.db.result # collab uses discrete PG_* vars
    "genos-jwt-secret"    = random_password.jwt.result
    "genos-django-secret" = random_password.django.result
    "genos-flask-secret"  = random_password.flask.result
  }

  # Secret containers we create but DON'T version here (add values manually,
  # see README). Optional AI/OAuth keys — the stack boots without them; only
  # the matching feature degrades.
  manual_secrets = [
    "genos-oauth-token-key",     # Fernet key for encrypted OAuth tokens (Calendar/GitHub tools)
    "genos-google-oauth-secret", # Google OAuth client secret
    "genos-github-oauth-secret", # GitHub OAuth client secret
    "genos-tavily-api-key",      # Tavily (agent web_search tool)
  ]
}

resource "google_secret_manager_secret" "managed" {
  for_each  = local.managed_secrets
  secret_id = each.key
  replication {
    auto {}
  }
  depends_on = [google_project_service.enabled]
}

resource "google_secret_manager_secret_version" "managed" {
  for_each    = local.managed_secrets
  secret      = google_secret_manager_secret.managed[each.key].id
  secret_data = each.value
}

resource "google_secret_manager_secret" "manual" {
  for_each  = toset(local.manual_secrets)
  secret_id = each.value
  replication {
    auto {}
  }
  depends_on = [google_project_service.enabled]
}
