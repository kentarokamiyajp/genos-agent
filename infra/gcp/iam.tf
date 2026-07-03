# --- Cloud Run runtime service account -----------------------------------
# The 4 services run as this SA. It authenticates to Vertex via ADC (no key
# file), reads secrets, and connects to Cloud SQL.
resource "google_service_account" "run" {
  account_id   = "genos-run"
  display_name = "Genos Cloud Run runtime"
  depends_on   = [google_project_service.enabled]
}

locals {
  run_roles = [
    "roles/aiplatform.user",              # Vertex: Gemini agent + embeddings
    "roles/secretmanager.secretAccessor", # read DB_URL / JWT / etc.
    "roles/cloudsql.client",              # Cloud SQL (private IP still needs this for auth)
    "roles/artifactregistry.reader",      # pull images
    "roles/logging.logWriter",
  ]
}

resource "google_project_iam_member" "run" {
  for_each = toset(local.run_roles)
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${google_service_account.run.email}"
}

# --- OpenSearch VM service account ----------------------------------------
resource "google_service_account" "opensearch" {
  account_id   = "genos-opensearch"
  display_name = "Genos OpenSearch VM"
  depends_on   = [google_project_service.enabled]
}

resource "google_project_iam_member" "opensearch" {
  for_each = toset(["roles/artifactregistry.reader", "roles/logging.logWriter"])
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${google_service_account.opensearch.email}"
}

# --- CI deploy service account (used by GitHub Actions via WIF, no JSON key)
resource "google_service_account" "deploy" {
  account_id   = "genos-deploy"
  display_name = "Genos CI deployer"
  depends_on   = [google_project_service.enabled]
}

locals {
  deploy_roles = [
    "roles/run.admin",               # deploy new Cloud Run revisions
    "roles/artifactregistry.writer", # push images
    "roles/cloudbuild.builds.editor",
    "roles/aiplatform.user",         # CI eval gate calls Vertex (embeddings + Gemini) via WIF — same path as prod
  ]
}

resource "google_project_iam_member" "deploy" {
  for_each = toset(local.deploy_roles)
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${google_service_account.deploy.email}"
}

# CI must be able to act AS the runtime SA to deploy services that run as it.
resource "google_service_account_iam_member" "deploy_actas_run" {
  service_account_id = google_service_account.run.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.deploy.email}"
}

# --- Workload Identity Federation: GitHub Actions -> deploy SA (keyless) ---
resource "google_iam_workload_identity_pool" "github" {
  workload_identity_pool_id = "genos-github"
  display_name              = "Genos GitHub Actions"
  depends_on                = [google_project_service.enabled]
}

resource "google_iam_workload_identity_pool_provider" "github" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github"
  display_name                       = "GitHub OIDC"

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
  }
  # Only tokens from our repos may map in.
  attribute_condition = "assertion.repository in ${jsonencode(var.github_repos)}"

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

# Let each allowed repo impersonate the deploy SA.
resource "google_service_account_iam_member" "github_wif" {
  for_each           = toset(var.github_repos)
  service_account_id = google_service_account.deploy.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/${each.value}"
}
