# Enable the GCP services this stack needs. `disable_on_destroy = false` so a
# `terraform destroy` of the app doesn't yank APIs another project resource uses.
locals {
  services = [
    "run.googleapis.com",
    "sqladmin.googleapis.com",
    "redis.googleapis.com",
    "compute.googleapis.com",
    "vpcaccess.googleapis.com",
    "servicenetworking.googleapis.com",
    "artifactregistry.googleapis.com",
    "secretmanager.googleapis.com",
    "aiplatform.googleapis.com", # Vertex AI (Gemini + embeddings)
    "iam.googleapis.com",        # service account creation
    "iamcredentials.googleapis.com",
    "cloudresourcemanager.googleapis.com", # project IAM bindings
    "serviceusage.googleapis.com",
    "cloudbuild.googleapis.com",
  ]
}

resource "google_project_service" "enabled" {
  for_each           = toset(local.services)
  service            = each.value
  disable_on_destroy = false
}
