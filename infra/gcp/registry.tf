# Docker images for the 4 services + the custom OpenSearch image live here.
resource "google_artifact_registry_repository" "genos" {
  location      = var.region
  repository_id = "genos"
  format        = "DOCKER"
  description   = "Genos service + opensearch-multilingual images."
  depends_on    = [google_project_service.enabled]
}
