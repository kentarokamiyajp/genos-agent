output "api_url" {
  value = google_cloud_run_v2_service.api.uri
}

output "sockets_url" {
  value = google_cloud_run_v2_service.sockets.uri
}

output "collab_url" {
  value = google_cloud_run_v2_service.collab.uri
}

output "frontend_url" {
  value = google_cloud_run_v2_service.frontend.uri
}

output "opensearch_internal_ip" {
  value = google_compute_instance.opensearch.network_interface[0].network_ip
}

output "db_private_ip" {
  value = google_sql_database_instance.pg.private_ip_address
}

output "redis_host" {
  value = google_redis_instance.cache.host
}

output "artifact_registry" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.genos.repository_id}"
}

output "deploy_service_account" {
  value = google_service_account.deploy.email
}

# Feed these two to google-github-actions/auth in CI (keyless WIF).
output "wif_provider" {
  value = google_iam_workload_identity_pool_provider.github.name
}
