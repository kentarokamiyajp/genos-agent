# Single-node OpenSearch on a Container-Optimized OS VM with a persistent data
# disk. No managed OpenSearch on GCP; this runs the custom multilingual image
# (JA/EN analyzers) that docker-compose builds. Internal IP only — reachable
# from Cloud Run via the VPC connector.

resource "google_compute_disk" "os_data" {
  name = "genos-os-data"
  type = "pd-ssd"
  zone = var.zone
  size = var.opensearch_data_disk_gb
}

resource "google_compute_instance" "opensearch" {
  name         = "genos-opensearch"
  machine_type = var.opensearch_machine_type
  zone         = var.zone
  tags         = ["opensearch"]

  boot_disk {
    initialize_params {
      image = "cos-cloud/cos-stable"
      size  = 20
    }
  }

  attached_disk {
    source      = google_compute_disk.os_data.id
    device_name = "genos-os-data" # -> /dev/disk/by-id/google-genos-os-data
  }

  network_interface {
    subnetwork = google_compute_subnetwork.subnet.id
    # No access_config block => no external IP.
  }

  service_account {
    email  = google_service_account.opensearch.email
    scopes = ["cloud-platform"]
  }

  metadata_startup_script = templatefile("${path.module}/opensearch-startup.sh.tftpl", {
    region           = var.region
    opensearch_image = var.opensearch_image
  })

  allow_stopping_for_update = true
  depends_on                = [google_project_iam_member.opensearch]

  lifecycle {
    precondition {
      condition     = var.opensearch_image != ""
      error_message = "Set var.opensearch_image to the pushed opensearch-multilingual image (see README bootstrap step 3)."
    }
  }
}
