# One custom VPC. Cloud Run reaches Cloud SQL / Memorystore / OpenSearch over
# private IPs through a Serverless VPC Access connector; public ingress to the
# services is unaffected.

resource "google_compute_network" "vpc" {
  name                    = "genos-vpc"
  auto_create_subnetworks = false
  depends_on              = [google_project_service.enabled]
}

resource "google_compute_subnetwork" "subnet" {
  name          = "genos-subnet"
  region        = var.region
  network       = google_compute_network.vpc.id
  ip_cidr_range = "10.10.0.0/24"

  # Required so the VM (and future services) can reach Google APIs privately.
  private_ip_google_access = true
}

# Serverless VPC Access connector — Cloud Run egress into the VPC.
resource "google_vpc_access_connector" "connector" {
  name          = "genos-connector"
  region        = var.region
  network       = google_compute_network.vpc.name
  ip_cidr_range = "10.8.0.0/28" # dedicated /28, must not overlap the subnet
  min_instances = 2
  max_instances = 3
  depends_on    = [google_project_service.enabled]
}

# --- Private Service Access: reserved range peered to Google's managed VPC,
# so Cloud SQL and Memorystore get private IPs reachable from our VPC. -------
resource "google_compute_global_address" "psa_range" {
  name          = "genos-psa-range"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.vpc.id
}

resource "google_service_networking_connection" "psa" {
  network                 = google_compute_network.vpc.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.psa_range.name]
  depends_on              = [google_project_service.enabled]
}

# --- Firewall ------------------------------------------------------------- #
# Allow the VPC connector range to reach OpenSearch (9200) on the VM.
resource "google_compute_firewall" "allow_opensearch_from_connector" {
  name    = "genos-allow-opensearch"
  network = google_compute_network.vpc.name

  allow {
    protocol = "tcp"
    ports    = ["9200"]
  }
  # Connector range + subnet (in case future in-VPC clients need it).
  source_ranges = ["10.8.0.0/28", "10.10.0.0/24"]
  target_tags   = ["opensearch"]
}

# SSH to the OpenSearch VM only via IAP (no public IP on the VM).
resource "google_compute_firewall" "allow_iap_ssh" {
  name    = "genos-allow-iap-ssh"
  network = google_compute_network.vpc.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
  source_ranges = ["35.235.240.0/20"] # Google IAP range
  target_tags   = ["opensearch"]
}
