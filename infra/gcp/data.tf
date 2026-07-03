# --- Cloud SQL (Postgres) — private IP only, reachable via the VPC connector.
resource "google_sql_database_instance" "pg" {
  name                = "genos-pg"
  database_version    = var.db_version
  region              = var.region
  deletion_protection = false # hackathon; flip to true for real prod

  settings {
    tier              = var.db_tier
    availability_type = "ZONAL"
    disk_type         = "PD_SSD"
    disk_autoresize   = true

    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.vpc.id
    }

    backup_configuration {
      enabled = true
    }
  }

  depends_on = [google_service_networking_connection.psa]
}

resource "google_sql_database" "origin" {
  name     = var.db_name
  instance = google_sql_database_instance.pg.name
}

resource "google_sql_user" "app" {
  name     = var.db_user
  instance = google_sql_database_instance.pg.name
  password = random_password.db.result
}

# --- Memorystore (Redis) — private, in-VPC.
resource "google_redis_instance" "cache" {
  name               = "genos-redis"
  tier               = "BASIC"
  memory_size_gb     = var.redis_memory_gb
  region             = var.region
  redis_version      = "REDIS_7_0"
  authorized_network = google_compute_network.vpc.id
  connect_mode       = "PRIVATE_SERVICE_ACCESS"

  depends_on = [google_service_networking_connection.psa]
}
