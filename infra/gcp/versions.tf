# Terraform + provider pins for the Genos GCP deployment.
#
# State: local by default (gitignored — it will contain generated secrets).
# For team use, uncomment the GCS backend below and create the bucket first
# (see README.md "Bootstrap").
terraform {
  required_version = ">= 1.6.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.8"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }

  # backend "gcs" {
  #   bucket = "genos-tfstate-amplified-album-496413-t2"
  #   prefix = "gcp/prod"
  # }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}
