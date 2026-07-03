variable "project_id" {
  type        = string
  description = "GCP project ID."
  default     = "amplified-album-496413-t2"
}

variable "region" {
  type    = string
  default = "asia-northeast1" # Tokyo
}

variable "zone" {
  type    = string
  default = "asia-northeast1-a"
}

# --- Vertex AI (Gemini) --------------------------------------------------- #
# The agent + embeddings route through Vertex (GEMINI_USE_VERTEX=true). The
# Cloud Run runtime service account authenticates via ADC — no key file.
variable "vertex_location" {
  type        = string
  default     = "asia-northeast1"
  description = "Vertex AI location for Gemini + embeddings. Switch to us-central1 if a required model isn't served in Tokyo."
}

variable "embedding_provider" {
  type    = string
  default = "vertex"
}

# --- Cloud SQL (Postgres) ------------------------------------------------- #
variable "db_tier" {
  type    = string
  default = "db-custom-1-3840" # 1 vCPU / 3.75GB — demo-grade
}

variable "db_version" {
  type    = string
  default = "POSTGRES_15" # match docker-compose (postgres:15)
}

variable "db_name" {
  type    = string
  default = "origin"
}

variable "db_user" {
  type    = string
  default = "genos"
}

# --- Memorystore (Redis) -------------------------------------------------- #
variable "redis_memory_gb" {
  type    = number
  default = 1
}

# --- OpenSearch (single-node on a GCE COS VM) ----------------------------- #
variable "opensearch_machine_type" {
  type    = string
  default = "e2-standard-2" # 2 vCPU / 8GB — OpenSearch heap is 1g (see below)
}

variable "opensearch_data_disk_gb" {
  type    = number
  default = 30
}

variable "opensearch_image" {
  type        = string
  description = "Full Artifact Registry path to the custom opensearch-multilingual image. Built + pushed in the bootstrap step."
  # e.g. asia-northeast1-docker.pkg.dev/<project>/genos/opensearch-multilingual:3.6.0
  default = ""
}

# --- Container images for the 4 Cloud Run services ------------------------ #
# Placeholder default lets `terraform apply` create the services before the
# real images exist; CI (or the bootstrap build) then deploys real revisions.
# Cloud Run `image` is under lifecycle.ignore_changes so CI deploys never drift.
variable "images" {
  type = object({
    api      = string
    sockets  = string
    collab   = string
    frontend = string
  })
  default = {
    api      = "us-docker.pkg.dev/cloudrun/container/hello"
    sockets  = "us-docker.pkg.dev/cloudrun/container/hello"
    collab   = "us-docker.pkg.dev/cloudrun/container/hello"
    frontend = "us-docker.pkg.dev/cloudrun/container/hello"
  }
}

# --- Public hostnames ----------------------------------------------------- #
# The frontend bakes VITE_* at BUILD time, so it needs stable backend URLs.
# Prefer custom domains on genosai.dev (set these + create Cloud Run domain
# mappings). If left empty, use the two-phase build in README (deploy backends,
# read their run.app URLs, then build+deploy the frontend with those).
variable "domains" {
  type = object({
    frontend = string
    api      = string
    sockets  = string
    collab   = string
  })
  default = {
    frontend = "" # e.g. genosai.dev
    api      = "" # e.g. api.genosai.dev
    sockets  = "" # e.g. ws.genosai.dev
    collab   = "" # e.g. collab.genosai.dev
  }
}

# GitHub "owner/repo" allowed to deploy via Workload Identity Federation.
# Used by iam.tf to let CI mint tokens for the deploy SA (no JSON key).
variable "github_repos" {
  type    = list(string)
  default = ["genos-tech/genos-api", "genos-tech/genos-sockets", "genos-tech/genos-collab", "genos-tech/genos-frontend"]
}
