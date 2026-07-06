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

# LLM-only region override (maps to GEMINI_LLM_LOCATION on the api service).
# Empty → the api falls back to vertex_location. Use this to point the agent
# LLM at a region/endpoint that serves a model vertex_location doesn't (e.g.
# "global" for a *-preview Gemini model) while embeddings + the reindex job
# stay on vertex_location — the region their OpenSearch index was built in.
# Changing vertex_location instead would re-region embeddings and can silently
# break retrieval (dim/region mismatch); this variable avoids that coupling.
variable "vertex_llm_location" {
  type        = string
  default     = ""
  description = "LLM-only Vertex region override (GEMINI_LLM_LOCATION). Empty falls back to vertex_location. Set to e.g. 'global' or 'us-central1' to reach a Gemini model not served in vertex_location without moving embeddings."
}

variable "embedding_provider" {
  type    = string
  default = "vertex"
}

# --- Cloud SQL (Postgres) ------------------------------------------------- #
variable "db_tier" {
  type = string
  # Shared-core, ~1.7GB — right-sized for the parallel demo env (10-20 light
  # users, low activity). Cloud SQL ENTERPRISE edition (not Plus) supports
  # shared-core tiers. Was db-custom-1-3840 (dedicated 1 vCPU / 3.75GB);
  # downsized 2026-07-07 for cost — bump back if this env takes real sustained
  # load.
  default = "db-g1-small"
}

variable "db_version" {
  type = string
  # Prod GCP Cloud SQL was upgraded 15 -> 18 (2026-07-05) to match the Railway
  # source DB for a data sync: Railway runs PG 18 and pg_dump/restore does not
  # support the 18 -> 15 downgrade direction. Local docker-compose stays on
  # postgres:15, so prod is intentionally ahead of local dev here.
  default = "POSTGRES_18"
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
  type = string
  # 2 vCPU / 4GB — the OpenSearch JVM heap is only 1g, so 4GB comfortably fits
  # heap + off-heap + COS. Was e2-standard-2 (8GB); downsized 2026-07-07 for
  # cost (the parallel demo env sees light traffic). Bump back to e2-standard-2
  # if indexing/query load grows.
  default = "e2-medium"
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

# Server-side OAuth login providers (handled by genos-api). Leave a client ID
# empty to keep that provider OFF in this env: with no env wired, the initiate
# endpoint returns 503 "…not configured on this server." and the frontend
# button no-ops. Set a client ID to wire that provider's ID (plain env) + its
# client secret (from Secret Manager) into the api Cloud Run service.
#
# ⚠️ Enabling a provider requires seeding secret VERSIONS *before* apply:
#     - genos-<provider>-oauth-secret   (that provider's client secret)
#     - genos-oauth-token-key           (shared Fernet key — the login callback
#                                        encrypts provider tokens on EVERY OAuth
#                                        sign-in, so it's required, not just for
#                                        the connect flow)
#   Cloud Run will NOT start a revision that references an unversioned secret,
#   so seed first. See README "Optional feature secrets".
variable "oauth" {
  type = object({
    google_client_id = string
    github_client_id = string
  })
  default = {
    google_client_id = "" # from Google Cloud Console → OAuth 2.0 Client ID
    github_client_id = "" # from GitHub → Developer settings → OAuth Apps
  }
}

# Enable the agent's live web search (Tavily) on the api service. When true,
# wires TAVILY_API_KEY from the genos-tavily-api-key secret. Seed that secret
# version BEFORE apply (Cloud Run won't start a revision referencing an
# unversioned secret). Default false ⇒ the search_web tool no-ops with a
# "not configured" message. Tavily free tier is 1000 searches/mo; per-use, no
# idle cost.
variable "tavily_enabled" {
  type    = bool
  default = false
}

# Feature-parity toggles for optional secrets Railway has but GCP lacked. Each
# wires env onto the api ONLY when enabled; seed the matching secret version(s)
# BEFORE apply. Mirror Railway's values: `railway variables -s backend-django`.

# Transactional email via Resend (verification, team invites, password reset).
# When false the api falls back to Django's console backend → emails aren't sent.
variable "email_enabled" {
  type    = bool
  default = false
}
variable "email_from" {
  type        = string
  default     = ""
  description = "DEFAULT_FROM_EMAIL, e.g. 'Genos Support <noreply@genosai.dev>' (a Resend-verified domain)."
}

# Anthropic Claude — needed because the model is user-selectable per request
# (LLM_PROVIDER stays gemini). Without CLAUDE_API_KEY, picking 'claude' errors.
variable "claude_enabled" {
  type    = bool
  default = false
}

# Web Push (browser notifications). public_key here MUST match the frontend's
# VITE_VAPID_PUBLIC_KEY, and the private key (genos-webpush-vapid-private-key
# secret) must be its pair — use the SAME keypair as Railway.
variable "webpush_enabled" {
  type    = bool
  default = false
}
variable "webpush_vapid_public_key" {
  type    = string
  default = ""
}
variable "webpush_vapid_admin_email" {
  type    = string
  default = ""
}

# GitHub "owner/repo" allowed to deploy via Workload Identity Federation.
# Used by iam.tf to let CI mint tokens for the deploy SA (no JSON key).
variable "github_repos" {
  type    = list(string)
  default = ["genos-tech/genos-api", "genos-tech/genos-sockets", "genos-tech/genos-collab", "genos-tech/genos-frontend"]
}
