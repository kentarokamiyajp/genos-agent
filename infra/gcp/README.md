# Genos on GCP (Terraform)

Full-stack deploy of Genos to **Google Cloud** in `asia-northeast1` (Tokyo):

| Piece | GCP resource |
| --- | --- |
| genos-api / sockets / collab / frontend | **Cloud Run** (v2) |
| Postgres (`origin`) | **Cloud SQL** (private IP) |
| Redis | **Memorystore** (private) |
| OpenSearch (custom multilingual image) | **GCE** COS VM + persistent disk (internal IP) |
| Agent LLM + embeddings | **Vertex AI** (Gemini), via the runtime SA (ADC, no key file) |
| Images | **Artifact Registry** |
| App secrets | **Secret Manager** |
| CI deploy auth | **Workload Identity Federation** (keyless) |

Cloud Run reaches the private data services (SQL / Redis / OpenSearch) through a
**Serverless VPC Access connector**. Public ingress is unaffected.

> ⚠️ **State holds generated secrets** (DB password, JWT, Django/Flask keys).
> Keep `terraform.tfstate` local + gitignored (already set), or switch to the
> locked-down GCS backend in `versions.tf`.

---

## Prerequisites

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project amplified-album-496413-t2
# Billing must be enabled on the project (hackathon credits count).
cp terraform.tfvars.example terraform.tfvars   # then edit
terraform init
```

## Bootstrap (order matters — image registry must exist before we push)

**1. Enable APIs + create the Artifact Registry repo first:**
```bash
terraform apply -target=google_project_service.enabled \
                -target=google_artifact_registry_repository.genos
gcloud auth configure-docker asia-northeast1-docker.pkg.dev
```

**2. Build + push the custom OpenSearch image** (the multilingual analyzers the
agent's JA/EN search needs — the same image `docker-compose` builds):
```bash
AR=asia-northeast1-docker.pkg.dev/amplified-album-496413-t2/genos
docker build -t $AR/opensearch-multilingual:3.6.0 ../../docker/opensearch
docker push $AR/opensearch-multilingual:3.6.0
```
Set `opensearch_image` in `terraform.tfvars` to that tag.

**3. Build + push the 4 app images** (reuse each repo's existing Dockerfile):
```bash
for s in api sockets collab; do
  docker build -t $AR/$s:latest ../../../genos-$s      # api uses genos-api/Dockerfile
  docker push $AR/$s:latest
done
```
> **Frontend is special — `VITE_*` is baked at BUILD time.** Build it with the
> *production* backend URLs (custom domains if you set them, else the run.app
> URLs from step 5's first apply):
> ```bash
> docker build -t $AR/frontend:latest \
>   --build-arg VITE_API_BASE_URL=https://api.genosai.dev/api/v2 \
>   --build-arg VITE_DJANGO_URL=https://api.genosai.dev \
>   --build-arg VITE_MEDIA_ROOT_DJANGO=https://api.genosai.dev/media \
>   --build-arg VITE_WS_BASE_URL=https://ws.genosai.dev \
>   --build-arg VITE_COLLAB_URL=wss://collab.genosai.dev \
>   ../../../genos-frontend
> ```
Uncomment + fill the `images` map in `terraform.tfvars`.

**4. Full apply:**
```bash
terraform apply
terraform output       # service URLs, DB/Redis/OpenSearch IPs, WIF provider
```

## Post-apply

- **DB schema** bootstraps on the api container's first boot (its Dockerfile
  runs `migrate` + `collectstatic`).
- **OpenSearch index + freshness** are owned by the **`genos-reindexer`
  scheduled job** (not the api boot): it runs `opensearch_setup` (idempotent —
  creates the index + mapping only if missing) then an incremental
  `opensearch_reindex` every 10 min. See
  [Scheduled maintenance jobs](#scheduled-maintenance-jobs). Before the first
  pass, **confirm the Vertex embedding dimension matches the index mapping**
  (docs flag a past 1536-vs-3072 mismatch → silent vector-search failure; see
  `docs/OPENSEARCH_COMMANDS.md`).
- **Optional feature secrets** (stack boots without them; only that feature
  degrades). The secret **containers** already exist (`manual_secrets` in
  `secrets.tf`); you add the **versions**:
  ```bash
  # Fernet key for OAuth tokens stored at rest. Shared by BOTH the agent tools
  # (Calendar / GitHub) AND OAuth *login* — the login callback encrypts the
  # provider token on every sign-in, so this is required for login, not just
  # the connect flow:
  KEY=$(python -c "from cryptography.fernet import Fernet;print(Fernet.generate_key().decode())")
  printf '%s' "$KEY" | gcloud secrets versions add genos-oauth-token-key --data-file=-
  printf '%s' "$TAVILY_KEY" | gcloud secrets versions add genos-tavily-api-key --data-file=-
  ```
  For the agent-tool keys above, then wire them into the api service env (add
  `env{value_source…}` blocks) and redeploy.

- **OAuth login (Google / GitHub sign-in buttons)** — off by default; a
  `terraform apply` turns it on once you've done all three of:
  1. **Register the OAuth apps** and set their callback URLs to exactly
     `https://<api-domain>/api/v2/oauth/<provider>/callback/` (with the domains
     in `tfvars`, that's `https://api.gcp.genosai.dev/api/v2/oauth/google/callback/`
     and `…/github/callback/`). Google also needs the frontend origin
     (`https://gcp.genosai.dev`) as an Authorized JavaScript origin. This yields
     a **client ID** (public) + **client secret** (sensitive) per provider.
  2. **Seed the secret versions** (client secrets + the shared Fernet key above):
     ```bash
     printf '%s' "$GOOGLE_OAUTH_CLIENT_SECRET" | gcloud secrets versions add genos-google-oauth-secret --data-file=-
     printf '%s' "$GITHUB_OAUTH_CLIENT_SECRET" | gcloud secrets versions add genos-github-oauth-secret --data-file=-
     # + genos-oauth-token-key from the block above if not already seeded
     ```
  3. **Set the client IDs** in `tfvars` and apply:
     ```hcl
     oauth = {
       google_client_id = "…apps.googleusercontent.com"
       github_client_id = "Iv1…"
     }
     ```
     ```bash
     terraform apply   # wires GOOGLE/GITHUB_OAUTH_CLIENT_ID/SECRET +
                       # OAUTH_TOKEN_ENCRYPTION_KEY into the api service
     ```
  Wiring is conditional on a non-empty client ID (`cloudrun.tf`), so an empty
  provider stays completely off. ⚠️ **Seed the secret versions *before* apply** —
  Cloud Run will not start a revision whose `secret_key_ref` points at a
  versionless secret. This change does **not** ride the image-only CI deploy;
  it needs an explicit `terraform apply`.
- **Custom domains** (recommended): `gcloud run domain-mappings create --service genos-frontend --domain genosai.dev` (repeat per service) and add the shown DNS records. Set `domains` in tfvars + re-apply so inter-service URLs are wired.
- **Vertex region:** if a required Gemini model isn't served in `asia-northeast1`, set `vertex_location = "us-central1"`. ⚠️ `vertex_location` also drives the Vertex **embedder** + reindex job — moving it re-regions embeddings and can silently break retrieval. To fix *only* the agent LLM (e.g. reach a `*-preview` model), set `vertex_llm_location = "global"` instead (maps to `GEMINI_LLM_LOCATION`), which leaves embeddings on `vertex_location`.

## Ongoing deploys

The Cloud Run `image` is under `lifecycle.ignore_changes`, so **CI owns image
tags** — after bootstrap, ship new revisions with `gcloud run deploy genos-<svc>
--image …` (this is what Part B's GitHub Actions pipeline does via the
`deploy_service_account` + `wif_provider` outputs). Terraform stays the source of
truth for infra, not for the app version.

## Scheduled maintenance jobs

Three recurring jobs mirror the Railway cron services (`jobs.tf`). Each is a
**Cloud Run Job** (the api image, serving CMD overridden to a management
command) triggered by a **Cloud Scheduler** job on the same cadence Railway
used. They run AS the `genos-run` runtime SA (so they already have Vertex,
Secret Manager and Cloud SQL access); a dedicated `genos-scheduler` SA holds
`roles/run.invoker` on each job — the minimum to execute it.

| Job | Schedule (UTC) | Command | Why it matters |
| --- | --- | --- | --- |
| `genos-reindexer` | `*/10 * * * *` | `opensearch_setup && opensearch_reindex --since-minutes 11` | The app writes to Postgres only; this makes new/changed content **searchable**. Without it, search silently goes stale. |
| `genos-demo-cleanup` | `0 3 * * *` | `cleanup_demo_users` | Sweeps `is_demo=True` users >24 h old + their team data + OpenSearch chunks. |
| `genos-judge-sampler` | `0 * * * *` | `agent_judge_sample` | Samples ~10 % of recent grounded agent runs for the online LLM-as-judge quality trend. |

> **Reindexer embedding provider must match the api.** The reindexer writes
> *document* vectors; the api embeds the *query* at search time. Both are pinned
> to `EMBEDDING_PROVIDER=vertex` here (via `local.job_env`). Diverge, and the
> k-NN lane returns noise — only BM25 keeps working. This is the #1 reindexer
> footgun; keep them in lockstep.

**Job images.** No CI owns the job images yet, so — unlike the services — they
are **not** under `ignore_changes`: `terraform apply` keeps them synced to
`var.images.api`. A `:latest` tag is resolved to a digest at apply time, so to
move a job onto a newer api build, re-apply or:
`gcloud run jobs update genos-reindexer --region asia-northeast1 --image <ref>`.

**First-fire auth check (do this once after apply).** Cloud Scheduler mints an
OAuth token as `genos-scheduler` to call the Cloud Run Admin API. This works out
of the box — `roles/cloudscheduler.serviceAgent` can mint tokens for
same-project SAs. If a trigger **403s**, grant the scheduler service agent
`roles/iam.serviceAccountTokenCreator` on the `genos-scheduler` SA and retry.
Verify without waiting for the schedule:

```bash
gcloud run jobs execute genos-reindexer --region asia-northeast1        # runs the job directly
gcloud scheduler jobs run genos-reindexer --location asia-northeast1     # exercises the scheduler→job auth path
gcloud scheduler jobs describe genos-reindexer --location asia-northeast1 --format='value(status)'
```

## Verify — the app must *work*, not just deploy

`terraform validate` can't catch these; check each after apply:

1. **Services up:** `curl -sS "$(terraform output -raw api_url)/..."` returns 200; frontend loads.
2. **Vertex model + region:** confirm `gemini-3.5-flash` **and** the embedding model are actually served on Vertex in `asia-northeast1`. If the LLM call 404s on a model the region doesn't serve (e.g. a `*-preview` pro model), set `vertex_llm_location = "global"` (LLM only — leaves embeddings put). If the *embedding* model itself 404s, move both with `vertex_location = "us-central1"` and re-apply, then re-check step 3. (Otherwise you just trade one 404 for another.)
3. **Vector search returns hits (the 1536-vs-3072 trap):** after the Vertex reindex, run a real **Spotlight ask** and confirm the `search_knowledge_base` tool comes back with results — a dimension mismatch fails **silently** (empty results, no error). This is the agent's core tool; don't assume it.
4. **Full agent loop:** ask → multi-step tool use → an **approval-gated write** (e.g. "create a task…") → cited answer.

```bash
terraform fmt -check && terraform validate
```
