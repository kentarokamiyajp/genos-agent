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

- **DB schema + OpenSearch index** bootstrap on the api container's first boot
  (its Dockerfile runs `migrate` + `opensearch_setup`).
- **Reindex for Vertex embeddings.** We switched `EMBEDDING_PROVIDER=vertex`
  (`gemini-embedding-001`). **Confirm the embedding dimension matches the index
  mapping** (docs flag a past 1536-vs-3072 mismatch → silent vector-search
  failure) and run the reindex command (see `docs/OPENSEARCH_COMMANDS.md`).
- **Optional feature secrets** (stack boots without them; only that feature
  degrades):
  ```bash
  # Fernet key for encrypted OAuth tokens (Calendar / GitHub agent tools):
  KEY=$(python -c "from cryptography.fernet import Fernet;print(Fernet.generate_key().decode())")
  printf '%s' "$KEY" | gcloud secrets versions add genos-oauth-token-key --data-file=-
  printf '%s' "$TAVILY_KEY" | gcloud secrets versions add genos-tavily-api-key --data-file=-
  ```
  Then wire them into the api service env (add `env{value_source…}` blocks) and
  redeploy.
- **Custom domains** (recommended): `gcloud run domain-mappings create --service genos-frontend --domain genosai.dev` (repeat per service) and add the shown DNS records. Set `domains` in tfvars + re-apply so inter-service URLs are wired.
- **Vertex region:** if a required Gemini model isn't served in `asia-northeast1`, set `vertex_location = "us-central1"`.

## Ongoing deploys

The Cloud Run `image` is under `lifecycle.ignore_changes`, so **CI owns image
tags** — after bootstrap, ship new revisions with `gcloud run deploy genos-<svc>
--image …` (this is what Part B's GitHub Actions pipeline does via the
`deploy_service_account` + `wif_provider` outputs). Terraform stays the source of
truth for infra, not for the app version.

## Verify — the app must *work*, not just deploy

`terraform validate` can't catch these; check each after apply:

1. **Services up:** `curl -sS "$(terraform output -raw api_url)/..."` returns 200; frontend loads.
2. **Vertex model + region:** confirm `gemini-3.5-flash` **and** the embedding model are actually served on Vertex in `asia-northeast1`. If a call 404s, set `vertex_location = "us-central1"` and re-apply. (Otherwise PR #5 just trades one 404 for another.)
3. **Vector search returns hits (the 1536-vs-3072 trap):** after the Vertex reindex, run a real **Spotlight ask** and confirm the `search_knowledge_base` tool comes back with results — a dimension mismatch fails **silently** (empty results, no error). This is the agent's core tool; don't assume it.
4. **Full agent loop:** ask → multi-step tool use → an **approval-gated write** (e.g. "create a task…") → cited answer.

```bash
terraform fmt -check && terraform validate
```
