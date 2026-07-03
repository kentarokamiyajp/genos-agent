# Environments: Railway ⇄ GCP (parallel, switchable)

Genos runs on **two independent, fully-live environments**. Neither depends on
the other; you can demo/operate GCP without disturbing Railway, and promote or
roll back between them by changing DNS. This doc is the map + the switch runbook.

> **Golden rule:** the two environments have **separate databases, search
> indices, and secrets**. They are *not* replicated. "Switching production" is a
> DNS change *plus* a data-parity step (below) — not just a DNS change.

## The two environments

| | **Railway** (current prod) | **GCP** (parallel) |
| --- | --- | --- |
| Frontend | `genosai.dev` | `gcp.genosai.dev` |
| API (django) | `api.genosai.dev` | `api.gcp.genosai.dev` |
| Sockets (flask) | (Railway) | `ws.gcp.genosai.dev` |
| Collab (node) | (Railway) | `collab.gcp.genosai.dev` |
| Compute | Railway services | **Cloud Run** ×4 |
| Postgres | Railway plugin | **Cloud SQL** (private IP) |
| Redis | Railway plugin | **Memorystore** (private) |
| OpenSearch | Railway service | **GCE** COS VM + disk (`opensearch-multilingual`) |
| Agent LLM + embeddings | Gemini (AI Studio or Vertex) | **Vertex AI** (same models) |
| Images | Nixpacks / Railway build | **Artifact Registry** (`…/genos/*`) |
| Secrets | Railway env vars | **Secret Manager** |
| Deploy | Railway GitHub integration (push→deploy) | **Terraform** (infra) + GitHub Actions `agent-evals` (evals-gated `gcloud run deploy`) |
| Runbook | `docs/RAILWAY_DEPLOY.md` | `infra/gcp/README.md` |

**DNS** (both hosted at Cloudflare): Railway lives on the **bare** `genosai.dev`
records; GCP lives on `gcp.*` subdomains. They never collide.

### GCP DNS records (Cloudflare, all **DNS-only / grey cloud**)
All four are subdomains → a single CNAME each. **Do not proxy (orange cloud)** or
Google can't provision the managed cert.

| Name | Type | Value |
| --- | --- | --- |
| `gcp` | CNAME | `ghs.googlehosted.com.` |
| `api.gcp` | CNAME | `ghs.googlehosted.com.` |
| `ws.gcp` | CNAME | `ghs.googlehosted.com.` |
| `collab.gcp` | CNAME | `ghs.googlehosted.com.` |

Leave the `gcloud domains verify` **TXT** record in place (deleting it un-verifies
the domain). Certs auto-provision ~15–60 min; check:
```bash
gcloud beta run domain-mappings describe --domain api.gcp.genosai.dev \
  --region asia-northeast1 --project amplified-album-496413-t2 \
  --format="value(status.conditions[].type, status.conditions[].status)"
```

## What differs between the environments (parity checklist)

- **The frontend bakes hostnames at BUILD time** (`VITE_*`). Each environment
  therefore has its **own frontend image**: the GCP image is built with
  `VITE_API_BASE_URL=https://api.gcp.genosai.dev/api/v2` (+ ws/collab `gcp.*`).
  Changing which domain the GCP frontend serves ⇒ **rebuild + redeploy**.
- **JWT**: within an environment, `JWT_SECRET_KEY` (api + sockets) and `JWT_SECRET`
  (collab) must share one value. The two environments may use *different* values —
  that's fine; each is internally consistent.
- **Data**: separate Postgres + OpenSearch per environment. No live sync.
- **Cost**: GCP ≈ $150/mo vs Railway (see `GCP_MIGRATION.md`); keep the GCP
  OpenSearch VM / Cloud SQL sized for demo, scale up only on promotion.

## Runbook A — Promote GCP to production (`genosai.dev` → GCP)

Do this when GCP becomes the canonical prod. **Reversible** (see Runbook B).

1. **Bring GCP data to parity.** GCP's Cloud SQL is independent, so snapshot
   Railway and restore into Cloud SQL, then reindex OpenSearch:
   ```bash
   # from a host that can reach both (or via a dump file)
   pg_dump "$RAILWAY_DATABASE_URL" -Fc -f railway.dump
   # restore into Cloud SQL (through the VPC / Cloud SQL Auth Proxy)
   pg_restore --no-owner -d "$GCP_DATABASE_URL" railway.dump
   # reindex search (Cloud Run Job or exec) — Vertex embeddings:
   gcloud run jobs execute genos-seed   # (or a dedicated reindex job)
   ```
   Freeze writes on Railway during the cutover window to avoid split-brain.
2. **Rebuild the GCP frontend for the BARE domain** and push:
   ```bash
   # NOTE: the frontend's Dockerfile is named Dockerfile.cloudrun (so Railway
   # doesn't auto-detect it) — pass it with -f.
   AR=asia-northeast1-docker.pkg.dev/amplified-album-496413-t2/genos
   docker buildx build --platform linux/amd64 -t $AR/frontend:latest --push \
     -f ../../../genos-frontend/Dockerfile.cloudrun \
     --build-arg VITE_API_BASE_URL=https://api.genosai.dev/api/v2 \
     --build-arg VITE_DJANGO_URL=https://api.genosai.dev \
     --build-arg VITE_MEDIA_ROOT_DJANGO=https://api.genosai.dev/media \
     --build-arg VITE_WS_BASE_URL=https://ws.genosai.dev \
     --build-arg VITE_COLLAB_URL=wss://collab.genosai.dev ../../../genos-frontend
   gcloud run deploy genos-frontend --region asia-northeast1 --image $AR/frontend:latest
   ```
3. **Point the bare domain at GCP.** Set `domains` in `infra/gcp/terraform.tfvars`
   to the bare names (`genosai.dev`, `api.genosai.dev`, …), `terraform apply`
   (updates `ALLOWED_HOSTS` / `CORS_ORIGINS` / inter-service URLs), then create the
   bare-name Cloud Run domain mappings and add the DNS at Cloudflare:
   - `genosai.dev` (apex) → **A** `216.239.32.21`, `216.239.34.21`, `216.239.36.21`,
     `216.239.38.21` + **AAAA** `2001:4860:4802:32::15`, `:34::15`, `:36::15`, `:38::15`
   - `api` / `ws` / `collab` → **CNAME** `ghs.googlehosted.com` (grey cloud)
   - Record the **old Railway values first** (see Runbook B).
4. **Verify** certs green + a Spotlight ask end-to-end on `https://genosai.dev`.

## Runbook B — Roll back to Railway

Because Railway's records were never touched, rollback is just restoring the bare
domain to Railway. **Capture these BEFORE step 3 of Runbook A** (current live
values, for reference):
- `genosai.dev` A → `69.46.46.71`
- `api.genosai.dev` → CNAME `p2bwi2xb.up.railway.app` (+ A `69.46.46.83`)

To roll back: in Cloudflare, restore the bare-domain records to the Railway values
above (delete the GCP A/AAAA/CNAMEs on the bare names). Railway keeps running
throughout, so this is near-instant (DNS TTL only). The `gcp.*` GCP environment
stays live regardless.

## Quick reference

- Bring GCP up from scratch / rebuild images: `infra/gcp/README.md`.
- GCP cost + migration effort: `GCP_MIGRATION.md`.
- Railway operations: `RAILWAY_DEPLOY.md`.
- CI that gates the GCP deploy on agent evals: `genos-api/.github/workflows/agent-evals.yml`.
