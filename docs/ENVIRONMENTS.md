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
| Deploy | Railway GitHub integration (push→deploy) | **Terraform** (infra) + per-repo **GitHub Actions CD** (merge→`gcloud run deploy` for all 4 services, keyless WIF), **gated on the `DEPLOY_TO_GCP` variable** — see [Continuous deployment](#continuous-deployment-gcp) |
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
  OpenSearch VM / Cloud SQL sized for demo, scale up only on promotion. To run
  **Railway-only** and actually stop the GCP spend, see
  [Going Railway-only](#going-railway-only-cost).

## Continuous deployment (GCP)

Each of the four services auto-deploys to Cloud Run **on merge to `main`**, via a
`deploy` job in that repo's own workflow (keyless WIF — repo secrets
`GCP_WIF_PROVIDER` + `GCP_DEPLOY_SA`, same values as genos-api):

| Service | Workflow | Gated on | Builds |
| --- | --- | --- | --- |
| api | `agent-evals.yml` | the eval quality gate | repo `Dockerfile` → `genos-api` |
| frontend | `ci.yml` (`deploy` job) | `frontend-test` | `Dockerfile.cloudrun` (`VITE_*` baked) → `genos-frontend` |
| sockets | `ci.yml` (`deploy` job) | `backend-flask` | repo `Dockerfile` → `genos-sockets` |
| collab | `ci.yml` (`deploy` job) | `collab` smoke | repo `Dockerfile` → `genos-collab` |

`gcloud run deploy --image` swaps **only the image** — Terraform stays the source of
truth for env/scaling/VPC/secrets (the Cloud Run `image` is under `ignore_changes`).
Railway is unaffected throughout; it auto-deploys each repo via its own GitHub
integration. (Before this, only api had CD, so a merged frontend/sockets/collab
change reached GCP only via a manual rebuild — the drift that once left a merged fix
invisible on GCP for days.)

### The `DEPLOY_TO_GCP` switch

Every GCP `deploy` job is gated on a GitHub Actions **variable**:
`if: … && vars.DEPLOY_TO_GCP == 'true'`.

- **Unset (the default) ⇒ OFF.** Merges are **Railway-only** and never touch GCP.
  Nothing to set — absence *is* the Railway-only state, and the `deploy` job simply
  shows as *skipped* (green), not failed. Test/eval jobs still run regardless.
- **Toggle it** (`true` = deploy to GCP on merge; anything else, or absent, = OFF),
  either way:

  - **GitHub UI** — per repo: **Settings → Secrets and variables → Actions → the
    Variables tab** (the *Variables* tab, not *Secrets*) → edit/add `DEPLOY_TO_GCP`. Direct links:
    [api](https://github.com/genos-tech/genos-api/settings/variables/actions) ·
    [frontend](https://github.com/genos-tech/genos-frontend/settings/variables/actions) ·
    [sockets](https://github.com/genos-tech/genos-sockets/settings/variables/actions) ·
    [collab](https://github.com/genos-tech/genos-collab/settings/variables/actions)
  - **CLI**:
    ```bash
    # per-repo (repo-admin is enough) — flip all four:
    for r in genos-api genos-frontend genos-sockets genos-collab; do
      gh variable set DEPLOY_TO_GCP --repo "genos-tech/$r" --body true; done   # false = off
    # OR one org-level variable (needs the admin:org scope: gh auth refresh -s admin:org):
    gh variable set DEPLOY_TO_GCP --org genos-tech --body true --visibility all
    ```

  It's currently wired **per-repo** (org-level needs `admin:org`, which the
  maintainer's token lacked). A repo-level value overrides the org one, so if you
  move to org-level later, delete the four repo copies. Changing the variable
  affects the **next** merge — it doesn't trigger a deploy by itself.

## Going Railway-only (cost)

> ⚠️ **`DEPLOY_TO_GCP` off freezes deploys — it does NOT stop the GCP bill.** The
> spend is always-on infra, independent of whether a deploy happens: Cloud SQL
> (`genos-pg`), the OpenSearch GCE VM, Redis Memorystore, and the warm Cloud Run
> min-instances all bill 24/7. The switch stops CI churn and pins GCP to its current
> revision; on its own it saves ≈ $0/mo.

To actually cut the bill, act on the running infra. Two tiers (both destructive /
outward — the operator's call):

**Tier 1 — Pause** (partial savings, keeps data, resume in minutes):

```bash
# Cloud Run: stop paying for warm instances
for s in genos-api genos-sockets genos-collab genos-frontend; do
  gcloud run services update "$s" --region asia-northeast1 --min-instances 0; done
# Cloud SQL: stop compute (storage still bills a few $/mo)
gcloud sql instances patch genos-pg --activation-policy NEVER
# OpenSearch VM: stop compute (disk still bills)
gcloud compute instances stop genos-opensearch --zone asia-northeast1-a
# Redis can't be "stopped" — delete it to stop billing (terraform recreates it)
```

Reverse with `--activation-policy ALWAYS`, `gcloud compute instances start …`, and
restoring the min-instance counts.

**Tier 2 — Destroy** (≈ $0, real re-standup later):

```bash
cd genos-platform/infra/gcp && terraform destroy
```

Safe **because GCP's data is a reproducible copy of Railway** (established
2026-07-05): the DB re-syncs from Railway, OpenSearch is rebuilt by the reindexer,
and generated secrets regenerate. Nothing irreplaceable is lost. The cost is
re-running the `infra/gcp/README.md` bootstrap on return (build/push images, re-seed
the manual secrets incl. the OAuth client secrets + the Fernet key, re-verify
domains, re-sync the DB).

Rule of thumb: **Tier 1** if you'll return within weeks and want a fast resume;
**Tier 2** if you're parking GCP indefinitely until funded. Either way, flip
`DEPLOY_TO_GCP` off first so CI doesn't try to redeploy into paused/destroyed infra.

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
- How GCP deploys work (per-repo CD + the `DEPLOY_TO_GCP` cost switch):
  [Continuous deployment](#continuous-deployment-gcp) above. The api eval gate that
  precedes its deploy: `genos-api/.github/workflows/agent-evals.yml`.
- Stop / reduce the GCP bill: [Going Railway-only](#going-railway-only-cost) above.
