# Genos — チームの文脈（チャット、タスクなど）を理解するContext OSとAI Agent

> **DevOps × AI Agent Hackathon 2026 submission.** An AI agent that reads your team's
> chats, tasks and notes as *connected project memory*, autonomously plans + uses tools
> to answer with citations, and takes approval-gated actions — running on **Vertex AI +
> Cloud Run**, shipped through a **CI/CD pipeline that gates every deploy on agent evals**.

**Live:** https://gcp.genosai.dev · **Demo login available** (a pre-seeded workspace — open `Cmd-K` and ask *"How is the Website Redesign project going?"*)

📐 **システム構成の詳細は [ARCHITECTURE.md](ARCHITECTURE.md)**（インフラ / API / フロントエンド + 図）。

---

## つくる → まわす → とどける

| | | |
| --- | --- | --- |
| **つくる (Build)** | The agent itself | ~60 tools, autonomously selected in a multi-step loop; **human-in-the-loop approval** on every state-changing write; hybrid BM25 + vector RAG with inline citations. |
| **まわす (Run)** | evals-in-CI | Every PR runs the agent eval suite; a regression **blocks the Cloud Run deploy**. Auth is keyless (Workload Identity Federation). *It has already caught a real bug in this repo.* |
| **とどける (Deliver)** | Google Cloud | Cloud Run ×4 + Cloud SQL + Memorystore + a GCE OpenSearch node + **Vertex AI** (Gemini generation + embeddings), all **Terraform-managed**. |

## What this repo is

A **curated, public extract** of the private Genos monorepo, focused on the two things this
hackathon is judged on: the **AI agent implementation** and the **DevOps loop around it**.
It is *not* meant to run standalone (it's carved out of a larger app); it's here so the
agent code, the IaC, and the eval-gated pipeline can be read and verified.

```
src/search_engine/     ← the Spotlight / Genos AI agent (the crown jewel)
  agent/
    controller.py      · the multi-step tool-calling loop (run_agent / resume_agent)
    tools/             · ~60 tools — reads (search, list, analytics) + approval-gated writes
    prompts.py         · system prompt, tool cheat-sheet, prompt-injection guard
    evals/             · the eval harness: cases.yaml, retrieval_cases.yaml, judge.py, runner.py
  llm/                 · provider-neutral LLM abstraction (Gemini/Vertex + Claude), per-request model choice
  search.py, reranker.py, query_rewriter.py · hybrid retrieval (BM25 + kNN + RRF + rerank)
infra/gcp/             ← Terraform: Cloud Run, Cloud SQL, Memorystore, OpenSearch VM, Vertex, WIF
.github/workflows/
  agent-evals.yml      ← the evals-in-CI quality gate → gated Cloud Run deploy
docs/                  ← architecture + the Railway⇄GCP environment/switch runbook
```

## The agent is genuinely agentic

Not a chatbot bolted on. `agent/controller.py` runs a **think → choose tool → observe →
repeat** loop over native provider function-calling (up to N steps). Reads run inline;
every **write** tool (`create_task`, `update_task`, `create_note`, calendar mutations, …)
**pauses the stream and emits an approval card** — nothing side-effecting fires without the
user clicking Approve. Answers carry `[task:123]` / `[chat:…]` citations that resolve to
clickable, ACL-checked links.

## The DevOps loop (the differentiator)

`.github/workflows/agent-evals.yml`:

1. On every PR: spin up Postgres + the multilingual OpenSearch image, seed a deterministic
   fixture, and run the **retrieval eval suite** (deterministic, no LLM) as a **hard gate** +
   a behavior/`--judge` smoke (real Gemini via Vertex, keyless WIF).
2. A regression below the calibrated floors **fails the build** → the deploy job is skipped.
3. On merge to `main` with a green gate: build the image and `gcloud run deploy`.

This is the literal "continuously improve the AI via DevOps" idea — and it's not theoretical:
the gate caught a **real index-mapping bug** (`opensearch_setup` had to run before the first
reindex, or vector search silently returned nothing) — retrieval dropped `61/64 → 8/64` and
the gate blocked the deploy until it was fixed.

Measured on the live GCP stack: retrieval **61/64**, mrr **0.85**, recall **0.97**,
multilingual (ICU + kuromoji) green, agent TTFT **~4.3s**.

## Stack

**Mandatory tech:** Cloud Run (compute) · Vertex AI — `gemini-3.5-flash` + `gemini-embedding-001` (AI).
**Also:** Cloud SQL, Memorystore, OpenSearch on GCE, Artifact Registry, Secret Manager, Terraform,
GitHub Actions + Workload Identity Federation. App services: Django REST, Flask+Socket.IO, Node/Yjs,
React 19 + Vite.

## Judging-criteria map

1. **Agent is the core value** — autonomous multi-step tool use + approval-gated writes (`agent/`).
2. **Problem approach** — "connected project memory": cross-workspace synthesis, decision archaeology.
3. **Usability** — `Cmd-K` Spotlight, streamed cited answers, approval cards, 4 contextual surfaces.
4. **Practical value** — reads *and* writes across the workspace; multilingual; production-grade.
5. **Implementation** — full GCP stack + IaC + evals-in-CI gating a keyless Cloud Run deploy.
