# HP-RAG Evaluation Harness

This repository evaluates Hyperlink-Driven Retrieval-Augmented Generation (HP-RAG) alongside a conventional FAISS-based vector retriever. Both systems ingest markdown contracts, retrieve relevant clauses, and feed them into an answer LLM so tariff and duty obligations can be extracted with high fidelity. Every contract question carries a GPT-5-generated ground-truth answer, and both retrievers are scored against that reference using DeepEval metrics.

## HP-RAG at a Glance
HP-RAG combines a table-of-contents–aware SQLite store with an LLM selector that proposes the most relevant section paths before generation. This two-stage flow keeps retrieval grounded in document structure (titles, hierarchy, schedules) while preserving generative flexibility. The companion vector pipeline offers a semantic baseline using FAISS shards populated with real OpenAI embeddings. Because vector search relies on cosine similarity, long or multi-faceted contract questions tend to flatten the similarity surface, making exact clause retrieval harder—precisely where HP-RAG’s section-level filtering provides leverage.

## Executive Summary
For a one-page overview of the corpus, question set, metrics, and current comparative results, see the <a href="executive_summary.md">executive summary</a>.

## Quickstart
1. Install dependencies: `uv sync`
2. Run an interactive shell: `uv run python`
3. Ingest contracts and build artifacts: `uv run python scripts/build_toc.py data/contracts artifacts/contracts --embedding-model text-embedding-3-small`
4. Evaluate HP-RAG vs. vector baseline: `uv run python scripts/run_evals.py --config configs/experiments/contracts_llm_vs_vector.yaml --table-output artifacts/contracts/comparison.md`

## Pipeline Overview
- Ingest markdown corpus into SQLite + FAISS: `uv run python scripts/build_toc.py ./docs ./artifacts`
- Run baseline evaluations: `uv run python scripts/run_evals.py ./datasets/questions.jsonl ./artifacts/eval.json --sqlite-db ./artifacts/hyperlink.db --faiss-dir ./artifacts/faiss_index`
- Render comparison report: `uv run python scripts/report.py ./artifacts/eval.json --output ./artifacts/report.md`

## Streaming Server
- Launch the FastAPI service with SSE streaming via `uv run uvicorn src.server.api:app --reload`. Ensure `.env` defines `OPENAI_API_KEY` (plus `LLM_MODEL`, `SELECTOR_MODEL`, and `EMBEDDING_MODEL` as needed).
- Create a run using `POST /api/chat/runs` (body: `{"system": "hp-rag", "message": "How are duties calculated?"}`) and consume events from `GET /api/chat/stream?run_id=<id>`.
- During development you can skip the handshake with `GET /api/chat/stream?system=hp-rag&q=<query>`. Each SSE carries `event: token` deltas or architect-facing `event: event` diagnostics, followed by `event: done`.
- `GET /api/runs/{run_id}` returns the persisted transcript + diagnostics (stored under `RUNS_DB_PATH`, default `./artifacts/server_runs.db`), while `GET /api/healthz` is a simple readiness probe.
- Configure retention with `RUN_TTL_SECONDS` (seconds, default one day); older runs are purged automatically.
- Event envelopes follow the schema in `src/rag/server-feature.md`, so UI layers can render both the user chat stream and the monitoring timeline without additional mapping.

## Document Browser
- Upload documents via `POST /api/documents/upload` (multipart `file`, optional `title`). Uploads queue for ingestion, populate SQLite (HP-RAG) and rebuild the FAISS index (Vector RAG) automatically.
- List and inspect documents through `GET /api/documents` and `GET /api/documents/{id}`; fetch a text rendering with `GET /api/documents/{id}/body`.
- Discover available document browsers with `GET /api/navigation` (`documents` section) and explore system-specific views through plugin routes:
  - HP-RAG TOC/hyperlink explorer: `/api/plugins/hp-rag/documents/{id}/toc`, `/hyperlinks`, `/chunks`.
  - Vector RAG chunk/search APIs: `/api/plugins/rag/documents/{id}/chunks` and `/search?q=...`.
- Purge all artifacts (SQLite sections, FAISS index, uploads) via `DELETE /api/documents?force=true`. Configure storage roots with `SQLITE_DB`, `FAISS_INDEX_DIR`, and `UPLOADS_DIR`.

## Traces & Navigation
- `GET /api/traces` lists recent chat runs (filter with `?q=`), `GET /api/traces/{run_id}` fetches one trace, and `GET /api/traces/compare?run_a=...&run_b=...` returns two traces side-by-side for investigation.
- `GET /api/navigation` exposes menu metadata for Documents, Chat (placeholder), and Traces so the UI can render the three top-level items with submenus sourced from registered plugins.

## Key Code Elements
- `src/hp_rag/retriever.py` — LLM-guided TOC filter that selects hyperlink sections from SQLite and returns structured contexts.
- `src/rag/retriever.py` — FAISS-backed semantic retriever that mirrors the HP-RAG interface for apples-to-apples comparisons.
- `src/orchestration/query_runner.py` — Shared answer-generation orchestrator; templates, system prompts, and LLM configuration are injected via experiment configs.
- `scripts/build_toc.py` — End-to-end ingestion pipeline that chunkifies markdown/pdfs, populates SQLite (HP-RAG) and FAISS (vector) stores.
- `scripts/prepare_contract_questions.py` — Flattens the contract QA spec into JSONL and attaches citation-based reference contexts for retrieval metrics.
- `scripts/run_evals.py` — Main evaluation CLI supporting config-driven experiments, prompt overrides, and comparison table rendering.
- `configs/experiments/` — Reusable experiment presets (e.g., `contracts_llm_vs_vector.yaml`) that capture storage paths, retriever options, and prompt templates.
- `configs/prompts/` — Dataset-specific answer prompts; the contracts template enforces the summary + JSON “Details” format expected by evaluators.
- `data/contracts/` — GPT-5 Pro–synthesized contract corpus plus the evaluation questions (`questions.jsonl`).

## What’s Been Tested Today
- Contract ingestion into SQLite + FAISS via `scripts/build_toc.py data/contracts artifacts/contracts --embedding-model text-embedding-3-small`.
- HP-RAG and vector retrieval smoke tests (`scripts/run_contract_smoke.py`) using dataset-aligned prompts.
- Full evaluation run with DeepEval metrics through `scripts/run_evals.py --config configs/experiments/contracts_llm_vs_vector.yaml`, producing the comparison table captured in the <a href="executive_summary.md">executive summary</a>.
- Question preparation with citation-based reference contexts using `scripts/prepare_contract_questions.py`.

## Future-Facing Design (NOT YET EXECUTED)
- Dataset adapters under `src/ingest/adapters/` and the orchestration CLI (`scripts/run_workflow.py`) support downloading public corpora (BEIR, HotpotQA, MIRACL, SQuAD) and constructing retriever artifacts automatically. These flows are design scaffolds and have not been validated in the current contracts-focused sprint.
- Experiment presets in `configs/datasets/` will pair with `run_workflow.py` for end-to-end ingestion → evaluation once external datasets are integrated.
- Additional prompt templates and retriever settings can be layered via new experiment configs to extend the framework beyond the contract corpus.

## Potential Optimizations
Future iterations could tighten HP-RAG’s efficiency by normalizing the table-of-contents strings (removing repeated contract identifiers or numbering boilerplate) before feeding them to the selector LLM, thereby reducing prompt tokens. Other levers include tuning `toc_limit` per corpus, compressing chunk text with key-point summaries, caching selector outputs for recurring queries, or exploring smaller LLMs for the TOC filtering stage while keeping answer generation on a higher-capacity model.

In high-stakes contract intelligence workflows, HP-RAG can also serve as the retrieval backbone for agentic search loops—iterative query chains that refine evidence until the exact clause is found. Pairing the selector/answer pipeline with a deterministic calculator tool (for duties, thresholds, and caps) would further boost numerical consistency, letting the agent delegate arithmetic to code while the LLM focuses on interpretation.
