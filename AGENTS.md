# Repository Guidelines

## Project Structure & Module Organization
- Core Python packages live under `src/`, grouped by domain: `ingest` for corpus loading, `retrievers` for FAISS/FTS search, `orchestration` for pipeline glue, `models` for pydantic schemas, and `eval` for metrics utilities.
- Utility CLIs in `scripts/` drive ingestion, evaluation, and reporting; keep reusable logic in `src/` instead of inline scripts.
- Use `main.py` for lightweight manual runs and persist experiment outputs under `./artifacts/` for reproducible comparisons.

## Build, Test, and Development Commands
- `uv sync` — install all runtime and dev dependencies declared in `pyproject.toml`.
- `uv run python main.py` — execute the current prototype pipeline while developing orchestration flows.
- `uv run python scripts/build_toc.py ./docs ./artifacts` — ingest docs into SQLite and FAISS artifacts.
- `uv run python scripts/run_evals.py <questions.jsonl> ./artifacts/eval.json --sqlite-db ./artifacts/hyperlink.db --faiss-dir ./artifacts/faiss_index` — launch benchmark suites against both retrievers.
- `uv run python scripts/report.py ./artifacts/eval.json --output ./artifacts/report.md` — render comparison reports for reviewers.

## Coding Style & Naming Conventions
- Target Python 3.12, four-space indentation, and type hints on public functions; mirror the existing module layout when introducing new files (e.g., `src/retrievers/faiss_vector.py`).
- Run `uv run ruff check .` before pushing; only ignore lint warnings with an inline comment that explains the exception.
- Use snake_case for functions and variables, PascalCase for classes, and align file names with the primary class or role.

## Testing Guidelines
- Prefer `pytest`; place unit tests next to the code (e.g., `src/retrievers/tests/test_faiss.py`) or in `tests/` for broader integration scenarios.
- Name tests with intent-driven prefixes (e.g., `test_retrieve_hyperlinks_returns_ranked_chunks`).
- Run `uv run pytest`; cover retrievers, orchestration flows, and metrics, and capture regression fixtures under `./datasets/` when adding cases.

## Commit & Pull Request Guidelines
- Write commits in imperative mood with focused scope (e.g., `Add hyperlink reranker fallback`) and include supporting context in the body when behavior shifts.
- For PRs, supply a concise summary, test results, linked issues or specs, and attach report artifacts or screenshots when outputs change.
- Keep diffs small and review-ready; note TODOs or follow-ups in the PR description to maintain shared visibility.

## Artifacts & Configuration Tips
- Load secrets from a local `.env` via `python-dotenv` and keep credentials out of version control.
- Purge bulky FAISS/SQLite assets in `./artifacts/` before sharing branches while retaining sample schema files that aid reproducibility.

## Reality Mode (Always‑Real) Policy
The default operating mode for this repo is “everything is real” in interactive and CLI runs (tests may still use stubs). Follow these rules:

- HP‑RAG selection is always LLM‑based.
  - Do not use FTS fallback or any non‑LLM heuristics for TOC filtering in production code or CLIs.
  - `HyperlinkRetriever` must be configured with a live LLM (`llm` or `llm_model`) and requires `OPENAI_API_KEY` in `.env`.

- Vector baseline requires real embeddings.
  - FAISS builds and loads must use a genuine embedding model (e.g., `text-embedding-3-small`) or a provided embedder instance.
  - Do not use local/dummy/offline embedding fallbacks.
  - Treat vector storage dependencies as mandatory; keep llama-index/FAISS imports at module scope so misconfigurations fail fast.
- Benchmark adapters are part of ingestion.
  - Use dataset adapters (BEIR, SQuAD, HotpotQA, MIRACL) when working with structured corpora instead of ad-hoc scripts.
  - Add new adapters under `src/ingest/adapters/` and register them via the factory so CLIs expose them automatically.

- CLI defaults assume real services.
  - `scripts/build_toc.py` always builds SQLite TOC/link data. It only builds FAISS when an embedding model is configured (via `--embedding-model` or env). No fake embeddings.
  - `scripts/run_evals.py` always uses LLM selection for HP‑RAG; use `--llm-model` or `LLM_MODEL` to set the model.
  - `--include-vector` expects a FAISS index built with real embeddings and will error if assets are missing.

- Env requirements.
  - `.env` should define `OPENAI_API_KEY`; optionally `LLM_MODEL` and `EMBEDDING_MODEL`.
  - CLIs call `load_dotenv()` early; avoid hardcoding keys in code.

- Exception: tests.
  - Unit tests may use deterministic stubs or selectors to avoid network while preserving public interfaces.

Quick commands
- Build artifacts (real embeddings):
  - `uv run python scripts/build_toc.py ./data/fakes ./artifacts --embedding-model text-embedding-3-small`
- Run side‑by‑side eval (LLM HP‑RAG + vector):
  - `uv run python scripts/run_evals.py --include-vector --llm-model gpt-4o-mini --table-format plain`

## Simplicity & Forward-only Rule
- This is a new project; do not reintroduce legacy APIs or compatibility layers.
- Prefer direct, simple implementations that align with the current specs instead of emulating older behaviors.
- When requirements shift, update the modern code path—never bolt on fallbacks to mimic past designs.
