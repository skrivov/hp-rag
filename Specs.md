# HP-RAG Evaluation Architecture

## Goals
- Build a reference implementation of Hyperlink-Driven RAG (HP-RAG) backed by SQLite + FTS5.
- Stand up a conventional RAG baseline using llama-index with FAISS.
- Provide a repeatable evaluation pipeline (DeepEval) to compare both systems over the same query set.

## High-Level Components
1. **Ingestion Layer**
   - `ingest/toc_builder.py`: Parse document corpus, extract hierarchical TOC (document → section → subsection → sub-subsection), and emit structured metadata (`SectionNode`).
   - `ingest/chunker.py`: Split section content into retrieval-ready spans while preserving hyperlink trails.
   - Outputs feed both storage backends.
2. **Storage Backends**
   - **Vector Store**: FAISS index managed via llama-index; stores embeddings + text payloads + TOC metadata.
   - **Hyperlink Store**: SQLite database with FTS5 virtual table keyed by hierarchical hyperlink path and section text. Auxiliary tables capture parent-child relationships and ordering for navigation.
3. **Retrieval Services**
   - `src/rag/retriever.py`: Standard semantic retrieval using vector similarity search over FAISS-backed embeddings.
   - `src/hp_rag/retriever.py`: Two-stage lookup driven exclusively by an LLM—first narrowing TOC entries, then fetching matching sections (with neighbor expansion for continuity).
   - `src/interfaces/retriever.py`: Defines the shared `BaseRetriever` contract so both strategies return normalized `RetrievedContext` payloads.
4. **LLM Orchestration Layer**
   - `orchestration/query_runner.py`: Accepts query, routes to selected retriever, formats prompt for llama-index LLM, and logs interactions.
   - Future `prompts/` directory will hold reusable templates (answering, summarization, evaluation).
5. **Evaluation Harness**
   - `eval/metrics.py`: Thin wrapper around DeepEval metrics definitions (Answer Correctness, Faithfulness, Context Precision/Recall, etc.).
   - `eval/run.py`: CLI helpers to execute evaluation suites for one or both systems, collate DeepEval outputs, and persist results (JSON summary).
   - `scripts/report.py`: Render aggregated metrics into Markdown for quick comparison across suites.

## Data Flow
1. **Corpus Preparation**: Documents ingested; sections + hyperlinks produced.
2. **Indexing**: Data stored simultaneously in FAISS (via llama-index) and SQLite FTS.
3. **Query Execution**: User query triggers selected retriever → contexts assembled → LLM generates answer → results logged.
4. **Evaluation**: DeepEval replays standardized question set across both retrievers; metrics captured and compared.

## Key Design Considerations
- **Consistency**: Shared `SectionNode` schema ensures identical text spans across both systems for fair comparisons.
- **Caching**: Persist LLM outputs and retrieval caches to avoid recomputation during repeated evaluations.
- **Configurability**: Use `.yaml` profiles in `configs/` for repository paths, model choices, and evaluation suites.
- **Observability**: Store retrieval traces, prompt inputs, token counts, latencies in SQLite or JSONL logs.
- **Reproducibility**: Fix random seeds, version corpora, and record dependency hashes (managed by uv lockfiles).

## Roadmap Highlights
- Implement ingestion pipeline with pluggable parsers (PDF, Markdown, HTML).
- Formalize TOC prompt for stage-1 HP-RAG filtering and evaluate LLM cost/latency.
- Extend evaluation harness to output delta metrics and significance tests.
- Optionally integrate dashboards (e.g., Streamlit) for interactive comparison of retrieval traces.

## Detailed Component Responsibilities
- **Ingestion (`src/ingest/`)**: `MarkdownTOCBuilder` parses markdown headings into hierarchical `SectionNode` trees, while `ParagraphChunker` breaks section bodies into retrieval chunks. `IngestionPipeline` orchestrates both steps and writes sections/chunks into the configured stores, returning an `IngestionResult` summary to support CLI/status reporting.
- **Storage (`src/hp_rag/` & `src/rag/`)**: `SQLiteHyperlinkStore` encapsulates table + FTS creation, recursive section upserts, neighbor lookups, and FTS-backed search. `FaissVectorStore` owns llama-index setup, embedding configuration, persistence, and metadata export so downstream retrievers obtain consistent payloads.
- **Retrievers (`src/hp_rag/`, `src/rag/`, `src/interfaces/`)**: `BaseRetriever` standardizes latency timing and metadata normalization. `VectorRetriever` wraps llama-index similarity search, translating hits into `RetrievedContext` objects. `HyperlinkRetriever` relies solely on LLM-driven TOC selection (no heuristic fallback), optionally expanding neighbors, and returns contexts annotated with retrieval-stage metadata.
- **Orchestration (`src/orchestration/query_runner.py`)**: `QueryRunner` currently exposes `retrieve_context` to fetch contexts via any `BaseRetriever` implementation and retains configuration for a future llama-index LLM client. `run` intentionally raises `NotImplementedError` until generation + logging glue is finalized.
- **Evaluation (`src/eval/`)**: Lightweight DeepEval placeholder. `run.py` ingests question datasets (JSON or JSONL), fans out queries across supplied retrievers, aggregates basic coverage/latency metrics, and persists an `EvaluationResult`. `metrics.py` declares `MetricDefinition` structures so we can swap in actual DeepEval metrics without touching pipeline orchestration.
- **Scripts (`scripts/`)**: CLI wrappers (`build_toc.py`, `run_evals.py`, `report.py`) compose the above building blocks into reproducible workflows—ensuring artifact directories exist, injecting configs, and emitting human-readable checkpoints.

## Data Contracts
- `src/models/section.py`: `SectionNode` holds TOC hierarchy (document id, breadcrumb path, title, body, level, parent linkage, order, children). `add_child` maintains sibling ordering and parent references for reliable navigation.
- `src/ingest/chunker.py`: `SectionChunk` captures chunk text plus parent path/title and intra-section order. `ChunkConfig`/`Chunker` allow future implementations to respect token budgets or custom overlap behavior.
- `src/models/context.py`: `RetrievedContext` is the shared payload returned by all retrievers, ensuring downstream orchestration/evaluators can assume `path`, `title`, `text`, `score`, and optional metadata.
- `src/interfaces/retriever.py`: `RetrievalOutput` wraps retrieved contexts alongside execution metadata (latency, counts) in a retriever-agnostic envelope, enabling apples-to-apples evaluation.

## Storage Schemas & Indexing Strategy
- SQLite tables (`src/hp_rag/storage.py`) persist normalized sections (`sections`), chunk metadata (`chunks`), and an FTS5 virtual table (`sections_fts`) with triggers that keep search indices in sync for inserts/updates/deletes. Neighbor queries reuse `order_index` and `parent_path` to fetch contiguous context windows.
- FAISS persistence (`src/rag/storage.py`) leverages llama-index `VectorStoreIndex` with optional custom embedding model injection via `Settings.embed_model`. Stored metadata mirrors chunk provenance (`path`, `chunk_order`, `parent_title`) so retrieval outputs can rehydrate TOC context.
- Both backends lazily initialize and tolerate re-use across CLI invocations; `FaissVectorConfig.overwrite` governs whether persisted artifacts are cleared before rebuilds.

## Pipelines & CLI Entry Points
- `scripts/build_toc.py`: Discovers corpus files, initializes both stores, runs `IngestionPipeline`, and prints ingestion counts + artifact locations. Defaults target `./artifacts/hyperlink.db` and `./artifacts/faiss_index` while allowing overrides.
- `scripts/run_evals.py`: Wires `VectorRetriever` and `HyperlinkRetriever` with CLI-provided artifacts, executes `run_suite`, and materializes evaluation JSON artifacts. Flags expose HP-RAG tuning knobs (`--max-sections`, `--neighbor-window`, `--top-k`).
- `scripts/report.py`: Loads one or more evaluation JSON payloads, renders a markdown summary (grouped by suite + metric), and either prints or writes to `--output`.
- `main.py`: Remains a guardrail raising `NotImplementedError`, steering users toward the purpose-built CLIs until a consolidated driver is introduced.

## Evaluation Workflow
- Question ingestion: `run_suite` accepts JSON arrays or JSONL questions. It tolerates `query` or `question` keys so datasets remain flexible.
- Retrieval loop: For each retriever id, the suite measures average contexts returned, token footprint (naively split on whitespace, pending tokenizer integration), and observed retrieval latency using `RetrievalOutput.metadata['duration_ms']`.
- Artifact persistence: Results land at `EvaluationConfig.output_path` as JSON with `suite_name`, aggregated metrics, and are ready for reporting scripts or downstream visualization. Markdown reports (`scripts/report.py`) surface the same metrics for reviewers.
- Future DeepEval integration: Replace the placeholder aggregation with real `MetricDefinition`-driven evaluators once metric adapters are implemented.

## Observability, Configuration, and Artifacts
- Configuration dataclasses (e.g., `TOCBuilderConfig`, `ChunkConfig`, `SQLiteHyperlinkConfig`, `FaissVectorConfig`, `HyperlinkRetrieverConfig`, `VectorRetrieverConfig`, `QueryRunnerConfig`, `EvaluationConfig`) capture tunables with sensible defaults, enabling both code and CLI layers to override specifics without cross-module coupling.
- Observability hooks currently record ingestion counts, retrieval latencies, hit counts, and basic chunk metadata. TODO: extend to structured logging (JSONL) and SQLite-backed trace storage once orchestration matures.
- Artifacts live under `./artifacts/` by default: SQLite databases, FAISS directories, evaluation JSON, and optional markdown reports. Scripts ensure directories exist but leave lifecycle management (rotation/cleanup) to operators.
- Secrets/environment: Rely on `python-dotenv` patterns (per repo guidelines) before introducing hosted LLM integrations; no secrets are read directly in the current implementation.

## Open Questions & Follow-Ups
- Finalize llama-index LLM integration within `QueryRunner.run`, including prompt templates from `prompts/` and structured logging of request/response metadata.
- Expand ingestion to support HTML/PDF (likely via new `TOCBuilder` implementations) and richer chunking strategies (token-aware, hyperlink-preserving spans).
- Integrate actual DeepEval metrics and significance testing into `src/eval/run.py`, capturing per-question artifacts that drive richer reporting/dashboards.
- Evaluate caching strategies for TOC selection (persisted LLM outputs, pre-filtered candidate maps) to reduce latency during iterative evaluation.
