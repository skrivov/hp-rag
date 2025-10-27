# HP-RAG Evaluation Harness

This repository explores Hyperlink-Driven Retrieval-Augmented Generation (HP-RAG) and compares it with a conventional vector-based RAG pipeline. The project uses uv for dependency management, llama-index for LLM orchestration, FAISS for vector search, SQLite + FTS5 for hyperlink retrieval, and DeepEval for benchmarking.

## Quickstart
1. Install dependencies: `uv sync`
2. Run an interactive shell: `uv run python`
3. Execute the main entry point (placeholder): `uv run python main.py`

## Pipeline Overview
- Ingest markdown corpus into SQLite + FAISS: `uv run python scripts/build_toc.py ./docs ./artifacts`
- Run baseline evaluations: `uv run python scripts/run_evals.py ./datasets/questions.jsonl ./artifacts/eval.json --sqlite-db ./artifacts/hyperlink.db --faiss-dir ./artifacts/faiss_index`
- Render comparison report: `uv run python scripts/report.py ./artifacts/eval.json --output ./artifacts/report.md`

## Benchmark Workflow
- List supported datasets: `uv run python scripts/run_workflow.py list`
- Download a dataset: `uv run python scripts/run_workflow.py download beir-fiqa`
- Ingest and build stores (optionally cleaning prior artifacts): `uv run python scripts/run_workflow.py ingest beir-fiqa --clean-stores --embedding-model text-embedding-3-small`
- Evaluate (with optional vector baseline): `uv run python scripts/run_workflow.py evaluate beir-fiqa --include-vector --embedding-model text-embedding-3-small`
- End-to-end run: `uv run python scripts/run_workflow.py full beir-fiqa --force-download --clean-stores --include-vector --embedding-model text-embedding-3-small`
- Remove cached dataset and artifacts: `uv run python scripts/run_workflow.py cleanup beir-fiqa --remove-artifacts`

## Next Steps
- Implement ingestion scripts that build TOC-aware document sections.
- Stand up dual retrievers (vector + HP) and wire them into a shared interface.
- Build DeepEval suites to compare answer quality, context precision, and latency.
