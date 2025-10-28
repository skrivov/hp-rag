Sample corpora and QA fixtures for local dry-runs.

Structure
- data/fakes/*.md — tiny Markdown corpus with headings and section text
- data/fakes/questions.jsonl — minimal eval set aligned to the fake corpus

Quick usage (HP-RAG only)
- Ingest to SQLite (and optionally FAISS) artifacts:
  uv run python scripts/build_toc.py ./data/fakes ./artifacts

- Run an HP-only sanity eval (uses FTS fallback by default in the CLI):
  uv run python scripts/run_evals.py ./data/fakes/questions.jsonl ./artifacts/eval.json \
    --sqlite-db ./artifacts/hyperlink.db --faiss-dir ./artifacts/faiss_index --top-k 3

Notes
- Vector path (FAISS) requires embeddings; if you only want HP-RAG, you can ignore the FAISS directory. The CLI enables FTS fallback for HP selection if no LLM is configured.

