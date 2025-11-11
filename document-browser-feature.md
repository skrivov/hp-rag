# Document Browser & Plugin Ingestion Service

## 1. Problem & Goals

We currently ingest corpora via offline scripts. Researchers want to upload ad-hoc documents, inspect their structure, and immediately run HP‑RAG or vector baselines against them. We also expect additional retrieval systems in the future. We need a first-class, plugin-friendly Document service that:

1. Accepts PDF/Markdown uploads, runs the ingestion pipeline (HP/FAISS) automatically, and tracks per-document status.
2. Provides CRUD APIs + UI surfaces to list documents and view whole-document renderings in a system-agnostic way.
3. Supports destructive maintenance actions (e.g., "delete all documents"), dropping SQLite + FAISS artifacts in a single operation.
4. Hooks into the FastAPI stack via pluggable adapters so HP‑RAG, baseline RAG, and future systems can implement their own browsing interfaces (TOC, hyperlink view, etc.) without coupling to the base document API.
5. For HP‑RAG specifically, expose a hyperlink-style browser (classic web navigation) that jumps between sections using the stored `path` hierarchy.

Non-goals: collaborative editing of document content, partial deletions (per section), or ingestion of remote corpora (still use CLI adapters for BEIR, etc.).

## 2. Requirements

### Functional
- Upload endpoint accepts `multipart/form-data` (one file per request) with metadata (title, tags). Supported formats: `.pdf`, `.md`. Files stored under `artifacts/uploads/` (configurable) before ingestion.
- Base ingestion automatically runs all registered retrieval plugins:
  - HP-RAG plugin: populate `sections`, `chunks`, `sections_fts` in SQLite (existing store) and build hyperlink navigation graph.
  - Vector baseline plugin: append to FAISS index with real embeddings (per Reality Mode rules). Rebuild on each ingestion until incremental updates land.
  - Future plugins can declare their own ingest hooks (schema migrations, vector stores, feature extractors).
- Every document keeps track of its contracting parties (tenants). Document listings and detail APIs must expose those tenants, and `/api/documents` must accept `tenant_id` so the UI can show only contracts for the selected party.
- Users can list all documents, filter by status, view entire document bodies (raw text or PDF preview), and delete everything (drops SQLite DB + FAISS dir + metadata tables + uploads).
- Document browser (core UI) is limited to:
  - Document table (name, type, size, status, ingested at).
  - Detail page showing document metadata and a full-document rendering (download link + text viewer).
- Plugin UIs extend the base view:
  - e.g., HP-RAG hyperlink explorer (TOC tree + clickable anchors).
  - e.g., Vector chunk inspector (embedding metadata, search, etc.).
- API surface mirrors that split:
  - `/api/documents/*` → system-agnostic listing + full-document fetch.
  - `/api/plugins/{plugin}/documents/*` → plugin-specific browsing (HP hyperlink, RAG chunk search, etc.).

### Non-Functional
- Ingestion jobs should be async/background to avoid blocking HTTP uploads (>10s). Use `asyncio.create_task` + in-memory queue; persist status in SQLite so restarts recover.
- Ensure deletions are atomic: the "delete all" endpoint must block until both stores are purged and metadata resets.
- Observability: log ingestion duration, chunk counts, and errors via existing instrumentation bus; surface status updates via SSE events when available.

## 3. High-Level Architecture

```
Client ──upload──▶ DocumentRouter ──▶ DocumentService
                                    │                         │
                               enqueue job                    │
                                    ▼                         ▼
                               Async worker ──▶ PluginRegistry ──▶ per-plugin ingest hooks
                                                      │
                                                      ├─ HP-RAG ingest → SQLiteHyperlinkStore (hyperlinks)
                                                      ├─ Vector ingest → FaissVectorStore
                                                      └─ Future plugin ingest targets
                                    ▼
                            DocumentStore (SQLite metadata) ──▶ UI listing & base document viewer
                                    │
                                    └─ PluginRouters (per system) expose specialized browse APIs (TOC, hyperlinks, embeddings…)
```

### Components
- **DocumentService** (new `src/server/documents/service.py`): orchestrates uploads, status tracking, cleanup, and plugin dispatch. It pushes ingestion work into the plugin registry.
- **DocumentStore**: extends the existing SQLite DB (same file as HP sections) with metadata tables to avoid dual databases. Tables:
  - `documents(id TEXT PK, name TEXT, source_path TEXT, mime_type TEXT, size_bytes INT, status TEXT, error TEXT, created_at TEXT, updated_at TEXT)`
  - `document_runs(document_id TEXT FK, job_id TEXT, stage TEXT, started_at TEXT, completed_at TEXT, stats JSON)` for auditing.
  - `uploads(id TEXT PK, document_id TEXT, original_filename TEXT, tmp_path TEXT, checksum TEXT)` (optional for dedupe/future multi-file support).
- **DocumentRouter** (new FastAPI router): handles REST endpoints for upload/list/delete + full-document rendering. Lives under `/api/documents`.
- **PluginRegistry**: central registry describing each retrieval system (HP-RAG, vector, future). Each plugin implements:
  - `name`, `display_name`.
  - `ingest(document_id, assets)` coroutine (given parsed sections/chunks, perform plugin-specific persistence).
  - `routes` (FastAPI APIRouter mounted at `/api/plugins/{name}`) for browsing.
  - `capabilities` metadata (e.g., `{"toc": true, "hyperlinks": true}`).
- **HP-RAG Plugin Router**: offers hyperlink browsing endpoints and TOC tree building. Generates hyperlink-style navigation payloads (list of nodes with `path`, `title`, `href`, `next_path`, `prev_path`) so the UI can mimic classic web navigation.
- **Async Worker**: minimal `asyncio.Queue` plus worker task started at app boot. When an upload arrives, service writes metadata row with `status='queued'`, saves file, enqueues job referencing document id. Worker pops jobs, runs ingestion pipeline, updates status/metrics, and triggers SSE progress events using existing instrumentation bus (reuse `EventBus` to emit `document.ingest.*` events for UIs that subscribe).

## 4. Data Model Changes

The existing SQLite schema (`sections`, `chunks`, `sections_fts`) already stores `document_id`. We extend it with:

```sql
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  source_path TEXT NOT NULL,
  mime_type TEXT,
  size_bytes INTEGER,
  status TEXT NOT NULL,
  error TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS document_runs (
  id TEXT PRIMARY KEY,
  document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  stage TEXT NOT NULL,
  stats TEXT,
  started_at TEXT NOT NULL,
  completed_at TEXT,
  status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS document_plugin_status (
  plugin_name TEXT NOT NULL,
  document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  status TEXT NOT NULL,
  error TEXT,
  stats TEXT,
  PRIMARY KEY (plugin_name, document_id)
);
```

We also add helper views:

- `document_sections` (document_id + section counts) for fast dashboard queries.
- `document_chunks` (document_id + chunk order + snippet) to drive the HP hyperlink browser and any plugin needing previews.

`SQLiteHyperlinkStore` gains methods:

- `list_documents()` returning metadata (joined with counts) for the base browser.
- `fetch_document_body(document_id)` to stream the full text for the common viewer.
- `delete_all_documents()` that truncates `documents`, `document_plugin_status`, `sections`, `chunks`, `sections_fts` within a transaction.
- HP-RAG plugin-only helpers: `fetch_toc(document_id)` for tree rendering and `fetch_hyperlink_graph(document_id)` returning ordered sections with `previous_path` / `next_path` / `href` so the UI can mimic classic hyperlink navigation.

FAISS store already rebuilds from chunk collections; for now we rebuild the entire index after each ingestion (fewer docs). Future work can add incremental vector insertion.

## 5. API Surface

### Base Document API (`/api/documents`)

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/api/documents/upload` | multipart upload (`file`, `title?`, `tags?`). Returns `{document_id, status}` immediately (status=`queued`). |
| `GET` | `/api/documents` | Lists documents with status counts, per-plugin states, and `tenants` metadata. Supports pagination/filtering plus `?tenant_id=...` to scope the table to a single party. |
| `GET` | `/api/documents/{document_id}` | Returns metadata, aggregated plugin statuses, and a download/preview URL for the entire document. |
| `GET` | `/api/documents/{document_id}/body` | Streams the whole document body (text/HTML) for the common viewer. |
| `DELETE` | `/api/documents` | Drops all documents (wipes SQLite + FAISS + uploads). Requires `?force=true` and confirmation payload. |
| (Future) `DELETE` `/api/documents/{document_id}` | Remove a single doc; will require selective FAISS rebuild. Not part of MVP but leave hooks. |

### Plugin APIs (`/api/plugins/{plugin}/…`)

Each plugin ships its own router. Examples:

| Plugin | Path | Description |
| --- | --- | --- |
| `hp-rag` | `/api/plugins/hp-rag/documents/{id}/toc` | Returns hierarchical TOC JSON (nodes: `path`, `title`, `level`, hyperlink metadata). |
| `hp-rag` | `/api/plugins/hp-rag/documents/{id}/hyperlinks` | Returns ordered sections with `href`, `prev_path`, `next_path`, enabling classic hyperlink navigation. |
| `hp-rag` | `/api/plugins/hp-rag/documents/{id}/chunks` | Returns HP-specific chunks/metadata for debugging selectors. |
| `rag` | `/api/plugins/rag/documents/{id}/chunks` | Returns vector chunks w/ scores, embedding metadata, search filters. |
| `rag` | `/api/plugins/rag/documents/{id}/search?q=…` | Runs semantic search within the ingested document to preview FAISS hits. |
| `future-*` | `/api/plugins/{name}/…` | Additional systems register their own browse routes via the plugin registry. |

### Responses & Status Tracking
- Upload response includes `run_url` for SSE subscription (`/api/documents/stream?document_id=...`). Document ingestion events reuse SSE with new `phase=document.ingest`. Example events: `document.ingest.queued`, `document.ingest.plugin.hp-rag`, `document.ingest.plugin.rag`, `document.ingest.completed`, `document.ingest.error`.
- Document listing aggregated stats come from `documents` + `document_plugin_status` so the UI can show per-plugin readiness.

## 6. UI / UX Notes

- **Documents page (core)**: grid listing metadata, per-plugin statuses, CTA to upload or purge. When the user picks a tenant, pass `tenant_id` on every `GET /api/documents` call so the grid only shows matching contracts. Use SSE to update row statuses without refresh.
- **Upload modal**: simple drag-and-drop; after submit, show job progress (subscribe to SSE channel). Block duplicate uploads by hashing file.
- **Document detail (core)**: default tab shows full-document text/preview + metadata timeline (ingest started/completed). Secondary tabs are injected by plugins.
- **HP-RAG plugin tab**: hyperlink explorer that mirrors "old style web pages": left nav = TOC tree; main pane renders selected section with inline hyperlinks for sibling/child sections using `prev_path`/`next_path`; clicking a link fetches the referenced section via plugin API.
- **RAG plugin tab**: chunk list with semantic search; displays vector scores and metadata to help debug FAISS results.
- **Delete all**: confirm dialog that explains artifacts removed (SQLite DB, FAISS index, uploads). Call `DELETE /api/documents?force=true`; show toast with result.

## 7. Operational Considerations

- **Config**: Extend `Settings` with `uploads_dir`, `documents_db_path`, `document_queue_size`, `document_ingest_workers`. Default `uploads_dir=artifacts/uploads`. Reuse existing `.env` for overrides.
- **Concurrency**: Use `asyncio.Queue(maxsize=Settings.document_queue_size)` to avoid unbounded uploads. When full, return 429 with retry-after.
- **Large files**: Accept up to, e.g., 20MB by default (FastAPI `UploadFile`). Persist to disk incrementally to avoid memory spikes.
- **Cleanup**: After successful ingestion, optionally delete raw uploaded file if `Settings.preserve_uploads` is false.
- **Error handling**: On ingestion failure, update `documents.status='failed'` + store stack trace snippet. Provide requeue endpoint later.

## 8. Implementation Plan (incremental)

1. **Data Layer**
   - Update `SQLiteHyperlinkStore` schema migrations to add `documents`, `document_runs`, `document_plugin_status` tables + helper views.
   - Add `DocumentMetadata` / `PluginStatus` models (`src/models/document.py`).
   - Implement `DocumentStore` wrapper to abstract CRUD, full-body streaming, plugin status updates.

2. **Plugin Registry**
   - Define `RetrievalPlugin` protocol (name, ingest hook, router factory, capabilities).
   - Implement HP-RAG plugin (uses existing SQLite store for TOC/hyperlinks) and RAG plugin (FAISS chunk search). Ensure registry is extendable.

3. **Service Layer**
   - Create `src/server/documents/service.py` + worker queue that ties uploads to DocumentStore, runs shared TOC/chunker pipeline, then dispatches to each plugin's `ingest()` coroutine.
   - Expose SSE events (`document.ingest.*`, `document.ingest.plugin.{name}`) using existing `EventBus`.

4. **API Layer**
   - Add base router (`src/server/documents/router.py`) for upload/list/detail/delete-all.
   - Mount plugin routers under `/api/plugins/{name}` via registry when FastAPI boots.

5. **UI/Client**
   - Implement base Documents page + detail view (full-document preview). Build HP hyperlink and RAG chunk tabs consuming plugin APIs (front-end work; spec the JSON payloads).

6. **Delete-All Command**
   - Implement `DocumentService.delete_all()` that drops DocumentStore tables, removes plugin artifacts (HP SQLite, FAISS directories), and deletes uploads. Tie into `DELETE /api/documents`.

7. **Testing**
   - Unit tests for `DocumentStore` migrations, ingestion queue, plugin registry dispatch, SSE events, delete-all.
   - Integration test simulating upload -> worker -> plugin-specific browse call (using temp files + stubbed embeddings).

## 9. Risks & Mitigations

- **Blocking uploads**: Large PDFs could still take time to reach disk. Mitigation: stream to temp file, enforce size limits, and return 413 when exceeded.
- **FAISS rebuild cost**: Rebuilding per upload is expensive. For MVP we accept the cost; document monitoring to revisit incremental updates when doc count > ~20.
- **Data loss on delete-all**: Provide double-confirmation UI and require `force=true` query param plus JSON body `{"confirm":"delete"}` to prevent accidents.
- **Queue starvation**: If ingestion worker crashes, queued docs would stall. Add watchdog to restart worker on failure and expose `/api/documents/health` to inspect queue length + last processed time.

## 10. Future Enhancements

- Partial deletes (per document) with selective FAISS rebuild using vector store delete APIs.
- Multi-user ownership & ACLs.
- Cloud storage (S3/Blob) for uploads with presigned URLs.
- Chunk annotations (allow analysts to flag key sections directly from UI).
