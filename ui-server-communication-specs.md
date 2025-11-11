# UI ↔ Server Communication Specs

This document enumerates every FastAPI endpoint and streaming contract exposed by the HP-RUG server so the React Router v7 UI (built in Codex) can call them consistently.

---
## 1. Base Conventions
- **Base URL (dev):** `http://localhost:8000`
- **Headers:** `Content-Type: application/json` for JSON payloads; multipart uploads use `multipart/form-data`.
- **Auth:** not required in current sprint.
- **SSE:** use `EventSource`; expect `event`, `data` pairs. All SSE payloads are JSON strings.
- **Error Contract:** FastAPI default (`{"detail": "..."}`) for 4xx/5xx.

---
## 2. Navigation Discovery
`GET /api/navigation`
```json
{
  "documents": [
    {"name": "hp-rag", "label": "HP-RAG", "capabilities": {"toc": true, "hyperlinks": true, "chunks": true}},
    {"name": "rag", "label": "Vector RAG", "capabilities": {"chunks": true, "search": true}}
  ],
  "chat": [],
  "traces": {"list": "/api/traces", "compare": "/api/traces/compare"}
}
```
*Use this to build the Documents submenu. Chat plugins will appear later.*

---
## 3. Documents API
### 3.1 Upload
`POST /api/documents/upload`
- `multipart/form-data` fields:
  - `file`: PDF or Markdown.
  - `title` (optional string).
```json
{
  "document_id": "uuid",
  "status": "queued",
  "plugins": [{"plugin_name": "hp-rag", "status": "queued", "stats": {}, "error": null}]
}
```

### 3.2 List
`GET /api/documents?tenant_id=optional`
```json
{
  "documents": [
    {
      "id": "uuid",
      "name": "Contracts.pdf",
      "status": "ready",
      "section_count": 120,
      "chunk_count": 540,
      "plugins": [{"plugin_name": "hp-rag", "status": "ready", "stats": {"sections": 120}}],
      "tenants": [
        {"tenant_id": "apex-robotics", "name": "Apex Robotics, Inc.", "role": "Buyer"}
      ]
    }
  ]
}
```
Pass `tenant_id` to scope the table to a single contracting party (e.g., when the user picks a tenant in the UI).

### 3.3 Detail
`GET /api/documents/{id}` → same shape as list item.

### 3.4 Full Body (Markdown-like)
`GET /api/documents/{id}/body`
```json
{"document_id": "uuid", "body": "# Heading\nSection text..."}
```

### 3.5 Delete All
`DELETE /api/documents?force=true`
```json
{"status": "deleted"}
```

---
## 4. Document Plugins
Each plugin exposes its own router under `/api/plugins/{name}`. Use navigation discovery to know which ones to show.

### 4.1 HP-RAG (hyperlink explorer)
- `GET /api/plugins/hp-rag/documents/{id}/toc`
```json
{"document_id": "uuid", "nodes": [{"path": "doc/section", "title": "Section", "level": 1, "parent_path": "doc", "order_index": 3}]}
```
- `GET /api/plugins/hp-rag/documents/{id}/hyperlinks`
```json
{
  "document_id": "uuid",
  "sections": [
    {"path": "doc/sec1", "title": "Intro", "level": 1, "parent_path": "doc", "body": "...", "prev_path": null, "next_path": "doc/sec2"}
  ]
}
```
- `GET /api/plugins/hp-rag/documents/{id}/chunks`
```json
{"document_id": "uuid", "chunks": [{"section_path": "doc/sec1", "order": 0, "text": "chunk", "section_title": "Intro"}]}
```

### 4.2 Vector RAG (chunk/search)
- `GET /api/plugins/rag/documents/{id}/chunks` → same chunk array as above.
- `GET /api/plugins/rag/documents/{id}/search?q=keyword`
```json
{"document_id": "uuid", "query": "keyword", "results": [{"path": "doc/sec1", "title": "Intro", "snippet": "...", "level": 1}]}
```

---
## 5. Tenants API
Use these endpoints to power the tenant picker and scope document/chat flows.

### 5.1 List
`GET /api/tenants?q=optional&limit=50`
```json
{
  "items": [
    {
      "tenant_id": "apex-robotics",
      "name": "Apex Robotics, Inc.",
      "role": "Buyer",
      "aliases": ["Buyer"],
      "document_count": 3
    }
  ]
}
```

### 5.2 Detail
`GET /api/tenants/{tenant_id}`
```json
{
  "tenant_id": "apex-robotics",
  "name": "Apex Robotics, Inc.",
  "role": "Buyer",
  "aliases": ["Buyer"],
  "document_count": 3,
  "documents": ["doc-1", "doc-9", "doc-12"]
}
```
Use the `documents` array to prefetch previews or to deep-link the document browser after a tenant is selected.

---
## 6. Chat API
### 5.1 Create Run
`POST /api/chat/runs`
```json
{"system": "hp-rag", "message": "How are duties calculated?", "tenant_id": "apex-robotics", "metadata": {...}}
```
Response:
```json
{"run_id": "uuid"}
```

### 5.2 Stream
`GET /api/chat/stream?run_id=uuid`
- SSE stream; events:
  - `event: token` → `{"text": "partial"}`
  - `event: event` → structured progress (see §5.4)
  - `event: done` → `{"finish_reason": "stop", "usage": {...}, "timings": {...}}`
- Dev shortcut: `GET /api/chat/stream?system=hp-rag&q=question` (auto-creates run).

### 5.3 Run Snapshot
`GET /api/runs/{run_id}` → same payload as traces detail (see §6.2).

### 5.4 Progress Event Schema
`event: event` data:
```json
{
  "version": 1,
  "run_id": "uuid",
  "step_id": "uuid",
  "phase": "selection|retrieval|llm|emit|system",
  "type": "start|progress|end|error",
  "name": "hp_rag.selector",
  "summary": "Selecting candidates",
  "detail": {...},
  "metrics": {"latency_ms": 12},
  "severity": "info"
}
```
UI should show `summary` and allow expansion of `detail` JSON.

---
## 7. Traces API
### 6.1 List
`GET /api/traces?limit=50&q=optional` → returns array of recent runs (ordered by created_at desc). Each item:
```json
{
  "run_id": "uuid",
  "system": "hp-rag",
  "message": "user query",
  "created_at": "2025-02-10T12:34:56Z",
  "status": "ready|error",
  "finish_reason": "stop",
  "usage": {...},
  "metadata": {...}
}
```

### 6.2 Detail
`GET /api/traces/{run_id}` → full trace from RunStore:
```json
{
  "run_id": "uuid",
  "system": "hp-rag",
  "created_at": "...",
  "tokens": "full answer",
  "events": [...],
  "done": {"finish_reason": "stop", "usage": {...}},
  "error": null
}
```

### 6.3 Compare
`GET /api/traces/compare?run_a=uuid&run_b=uuid2`
```json
{"run_a": {...trace...}, "run_b": {...trace...}}
```
UI can align tokens/events by `step_id` or timestamps for side-by-side diff.

---
## 8. SSE Heartbeats
- Event bus emits `event: heartbeat` every `SSE_HEARTBEAT_SECONDS` (default 15). Data: `{ "ts": ISO8601 }`. UI can ignore or use to show liveness.

---
## 9. Configurable Paths
All storage paths (SQLite DB, FAISS, uploads, runs DB) live under `/artifacts/...` by default. No UI action required, but uploads >20MB may be rejected in the future (surfaced as HTTP 413).

---
## 10. Future Hooks
- `chat` section in `/api/navigation` will start listing chat plugins once the registry exposes them. UI should render gracefully when array is empty.
- More document plugins can be added without frontend code changes thanks to discovery endpoint.
