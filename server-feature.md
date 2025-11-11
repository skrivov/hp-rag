feature-server.md
# AgentLab – Feature Plan (Server Side)

**Goal:** Add a FastAPI service that exposes *chat* over both **HP‑RAG** and **baseline RAG**, with **token streaming** and a **structured stream of “thinking/progress” events** via SSE. The stream must serve two audiences simultaneously:

1) **User chat** — token-by-token assistant output (`event: token`).  
2) **Architect monitor** — succinct stage summaries *and* highly detailed diagnostic events (`event: event`) for each step.

The design is development‑first (architect-centric), but production‑safe.

---

## 0) Current Context (from repo)

- Retrieval code exists in:
  - `src/hp_rag/retriever.py` (HP‑RAG)  
  - `src/rag/retriever.py` (vector/FAISS baseline)  
- Shared orchestration logic is centered around `src/orchestration/query_runner.py`.  
- Tooling uses `uv` and Python virtual envs per README. :contentReference[oaicite:2]{index=2}

---

## 1) New Top‑Level Layout



src/
server/
init.py
api.py # FastAPI app and routers
sse.py # SSE helpers (publisher, encoders)
models.py # Pydantic request/response/event models
instrumentation.py # Context managers & decorators to emit events
chat_service.py # Orchestration adaptor for chat streaming
settings.py # Config (model names, limits, CORS, etc.)
orchestration/
query_runner.py # (augment, do not replace)
hp_rag/retriever.py # (wrap with instrumentation)
rag/retriever.py # (wrap with instrumentation)


### Python deps to add
- `sse-starlette` (SSE for FastAPI) :contentReference[oaicite:3]{index=3}
- `openai` (>=1.0) for async streaming (Chat/Responses APIs) :contentReference[oaicite:4]{index=4}
- `pydantic` (v2), `uvicorn`, `python-dotenv` (optional)

---

## 2) Endpoints

> All streaming endpoints are **GET** (EventSource requires GET). For POST‑style inputs, we use a pre‑create handshake returning a `run_id`, then connect via GET `/stream?run_id=…`. For dev we also allow a simple GET with query params.

- `POST /api/chat/runs`
  - Body: `{ system: "hp-rag"|"rag", message: string, metadata?: {...} }`
  - Returns: `{ run_id }`
- `GET /api/chat/stream?run_id=…`
  - **SSE** stream. Multiplexes:
    - `event: token` → `{ text: "<delta>" }` (assistant tokens)
    - `event: event` → rich progress diagnostics (see §4)
    - `event: done` → `{ usage, timings, finish_reason }`
- `GET /api/healthz` → 200
- (Optional) `GET /api/runs/:id` → returns persisted transcript + event log (for replay in UI)

**Dev convenience (no pre‑create):**

- `GET /api/chat/stream?system=hp-rag&q=...`  
  Useful while iterating from the UI.

---

## 3) Streaming & SSE

We use **sse‑starlette**’s `EventSourceResponse` with an async generator. Producer code pushes JSON events into an `asyncio.Queue` which the generator reads and yields as SSE. :contentReference[oaicite:5]{index=5}

### Server skeleton

```python
# src/server/api.py
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette import EventSourceResponse, JSONServerSentEvent
from .settings import Settings, get_settings
from .chat_service import start_run, stream_run

app = FastAPI(title="AgentLab API", version="0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/chat/runs")
async def create_run(payload: dict, settings: Settings = Depends(get_settings)):
    run_id = await start_run(payload, settings=settings)
    return {"run_id": run_id}

@app.get("/api/chat/stream")
async def chat_stream(request: Request, run_id: str | None = None,
                      system: str | None = None, q: str | None = None,
                      settings: Settings = Depends(get_settings)):
    # Allow ad-hoc dev flow: create ephemeral run if not provided
    if not run_id:
        if not (system and q):
            raise HTTPException(400, "Provide run_id or (system & q)")
        run_id = await start_run({"system": system, "message": q}, settings)

    async def gen():
        async for sse in stream_run(run_id, request, settings):
            yield sse
    return EventSourceResponse(gen())
```

## 4) Event Model (SSE “thinking monitor”)

We use two wire-level event types:

event: token — minimal token deltas for the chat transcript.

event: event — structured progress diagnostics for the architect.

Every “progress” event uses a normalized envelope:

{
  "version": 1,
  "run_id": "uuid",
  "step_id": "uuid",
  "parent_step_id": "uuid|null",
  "ts": "2025-11-10T12:34:56.789Z",
  "phase": "retrieval|selection|rerank|llm|tool|guard|emit|system",
  "type": "start|progress|end|error",
  "name": "hp_rag.selector" ,
  "summary": "Selecting candidate sections from TOC…",
  "detail": {
    "...": "rich, typed payload for architects (see below)"
  },
  "metrics": {
    "latency_ms": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 0
  },
  "severity": "info|debug|warn|error"
}


Short vs detailed: UI shows summary by default. A “Details” disclosure renders detail JSON with pretty print for architect review.
Chain‑of‑thought safety: never log hidden model reasoning text. If a model emits a reasoning summary field, include it in detail.reasoning_summary but do not expose raw chain‑of‑thought. 
OpenAI Platform

Canonical events to emit

Run lifecycle

system.run start|end|error (captures model names, config snapshot)

Query intake

query.received (user input, run options, redactions applied)

HP‑RAG selection (SQLite/TOC)

hp_rag.selector start → { k, prompt_template_id }

hp_rag.selector progress → { candidate_paths: [...], scores: [...] }

hp_rag.selector end → { chosen_paths: [...], usage, latency_ms }

Baseline RAG (FAISS)

rag.search start → { k }

rag.search end → { doc_ids: [...], scores: [...] }

Reranker (if used)

rerank start|end → { model, before_k, after_k, scores }

Context assembly

context.assembled end → { chunks: [{id, path, char_range}], total_chars, token_estimate }

LLM call

llm.call start → { provider: "openai", model, temperature, prompt_template_id }

llm.token delta → not in event; these become event: token with {text}

llm.call end → { usage: {prompt, completion, total}, latency_ms, finish_reason }

Tools

tool.call start|end → { tool_name, args, result_size }

Citations

emit.citations end → { items: [{doc_id, path, spans: [...] }...] }

Guards

guardrail start|end → { policy, action }

Errors

*.error with { message, exc_type, stack_excerpt } (omit secrets)

All end events may include usage/cost/latency where relevant.

## 5) Chat Service Orchestration

chat_service.py adapts your existing query_runner to a streaming style:
```python
# src/server/chat_service.py
import asyncio, json
from typing import AsyncIterator
from sse_starlette import JSONServerSentEvent
from openai import AsyncOpenAI
from .instrumentation import EventBus, step
from .models import RunConfig
from src.orchestration.query_runner import run_query_streaming

client = AsyncOpenAI()

_runs: dict[str, EventBus] = {}  # run_id -> EventBus

async def start_run(payload: dict, settings) -> str:
    cfg = RunConfig.from_payload(payload, settings)
    bus = EventBus()
    _runs[cfg.run_id] = bus
    # Fire-and-forget task to execute the run:
    asyncio.create_task(_execute(cfg, bus, settings))
    return cfg.run_id

async def stream_run(run_id: str, request, settings) -> AsyncIterator[JSONServerSentEvent]:
    bus = _runs[run_id]
    async for evt in bus.subscribe(request):
        yield evt

async def _execute(cfg: RunConfig, bus: EventBus, settings):
    async with step(bus, phase="system", name="system.run", type="start", summary="Run started"):
        async for item in run_query_streaming(cfg, bus=bus, settings=settings, client=client):
            if item.kind == "token":
                await bus.send("token", {"text": item.text})
            elif item.kind == "event":
                await bus.send("event", item.payload)
    await bus.send("done", {"finish_reason": "completed"})
```

The run_query_streaming(...) function is your orchestrator’s streaming wrapper (see below). The EventBus handles queues → SSE conversion.
 

## 6) Instrumentation Helpers

```python 
# src/server/sse.py
from sse_starlette import JSONServerSentEvent
import asyncio, contextlib, time

class EventBus:
    def __init__(self): self._q = asyncio.Queue()
    async def send(self, event: str, data: dict):
        await self._q.put(JSONServerSentEvent(data, event=event))
    async def subscribe(self, request):
        while True:
            if await request.is_disconnected(): break
            evt = await self._q.get()
            yield evt

# src/server/instrumentation.py
from contextlib import asynccontextmanager
import time, uuid

@asynccontextmanager
async def step(bus, *, phase, name, type="start", summary="", detail=None, parent_step_id=None):
    step_id = str(uuid.uuid4())
    ts = time.time()
    await bus.send("event", {
      "version": 1, "run_id": "TBD", "step_id": step_id, "parent_step_id": parent_step_id,
      "phase": phase, "type": "start", "name": name, "summary": summary, "detail": detail or {},
      "metrics": {}, "severity": "info"
    })
    try:
        yield step_id
        await bus.send("event", {
          "version": 1, "run_id": "TBD", "step_id": step_id,
          "phase": phase, "type": "end", "name": name,
          "summary": f"{summary} — done", "detail": {}, "metrics": {"latency_ms": int((time.time()-ts)*1000)}
        })
    except Exception as e:
        await bus.send("event", {
          "version": 1, "run_id": "TBD", "step_id": step_id,
          "phase": phase, "type": "error", "name": name, "summary": str(e),
          "detail": {"exc_type": type(e).__name__}, "severity": "error"
        })
        raise

```
## 7) Orchestrator (wrapping your existing code)

Augment src/orchestration/query_runner.py with a streaming generator that emits both events and tokens. This function adapts your current batch flow to a token‑streamed answer:

```python
# src/orchestration/query_runner.py  (new streaming adaptor)
from typing import AsyncIterator, NamedTuple, Literal
from src.hp_rag.retriever import HPRetriever
from src.rag.retriever import VectorRetriever

class StreamItem(NamedTuple):
    kind: Literal["event","token"]
    text: str | None = None
    payload: dict | None = None

async def run_query_streaming(cfg, bus, settings, client) -> AsyncIterator[StreamItem]:
    # Choose system
    system = cfg.system  # 'hp-rag' or 'rag'

    # 1) Retrieval
    if system == "hp-rag":
        with bus.step(phase="selection", name="hp_rag.selector", summary="Selecting candidates"):
            contexts = await HPRetriever(...).select_and_fetch(cfg.message)
    else:
        with bus.step(phase="retrieval", name="rag.search", summary="Vector search"):
            contexts = await VectorRetriever(...).search(cfg.message)

    yield StreamItem("event", payload={"phase":"context","type":"end","name":"context.assembled","detail":{"n": len(contexts)}})

    # 2) LLM call (OpenAI streaming)
    #    Use Chat Completions or Responses streaming; forward deltas as tokens. :contentReference[oaicite:7]{index=7}
    stream = await client.chat.completions.create(
        model=settings.answer_model,
        messages=[{"role":"system","content": cfg.system_prompt(contexts)},
                  {"role":"user","content": cfg.message}],
        stream=True,
        temperature=settings.temperature
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta: yield StreamItem("token", text=delta)
```

You may prefer the Responses API with client.responses.stream(...); either way, the UI only cares that token events contain deltas. 
OpenAI Platform

## 8) Configuration & Dev

src/server/settings.py for model names, temperature, timeouts, and SSE heartbeat (optional comment frames every 15s).

CORS enabled for RR7 dev ports.

Start API: uv run uvicorn src.server.api:app --reload (consistent with your repo’s use of uv). 
GitHub

## 9) Testing & Observability

Unit tests for event emission (snapshot of envelopes).

Integration test for /api/chat/stream reading until done.

Log counts: total tokens, per‑phase latency, retrieval hit counts.

Optional: persist {run_id, messages, events} in SQLite for replay.

## 10) Acceptance Criteria

 /api/chat/stream streams tokens for both hp‑rag and rag.

 Progress events cover: selection/retrieval, context, LLM start/end, citations, errors.

 Architect monitor receives summary + detail.

 No raw chain‑of‑thought logged.

 Works with the current retrievers and orchestrator.


---
