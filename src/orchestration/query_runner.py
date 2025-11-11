from __future__ import annotations

import json
import os
import asyncio
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, AsyncIterator, Iterable, List, Literal, NamedTuple, Sequence

from dotenv import load_dotenv
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI

from src.hp_rag.retriever import HyperlinkRetriever, HyperlinkRetrieverConfig
from src.hp_rag.storage import SQLiteHyperlinkConfig, SQLiteHyperlinkStore
from src.interfaces import BaseRetriever, RetrievalOutput
from src.models.context import RetrievedContext
from src.models.tenant import TenantSelection
from src.rag.retriever import VectorRetriever, VectorRetrieverConfig
from src.rag.storage import FaissVectorConfig
from src.server.instrumentation import EventFactory


@dataclass(slots=True)
class QueryRunnerConfig:
    """Configuration for orchestrating LLM queries."""

    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.0
    system_prompt: str = (
        "You are a helpful assistant that answers questions using the provided context."
    )
    default_top_k: int = 5
    prompt_template: str | None = None


class QueryRunner:
    """Routes queries through a retriever and LLM."""

    def __init__(
        self,
        retriever: BaseRetriever,
        config: QueryRunnerConfig | None = None,
        *,
        llm: Any | None = None,
    ) -> None:
        self.retriever = retriever
        self.config = config or QueryRunnerConfig()
        self._llm = llm

    def retrieve_context(self, query: str, *, top_k: int | None = None) -> RetrievalOutput:
        requested = top_k or self.config.default_top_k
        return self.retriever.retrieve(query, top_k=requested)

    def run(self, query: str, *, top_k: int | None = None) -> "QueryResult":
        retrieval_output = self.retrieve_context(query, top_k=top_k)
        generated = self.generate_answer(query, retrieval_output.contexts)
        return QueryResult(
            query=query,
            answer=generated.answer,
            prompt=generated.prompt,
            retrieval=retrieval_output,
        )

    def generate_answer(
        self,
        query: str,
        contexts: Sequence[RetrievedContext] | Iterable[RetrievedContext],
    ) -> "GeneratedAnswer":
        context_list = list(contexts)
        prompt = self._format_prompt(query, context_list)
        llm = self._resolve_llm()
        self._attach_contexts(llm, context_list)
        completion = self._invoke_llm(llm, prompt)
        answer = self._extract_text(completion)
        return GeneratedAnswer(prompt=prompt, answer=answer)

    def _resolve_llm(self) -> Any:
        if self._llm is not None:
            return self._llm

        # Load .env so OPENAI_API_KEY is available for the client
        load_dotenv()

        model_name = os.getenv("LLM_MODEL", self.config.llm_model)
        self._llm = OpenAI(model=model_name, temperature=self.config.temperature)
        return self._llm

    def _format_prompt(
        self,
        query: str,
        contexts: Iterable[RetrievedContext | str | dict[str, Any]],
    ) -> str:
        context_snippets: list[str] = []
        for idx, context in enumerate(contexts, start=1):
            title, text = self._context_payload(idx, context)
            snippet = f"[{idx}] {title}\n{text}".strip()
            context_snippets.append(snippet)

        context_block = "\n\n".join(context_snippets) or "No relevant context provided."
        template = self.config.prompt_template or self._default_prompt_template()
        return template.format(
            system_prompt=self.config.system_prompt,
            query=query,
            context=context_block,
        ).strip()

    def _attach_contexts(
        self,
        llm: Any,
        contexts: Sequence[RetrievedContext],
    ) -> None:
        if llm is None:
            return
        context_snapshot: List[dict[str, Any]] = [
            {
                "path": ctx.path,
                "title": ctx.title,
                "text": ctx.text,
                "score": ctx.score,
                "metadata": ctx.metadata or {},
            }
            for ctx in contexts
        ]
        if hasattr(llm, "set_contexts"):
            llm.set_contexts(context_snapshot)
        elif hasattr(llm, "context_snapshot"):
            setattr(llm, "context_snapshot", context_snapshot)

    def _invoke_llm(self, llm: Any, prompt: str) -> Any:
        if hasattr(llm, "complete"):
            return llm.complete(prompt)
        if hasattr(llm, "predict"):
            return llm.predict(prompt)
        if hasattr(llm, "chat"):
            message = ChatMessage(role=MessageRole.USER, content=prompt)
            return llm.chat([message])

        raise AttributeError("LLM must implement complete, predict, or chat APIs")

    def _extract_text(self, completion: Any) -> str:
        if completion is None:
            return ""
        if isinstance(completion, str):
            return completion.strip()

        text = getattr(completion, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()

        message = getattr(completion, "message", None)
        if message is not None:
            content = getattr(message, "content", None)
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                pieces = []
                for chunk in content:
                    if isinstance(chunk, str):
                        pieces.append(chunk)
                    elif isinstance(chunk, dict):
                        pieces.append(str(chunk.get("text", "")))
                candidate = "".join(pieces).strip()
                if candidate:
                    return candidate

        return str(completion).strip()

    @staticmethod
    def _context_payload(index: int, context: RetrievedContext | str | dict[str, Any]) -> tuple[str, str]:
        if isinstance(context, RetrievedContext):
            return context.title or f"Context {index}", context.text
        if isinstance(context, str):
            return f"Context {index}", context
        if isinstance(context, dict):
            title = str(context.get("title", f"Context {index}"))
            text = str(context.get("text", ""))
            return title, text
        title = getattr(context, "title", f"Context {index}")
        text = getattr(context, "text", "")
        return str(title), str(text)

    @staticmethod
    def _default_prompt_template() -> str:
        return (
            "{system_prompt}\n\n"
            "Question: {query}\n\n"
            "Retrieved Context:\n{context}\n\n"
            "Answer:"
        )


@dataclass(slots=True)
class QueryResult:
    """Represents a completed query run including LLM answer and retrieval data."""

    query: str
    answer: str
    prompt: str
    retrieval: RetrievalOutput


@dataclass(slots=True)
class GeneratedAnswer:
    """Container for LLM prompt and answer text."""

    prompt: str
    answer: str


class StreamItem(NamedTuple):
    kind: Literal["token", "event", "summary"]
    text: str | None = None
    payload: dict[str, Any] | None = None


async def run_query_streaming(
    cfg,
    factory: EventFactory,
    settings: Any,
    client: Any,
) -> AsyncIterator[StreamItem]:
    """Stream retrieval + LLM progress events and token deltas."""

    total_start = perf_counter()
    yield StreamItem(
        "event",
        payload=factory.event(
            phase="system",
            type="start",
            name="query.received",
            summary="User query received",
            detail={"query": cfg.message, "system": cfg.system},
        ),
    )

    tenant: TenantSelection | None = None
    tenant_events: List[dict[str, Any]] = []
    if cfg.system == "hp-rag":
        tenant, tenant_events = await _resolve_tenant_context(cfg, settings, factory, client)
        if tenant:
            cfg.metadata = dict(cfg.metadata or {})
            cfg.metadata.update(
                {
                    "tenant_id": tenant.tenant_id,
                    "tenant_role": tenant.role,
                    "tenant_source": tenant.source,
                }
            )
    for event in tenant_events:
        yield StreamItem("event", payload=event)

    contexts, retrieval_meta, retrieval_events = await _retrieve_with_events(
        cfg, factory, settings, tenant
    )
    for event in retrieval_events:
        yield StreamItem("event", payload=event)

    yield StreamItem(
        "event",
        payload=_context_event(factory, contexts, tenant),
    )

    llm_summary: dict[str, Any] | None = None
    async for item in _stream_llm_answer(cfg, factory, contexts, client, settings):
        if item.kind == "summary":
            llm_summary = item.payload or {}
        else:
            yield item

    yield StreamItem("event", payload=_citation_event(factory, contexts, tenant))

    total_ms = int((perf_counter() - total_start) * 1000)
    timings = {"total_ms": total_ms}
    if retrieval_meta.get("duration_ms") is not None:
        timings["retrieval_ms"] = retrieval_meta["duration_ms"]
    if llm_summary and "timings" in llm_summary:
        timings.update(llm_summary["timings"])

    summary_payload = {
        "usage": (llm_summary or {}).get("usage", {}),
        "finish_reason": (llm_summary or {}).get("finish_reason", "stop"),
        "timings": timings,
    }
    if tenant:
        summary_payload["tenant_id"] = tenant.tenant_id
        summary_payload["tenant_role"] = tenant.role
    yield StreamItem("summary", payload=summary_payload)


async def _retrieve_with_events(
    cfg,
    factory: EventFactory,
    settings: Any,
    tenant: TenantSelection | None,
) -> tuple[list[RetrievedContext], dict[str, Any], list[dict[str, Any]]]:
    retriever = _build_retriever(cfg.system, settings, top_k=cfg.top_k, tenant=tenant)
    step_name = "hp_rag.selector" if cfg.system == "hp-rag" else "rag.search"
    phase = "selection" if cfg.system == "hp-rag" else "retrieval"
    tracker = factory.step(
        phase=phase,
        name=step_name,
        summary="Selecting candidate sections" if cfg.system == "hp-rag" else "Vector search",
        detail={"k": cfg.top_k},
    )
    events = [tracker.start_event()]
    retrieval_output: RetrievalOutput = await asyncio.to_thread(retriever.retrieve, cfg.message, top_k=cfg.top_k)
    metadata = dict(retrieval_output.metadata)
    if tenant:
        metadata.setdefault("tenant_id", tenant.tenant_id)
        metadata.setdefault("tenant_role", tenant.role)

    if cfg.system == "hp-rag":
        if metadata.get("candidate_paths"):
            events.append(
                tracker.progress_event(
                    summary="TOC candidates scored",
                    detail={
                        "candidate_paths": metadata.get("candidate_paths"),
                        "scores": metadata.get("candidate_scores"),
                    },
                )
            )
        events.append(
            tracker.end_event(
                detail={
                    **{
                        "chosen_paths": [ctx.path for ctx in retrieval_output.contexts],
                        "usage": metadata.get("selector_usage", {}),
                    },
                    **({"tenant": _tenant_detail_dict(tenant)} if tenant else {}),
                }
            )
        )
    else:
        detail = {
            "doc_ids": [ctx.path for ctx in retrieval_output.contexts],
            "scores": [ctx.score for ctx in retrieval_output.contexts],
        }
        tenant_detail = _tenant_detail_dict(tenant)
        if tenant_detail:
            detail["tenant"] = tenant_detail
        events.append(tracker.end_event(detail=detail))

    return retrieval_output.contexts, metadata, events


def _build_retriever(
    system: str,
    settings: Any,
    *,
    top_k: int,
    tenant: TenantSelection | None = None,
) -> BaseRetriever:
    if system == "hp-rag":
        sqlite_path = Path(getattr(settings, "sqlite_db_path", "artifacts/hyperlink.db"))
        if not sqlite_path.exists():
            raise FileNotFoundError(f"SQLite hyperlink DB not found at {sqlite_path}")
        selector_model = getattr(settings, "selector_model", getattr(settings, "answer_model", None))
        if not selector_model:
            raise ValueError("selector_model must be configured for HP-RAG runs.")
        config = HyperlinkRetrieverConfig(
            sqlite_config=SQLiteHyperlinkConfig(db_path=sqlite_path),
            toc_limit=getattr(settings, "hp_toc_limit", 200),
            llm_model=selector_model,
            max_sections=getattr(settings, "hp_max_sections", min(top_k, 8)),
            tenant_id=tenant.tenant_id if tenant else None,
        )
        return HyperlinkRetriever(config)

    faiss_dir = Path(getattr(settings, "faiss_index_path", "artifacts/faiss_index"))
    if not faiss_dir.exists():
        raise FileNotFoundError(f"FAISS index directory not found at {faiss_dir}")
    embedding_model = getattr(settings, "embedding_model", None)
    if not embedding_model:
        raise ValueError("embedding_model must be configured for vector RAG runs.")
    config = VectorRetrieverConfig(
        faiss_config=FaissVectorConfig(index_path=faiss_dir, embed_model_name=embedding_model),
        similarity_top_k=top_k,
    )
    return VectorRetriever(config)


async def _stream_llm_answer(
    cfg,
    factory: EventFactory,
    contexts: Sequence[RetrievedContext],
    client: Any,
    settings: Any,
) -> AsyncIterator[StreamItem]:
    detail = {
        "provider": "openai",
        "model": getattr(settings, "answer_model", "gpt-4o-mini"),
        "temperature": cfg.temperature,
    }
    tracker = factory.step(phase="llm", name="llm.call", summary="Generating answer", detail=detail)
    yield StreamItem("event", payload=tracker.start_event())

    prompt = _format_stream_prompt(cfg, contexts)
    messages = [{"role": "user", "content": prompt}]

    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    finish_reason = "stop"
    llm_start = perf_counter()
    stream = await client.chat.completions.create(
        model=getattr(settings, "answer_model", "gpt-4o-mini"),
        messages=messages,
        temperature=cfg.temperature,
        stream=True,
    )
    async for chunk in stream:
        choice = chunk.choices[0] if chunk.choices else None
        if choice:
            delta = choice.delta
            text = _delta_to_text(delta.content if hasattr(delta, "content") else None)
            if text:
                yield StreamItem("token", text=text)
            if choice.finish_reason:
                finish_reason = choice.finish_reason
        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage:
            usage = {
                "prompt_tokens": getattr(chunk_usage, "prompt_tokens", usage["prompt_tokens"]),
                "completion_tokens": getattr(chunk_usage, "completion_tokens", usage["completion_tokens"]),
                "total_tokens": getattr(chunk_usage, "total_tokens", usage["total_tokens"]),
            }

    if hasattr(stream, "aclose"):
        await stream.aclose()

    llm_ms = int((perf_counter() - llm_start) * 1000)
    yield StreamItem(
        "event",
        payload=tracker.end_event(detail={"usage": usage, "finish_reason": finish_reason}),
    )
    yield StreamItem(
        "summary",
        payload={"usage": usage, "finish_reason": finish_reason, "timings": {"llm_ms": llm_ms}},
    )


def _context_event(
    factory: EventFactory,
    contexts: Sequence[RetrievedContext],
    tenant: TenantSelection | None,
) -> dict[str, Any]:
    total_chars = sum(len(ctx.text or "") for ctx in contexts)
    chunks = [
        {
            "id": idx,
            "path": ctx.path,
            "title": ctx.title,
            "score": ctx.score,
            "char_len": len(ctx.text or ""),
        }
        for idx, ctx in enumerate(contexts, start=1)
    ]
    detail = {
        "chunks": chunks,
        "total_chars": total_chars,
        "token_estimate": total_chars // 4 if total_chars else 0,
    }
    tenant_detail = _tenant_detail_dict(tenant)
    if tenant_detail:
        detail["tenant"] = tenant_detail
    return factory.event(
        phase="context",
        type="end",
        name="context.assembled",
        summary="Context window assembled",
        detail=detail,
    )


def _citation_event(
    factory: EventFactory,
    contexts: Sequence[RetrievedContext],
    tenant: TenantSelection | None,
) -> dict[str, Any]:
    items = [
        {
            "doc_id": ctx.path or f"ctx-{idx}",
            "path": ctx.path,
            "title": ctx.title,
            "score": ctx.score,
        }
        for idx, ctx in enumerate(contexts, start=1)
    ]
    detail = {"items": items}
    tenant_detail = _tenant_detail_dict(tenant)
    if tenant_detail:
        detail["tenant"] = tenant_detail
    return factory.event(
        phase="emit",
        type="end",
        name="emit.citations",
        summary="Citations prepared",
        detail=detail,
    )


def _format_stream_prompt(cfg, contexts: Sequence[RetrievedContext]) -> str:
    context_snippets: list[str] = []
    for idx, ctx in enumerate(contexts, start=1):
        title, text = QueryRunner._context_payload(idx, ctx)
        context_snippets.append(f"[{idx}] {title}\n{text}".strip())

    context_block = "\n\n".join(context_snippets) or "No relevant context provided."
    template = cfg.prompt_template or QueryRunner._default_prompt_template()
    return template.format(
        system_prompt=cfg.system_prompt,
        query=cfg.message,
        context=context_block,
    ).strip()


def _delta_to_text(content: Any) -> str:
    if not content:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for chunk in content:
            if isinstance(chunk, str):
                pieces.append(chunk)
            elif isinstance(chunk, dict):
                if "text" in chunk:
                    pieces.append(str(chunk["text"]))
                elif "content" in chunk:
                    pieces.append(str(chunk["content"]))
        return "".join(pieces)
    return str(content)


async def _resolve_tenant_context(
    cfg,
    settings: Any,
    factory: EventFactory,
    client: Any,
) -> tuple[TenantSelection | None, List[dict[str, Any]]]:
    events: List[dict[str, Any]] = []
    tenant_id = _requested_tenant_id(cfg)

    sqlite_path = Path(getattr(settings, "sqlite_db_path", "artifacts/hyperlink.db"))
    if not sqlite_path.exists():
        return None, events

    store = SQLiteHyperlinkStore(SQLiteHyperlinkConfig(db_path=sqlite_path))
    store.initialize()
    try:
        if tenant_id:
            row = store.fetch_tenant(tenant_id)
            if row:
                selection = _tenant_from_row(row, source="user")
                events.append(_tenant_event(factory, selection, "Tenant provided by client"))
                return selection, events
            events.append(
                factory.event(
                    phase="selection",
                    type="warning",
                    name="tenant.resolve",
                    summary="Requested tenant not found",
                    detail={"tenant_id": tenant_id},
                )
            )

        detected = await _detect_tenant_from_question(cfg.message, store, settings, client)
        if detected:
            events.append(_tenant_event(factory, detected, "Tenant inferred from query"))
            return detected, events
    finally:
        store.close()

    return None, events


def _requested_tenant_id(cfg) -> str | None:
    candidate = getattr(cfg, "tenant_id", None)
    if candidate:
        return str(candidate)
    metadata = getattr(cfg, "metadata", {}) or {}
    tenant_id = metadata.get("tenant_id")
    return str(tenant_id) if tenant_id else None


def _tenant_from_row(
    row,
    *,
    source: str,
    confidence: float | None = None,
    reason: str | None = None,
) -> TenantSelection:
    selection = TenantSelection(
        tenant_id=row["tenant_id"],
        name=row["name"],
        role=row["role"] or "",
        source=source,
        confidence=confidence,
        reason=reason,
    )
    return selection


def _tenant_event(
    factory: EventFactory,
    tenant: TenantSelection,
    summary: str,
) -> dict[str, Any]:
    return factory.event(
        phase="selection",
        type="info",
        name="tenant.resolve",
        summary=summary,
        detail=_tenant_detail_dict(tenant),
    )


async def _detect_tenant_from_question(
    question: str,
    store: SQLiteHyperlinkStore,
    settings: Any,
    client: Any,
) -> TenantSelection | None:
    rows = store.list_tenants(limit=25)
    if not rows:
        return None

    tenant_summary = []
    allowed_ids: List[str] = []
    for row in rows:
        aliases: List[str] = []
        raw_aliases = row["aliases"]
        if raw_aliases:
            try:
                aliases = json.loads(raw_aliases)
            except json.JSONDecodeError:
                aliases = []
        tenant_summary.append(
            {
                "tenant_id": row["tenant_id"],
                "name": row["name"],
                "role": row["role"],
                "aliases": aliases,
                "documents": row["document_count"],
            }
        )
        allowed_ids.append(row["tenant_id"])

    prompt = _TENANT_SELECTOR_PROMPT.format(
        question=question.strip(),
        tenants=json.dumps(tenant_summary, ensure_ascii=False, indent=2),
    )
    model_name = getattr(settings, "selector_model", getattr(settings, "answer_model", "gpt-4o-mini"))
    response = await client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You match user questions to the most relevant contract party. "
                    "Use the provided tenant list. If unsure, set tenant_id to null.\n"
                    'Return JSON: {"tenant_id": <string|null>, "role": <string>, "confidence": <0-1>, "reason": <string>}.'
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    text = _response_message_text(response)
    data = _safe_json_object(text)
    if not data:
        return None

    tenant_id = data.get("tenant_id")
    if tenant_id not in allowed_ids:
        return None

    row = next((r for r in rows if r["tenant_id"] == tenant_id), None)
    if row is None:
        return None

    confidence = _safe_confidence(data.get("confidence"))
    reason = data.get("reason") or data.get("explanation")
    selection = _tenant_from_row(row, source="detector", confidence=confidence, reason=reason)
    if not selection.role and data.get("role"):
        selection.role = str(data["role"])
    return selection


def _response_message_text(response: Any) -> str:
    choice = response.choices[0] if getattr(response, "choices", None) else None
    if not choice:
        return ""
    message = getattr(choice, "message", None)
    if message is None:
        return str(choice)
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: List[str] = []
        for chunk in content:
            if isinstance(chunk, str):
                pieces.append(chunk)
            elif isinstance(chunk, dict):
                pieces.append(str(chunk.get("text", "")))
        return "".join(pieces)
    return str(content)


def _safe_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        candidate = ""
        for segment in parts[1:]:
            segment = segment.strip()
            if not segment:
                continue
            candidate = segment
            break
        if candidate.lower().startswith("json"):
            candidate = candidate[4:].strip()
        cleaned = candidate or cleaned
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = cleaned[start : end + 1]
    try:
        payload = json.loads(snippet)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _safe_confidence(value: Any) -> float | None:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return None
    if conf < 0 or conf > 1:
        return None
    return conf


def _tenant_detail_dict(tenant: TenantSelection | None) -> dict[str, Any] | None:
    if not tenant:
        return None
    detail = {
        "tenant_id": tenant.tenant_id,
        "name": tenant.name,
        "role": tenant.role,
        "source": tenant.source,
    }
    if tenant.confidence is not None:
        detail["confidence"] = tenant.confidence
    if tenant.reason:
        detail["reason"] = tenant.reason
    return detail


_TENANT_SELECTOR_PROMPT = (
    "Question:\n{question}\n\n"
    "Tenants (JSON array):\n{tenants}\n\n"
    "Pick the tenant_id whose perspective best matches the question. If none, reply with tenant_id = null."
)


__all__ = [
    "GeneratedAnswer",
    "QueryResult",
    "QueryRunner",
    "QueryRunnerConfig",
    "StreamItem",
    "run_query_streaming",
]
