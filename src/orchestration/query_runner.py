from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence

from dotenv import load_dotenv
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI

from src.interfaces import BaseRetriever, RetrievalOutput
from src.models.context import RetrievedContext


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


__all__ = ["QueryRunner", "QueryRunnerConfig", "QueryResult", "GeneratedAnswer"]
