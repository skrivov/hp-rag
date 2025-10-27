from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from dotenv import load_dotenv
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI

from src.hp_rag.storage import SQLiteHyperlinkConfig, SQLiteHyperlinkStore
from src.interfaces import BaseRetriever
from src.models.context import RetrievedContext


@dataclass(slots=True)
class HyperlinkRetrieverConfig:
    """Configuration for HP-RAG hyperlink retrieval."""

    sqlite_config: SQLiteHyperlinkConfig
    selection_prompt_template: str = (
        "You are given a table of contents for a knowledge base.\n"
        "Given the user query, return a JSON array of at most {limit} section\n"
        "paths (strings) that are most relevant. Only use paths that appear\n"
        "in the provided list.\n\n"
        "Query: {query}\n\n"
        "TOC (path: title):\n{toc}\n\n"
        "Return JSON array only, no commentary."
    )
    toc_limit: int = 200
    llm: Optional[Any] = None
    llm_model: Optional[str] = None
    max_sections: int = 10
    neighbor_window: int = 0  # Ignored; retained for backwards compatibility
    retriever_id: str = "hyperlink"

    def __post_init__(self) -> None:
        if self.llm is None and not self.llm_model:
            raise ValueError("HyperlinkRetrieverConfig requires `llm` instance or `llm_model` name.")


class HyperlinkRetriever(BaseRetriever):
    """LLM-based TOC filtering + targeted section lookup."""

    def __init__(self, config: HyperlinkRetrieverConfig) -> None:
        self.config = config
        super().__init__(config.retriever_id)
        self._store = SQLiteHyperlinkStore(config.sqlite_config)
        self._store.initialize()
        self._llm = config.llm

    def _retrieve(
        self,
        query: str,
        *,
        top_k: int,
    ) -> Tuple[List[RetrievedContext], Dict[str, int]]:
        toc_payload = self._collect_toc()

        candidate_paths = self._select_with_llm(query, toc_payload)

        seen: Set[str] = set()
        contexts: List[RetrievedContext] = []

        for path in candidate_paths:
            if path in seen:
                continue
            row = self._store.fetch_by_path(path)
            if not row:
                continue
            contexts.append(
                RetrievedContext(
                    path=row["path"],
                    title=row["title"],
                    text=row["body"],
                    score=1.0,
                    metadata={
                        "level": row["level"],
                        "parent_path": row["parent_path"],
                        "order_index": row["order_index"],
                        "retrieval_stage": "toc_filter",
                    },
                )
            )
            seen.add(path)

            if len(contexts) >= top_k:
                break

        contexts = contexts[:top_k]
        metadata = {
            "toc_candidates": len(candidate_paths),
            "toc_entries": len(toc_payload),
            "selector": "llm",
        }
        return contexts, metadata

    # Helpers --------------------------------------------------------------

    def _collect_toc(self) -> List[dict]:
        rows = self._store.iter_sections()
        toc = [
            {
                "path": row["path"],
                "title": row["title"],
                "level": row["level"],
                "parent_path": row["parent_path"],
                "order_index": row["order_index"],
            }
            for row in rows
        ]
        return toc[: self.config.toc_limit] if self.config.toc_limit else toc

    def _ensure_llm(self) -> Any:
        if self._llm is not None:
            return self._llm
        if self.config.llm_model:
            load_dotenv()
            self._llm = OpenAI(model=self.config.llm_model)
            return self._llm
        raise RuntimeError(
            "LLM-based TOC selection requires `llm` or `llm_model` to be provided and OPENAI_API_KEY to be set."
        )

    def _select_with_llm(self, query: str, toc_payload: Sequence[dict]) -> List[str]:
        llm = self._ensure_llm()
        toc_lines: List[str] = [f"- {item['path']}: {item['title']}" for item in toc_payload]
        toc_text = "\n".join(toc_lines)
        prompt = self.config.selection_prompt_template.format(
            query=query,
            toc=toc_text,
            limit=self.config.max_sections,
        )

        response = self._llm_complete(llm, prompt)
        paths = self._parse_paths_from_response(response, {item["path"] for item in toc_payload})
        return paths[: self.config.max_sections]

    def _llm_complete(self, llm: Any, prompt: str) -> str:
        if hasattr(llm, "complete"):
            res = llm.complete(prompt)
            text = getattr(res, "text", None)
            return text if isinstance(text, str) else str(res)
        if hasattr(llm, "predict"):
            return str(llm.predict(prompt))
        if hasattr(llm, "chat"):
            msg = ChatMessage(role=MessageRole.USER, content=prompt)
            res = llm.chat([msg])
            message = getattr(res, "message", None)
            if message is not None:
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
                    joined = "".join(pieces).strip()
                    if joined:
                        return joined
            text = getattr(res, "text", None)
            return text if isinstance(text, str) else str(res)
        raise AttributeError("LLM must provide complete, predict, or chat")

    def _parse_paths_from_response(self, text: Any, valid_paths: Set[str]) -> List[str]:
        raw = text
        if not isinstance(raw, str):
            raw = getattr(text, "text", None) or getattr(text, "message", None) or str(text)
        s = str(raw)
        try:
            data = json.loads(self._extract_json_block(s))
            if isinstance(data, list):
                return [p for p in data if isinstance(p, str) and p in valid_paths]
        except Exception:
            pass
        candidates: List[str] = []
        for path in valid_paths:
            if path in s:
                candidates.append(path)
        seen: Set[str] = set()
        ordered: List[str] = []
        for p in candidates:
            if p not in seen:
                seen.add(p)
                ordered.append(p)
        return ordered

    @staticmethod
    def _extract_json_block(text: str) -> str:
        m = re.search(r"\[[\s\S]*\]", text)
        return m.group(0) if m else text


__all__ = ["HyperlinkRetriever", "HyperlinkRetrieverConfig"]
