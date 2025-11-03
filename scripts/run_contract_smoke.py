from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable, List

from dotenv import load_dotenv

from src.hp_rag.retriever import HyperlinkRetriever, HyperlinkRetrieverConfig
from src.hp_rag.storage import SQLiteHyperlinkConfig
from src.orchestration.query_runner import QueryRunner, QueryRunnerConfig
from src.rag.retriever import VectorRetriever, VectorRetrieverConfig
from src.rag.storage import FaissVectorConfig


DEFAULT_QUESTIONS = Path("data/contracts/questions.jsonl")


@dataclass(slots=True)
class SmokeSample:
    identifier: str
    question: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a short HP-RAG or vector RAG smoke test on contract questions."
    )
    parser.add_argument(
        "--retriever",
        choices=["hp", "vector"],
        default="hp",
        help="Retriever to exercise: hp (HP-RAG) or vector (FAISS baseline).",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS,
        help="Path to questions JSONL file (default: data/contracts/questions.jsonl).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of questions to run (default: 3).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file. Defaults to artifacts/contracts/<retriever>_smoke.txt.",
    )
    parser.add_argument(
        "--sqlite-db",
        type=Path,
        default=Path("artifacts/contracts/hyperlink.db"),
        help="SQLite hyperlink store for HP-RAG (default: artifacts/contracts/hyperlink.db).",
    )
    parser.add_argument(
        "--faiss-dir",
        type=Path,
        default=Path("artifacts/contracts/faiss_index"),
        help="FAISS index directory for vector RAG (default: artifacts/contracts/faiss_index).",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Override LLM model used for selection/answering (defaults to env LLM_MODEL or gpt-4o-mini).",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Embedding model name for FAISS baseline (defaults to env EMBEDDING_MODEL).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of contexts to retrieve per query (default: 3).",
    )
    parser.add_argument(
        "--toc-limit",
        type=int,
        default=100,
        help="Max TOC entries exposed to HP-RAG selector (default: 100).",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Override system prompt for answer generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the answer LLM (default: 0.0).",
    )
    parser.add_argument(
        "--prompt-template-file",
        type=Path,
        default=None,
        help="Optional path to a prompt template used for answer generation.",
    )
    return parser.parse_args()


def load_samples(path: Path, limit: int) -> List[SmokeSample]:
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")
    samples: List[SmokeSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            question = str(payload.get("question") or payload.get("query") or "").strip()
            identifier = str(payload.get("id") or payload.get("slug") or len(samples)).strip()
            if not question:
                continue
            samples.append(SmokeSample(identifier=identifier, question=question))
            if len(samples) >= limit:
                break
    if not samples:
        raise ValueError(f"No questions were loaded from {path}")
    return samples


def build_hp_rag(args: argparse.Namespace):
    llm_model = args.llm_model or os.getenv("LLM_MODEL") or "gpt-4o-mini"
    config = HyperlinkRetrieverConfig(
        sqlite_config=SQLiteHyperlinkConfig(db_path=args.sqlite_db),
        llm_model=llm_model,
        toc_limit=args.toc_limit,
        max_sections=args.top_k,
    )
    return HyperlinkRetriever(config)


def build_vector_rag(args: argparse.Namespace):
    embedding_model = args.embedding_model or os.getenv("EMBEDDING_MODEL") or os.getenv(
        "OPENAI_EMBEDDING_MODEL"
    )
    if embedding_model is None:
        raise ValueError(
            "Vector smoke test requires an embedding model name via --embedding-model or EMBEDDING_MODEL env."
        )
    faiss_config = FaissVectorConfig(
        index_path=args.faiss_dir,
        embed_model_name=embedding_model,
    )
    return VectorRetriever(
        VectorRetrieverConfig(
            faiss_config=faiss_config,
            similarity_top_k=args.top_k,
        )
    )


def ensure_output_path(path: Path | None, retriever: str) -> Path:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    default_path = Path("artifacts/contracts") / f"{retriever}_smoke.txt"
    default_path.parent.mkdir(parents=True, exist_ok=True)
    return default_path


def render_contexts(contexts: Iterable, limit: int) -> List[str]:
    lines: List[str] = []
    for idx, ctx in enumerate(contexts, start=1):
        if idx > limit:
            break
        title = getattr(ctx, "title", "") or "(untitled)"
        path = getattr(ctx, "path", "") or "(unknown)"
        snippet = getattr(ctx, "text", "")[:500].replace("\n", " ").strip()
        lines.append(f"[{idx}] {title} ({path}) :: {snippet}")
    return lines


def main() -> None:
    args = parse_args()
    load_dotenv()

    samples = load_samples(args.questions, args.limit)

    if args.retriever == "hp":
        retriever = build_hp_rag(args)
    else:
        retriever = build_vector_rag(args)

    prompt_template = (
        Path(args.prompt_template_file).read_text(encoding="utf-8") if args.prompt_template_file else None
    )
    base_runner = QueryRunnerConfig()
    system_prompt = args.system_prompt or base_runner.system_prompt
    answer_llm = args.llm_model or os.getenv("LLM_MODEL") or "gpt-4o-mini"

    runner_config = QueryRunnerConfig(
        llm_model=answer_llm,
        temperature=args.temperature,
        system_prompt=system_prompt,
        prompt_template=prompt_template,
        default_top_k=args.top_k,
    )
    runner = QueryRunner(retriever, runner_config)

    output_path = ensure_output_path(args.output, args.retriever)
    lines: List[str] = []
    lines.append(f"Retriever: {retriever.retriever_id}")
    lines.append(f"Questions source: {args.questions}")
    lines.append("")

    for sample in samples:
        result = runner.run(sample.question, top_k=args.top_k)
        contexts = render_contexts(result.retrieval.contexts, args.top_k)
        metadata_json = json.dumps(result.retrieval.metadata, indent=2, sort_keys=True)
        lines.append(f"=== {sample.identifier} ===")
        lines.append(f"Question: {sample.question}")
        lines.append("Answer:")
        lines.append(result.answer.strip())
        lines.append("Contexts:")
        lines.extend(f"  {ctx}" for ctx in contexts)
        lines.append("Metadata:")
        lines.extend(f"  {line}" for line in metadata_json.splitlines())
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote smoke test results to {output_path}")


if __name__ == "__main__":
    main()
