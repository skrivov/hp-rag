from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, TYPE_CHECKING

from dotenv import load_dotenv

# Load .env early so keys/models are available
if Path(".env").exists():
    load_dotenv(".env", override=False)

from src.eval import EvaluationConfig, run_suite
from src.ingest import (
    IngestionPipeline,
    MarkdownTOCBuilder,
    MarkdownTOCBuilderConfig,
    ParagraphChunker,
    PyMuPDFTOCBuilder,
    PyMuPDFTOCBuilderConfig,
    TenantExtractor,
)
from src.ingest.adapters.factory import adapter_choices, create_adapter
from src.orchestration.config_loader import (
    PreparedDataset,
    load_dataset_config,
    load_experiment_config,
    prepare_datasets,
)
from src.orchestration.query_runner import QueryRunnerConfig
from src.hp_rag.retriever import HyperlinkRetriever, HyperlinkRetrieverConfig
from src.hp_rag.storage import SQLiteHyperlinkConfig, SQLiteHyperlinkStore
from src.rag.retriever import VectorRetriever, VectorRetrieverConfig
from src.rag.storage import FaissVectorConfig, FaissVectorStore

if TYPE_CHECKING:
    from src.models.configs import ExperimentConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run HP-RAG (default) and optional vector baseline evaluations with minimal flags."
    )

    # Positional but optional for easy dry runs
    parser.add_argument(
        "questions",
        nargs="?",
        type=Path,
        default=Path("data/fakes/questions.jsonl"),
        help="Path to question dataset (JSON/JSONL). Default: data/fakes/questions.jsonl",
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("artifacts/eval.json"),
        help="Where to store evaluation results (JSON). Default: artifacts/eval.json",
    )
    parser.add_argument(
        "--suite", default="baseline", help="Suite identifier (default: baseline)"
    )

    # Storage defaults to artifacts/
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/fakes"),
        help="Path to corpus directory (default: data/fakes)",
    )
    parser.add_argument(
        "--pattern",
        default="**/*.md",
        help="Glob pattern for corpus discovery (default: **/*.md)",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(adapter_choices().keys()),
        help="Dataset adapter for structured corpora (beir, squad, hotpotqa, miracl)",
    )
    parser.add_argument(
        "--miracl-language",
        help="Language code for MIRACL adapter when ingesting a directory",
    )
    parser.add_argument(
        "--sqlite-db",
        type=Path,
        default=Path("artifacts/hyperlink.db"),
        help="SQLite hyperlink store path (default: artifacts/hyperlink.db)",
    )
    parser.add_argument(
        "--faiss-dir",
        type=Path,
        default=Path("artifacts/faiss_index"),
        help="Persisted FAISS index directory (default: artifacts/faiss_index)",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL") or os.getenv("OPENAI_EMBEDDING_MODEL"),
        help="Embedding model used by FAISS retriever at load time (e.g., text-embedding-3-small). Defaults from env.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Purge and rebuild both stores from the --corpus before evaluation",
    )

    # HP-RAG knobs
    parser.add_argument(
        "--max-sections",
        type=int,
        default=5,
        help="Max sections to consider after selection (default: 5)",
    )
    parser.add_argument(
        "--neighbor-window",
        type=int,
        default=1,
        help="Neighbor window for HP-RAG retrieval (default: 1)",
    )
    parser.add_argument(
        "--toc-limit",
        type=int,
        default=100,
        help="Max TOC entries shown to selector to speed selection (default: 100)",
    )

    # Retrieval count
    parser.add_argument(
        "--top-k", type=int, default=3, help="Contexts to retrieve per query (default: 3)"
    )

    # LLM controls: always on; model from env or default
    parser.add_argument(
        "--llm-model",
        default=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        help="LLM model for selection and answering (default from env LLM_MODEL or gpt-4o-mini)",
    )

    # Optional vector baseline
    parser.add_argument(
        "--include-vector",
        action="store_true",
        help="Also run vector baseline (requires FAISS artifacts)",
    )

    # Table rendering/output options
    parser.add_argument(
        "--table-output",
        type=Path,
        default=None,
        help="Optional path to write the comparison table (md, csv, or tsv)",
    )
    parser.add_argument(
        "--table-format",
        choices=["md", "csv", "tsv", "plain"],
        default=None,
        help="Table format for stdout; defaults to md. If --table-output is set, the format is inferred from extension unless overridden here.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to an experiment config file (YAML/TOML/JSON) to seed arguments",
    )
    parser.add_argument(
        "--prompt-template-file",
        type=Path,
        default=None,
        help="Optional prompt template file for answer generation (overrides experiment config).",
    )

    return parser


def _collect_parser_defaults(parser: argparse.ArgumentParser) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}
    for action in parser._actions:
        dest = getattr(action, "dest", None)
        if not dest or dest == argparse.SUPPRESS:
            continue
        defaults[dest] = action.default
    return defaults


def _apply_experiment_overrides(
    args: argparse.Namespace,
    defaults: Dict[str, Any],
    experiment: "ExperimentConfig",
    prepared_datasets: List[PreparedDataset],
) -> Dict[str, Any]:
    adapter_kwargs: Dict[str, Any] = {}

    def override(field: str, value: Any) -> None:
        if value is None:
            return
        if field not in defaults or getattr(args, field, None) == defaults[field]:
            setattr(args, field, value)

    eval_cfg = experiment.evaluation
    override("suite", eval_cfg.suite)
    override("questions", eval_cfg.questions)
    override("output", eval_cfg.output)
    override("table_output", eval_cfg.table_output)
    override("table_format", eval_cfg.table_format)

    override("corpus", experiment.corpus)
    override("pattern", experiment.pattern)
    override("dataset", experiment.dataset)
    override("miracl_language", experiment.miracl_language)
    override("sqlite_db", experiment.sqlite_db)
    override("faiss_dir", experiment.faiss_dir)
    override("embedding_model", experiment.embedding_model)

    if experiment.rebuild:
        override("rebuild", True)
    if experiment.include_vector:
        override("include_vector", True)

    hp_params = experiment.hp_rag or {}
    override("llm_model", hp_params.get("llm_model"))
    override("max_sections", hp_params.get("max_sections"))
    override("neighbor_window", hp_params.get("neighbor_window"))
    override("toc_limit", hp_params.get("toc_limit"))
    override("top_k", hp_params.get("top_k"))

    for retr in experiment.retrievers:
        rtype = retr.type.lower()
        if rtype in {"hp_rag", "hyperlink"}:
            override("llm_model", retr.llm_model)
            if retr.max_sections is not None:
                override("max_sections", retr.max_sections)
            if retr.neighbor_window is not None:
                override("neighbor_window", retr.neighbor_window)
            if retr.toc_limit is not None:
                override("toc_limit", retr.toc_limit)
            if retr.top_k is not None:
                override("top_k", retr.top_k)
        elif rtype in {"vector", "faiss"}:
            override("include_vector", True)
            if retr.embedding_model:
                override("embedding_model", retr.embedding_model)
            if retr.faiss_index:
                override("faiss_dir", retr.faiss_index)
            if retr.top_k is not None:
                override("top_k", retr.top_k)
        else:
            print(f"[warn] Unknown retriever type {retr.type!r} in experiment config; ignoring")

    if prepared_datasets:
        primary = prepared_datasets[0]
        override("dataset", primary.config.adapter)
        override("corpus", primary.corpus_path)
        if primary.questions_path:
            override("questions", primary.questions_path)
        if "language" in primary.adapter_kwargs:
            override("miracl_language", primary.adapter_kwargs["language"])
        adapter_kwargs = dict(primary.adapter_kwargs)
        if len(prepared_datasets) > 1:
            others = ", ".join(pd.config.name for pd in prepared_datasets[1:])
            print(
                "[warn] Multiple dataset configs provided; defaulting to"
                f" {primary.config.name!r} for CLI arguments. Ignoring: {others}",
            )

    return adapter_kwargs


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    defaults = _collect_parser_defaults(parser)

    args._adapter_kwargs = {}
    args._prepared_datasets = []
    args._hp_rag_overrides = {}
    args._query_runner_config = {}

    if args.config:
        args.config = args.config.resolve()
        experiment_config = load_experiment_config(args.config)
        dataset_configs = [load_dataset_config(path) for path in experiment_config.dataset_configs]
        prepared_datasets = prepare_datasets(dataset_configs) if dataset_configs else []
        args._adapter_kwargs = _apply_experiment_overrides(
            args, defaults, experiment_config, prepared_datasets
        )
        args._prepared_datasets = prepared_datasets
        args._hp_rag_overrides = experiment_config.hp_rag or {}
        args._query_runner_config = experiment_config.query_runner or {}
        print(
            f"[info] Loaded experiment config {experiment_config.experiment_name!r} from {args.config}"
        )

    # Resolve dataset and output dirs
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Auto-detect questions file under the corpus if the provided/default path is missing
    def _auto_discover_questions(default_path: Path, corpus: Path) -> Path:
        if default_path.exists():
            return default_path
        search_root = corpus if corpus.is_dir() else corpus.parent
        candidates: list[Path] = []
        for name in ("questions.jsonl", "questions.json"):
            p = search_root / name
            if p.exists():
                candidates.append(p)
        nested = search_root / "eval" / "questions.jsonl"
        if nested.exists():
            candidates.append(nested)
        candidates.extend(sorted([p for p in search_root.glob("*.jsonl") if p.is_file()]))
        candidates.extend(sorted([p for p in search_root.glob("**/*.jsonl") if p.is_file()]))
        fallback = Path("data/fakes/questions.jsonl")
        if fallback.exists():
            candidates.append(fallback)
        if not candidates:
            raise FileNotFoundError(
                f"Questions dataset not found. Tried {default_path} and common patterns under {search_root}"
            )
        return candidates[0]

    orig_questions = args.questions
    if not args.questions.exists():
        detected = _auto_discover_questions(orig_questions, args.corpus)
        if detected != orig_questions:
            print(f"[info] Using questions dataset: {detected}")
        args.questions = detected

    if args.rebuild:
        if args.sqlite_db.exists():
            try:
                args.sqlite_db.unlink()
            except Exception as e:
                print(f"[warn] Failed to remove SQLite DB {args.sqlite_db}: {e}")
        if args.faiss_dir.exists():
            for p in sorted(args.faiss_dir.glob("**/*"), reverse=True):
                try:
                    if p.is_file():
                        p.unlink()
                except Exception as e:
                    print(f"[warn] Failed to remove {p}: {e}")

        sqlite_store = SQLiteHyperlinkStore(SQLiteHyperlinkConfig(db_path=args.sqlite_db))
        faiss_store = None
        if args.embedding_model:
            faiss_store = FaissVectorStore(
                FaissVectorConfig(
                    index_path=args.faiss_dir,
                    embed_model_name=args.embedding_model,
                    overwrite=True,
                )
            )
        elif args.include_vector:
            raise ValueError(
                "--include-vector requires --embedding-model (or EMBEDDING_MODEL in env) when rebuilding."
            )

        pipeline = IngestionPipeline(
            toc_builder=None,
            chunker=ParagraphChunker(),
            sqlite_store=sqlite_store,
            vector_store=faiss_store,
            tenant_extractor=TenantExtractor(),
        )
        sections_written = 0
        chunks_written = 0

        if args.dataset:
            adapter_kwargs: Dict[str, object] = dict(getattr(args, "_adapter_kwargs", {}))
            if args.dataset == "miracl":
                if args.corpus.is_dir() and not args.miracl_language:
                    raise ValueError(
                        "--miracl-language is required when using MIRACL adapter with a directory corpus"
                    )
                if args.miracl_language:
                    adapter_kwargs["language"] = args.miracl_language
            adapter = create_adapter(args.dataset, args.corpus, **adapter_kwargs)
            ingestion_result = pipeline.ingest_sections(adapter.iter_section_roots())
            documents_processed = ingestion_result.documents_processed
            sections_written = ingestion_result.sections_written
            chunks_written = ingestion_result.chunks_written
        else:
            documents_processed = 0

            pdf_docs: List[Path] = []
            markdown_docs: List[Path] = []

            if args.corpus.is_file():
                if args.corpus.suffix.lower() == ".pdf":
                    pdf_docs = [args.corpus]
                else:
                    markdown_docs = [args.corpus]
            else:
                pdf_docs = sorted([p for p in args.corpus.glob("**/*.pdf") if p.is_file()])
                markdown_docs = sorted([p for p in args.corpus.glob(args.pattern) if p.is_file()])

            if pdf_docs:
                pdf_builder = PyMuPDFTOCBuilder(PyMuPDFTOCBuilderConfig())
                pdf_roots = [pdf_builder.build(path) for path in pdf_docs]
                pdf_result = pipeline.ingest_sections(pdf_roots)
                documents_processed += pdf_result.documents_processed
                sections_written += pdf_result.sections_written
                chunks_written += pdf_result.chunks_written

            if markdown_docs:
                pipeline.toc_builder = MarkdownTOCBuilder(MarkdownTOCBuilderConfig())
                md_result = pipeline.ingest(markdown_docs)
                documents_processed += md_result.documents_processed
                sections_written += md_result.sections_written
                chunks_written += md_result.chunks_written

            if not pdf_docs and not markdown_docs:
                raise FileNotFoundError(
                    f"No documents found in {args.corpus} matching {args.pattern} or containing PDFs"
                )

        print(
            "Rebuilt stores",
            {
                "documents": documents_processed,
                "sections": sections_written,
                "chunks": chunks_written,
                "sqlite_db": str(args.sqlite_db),
                "faiss_dir": str(args.faiss_dir) if faiss_store else None,
            },
        )

    retrievers: dict[str, object] = {}
    retriever_ids: list[str] = []

    selection_prompt = (
        "Return a JSON array of at most {limit} exact section paths most relevant to the query.\n"
        "Only include paths that appear in the TOC list. No commentary.\n\n"
        "Query: {query}\n\nTOC (path: title):\n{toc}"
    )
    hp_overrides = dict(getattr(args, "_hp_rag_overrides", {}))
    prompt = hp_overrides.pop("selection_prompt_template", selection_prompt)
    hp_config = HyperlinkRetrieverConfig(
        sqlite_config=SQLiteHyperlinkConfig(db_path=args.sqlite_db),
        selection_prompt_template=prompt,
        toc_limit=hp_overrides.pop("toc_limit", args.toc_limit),
        llm_model=hp_overrides.pop("llm_model", args.llm_model),
        max_sections=hp_overrides.pop("max_sections", args.max_sections),
        neighbor_window=hp_overrides.pop("neighbor_window", args.neighbor_window),
        **hp_overrides,
    )
    retrievers["hyperlink"] = HyperlinkRetriever(hp_config)
    retriever_ids.append("hyperlink")

    if args.include_vector:
        faiss_ok = args.faiss_dir.exists() and any(args.faiss_dir.iterdir())
        if not faiss_ok:
            print(f"[warn] Skipping vector baseline: FAISS dir missing or empty at {args.faiss_dir}")
        else:
            try:
                retrievers["vector"] = VectorRetriever(
                    VectorRetrieverConfig(
                        faiss_config=FaissVectorConfig(
                            index_path=args.faiss_dir,
                            embed_model_name=args.embedding_model,
                        ),
                        similarity_top_k=args.top_k,
                    )
                )
                retriever_ids.insert(0, "vector")
            except Exception as e:
                print(f"[warn] Skipping vector baseline: {e}")

    evaluation_config = EvaluationConfig(
        suite_name=args.suite,
        questions_path=str(args.questions),
        output_path=str(args.output),
        retriever_ids=retriever_ids,
        top_k=args.top_k,
    )

    runner_overrides: dict[str, object] = dict(getattr(args, "_query_runner_config", {}))
    if args.prompt_template_file:
        runner_overrides["prompt_template_path"] = args.prompt_template_file

    prompt_template_text: str | None = None
    prompt_template_path = runner_overrides.get("prompt_template_path")
    if prompt_template_path:
        prompt_template_text = Path(str(prompt_template_path)).read_text(encoding="utf-8")
    else:
        prompt_template = runner_overrides.get("prompt_template")
        if isinstance(prompt_template, str):
            prompt_template_text = prompt_template

    default_runner = QueryRunnerConfig()
    runner_cfg = QueryRunnerConfig(
        llm_model=str(runner_overrides.get("llm_model") or args.llm_model),
        temperature=float(runner_overrides.get("temperature", 0.0)),
        system_prompt=str(runner_overrides.get("system_prompt") or default_runner.system_prompt),
        prompt_template=prompt_template_text,
        default_top_k=args.top_k,
    )
    use_stub_llm = bool(runner_overrides.get("use_stub_llm", False))

    result = run_suite(
        evaluation_config,
        retrievers,  # type: ignore[arg-type]
        runner_config=runner_cfg,
        use_stub_llm=use_stub_llm,
    )

    def _collect_rows(metrics: dict[str, float], retrievers: list[str]):
        keys: list[str] = [
            "avg_contexts",
            "avg_tokens",
            "avg_latency_ms",
            "answer_token_f1",
            "answer_embedding_similarity",
            "answer_llm_correctness",
            "context_precision",
            "context_recall",
            "context_f1",
        ]
        rows: list[list[str]] = []
        for k in keys:
            row = [k]
            for rid in retrievers:
                v = metrics.get(f"{rid}/{k}")
                row.append(f"{v:.3f}" if isinstance(v, (int, float)) else "-")
            rows.append(row)
        return rows

    def render_markdown_table(metrics: dict[str, float], retrievers: list[str]) -> str:
        rows = _collect_rows(metrics, retrievers)
        header = ["Metric"] + [rid for rid in retrievers]
        rows = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
        for row in _collect_rows(metrics, retrievers):
            rows.append("| " + " | ".join(row) + " |")
        return "\n".join(rows)

    def render_delimited_table(metrics: dict[str, float], retrievers: list[str], sep: str) -> str:
        header = sep.join(["Metric"] + retrievers)
        body_lines = [sep.join(row) for row in _collect_rows(metrics, retrievers)]
        return "\n".join([header] + body_lines)

    def render_plain_table(metrics: dict[str, float], retrievers: list[str]) -> str:
        cols = ["Metric"] + retrievers
        data = _collect_rows(metrics, retrievers)
        widths = [max(len(cols[0]), max(len(r[0]) for r in data))]
        for i in range(1, len(cols)):
            widths.append(max(len(cols[i]), max(len(r[i]) for r in data)))

        def fmt_row(values: list[str]) -> str:
            return " | ".join(v.ljust(w) for v, w in zip(values, widths))

        lines = [fmt_row(cols), "-+-".join("-" * w for w in widths)]
        for row in data:
            lines.append(fmt_row(row))
        return "\n".join(lines)

    fmt = args.table_format or "md"
    if args.table_output and not args.table_format:
        ext = args.table_output.suffix.lower().lstrip(".")
        if ext in {"md", "markdown"}:
            fmt = "md"
        elif ext == "csv":
            fmt = "csv"
        elif ext in {"tsv", "tab"}:
            fmt = "tsv"
        else:
            fmt = "plain"

    print("\n=== Comparison ===")
    if fmt == "md":
        table_text = render_markdown_table(result.metrics, retriever_ids)
    elif fmt == "csv":
        table_text = render_delimited_table(result.metrics, retriever_ids, ",")
    elif fmt == "tsv":
        table_text = render_delimited_table(result.metrics, retriever_ids, "	")
    else:
        table_text = render_plain_table(result.metrics, retriever_ids)
    print(table_text)

    if args.table_output:
        args.table_output.parent.mkdir(parents=True, exist_ok=True)
        args.table_output.write_text(table_text, encoding="utf-8")
        print(f"\n[info] Wrote comparison table to {args.table_output}")
if __name__ == "__main__":
    main()
