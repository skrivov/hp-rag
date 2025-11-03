from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from types import SimpleNamespace

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, DisplayConfig
from deepeval.test_case import LLMTestCase

from src.eval.metrics import DEFAULT_METRICS, MetricDefinition, instantiate_metrics
from src.orchestration import QueryRunner, QueryRunnerConfig
from src.interfaces import BaseRetriever, RetrievalOutput


os.environ.setdefault("DEEPEVAL_TELEMETRY_OPTOUT", "1")


AnswerGenerator = Callable[[str, Mapping[str, object], RetrievalOutput], str]
QueryRunnerFactory = Callable[[str, BaseRetriever], QueryRunner]


class ContextualConcatLLM:
    """Offline-friendly LLM stub that concatenates retrieved contexts."""

    def __init__(self, joiner: str = "\n\n") -> None:
        self.joiner = joiner
        self._contexts: List[Mapping[str, Any]] = []

    def set_contexts(self, contexts: Sequence[Mapping[str, Any]]) -> None:
        self._contexts = list(contexts)

    def complete(self, prompt: str) -> SimpleNamespace:
        texts = [str(ctx.get("text", "")) for ctx in self._contexts if ctx.get("text")]
        return SimpleNamespace(text=self.joiner.join(texts).strip())


@dataclass(slots=True)
class EvaluationConfig:
    """Parameters controlling evaluation runs."""

    suite_name: str
    questions_path: str
    output_path: str
    retriever_ids: List[str]
    top_k: int = 5
    metrics: Sequence[MetricDefinition] | None = None


@dataclass(slots=True)
class EvaluationResult:
    """Aggregate evaluation outputs for one run."""

    suite_name: str
    metrics: Mapping[str, float]


def load_questions(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Question dataset not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    samples = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        samples.append(json.loads(line))
    return samples


def run_suite(
    config: EvaluationConfig,
    retrievers: Mapping[str, BaseRetriever],
    answer_generator: Optional[AnswerGenerator] = None,
    query_runner_factory: Optional[QueryRunnerFactory] = None,
    *,
    runner_config: QueryRunnerConfig | None = None,
    use_stub_llm: bool = True,
) -> EvaluationResult:
    """Execute evaluation suite using DeepEval metrics and retriever statistics."""

    dataset = load_questions(Path(config.questions_path))
    metric_definitions = list(config.metrics) if config.metrics is not None else list(DEFAULT_METRICS)
    aggregated: dict[str, float] = {}
    answer_fn = answer_generator or _default_answer
    query_runners: Dict[str, QueryRunner] = {}
    base_runner_config = runner_config or QueryRunnerConfig()

    def _build_runner_config() -> QueryRunnerConfig:
        return QueryRunnerConfig(
            llm_model=base_runner_config.llm_model,
            temperature=base_runner_config.temperature,
            system_prompt=base_runner_config.system_prompt,
            prompt_template=base_runner_config.prompt_template,
            default_top_k=config.top_k,
        )

    def resolve_runner(rid: str, retriever_obj: BaseRetriever) -> QueryRunner:
        if rid not in query_runners:
            if query_runner_factory is not None:
                query_runners[rid] = query_runner_factory(rid, retriever_obj)
            else:
                query_runners[rid] = QueryRunner(
                    retriever_obj,
                    _build_runner_config(),
                    llm=ContextualConcatLLM() if use_stub_llm else None,
                )
        return query_runners[rid]

    for retriever_id in config.retriever_ids:
        retriever = retrievers.get(retriever_id)
        if retriever is None:
            raise KeyError(f"Retriever '{retriever_id}' not provided")

        test_cases: List[LLMTestCase] = []
        context_counts: List[int] = []
        token_counts: List[int] = []
        latencies: List[float] = []

        for sample in dataset:
            query = (sample.get("query") or sample.get("question") or "").strip()
            if not query:
                continue

            retrieval_output = retriever.retrieve(query, top_k=config.top_k)
            contexts = retrieval_output.contexts
            context_counts.append(len(contexts))
            token_counts.append(sum(len(ctx.text.split()) for ctx in contexts))
            latencies.append(float(retrieval_output.metadata.get("duration_ms", 0.0)))

            contexts_text = [ctx.text for ctx in contexts]
            expected_answer = (
                sample.get("expected_answer")
                or sample.get("answer")
                or sample.get("ground_truth_answer")
                or ""
            )
            raw_refs = (
                sample.get("reference_contexts")
                or sample.get("ground_truth_contexts")
                or []
            )
            if isinstance(raw_refs, str):
                reference_contexts = [raw_refs]
            elif isinstance(raw_refs, Iterable):
                reference_contexts = list(raw_refs)
            else:
                reference_contexts = None
            actual_answer = sample.get("actual_answer")
            generated_prompt = ""
            if not isinstance(actual_answer, str) or not actual_answer.strip():
                if answer_generator is not None:
                    actual_answer = answer_fn(retriever_id, sample, retrieval_output)
                else:
                    runner = resolve_runner(retriever_id, retriever)
                    generated = runner.generate_answer(query, contexts)
                    actual_answer = generated.answer
                    generated_prompt = generated.prompt

            test_cases.append(
                LLMTestCase(
                    input=query,
                    actual_output=actual_answer,
                    expected_output=expected_answer,
                    context=reference_contexts,
                    retrieval_context=contexts_text,
                    additional_metadata={
                        "retriever_id": retriever_id,
                        "retrieval_metadata": retrieval_output.metadata,
                        "prompt": generated_prompt,
                    },
                )
            )

        aggregated[f"{retriever_id}/avg_contexts"] = _safe_mean(context_counts)
        aggregated[f"{retriever_id}/avg_tokens"] = _safe_mean(token_counts)
        aggregated[f"{retriever_id}/avg_latency_ms"] = _safe_mean(latencies)

        if not test_cases or not metric_definitions:
            continue

        deepeval_metrics = instantiate_metrics(metric_definitions)
        evaluation = evaluate(
            test_cases,
            metrics=deepeval_metrics,
            display_config=DisplayConfig(show_indicator=False, print_results=False, verbose_mode=False),
            async_config=AsyncConfig(run_async=False),
            cache_config=CacheConfig(write_cache=False, use_cache=False),
        )

        aggregated.update(_collect_metric_scores(retriever_id, evaluation.test_results))

    result = EvaluationResult(suite_name=config.suite_name, metrics=aggregated)
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(result), indent=2))
    return result


def _default_answer(
    retriever_id: str,
    sample: Mapping[str, object],
    retrieval_output: RetrievalOutput,
) -> str:
    """Combine retrieved contexts into a naive reference answer."""

    texts = [ctx.text for ctx in retrieval_output.contexts]
    return "\n\n".join(texts).strip()


def _safe_mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _slugify_metric(name: str) -> str:
    return "_".join(name.lower().split())


def _collect_metric_scores(
    retriever_id: str,
    test_results: Sequence[Any],
) -> Dict[str, float]:
    """Aggregate metric scores across DeepEval test results."""

    buckets: Dict[str, List[float]] = defaultdict(list)
    for result in test_results:
        for metric_data in getattr(result, "metrics_data", []):
            slug = _slugify_metric(metric_data.name)
            score = metric_data.score if metric_data.score is not None else 0.0
            buckets[slug].append(float(score))

    return {
        f"{retriever_id}/{metric}": _safe_mean(scores)
        for metric, scores in buckets.items()
    }


__all__ = ["EvaluationConfig", "EvaluationResult", "run_suite"]
