from types import SimpleNamespace

from src.models.context import RetrievedContext
from src.orchestration.query_runner import GeneratedAnswer, QueryResult, QueryRunner
from src.interfaces import BaseRetriever


class DummyRetriever(BaseRetriever):
    def __init__(self) -> None:
        super().__init__("dummy")

    def _retrieve(self, query: str, *, top_k: int):
        context = RetrievedContext(
            path="doc/section",
            title="Section",
            text=f"answer for {query}",
            score=1.0,
        )
        return [context], {"source": "dummy"}

class DummyLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.snapshots: list[list[dict[str, object]]] = []

    def set_contexts(self, contexts):
        self.snapshots.append(list(contexts))

    def complete(self, prompt: str) -> SimpleNamespace:
        self.prompts.append(prompt)
        return SimpleNamespace(text="Final answer.")


def test_query_runner_retrieve_and_run_with_stub_llm():
    llm = DummyLLM()
    runner = QueryRunner(DummyRetriever(), llm=llm)

    output = runner.retrieve_context("question")

    assert output.retriever_id == "dummy"
    assert output.metadata["retrieved"] == 1
    assert output.contexts[0].text.startswith("answer for")

    generated = runner.generate_answer("question", output.contexts)
    assert isinstance(generated, GeneratedAnswer)
    assert generated.answer == "Final answer."
    assert llm.snapshots and llm.snapshots[-1][0]["text"].startswith("answer for")

    result = runner.run("question")
    assert isinstance(result, QueryResult)
    assert result.answer == "Final answer."
    assert "Question: question" in llm.prompts[0]
    assert "Section" in llm.prompts[0]
    assert result.retrieval.metadata["retrieved"] == 1
