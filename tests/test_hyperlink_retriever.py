from pathlib import Path
from types import SimpleNamespace

from src.models.section import SectionNode
from src.hp_rag.retriever import HyperlinkRetriever, HyperlinkRetrieverConfig
from src.hp_rag.storage import SQLiteHyperlinkConfig, SQLiteHyperlinkStore


def _build_sections() -> SectionNode:
    root = SectionNode(
        document_id="doc",
        path="doc",
        title="Doc",
        body="Root",
        level=0,
    )
    intro = SectionNode(
        document_id="doc",
        path="doc/intro",
        title="Intro",
        body="Intro body",
        level=1,
    )
    details = SectionNode(
        document_id="doc",
        path="doc/details",
        title="Details",
        body="Detailed wizard info",
        level=1,
    )
    root.add_child(intro)
    root.add_child(details)
    return root


class StubLLM:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def complete(self, prompt: str) -> SimpleNamespace:
        self.calls.append(prompt)
        return SimpleNamespace(text='["doc/details"]')


def test_hyperlink_retriever_uses_llm(tmp_path):
    db_path = tmp_path / "hyperlink.db"
    seed_store = SQLiteHyperlinkStore(SQLiteHyperlinkConfig(db_path=Path(db_path)))
    seed_store.initialize()
    seed_store.upsert_sections([_build_sections()])
    seed_store.close()

    stub_llm = StubLLM()
    config = HyperlinkRetrieverConfig(
        sqlite_config=SQLiteHyperlinkConfig(db_path=Path(db_path)),
        max_sections=5,
        neighbor_window=1,
        llm=stub_llm,
    )

    retriever = HyperlinkRetriever(config)
    output = retriever.retrieve("wizard", top_k=3)

    assert output.metadata["retrieved"] == 1
    paths = [context.path for context in output.contexts]
    assert paths == ["doc/details"]
    assert output.contexts[0].metadata["retrieval_stage"] == "toc_filter"
    assert output.metadata["toc_candidates"] == 1
    assert stub_llm.calls, "LLM selector should be invoked"
