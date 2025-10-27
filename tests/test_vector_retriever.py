from types import SimpleNamespace

from src.rag.retriever import VectorRetriever, VectorRetrieverConfig
from src.rag.storage import FaissVectorConfig, FaissVectorStore


def test_vector_retriever_maps_hits_to_contexts(monkeypatch, tmp_path):
    config = VectorRetrieverConfig(
        faiss_config=FaissVectorConfig(index_path=tmp_path / "faiss"),
        similarity_top_k=2,
    )
    retriever = VectorRetriever(config)

    class FakeHit:
        def __init__(self, text, metadata, score):
            node = SimpleNamespace(
                metadata=metadata,
                get_content=lambda metadata_mode="all": text,
            )
            self.node = node
            self.score = score

    hits = [
        FakeHit(
            text="Chunk text",
            metadata={"path": "doc/section", "parent_title": "Section"},
            score=0.42,
        )
    ]

    def fake_as_retriever(self, similarity_top_k: int):
        class StubRetriever:
            def retrieve(self_inner, query: str):
                return hits

        return StubRetriever()

    monkeypatch.setattr(FaissVectorStore, "as_retriever", fake_as_retriever)

    output = retriever.retrieve("query", top_k=1)

    assert output.metadata["raw_hits"] == 1
    assert output.contexts[0].path == "doc/section"
    assert output.contexts[0].title == "Section"
    assert output.contexts[0].text == "Chunk text"
    assert output.contexts[0].score == 0.42
