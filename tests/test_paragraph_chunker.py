from src.ingest.simple_chunker import ParagraphChunker
from src.models.section import SectionNode


def test_paragraph_chunker_splits_on_blank_lines():
    section = SectionNode(
        document_id="doc",
        path="doc/section",
        title="Section",
        body="First paragraph.\n\nSecond paragraph.\nStill second.\n\nThird.",
        level=1,
    )

    chunker = ParagraphChunker()
    chunks = list(chunker.chunk(section))

    assert [chunk.order for chunk in chunks] == [0, 1, 2]
    assert chunks[0].text == "First paragraph."
    assert chunks[1].text == "Second paragraph.\nStill second."
    assert chunks[2].parent_title == "Section"
