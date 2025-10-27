from src.models.context import RetrievedContext
from src.models.section import SectionNode


def test_section_node_add_child_tracks_parent_and_order():
    root = SectionNode(
        document_id="doc",
        path="doc",
        title="Doc",
        body="root",
        level=0,
    )
    child_a = SectionNode(
        document_id="doc",
        path="doc/a",
        title="A",
        body="child",
        level=1,
    )
    child_b = SectionNode(
        document_id="doc",
        path="doc/b",
        title="B",
        body="child",
        level=1,
    )

    root.add_child(child_a)
    root.add_child(child_b)

    assert child_a.parent_path == "doc"
    assert child_b.parent_path == "doc"
    assert child_a.order == 0
    assert child_b.order == 1
    assert root.children == [child_a, child_b]


def test_retrieved_context_defaults_metadata_none():
    context = RetrievedContext(
        path="doc/section",
        title="Section",
        text="content",
        score=0.8,
    )

    assert context.metadata is None
