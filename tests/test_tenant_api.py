import importlib
from pathlib import Path

import src.server.settings as server_settings
from src.hp_rag.storage import SQLiteHyperlinkConfig, SQLiteHyperlinkStore
from src.models.section import SectionNode
from src.models.tenant import TenantRecord


def _prepare_db(path):
    store = SQLiteHyperlinkStore(SQLiteHyperlinkConfig(db_path=path))
    store.initialize()
    root = SectionNode(
        document_id="doc-1",
        path="doc-1",
        title="Doc",
        body="Root",
        level=0,
    )
    store.upsert_sections([root])
    store.register_document_tenants(
        "doc-1",
        [TenantRecord(tenant_id="apex-robotics", name="Apex Robotics, Inc.", role="Buyer", aliases=["Buyer"])],
    )
    assert store.list_tenants(), "tenant registration failed"
    store.close()


def test_tenant_router_functions(tmp_path):
    db_path = tmp_path / "hyperlink.db"
    _prepare_db(db_path)

    tenants_router_module = importlib.import_module("src.server.tenants.router")

    store = SQLiteHyperlinkStore(SQLiteHyperlinkConfig(db_path=db_path))
    store.initialize()

    response = tenants_router_module.list_tenants(q=None, limit=50, offset=0, store=store)
    assert response.items and response.items[0].tenant_id == "apex-robotics"

    detail = tenants_router_module.get_tenant("apex-robotics", store=store)
    assert detail.tenant_id == "apex-robotics"
    assert detail.documents == ["doc-1"]

    store.close()
