from src.ingest.tenant_extractor import TenantExtractor
from src.models.section import SectionNode


def test_tenant_extractor_identifies_parties():
    body = """**Buyer**: Apex Robotics, Inc., a Delaware corporation with offices at 200 Market Street ("Buyer").
**Seller**: Shenzhen Nova Electronics Co., Ltd., organized under PRC law ("Seller").

## Table of Contents
1. Definitions
"""
    root = SectionNode(
        document_id="doc",
        path="doc",
        title="Doc",
        body=body,
        level=0,
    )
    extractor = TenantExtractor()
    tenants = extractor.extract(root)
    assert len(tenants) == 2
    buyer = next(t for t in tenants if t.role == "Buyer")
    assert buyer.name.startswith("Apex Robotics")
    assert "Buyer" in buyer.aliases
