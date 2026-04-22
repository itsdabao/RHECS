import json
import os

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

# For Phase 3 local PoC, we bootstrap an in-memory client
client = QdrantClient(":memory:")


def setup_mock_qdrant(tenant_id: str):
    """Initializes dummy Qdrant data for testing."""
    if not client.collection_exists("truth_base"):
        client.create_collection(
            collection_name="truth_base",
            vectors_config=VectorParams(size=3, distance=Distance.COSINE),
        )

    # Mock evidence locked strictly to tenant_123
    client.upsert(
        collection_name="truth_base",
        points=[
            PointStruct(
                id=1,
                vector=[1.0, 0.0, 0.0],
                payload={
                    "tenant_id": "tenant_123",
                    "text": "Gustave Eiffel established his reputation by building the internal frame for the Statue of Liberty.",
                },
            ),
            PointStruct(
                id=2,
                vector=[0.0, 1.0, 0.0],
                payload={
                    "tenant_id": "tenant_123",
                    "text": "In 2023, OpenAI releases ChatGPT-4, and its safety protocol has been improved, says Sam Altman.",
                },
            ),
        ],
    )


def search_evidence(query: str, top_k: int = 1) -> list:
    """
    Searches Qdrant for evidence.
    Dual-Mode Search:
    - Mode A: Environment-Locked Querying if TENANT_ID is set.
    - Mode B: Global Knowledge Querying if TENANT_ID is absent.
    If the Root Planner attempts to pass a tenant_id in the query, we ignore it natively.
    """
    tenant_id = os.environ.get("TENANT_ID")
    tenant_filter = None

    if not client.collection_exists("truth_base"):
        setup_mock_qdrant(tenant_id or "tenant_123")

    if tenant_id:
        # Mode A (Enterprise Isolation): Strongly lock queries to Tenant bounds
        tenant_filter = Filter(
            must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
        )
    else:
        # Mode B (Global Knowledge): Broad search when tenant_id is omitted
        tenant_filter = None

    # Dummy embedding vector (In prod, we use SentenceTransformers or API embeddings)
    dummy_vector = [1.0, 0.0, 0.0]

    if hasattr(client, "search"):
        results = client.search(
            collection_name="truth_base",
            query_vector=dummy_vector,
            query_filter=tenant_filter,
            limit=top_k,
        )
    else:
        query_response = client.query_points(
            collection_name="truth_base",
            query=dummy_vector,
            query_filter=tenant_filter,
            limit=top_k,
        )
        results = getattr(query_response, "points", query_response)

    output = []
    for hit in results:
        payload = hit.payload if hasattr(hit, "payload") else hit.get("payload", {})
        text = payload.get("text")
        if text:
            output.append(text)

    return output
