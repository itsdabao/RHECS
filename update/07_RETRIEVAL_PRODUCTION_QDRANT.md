# 07 Retrieval Production Mode (Qdrant + Embeddings)

## Why
Current in-memory and dummy-vector retrieval does not represent production behavior.

## Goals
- Use external Qdrant collection
- Use real semantic embeddings
- Keep tenant-level isolation mandatory

## Target Architecture
- Embedding adapter returns dense vector for claim/query
- Qdrant queried with vector + tenant filter
- Retrieval scores and ids included in evidence metadata

## Required Config
- QDRANT_HOST
- QDRANT_PORT
- QDRANT_COLLECTION
- EMBEDDING_PROVIDER
- EMBEDDING_MODEL
- EMBEDDING_API_KEY (if remote)

## Repository Changes

### Add Module
- `rhecs_core/retrieval/embedding_adapter.py`
- `rhecs_core/retrieval/qdrant_client_factory.py`

### Modify Module
- `rhecs_core/verification/sandbox_helpers.py`
  - replace dummy vector path with embedding call
  - keep compatibility with `search` and `query_points`

## Retrieval Metrics to Track
- hit@1
- hit@3
- hit@5
- avg evidence score
- empty retrieval rate

## Implementation Steps
1. Build embedding interface and provider implementation.
2. Wire Qdrant config from env.
3. Query with vector + tenant filter.
4. Return ranked evidence with score metadata.
5. Add retrieval metrics collection script.

## Definition of Done
- Verification stage uses real vector search.
- Tenant filter is always applied.
- hit@k metrics are reported on benchmark set.

## Validation
- Connectivity test to Qdrant
- Embedding generation test
- Retrieval quality smoke benchmark
