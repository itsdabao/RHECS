# 04 Role-Based Model Strategy and Routing

## Why
A single model for all stages is expensive and unstable. RLM patterns suggest role-aware model usage and controlled fallback.

## Goals
- Assign best-fit models per stage
- Keep fallback chains explicit
- Track usage and failures per role

## Stage Roles
- extractor_model
- planner_model
- judge_model
- rewriter_model

## Suggested Config Contract
Use env vars or config object:
- `RHECS_MODEL_EXTRACTOR`
- `RHECS_MODEL_PLANNER`
- `RHECS_MODEL_JUDGE`
- `RHECS_MODEL_REWRITER`
- `RHECS_FALLBACK_EXTRACTOR`
- `RHECS_FALLBACK_PLANNER`
- `RHECS_FALLBACK_JUDGE`
- `RHECS_FALLBACK_REWRITER`

Fallback format example:
- comma-separated model IDs

## Repository Changes

### Modify
- `rhecs_core/llm/model_router.py`
  - add role-aware candidate resolution
  - add stage-level budget and retry policy hooks

- `rhecs_core/extraction/extractor.py`
- `rhecs_core/verification/root_planner.py`
- `rhecs_core/verification/nli_judge.py`
- `rhecs_core/restoration/rewriter.py`
  - pass role to router

## Routing Policy
1. Resolve primary model by role.
2. Resolve role fallback chain.
3. Retry on retryable provider failures only.
4. Emit degrade event when role chain exhausted.
5. Never retry policy_error.

## Budget Policy
Per request stage caps:
- max_calls_extract
- max_calls_verify
- max_calls_restore
- max_total_calls

## Definition of Done
- Each stage can use different models without code changes.
- Fallback events are visible in trajectory logs.
- Cost and failure stats can be grouped by role.

## Validation
- Unit tests for candidate resolution order
- Simulated 429/503/404 sequences per role
- Integration test with mixed model chain
