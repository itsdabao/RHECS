# 01 Runtime Contract

## Why
RHECS currently runs end-to-end, but runtime lifecycle is implicit. This makes failures hard to classify and recover.

RLM strength to reuse:
- Explicit iteration lifecycle
- Structured stop conditions
- Strong execution boundaries

## Current Gaps in RHECS
- No explicit request state machine
- No claim-level lifecycle states
- No normalized error taxonomy across modules
- No consistent degrade policy on repeated failures

## Target Contract

### Request Lifecycle
Define enum-like states:
- RECEIVED
- CLAIMS_EXTRACTED
- VERIFICATION_IN_PROGRESS
- VERIFICATION_DONE
- RESTORATION_IN_PROGRESS
- RESTORATION_DONE
- FINALIZED
- DEGRADED
- FAILED

### Claim Lifecycle
For each claim:
- CLAIM_CREATED
- PLAN_GENERATED
- SANDBOX_EXECUTED
- VERDICT_ASSIGNED
- PATCH_GENERATED
- PATCH_APPLIED
- CLAIM_FAILED

### Error Taxonomy
Every failure must map to one of:
- provider_error
- policy_error
- execution_error
- data_error
- timeout_error
- unknown_error

## Repository Changes

### Add New Module
- `rhecs_core/runtime/contracts.py`
  - dataclasses or pydantic models:
    - `RequestRuntimeState`
    - `ClaimRuntimeState`
    - `RuntimeErrorInfo`
    - `RuntimeTransition`

- `rhecs_core/runtime/state_machine.py`
  - transition guards
  - invalid transition handling

### Modify Existing Module
- `rhecs_core/pipeline.py`
  - create `request_id` at entry
  - initialize runtime state
  - enforce legal transitions at each stage
  - mark DEGRADED on controlled fallback exhaustion
  - return final runtime status in output payload

## Implementation Steps
1. Define state enums and transition map.
2. Add helper `transition(current, event) -> next_state`.
3. Instrument `process_document` with explicit transitions.
4. Instrument `_verify_claim_worker` and `_restore_claim_worker` with claim-level transitions.
5. Map all exceptions to `RuntimeErrorInfo` with taxonomy.
6. Add top-level output section: `runtime_status` and `runtime_errors`.

## Definition of Done
- Every request has explicit final state.
- Every claim has auditable stage transitions.
- Invalid transition throws deterministic error.
- Pipeline output includes request-level runtime summary.

## Validation
- Unit tests for transition map
- Integration test for success path and degraded path
- Synthetic fail cases for each error taxonomy

## Risks
- Risk: over-coupling runtime contract with business logic.
- Mitigation: keep contract layer isolated in `rhecs_core/runtime`.
