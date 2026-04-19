# 10 Agent Editing Checklist

## Goal
Provide a strict execution checklist so coding agents can modify RHECS safely and consistently.

## Before Editing
1. Read the relevant spec in `update/` first.
2. Confirm target files and expected outputs.
3. Confirm no overlapping risky edits in unrelated modules.
4. Define rollback-safe plan (additive first, refactor later).

## During Editing
1. Keep changes scoped to one concern per commit-sized unit.
2. Preserve existing public payload fields unless versioned.
3. Add structured logging when adding new behavior.
4. Add explicit error taxonomy mapping for new exceptions.
5. Avoid hidden side effects in global state.

## Runtime Contract Checks
- Every stage transition is valid.
- Failures end in FAILED or DEGRADED explicitly.
- Request output contains runtime status.

## Telemetry Checks
- Every stage emits start and end/failure events.
- Every retry emits retry event with reason.
- Every event includes request_id and stage.

## Sandbox Guardrail Checks
- Policy check runs before subprocess execution.
- Policy violations are non-retryable by default.
- Forbidden imports/calls are blocked deterministically.

## Model Routing Checks
- Stage role resolves primary model correctly.
- Fallback order is deterministic.
- Retry only on retryable provider errors.

## API Checks
- `/health` and `/verify` return contract-compliant payloads.
- Error responses include request_id and retryable flag.

## Evaluation Checks
- Eval run can resume after interruption.
- Metrics are exported in expected files.
- Failure samples include trace references.

## Test and Validation Checks
1. Run unit tests for modified modules.
2. Run at least one integration smoke test.
3. Validate generated artifacts paths and schemas.
4. Verify no syntax errors in changed files.

## Done Criteria for an Agent Task
- Code changes match one spec section.
- Logs and error taxonomy are updated.
- Tests for changed behavior exist or are updated.
- Results are reproducible with documented command.

## Suggested Command Set (Windows)
- Install editable package: `D:/miniconda3/envs/rhecs/python.exe -m pip install -e .`
- Run one script: `D:/miniconda3/envs/rhecs/python.exe path/to/script.py`
- Syntax check file: `D:/miniconda3/envs/rhecs/python.exe -m py_compile path/to/file.py`
- Run pytest (if configured): `D:/miniconda3/envs/rhecs/python.exe -m pytest -q`
