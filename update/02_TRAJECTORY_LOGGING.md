# 02 Trajectory Logging and Telemetry

## Why
The biggest upgrade from PoC to production-agent quality is traceability. RLM provides trajectory metadata per iteration; RHECS should provide similar request and claim traces.

## Goals
- Reconstruct full request history from logs only
- Diagnose where and why a claim failed
- Quantify latency, retries, and model usage by stage

## Log Structure (JSONL)
One event per line. Required fields:
- timestamp
- request_id
- trace_id
- claim_id (nullable at request-level events)
- stage (`extract`, `plan`, `sandbox`, `judge`, `restore`, `replace`, `finalize`)
- event_type (`start`, `success`, `failure`, `retry`, `degraded`)
- model_name
- provider
- latency_ms
- retry_count
- error_type
- error_message
- metadata (object)

## Output Locations
- `artifacts/trajectory/YYYY-MM-DD/<request_id>.jsonl`
- Optional compact summary:
  - `artifacts/trajectory/YYYY-MM-DD/<request_id>.summary.json`

## Repository Changes

### Add New Module
- `rhecs_core/logger/trajectory.py`
  - `TrajectoryLogger`
  - `log_event(...)`
  - `flush_summary(...)`

### Modify Existing Modules
- `rhecs_core/pipeline.py`
- `rhecs_core/verification/root_planner.py`
- `rhecs_core/verification/sandbox_manager.py`
- `rhecs_core/verification/nli_judge.py`
- `rhecs_core/restoration/rewriter.py`

## Implementation Steps
1. Implement logger with buffered writes and fail-safe flush.
2. Inject logger instance at pipeline entry.
3. Log start/success/failure around each stage.
4. Emit retry events from fallback router and sandbox retry loops.
5. Emit final summary event with totals:
   - total_claims
   - total_failures
   - total_retries
   - total_latency_ms

## Summary Metrics to Compute
- Request latency: p50, p95, p99
- Claim-level failure rates by stage
- Provider/model failure distribution
- Retry amplification factor

## Definition of Done
- One trajectory file per request exists.
- Any failure can be traced to exact stage and model.
- Summary report can be generated from trajectory files only.

## Risks
- Log volume explosion.
- Mitigation: keep payload fields minimal and avoid full text dumps.
