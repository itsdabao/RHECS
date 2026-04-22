# 09 Fourteen-Day Execution Plan

## Objective
Deliver production-oriented RHECS baseline in 14 days with measurable outcomes.

## Week 1: Core Runtime Reliability

### Day 1
- Implement runtime state enums and transition map
- Add request_id, trace_id generation at pipeline entry
Output:
- runtime contract module scaffold

### Day 2
- Add claim-level lifecycle transitions
- Add normalized error taxonomy mapping
Output:
- runtime status in pipeline response

### Day 3
- Implement trajectory logger with JSONL output
- Wire start/success/failure events in pipeline
Output:
- one trajectory file per request

### Day 4
- Add policy guard module with AST checks
- Integrate guard in sandbox manager pre-execution
Output:
- policy violation structured responses

### Day 5
- Add role-based model config and stage-aware routing
- Add fallback telemetry events
Output:
- per-stage model selection and fallback traces

### Day 6
- Create FastAPI skeleton (`/health`, `/ready`, `/verify`)
- Add request middleware for request_id and timing
Output:
- basic API service running locally

### Day 7
- Stabilization day: bug fixes, docs update, smoke tests
Output:
- week 1 checkpoint report

## Week 2: Quality Measurement and Hardening

### Day 8
- Build eval runner scaffold and data loaders
- Implement result persistence and resume mechanism
Output:
- `eval_results.jsonl` generated on sample subset

### Day 9
- Implement metrics computations and summary outputs
- Add confusion matrix export
Output:
- `eval_summary.json` and `eval_confusion_matrix.json`

### Day 10
- Implement embedding adapter and Qdrant config factory
- Integrate retrieval path in verification helpers
Output:
- production retrieval mode enabled

### Day 11
- Add retrieval quality script and hit@k reporting
- Validate tenant filter enforcement
Output:
- retrieval benchmark report

### Day 12
- Add unit tests for runtime, router, policy guard, sandbox
Output:
- baseline unit test suite

### Day 13
- Add integration tests for API and pipeline
- Add CI workflow for lint + tests
Output:
- first CI pipeline passing

### Day 14
- Final hardening, run full eval, publish upgrade report
Output:
- go/no-go readiness summary

## Exit Deliverables
- API service with structured contracts
- Runtime trajectory logs for every request
- Policy guardrails enforced before sandbox
- Eval artifacts and reproducible metrics
- Retrieval production mode and hit@k report
- Test suite + CI checks
