# RHECS Update Specs (Based on RLM)

## Purpose
This folder contains implementation-grade markdown specs for upgrading the RHECS project using proven patterns from the RLM paper and the `rlm` repository.

The docs are written for coding agents and engineers. Each doc includes:
- Why this change matters
- Exact repository areas to modify
- Step-by-step implementation plan
- Definition of Done (DoD)
- Risks and validation strategy

## Scope
Target repository: `d:/CS431`
Primary upgrade target: `rhecs_core`
Reference architecture: `d:/CS431/rlm`

## Document Index
1. `01_RUNTIME_CONTRACT.md`
2. `02_TRAJECTORY_LOGGING.md`
3. `03_POLICY_GUARDRAILS_SANDBOX.md`
4. `04_MODEL_STRATEGY_ROUTING.md`
5. `05_API_SERVICE_FASTAPI.md`
6. `06_EVAL_RUNNER.md`
7. `07_RETRIEVAL_PRODUCTION_QDRANT.md`
8. `08_TESTING_CI.md`
9. `09_14_DAY_EXECUTION_PLAN.md`
10. `10_AGENT_EDITING_CHECKLIST.md`

## Recommended Implementation Order
Use this order unless blocked:
1. Runtime contract
2. Trajectory logging
3. Policy guardrails
4. Role-based model strategy
5. API service
6. Eval runner
7. Retrieval production mode
8. Tests and CI hardening

## Global Constraints
- Keep existing pipeline behavior unless explicitly redesigned.
- Prefer additive changes before destructive refactors.
- Keep tenant isolation strict.
- Every new behavior must be traceable in structured logs.
- Every external call should have timeout + retry policy.

## Cross-Cutting Non-Functional Requirements
- Reliability: controlled retries, bounded failure loops
- Observability: request and claim level traces
- Reproducibility: deterministic artifacts for evaluation
- Safety: code execution policy checks before sandbox run
- Maintainability: typed schemas and test coverage for critical paths

## Exit Criteria (Project-Level)
RHECS is considered production-ready when all are true:
- API endpoint exists and is stable (`/health`, `/verify`)
- Runtime states and failures are fully traceable
- Sandbox code is policy-checked before execution
- Eval runner outputs reproducible metrics and artifacts
- Retrieval uses real embeddings + external Qdrant
- Test suite protects extraction, verification, restoration, and fallback behavior
