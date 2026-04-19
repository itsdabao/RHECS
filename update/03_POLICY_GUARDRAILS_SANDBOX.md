# 03 Policy Guardrails Before Sandbox Execution

## Why
RHECS executes model-generated Python code. Current subprocess isolation helps, but a pre-execution policy gate is required for production safety.

## Objectives
- Block unsafe scripts before execution
- Return deterministic policy errors
- Reduce useless retries on non-executable policy-violating code

## Guardrail Strategy
Use static AST checks + keyword checks before `subprocess.run`.

## Allowlist / Denylist Design

### Allowed Imports (initial set)
- json
- math
- re
- typing
- collections
- datetime
- qdrant_client (if required)
- rhecs_core.verification.sandbox_helpers

### Denied Capabilities
- socket
- requests
- urllib
- subprocess
- multiprocessing
- threading
- os.system
- eval / exec / compile
- file writes outside temp path

## Repository Changes

### Add New Module
- `rhecs_core/verification/policy_guard.py`
  - `validate_script(code: str) -> PolicyResult`
  - `PolicyViolation` model
  - checks:
    - forbidden imports
    - forbidden call patterns
    - max AST node count
    - max code length

### Modify Existing Module
- `rhecs_core/verification/sandbox_manager.py`
  - run policy check before file creation and subprocess execution
  - return structured error:
    - success: false
    - error_type: policy_error
    - error_code: POLICY_VIOLATION
    - violations: []

## Implementation Steps
1. Parse script into AST safely.
2. Walk nodes and collect violations.
3. Fail fast if any high-severity violation found.
4. Add policy version in response metadata.
5. Update retry logic to avoid retrying policy errors.

## Definition of Done
- Violating scripts are rejected before subprocess.
- Policy violations are traceable in trajectory logs.
- Retry loop does not repeat policy-violating scripts blindly.

## Validation
- Unit tests with safe and unsafe script fixtures.
- Regression tests for known bad patterns.

## Risks
- False positives blocking valid scripts.
- Mitigation: severity levels + monitored allowlist expansion.
