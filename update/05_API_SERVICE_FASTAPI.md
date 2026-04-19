# 05 API Service (FastAPI)

## Why
RHECS needs a stable integration surface for external systems and reproducible operations.

## API Objectives
- Health and readiness checks
- Single verify endpoint with structured request/response
- Deterministic error schema
- Request-level tracing

## Proposed Structure
- `rhecs_api/main.py`
- `rhecs_api/routes/health.py`
- `rhecs_api/routes/verify.py`
- `rhecs_api/schemas.py`
- `rhecs_api/middleware.py`

## Endpoint Contract

### GET /health
Response:
- status
- version
- timestamp

### GET /ready
Response:
- status
- model_provider_reachable
- vector_store_reachable

### POST /verify
Request:
- text: string
- tenant_id: string
- options: object (optional)

Response:
- request_id
- original_text
- restored_text
- metrics
- runtime_status
- audit_trail
- trajectory_path

## Error Contract
Always return:
- error.code
- error.message
- error.type
- error.retryable
- error.request_id

## Middleware Requirements
- Generate or propagate `X-Request-ID`
- Add timing header
- Log request start/end with status code

## Repository Changes
- Create `rhecs_api` package
- Add CLI start command in docs
- Add integration tests in `tests/api`

## Definition of Done
- API can process one successful verify request end-to-end.
- API returns structured errors for known failure classes.
- Health and readiness endpoints reflect dependency status.

## Validation
- curl smoke tests
- pytest integration tests
- load sanity test with small batch
