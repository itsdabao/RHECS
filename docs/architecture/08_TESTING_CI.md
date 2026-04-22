# 08 Testing and CI Hardening

## Why
RHECS now has enough moving parts that regressions are likely without systematic tests.

## Test Pyramid

### Unit Tests
Focus areas:
- extractor response parsing
- model router retry and fallback ordering
- policy guard AST checks
- sandbox JSON output parsing
- replacer edge cases

### Integration Tests
Focus areas:
- end-to-end pipeline with mocked model calls
- API `/verify` happy path and failure path
- retrieval integration with test Qdrant fixture

### Regression Tests
Focus areas:
- known failure signatures from past runs
- retry storms and degrade behavior
- unicode replacement edge cases

## Suggested Test Structure
- `tests/unit/...`
- `tests/integration/...`
- `tests/regression/...`

## CI Pipeline (GitHub Actions)
Stages:
1. lint
2. unit tests
3. integration tests (with service containers if needed)
4. artifact upload (coverage + test report)

## Repository Changes
- Add test config (`pytest.ini` or pyproject section)
- Add CI workflow file in `.github/workflows/`
- Add minimal fixtures and fake model provider

## Coverage Target
Initial critical coverage target:
- 70 percent on `rhecs_core/pipeline.py`
- 80 percent on `rhecs_core/llm/model_router.py`
- 80 percent on `rhecs_core/verification/sandbox_manager.py` and policy checks

## Definition of Done
- PR checks fail on regressions.
- Critical modules have baseline coverage.
- Test reports are available as CI artifacts.

## Validation
- Introduce one intentional failing test and confirm CI blocks merge.
