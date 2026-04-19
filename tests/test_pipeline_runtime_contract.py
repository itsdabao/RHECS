from types import SimpleNamespace

from rhecs_core.pipeline import RHECSPipeline
from rhecs_core.verification.nli_judge import NLIStatus, VerificationResult


class _FakeClaim:
    def __init__(self, payload):
        self._payload = payload

    def dict(self):
        return self._payload


async def _fake_extract(_text):
    return SimpleNamespace(claims=[_FakeClaim({"subject": "A", "predicate": "is", "object": "B"})])


def _fake_generate_script(_claim, _error_trace=None):
    return "import json\nprint(json.dumps({'evidence': ['A is B']}))"


def _fake_execute_sandbox(_code, _tenant_id):
    return {"success": True, "output": {"evidence": ["A is B"]}}


async def _fake_judge(_original, _triplet, _evidence):
    return VerificationResult(
        status=NLIStatus.SUPPORTED,
        reasoning="Evidence supports claim.",
        error_category=None,
        fault_span=None,
    )


def test_pipeline_runtime_contract_success(monkeypatch):
    monkeypatch.setattr("rhecs_core.pipeline.extract_vietnamese_claims", _fake_extract)
    monkeypatch.setattr("rhecs_core.pipeline.generate_verification_script", _fake_generate_script)
    monkeypatch.setattr("rhecs_core.pipeline.execute_sandbox_code", _fake_execute_sandbox)
    monkeypatch.setattr("rhecs_core.pipeline.judge_evidence", _fake_judge)

    pipeline = RHECSPipeline(tenant_id="tenant-test")

    import asyncio

    result = asyncio.run(pipeline.process_document("sample text"))

    assert result["runtime_status"]["request_state"] == "FINALIZED"
    assert result["metrics"]["claims_extracted"] == 1
    assert result["runtime_errors"] == []
    assert "trajectory_path" in result and result["trajectory_path"]
