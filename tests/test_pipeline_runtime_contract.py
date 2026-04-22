from types import SimpleNamespace

from rhecs_core.pipeline import RHECSPipeline
from rhecs_core.runtime import RuntimeConfig, VerificationStrategy
from rhecs_core.verification.nli_judge import NLIStatus, VerificationResult


class _FakeClaim:
    def __init__(self, payload):
        self._payload = payload

    def dict(self):
        return self._payload


async def _fake_extract(_text):
    return SimpleNamespace(
        claims=[_FakeClaim({"subject": "A", "predicate": "is", "object": "B"})]
    )


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
    monkeypatch.setattr(
        "rhecs_core.pipeline.generate_verification_script", _fake_generate_script
    )
    monkeypatch.setattr(
        "rhecs_core.pipeline.execute_sandbox_code", _fake_execute_sandbox
    )
    monkeypatch.setattr("rhecs_core.pipeline.judge_evidence", _fake_judge)

    pipeline = RHECSPipeline(tenant_id="tenant-test")

    import asyncio

    result = asyncio.run(pipeline.process_document("sample text"))

    assert result["runtime_status"]["request_state"] == "FINALIZED"
    assert result["metrics"]["claims_extracted"] == 1
    assert result["runtime_errors"] == []
    assert result["runtime_status"]["verification_strategy_requested"] == "direct_llm"
    assert result["runtime_status"]["verification_strategy"] == "direct_llm"
    assert result["runtime_status"]["strategy_fallback_used"] is False
    assert result["runtime_status"]["strategy_fallback_reason"] is None
    assert result["runtime_status"]["request_events"] == []
    assert "trajectory_path" in result and result["trajectory_path"]


def test_pipeline_runtime_recursive_strategy_records_router_events(monkeypatch):
    monkeypatch.setattr("rhecs_core.pipeline.extract_vietnamese_claims", _fake_extract)
    monkeypatch.setattr(
        "rhecs_core.pipeline.generate_verification_script", _fake_generate_script
    )
    monkeypatch.setattr(
        "rhecs_core.pipeline.execute_sandbox_code", _fake_execute_sandbox
    )
    monkeypatch.setattr("rhecs_core.pipeline.judge_evidence", _fake_judge)

    pipeline = RHECSPipeline(
        tenant_id="tenant-test",
        verification_strategy=VerificationStrategy.RLM_RECURSIVE,
    )

    import asyncio

    result = asyncio.run(pipeline.process_document("sample text"))
    runtime_status = result["runtime_status"]

    assert runtime_status["verification_strategy_requested"] == "rlm_recursive"
    assert runtime_status["verification_strategy"] == "rlm_recursive"
    assert runtime_status["strategy_fallback_used"] is False
    assert runtime_status["strategy_fallback_reason"] is None
    assert len(runtime_status["request_events"]) == 2
    assert runtime_status["request_events"][0]["event_type"] == "RLM_SUBCALL_STARTED"
    assert runtime_status["request_events"][1]["event_type"] == "RLM_SUBCALL_FAILED"
    assert (
        runtime_status["request_events"][1]["payload"]["strategy_used"] == "direct_llm"
    )
    assert runtime_status["request_events"][1]["payload"]["degraded"] is True


def test_pipeline_runtime_strategy_fallback_when_unknown_strategy_requested(
    monkeypatch,
):
    monkeypatch.setattr("rhecs_core.pipeline.extract_vietnamese_claims", _fake_extract)
    monkeypatch.setattr(
        "rhecs_core.pipeline.generate_verification_script", _fake_generate_script
    )
    monkeypatch.setattr(
        "rhecs_core.pipeline.execute_sandbox_code", _fake_execute_sandbox
    )
    monkeypatch.setattr("rhecs_core.pipeline.judge_evidence", _fake_judge)

    pipeline = RHECSPipeline(
        tenant_id="tenant-test",
        verification_strategy="unknown_mode",
    )

    import asyncio

    result = asyncio.run(pipeline.process_document("sample text"))
    runtime_status = result["runtime_status"]

    assert runtime_status["verification_strategy_requested"] == "unknown_mode"
    assert runtime_status["verification_strategy"] == "direct_llm"
    assert runtime_status["strategy_fallback_used"] is True
    assert (
        runtime_status["strategy_fallback_reason"]
        == "unknown_requested_strategy:unknown_mode"
    )
    assert len(runtime_status["request_events"]) == 1
    event = runtime_status["request_events"][0]
    assert event["event_type"] == "RLM_SUBCALL_FAILED"
    assert event["payload"]["requested_strategy"] == "unknown_mode"
    assert event["payload"]["effective_strategy"] == "direct_llm"
    assert (
        event["payload"]["fallback_reason"] == "unknown_requested_strategy:unknown_mode"
    )


def test_pipeline_uses_runtime_config_default_strategy(monkeypatch):
    monkeypatch.setattr("rhecs_core.pipeline.extract_vietnamese_claims", _fake_extract)
    monkeypatch.setattr(
        "rhecs_core.pipeline.generate_verification_script", _fake_generate_script
    )
    monkeypatch.setattr(
        "rhecs_core.pipeline.execute_sandbox_code", _fake_execute_sandbox
    )
    monkeypatch.setattr("rhecs_core.pipeline.judge_evidence", _fake_judge)

    runtime_config = RuntimeConfig(
        default_strategy=VerificationStrategy.RLM_RECURSIVE,
        max_recursion=3,
        call_budget=12,
        token_budget=6000,
        timeout_budget=45000,
    )
    pipeline = RHECSPipeline(tenant_id="tenant-test", runtime_config=runtime_config)

    import asyncio

    result = asyncio.run(pipeline.process_document("sample text"))
    runtime_status = result["runtime_status"]

    assert runtime_status["verification_strategy_requested"] == "rlm_recursive"
    assert runtime_status["verification_strategy"] == "rlm_recursive"
    assert runtime_status["strategy_fallback_used"] is False
    assert runtime_status["strategy_fallback_reason"] is None
    assert runtime_status["runtime_config"] == {
        "default_strategy": "rlm_recursive",
        "max_recursion": 3,
        "call_budget": 12,
        "token_budget": 6000,
        "timeout_budget": 45000,
    }
