import asyncio

from rhecs_core.query_strategy import (
    DirectLLMAdapter,
    QueryRequest,
    QueryRouter,
    QueryStrategy,
    RLMBridge,
)
from rhecs_core.query_strategy.contracts import RLMRecursiveRequest


def test_rlm_bridge_parses_final_payload_and_trajectory():
    def _mock_rlm_query(payload, _model=None):
        assert "claim" in payload
        return {
            "FINAL": {
                "status": "CONTRADICTED",
                "reasoning": "Claim contradicts retrieved evidence.",
                "evidence": ["Nguon A", "Nguon B"],
                "confidence": 0.91,
            }
        }

    bridge = RLMBridge(rlm_query_fn=_mock_rlm_query)
    req = RLMRecursiveRequest(
        original_sentence="A la B",
        claim={"entity": "A", "relationship": "la", "target": "B"},
    )

    response = bridge.query_recursive(req)

    assert response.verdict == "CONTRADICTED"
    assert response.reasoning == "Claim contradicts retrieved evidence."
    assert response.evidence == ["Nguon A", "Nguon B"]
    assert response.confidence == 0.91
    assert response.subcall_count == 1
    assert len(response.trajectory) >= 3
    assert any(
        evt.action == "rlm_query" and evt.output_summary == "finished"
        for evt in response.trajectory
    )


def test_rlm_bridge_autonomous_fallback_to_llm_query():
    def _mock_llm_query(payload, _model=None):
        assert payload.get("mode") == "autonomous_llm_fallback"
        return {
            "status": "NOT_MENTIONED",
            "reasoning": "Insufficient evidence from retrieval.",
            "evidence": payload.get("evidence", []),
        }

    bridge = RLMBridge(
        llm_query_fn=_mock_llm_query,
        search_evidence_fn=lambda _query: ["Evidence snippet"],
    )
    req = RLMRecursiveRequest(
        original_sentence="A la B",
        claim={"entity": "A", "relationship": "la", "target": "B"},
    )

    response = bridge.query_recursive(req)

    assert response.verdict == "NOT_MENTIONED"
    assert response.evidence == ["Evidence snippet"]
    assert any(evt.action == "search_evidence" for evt in response.trajectory)
    assert any(
        evt.action == "llm_query" and evt.output_summary == "finished"
        for evt in response.trajectory
    )


async def _fake_judge(_original_sentence, _claim, _evidence):
    return {
        "status": "SUPPORTED",
        "reasoning": "Fallback direct judge succeeded.",
        "error_category": None,
        "fault_span": None,
    }


def test_query_router_degrades_to_direct_when_recursive_fails():
    def _broken_rlm_query(_payload, _model=None):
        raise RuntimeError("mock_recursive_failure")

    router = QueryRouter(
        direct_adapter=DirectLLMAdapter(judge_fn=_fake_judge).execute,
        rlm_bridge=RLMBridge(rlm_query_fn=_broken_rlm_query),
    )

    request = QueryRequest(
        original_sentence="A la B",
        claim={"entity": "A", "relationship": "la", "target": "B"},
        strategy=QueryStrategy.RLM_RECURSIVE,
        context={"evidence": ["A la B"]},
    )

    response = asyncio.run(router.route_async(request))

    assert response.strategy_used == QueryStrategy.DIRECT_LLM
    assert response.verdict == "SUPPORTED"
    assert response.degraded is True
    assert response.fallback_count >= 1
    assert response.error is not None and "rlm_recursive_failed" in response.error
    assert response.raw is not None
    assert response.raw.get("strategy_requested") == QueryStrategy.RLM_RECURSIVE.value
    assert response.raw.get("strategy_used") == QueryStrategy.DIRECT_LLM.value
