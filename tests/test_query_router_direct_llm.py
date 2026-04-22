import asyncio

from rhecs_core.query_strategy import (
    DirectLLMAdapter,
    QueryRequest,
    QueryRouter,
    QueryStrategy,
)


class _FakeVerdict:
    def __init__(self):
        self.status = "SUPPORTED"
        self.reasoning = "Evidence supports claim."
        self.error_category = None
        self.fault_span = None

    def model_dump(self):
        return {
            "status": self.status,
            "reasoning": self.reasoning,
            "error_category": self.error_category,
            "fault_span": self.fault_span,
        }


async def _fake_judge(_original_sentence, _claim, _evidence):
    return _FakeVerdict()


def test_direct_llm_adapter_maps_judge_output_and_telemetry():
    adapter = DirectLLMAdapter(judge_fn=_fake_judge)
    router = QueryRouter(direct_adapter=adapter.execute)

    request = QueryRequest(
        original_sentence="A là B",
        claim={"entity": "A", "relationship": "là", "target": "B"},
        strategy=QueryStrategy.DIRECT_LLM,
        context={"evidence": ["A là B trong tài liệu gốc."]},
        metadata={"model_used": "gemini-2.5-flash", "fallback_count": 1},
    )

    response = asyncio.run(router.route_async(request))

    assert response.strategy_used == QueryStrategy.DIRECT_LLM
    assert response.verdict == "SUPPORTED"
    assert response.reasoning == "Evidence supports claim."
    assert response.evidence == ["A là B trong tài liệu gốc."]
    assert response.model_used == "gemini-2.5-flash"
    assert response.fallback_count == 1


def test_route_sync_raises_inside_event_loop():
    adapter = DirectLLMAdapter(judge_fn=_fake_judge)
    router = QueryRouter(direct_adapter=adapter.execute)

    request = QueryRequest(
        original_sentence="A là B",
        claim={"entity": "A", "relationship": "là", "target": "B"},
        strategy=QueryStrategy.DIRECT_LLM,
    )

    async def _call_sync_route():
        try:
            router.route(request)
        except RuntimeError as exc:
            assert "route_async" in str(exc)
            return
        raise AssertionError(
            "Expected RuntimeError when calling route() inside event loop"
        )

    asyncio.run(_call_sync_route())
