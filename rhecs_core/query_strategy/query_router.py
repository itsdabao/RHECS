import asyncio
from typing import Any, Awaitable, Callable, Optional

from rhecs_core.query_strategy.contracts import (
    DirectLLMRequest,
    DirectLLMResponse,
    QueryRequest,
    QueryResponse,
    QueryStrategy,
    RLMRecursiveRequest,
)
from rhecs_core.query_strategy.rlm_bridge import RLMBridge
from rhecs_core.verification.nli_judge import judge_evidence


class DirectLLMAdapter:
    """
    Default direct_llm adapter that preserves current judge behavior.

    WS1-02 scope:
    - map QueryStrategy direct request to existing judge call
    - normalize response contract for query strategy layer
    - attach lightweight telemetry fields (model_used, fallback_count)
    """

    def __init__(
        self,
        judge_fn: Optional[
            Callable[[str, dict[str, Any], list[str]], Awaitable[Any]]
        ] = None,
    ) -> None:
        self._judge_fn = judge_fn or judge_evidence

    async def execute(self, request: DirectLLMRequest) -> DirectLLMResponse:
        evidence_list = _extract_evidence_list(request.context)
        verdict = await self._judge_fn(
            request.original_sentence, request.claim, evidence_list
        )

        verdict_payload = _model_to_dict(verdict)
        status = verdict_payload.get("status")
        reasoning = str(verdict_payload.get("reasoning", ""))

        telemetry_model = _read_first_nonempty_str(
            request.metadata,
            keys=("model_used", "judge_model", "model"),
        )
        fallback_count = _read_int(request.metadata.get("fallback_count"), default=0)
        if fallback_count == 0:
            fallback_count = _read_int(
                request.metadata.get("judge_fallback_count"), default=0
            )

        raw_payload = dict(verdict_payload)
        raw_payload["evidence_count"] = len(evidence_list)

        return DirectLLMResponse(
            verdict=str(status),
            reasoning=reasoning,
            evidence=evidence_list,
            confidence=_read_float(verdict_payload.get("confidence")),
            model_used=telemetry_model,
            fallback_count=fallback_count,
            raw=raw_payload,
        )


class QueryRouter:
    """
    Strategy router for verification queries.

    WS1-01 scope provides contracts and routing skeleton.
    WS1-02 and WS1-03 will plug concrete direct and recursive adapters.
    """

    def __init__(
        self,
        direct_adapter: Optional[
            Callable[[DirectLLMRequest], Awaitable[DirectLLMResponse]]
        ] = None,
        rlm_bridge: Optional[RLMBridge] = None,
    ) -> None:
        self.direct_adapter = direct_adapter or DirectLLMAdapter().execute
        self.rlm_bridge = rlm_bridge

    async def route_async(self, request: QueryRequest) -> QueryResponse:
        if request.strategy == QueryStrategy.DIRECT_LLM:
            direct_request = DirectLLMRequest(
                original_sentence=request.original_sentence,
                claim=request.claim,
                claim_id=request.claim_id,
                context=request.context,
                metadata=request.metadata,
            )
            if self.direct_adapter is None:
                raise RuntimeError("direct_llm adapter is not configured")
            return QueryResponse.from_direct(await self.direct_adapter(direct_request))

        recursive_request = RLMRecursiveRequest(
            original_sentence=request.original_sentence,
            claim=request.claim,
            claim_id=request.claim_id,
            context=request.context,
            budget=request.budget,
            metadata=request.metadata,
        )
        if self.rlm_bridge is None:
            raise RuntimeError("rlm_recursive bridge is not configured")
        try:
            return QueryResponse.from_recursive(
                self.rlm_bridge.query_recursive(recursive_request)
            )
        except Exception as exc:
            if self.direct_adapter is None:
                raise

            fallback_direct_request = DirectLLMRequest(
                original_sentence=request.original_sentence,
                claim=request.claim,
                claim_id=request.claim_id,
                context=request.context,
                metadata=request.metadata,
            )
            direct_response = await self.direct_adapter(fallback_direct_request)
            fallback_response = QueryResponse.from_direct(direct_response)
            fallback_response.degraded = True
            fallback_response.error = f"rlm_recursive_failed:{type(exc).__name__}:{exc}"
            fallback_response.fallback_count = max(1, fallback_response.fallback_count)
            fallback_response.raw = {
                "strategy_requested": QueryStrategy.RLM_RECURSIVE.value,
                "strategy_used": QueryStrategy.DIRECT_LLM.value,
                "fallback_reason": str(exc),
                "recursive_error_type": type(exc).__name__,
                "direct_raw": direct_response.raw,
            }
            return fallback_response

    def route(self, request: QueryRequest) -> QueryResponse:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.route_async(request))

        raise RuntimeError(
            "route() cannot be used inside an active event loop; use route_async()"
        )


def _model_to_dict(model_obj: Any) -> dict[str, Any]:
    if hasattr(model_obj, "model_dump"):
        return model_obj.model_dump()
    if hasattr(model_obj, "dict"):
        return model_obj.dict()
    if isinstance(model_obj, dict):
        return model_obj
    return {"raw": model_obj}


def _extract_evidence_list(context: Optional[dict[str, Any]]) -> list[str]:
    if not context:
        return []

    if isinstance(context.get("evidence"), list):
        return [str(item) for item in context["evidence"] if str(item).strip()]
    if isinstance(context.get("evidence_list"), list):
        return [str(item) for item in context["evidence_list"] if str(item).strip()]

    context_text = context.get("context") or context.get("text")
    if isinstance(context_text, str) and context_text.strip():
        return [context_text]

    return []


def _read_first_nonempty_str(
    data: dict[str, Any], keys: tuple[str, ...]
) -> Optional[str]:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _read_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _read_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
