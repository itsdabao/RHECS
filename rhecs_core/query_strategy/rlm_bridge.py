import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from rhecs_core.query_strategy.contracts import (
    RLMRecursiveRequest,
    RLMRecursiveResponse,
    TrajectoryEvent,
)


class RLMBridgeError(RuntimeError):
    pass


@dataclass(frozen=True)
class RLMBridgeConfig:
    max_recursion: int = 2
    call_budget: int = 10
    token_budget: int = 5000
    timeout_budget_ms: int = 30000


class RLMBridge:
    """
    Adapter boundary for recursive query strategy.

    This class mirrors the rlm concepts at design level (llm_query/rlm_query/search_evidence)
    without coupling query routing code to private rlm internals.
    """

    def __init__(
        self,
        config: Optional[RLMBridgeConfig] = None,
        llm_query_fn: Optional[Callable[[Any, Optional[str]], Any]] = None,
        rlm_query_fn: Optional[Callable[[Any, Optional[str]], Any]] = None,
        search_evidence_fn: Optional[Callable[[str], list[str]]] = None,
    ) -> None:
        self.config = config or RLMBridgeConfig()
        self._llm_query_fn = llm_query_fn
        self._rlm_query_fn = rlm_query_fn
        self._search_evidence_fn = search_evidence_fn
        self._step = 0

    def llm_query(self, payload: Any, model: Optional[str] = None) -> Any:
        if self._llm_query_fn is None:
            raise RLMBridgeError("llm_query function is not configured")
        return self._llm_query_fn(payload, model)

    def rlm_query(self, payload: Any, model: Optional[str] = None) -> Any:
        if self._rlm_query_fn is None:
            raise RLMBridgeError("rlm_query function is not configured")
        return self._rlm_query_fn(payload, model)

    def search_evidence(self, query: str) -> list[str]:
        if self._search_evidence_fn is None:
            raise RLMBridgeError("search_evidence function is not configured")
        return self._search_evidence_fn(query)

    def query_recursive(self, request: RLMRecursiveRequest) -> RLMRecursiveResponse:
        self._step = 0
        started_at = time.perf_counter()
        trajectory: list[TrajectoryEvent] = []

        max_recursion = request.budget.max_recursion or self.config.max_recursion
        call_budget = request.budget.call_budget or self.config.call_budget
        token_budget = request.budget.token_budget or self.config.token_budget
        timeout_budget_ms = (
            request.budget.timeout_budget_ms or self.config.timeout_budget_ms
        )

        runtime = {
            "calls": 0,
            "token_estimate": 0,
            "max_depth_reached": 0,
            "started_at": started_at,
        }

        root_payload = {
            "claim": request.claim,
            "original_sentence": request.original_sentence,
            "context": request.context or {},
            "metadata": request.metadata,
            "budget": {
                "max_recursion": max_recursion,
                "call_budget": call_budget,
                "token_budget": token_budget,
                "timeout_budget_ms": timeout_budget_ms,
            },
        }

        self._push_event(
            trajectory,
            action="route",
            input_summary="strategy=rlm_recursive",
            output_summary="router_autonomy_enabled",
            model=(
                _read_str(request.metadata.get("model")) if request.metadata else None
            ),
        )

        raw_result = self._run_recursive(
            payload=root_payload,
            depth=0,
            max_recursion=max_recursion,
            call_budget=call_budget,
            token_budget=token_budget,
            timeout_budget_ms=timeout_budget_ms,
            trajectory=trajectory,
            runtime=runtime,
        )

        verdict, reasoning, evidence, confidence, raw = self._parse_recursive_output(
            raw_result
        )

        return RLMRecursiveResponse(
            verdict=verdict,
            reasoning=reasoning,
            evidence=evidence,
            confidence=confidence,
            trajectory=trajectory,
            subcall_count=runtime["calls"],
            max_depth_reached=runtime["max_depth_reached"],
            raw=raw,
        )

    def _run_recursive(
        self,
        payload: dict[str, Any],
        depth: int,
        max_recursion: int,
        call_budget: int,
        token_budget: int,
        timeout_budget_ms: int,
        trajectory: list[TrajectoryEvent],
        runtime: dict[str, Any],
    ) -> Any:
        runtime["max_depth_reached"] = max(runtime["max_depth_reached"], depth)

        self._assert_guardrails(
            payload=payload,
            runtime=runtime,
            call_budget=call_budget,
            token_budget=token_budget,
            timeout_budget_ms=timeout_budget_ms,
        )

        if depth >= max_recursion:
            self._push_event(
                trajectory,
                action="route",
                input_summary=f"depth={depth}",
                output_summary="recursion_stop_fallback_to_llm_query",
            )
            return self._invoke_llm_query(payload, trajectory, runtime)

        # Prefer an explicit recursive runtime if provided.
        if self._rlm_query_fn is not None:
            return self._invoke_rlm_query(payload, trajectory, runtime)

        # Autonomous fallback when no external recursive runtime is configured.
        return self._invoke_autonomous_mode(
            payload, depth, max_recursion, trajectory, runtime
        )

    def _invoke_rlm_query(
        self,
        payload: dict[str, Any],
        trajectory: list[TrajectoryEvent],
        runtime: dict[str, Any],
    ) -> Any:
        model = (
            _read_str(payload.get("metadata", {}).get("model"))
            if isinstance(payload, dict)
            else None
        )
        preview = self._claim_preview(payload)

        started_at = time.perf_counter()
        self._push_event(
            trajectory,
            action="rlm_query",
            input_summary=preview,
            output_summary="started",
            model=model,
        )

        runtime["calls"] += 1
        raw = self.rlm_query(payload, model=model)

        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        self._push_event(
            trajectory,
            action="rlm_query",
            input_summary=preview,
            output_summary="finished",
            model=model,
            latency_ms=elapsed_ms,
            tokens=_estimate_tokens(raw),
        )
        return raw

    def _invoke_llm_query(
        self,
        payload: dict[str, Any],
        trajectory: list[TrajectoryEvent],
        runtime: dict[str, Any],
    ) -> Any:
        if self._llm_query_fn is None:
            raise RLMBridgeError("llm_query function is not configured")

        model = (
            _read_str(payload.get("metadata", {}).get("model"))
            if isinstance(payload, dict)
            else None
        )
        preview = self._claim_preview(payload)
        started_at = time.perf_counter()

        self._push_event(
            trajectory,
            action="llm_query",
            input_summary=preview,
            output_summary="started",
            model=model,
        )

        runtime["calls"] += 1
        raw = self.llm_query(payload, model=model)

        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        self._push_event(
            trajectory,
            action="llm_query",
            input_summary=preview,
            output_summary="finished",
            model=model,
            latency_ms=elapsed_ms,
            tokens=_estimate_tokens(raw),
        )
        return raw

    def _invoke_autonomous_mode(
        self,
        payload: dict[str, Any],
        depth: int,
        max_recursion: int,
        trajectory: list[TrajectoryEvent],
        runtime: dict[str, Any],
    ) -> Any:
        claim_text = self._claim_preview(payload)
        self._push_event(
            trajectory,
            action="route",
            input_summary=f"depth={depth}",
            output_summary="autonomous_mode",
        )

        evidence: list[str] = []
        if self._search_evidence_fn is not None:
            try:
                evidence = self.search_evidence(claim_text)
            except Exception as exc:
                self._push_event(
                    trajectory,
                    action="search_evidence",
                    input_summary=claim_text,
                    output_summary=f"error:{exc}",
                )
                evidence = []
            else:
                self._push_event(
                    trajectory,
                    action="search_evidence",
                    input_summary=claim_text,
                    output_summary=f"hits={len(evidence)}",
                )

        if depth + 1 <= max_recursion and _looks_complex_claim(claim_text):
            self._push_event(
                trajectory,
                action="route",
                input_summary=f"depth={depth}",
                output_summary="decompose_to_subcall",
            )
            sub_payload = dict(payload)
            sub_payload["metadata"] = {
                **(
                    payload.get("metadata", {})
                    if isinstance(payload.get("metadata"), dict)
                    else {}
                ),
                "decomposed": True,
            }
            return self._run_recursive(
                payload=sub_payload,
                depth=depth + 1,
                max_recursion=max_recursion,
                call_budget=payload["budget"]["call_budget"],
                token_budget=payload["budget"]["token_budget"],
                timeout_budget_ms=payload["budget"]["timeout_budget_ms"],
                trajectory=trajectory,
                runtime=runtime,
            )

        llm_payload = {
            **payload,
            "evidence": evidence,
            "mode": "autonomous_llm_fallback",
        }
        return self._invoke_llm_query(llm_payload, trajectory, runtime)

    def _assert_guardrails(
        self,
        payload: dict[str, Any],
        runtime: dict[str, Any],
        call_budget: int,
        token_budget: int,
        timeout_budget_ms: int,
    ) -> None:
        elapsed_ms = int((time.perf_counter() - runtime["started_at"]) * 1000)
        if elapsed_ms > timeout_budget_ms:
            raise RLMBridgeError(
                f"timeout_budget_exceeded:{elapsed_ms}>{timeout_budget_ms}"
            )

        if runtime["calls"] >= call_budget:
            raise RLMBridgeError(
                f"call_budget_exceeded:{runtime['calls']}>={call_budget}"
            )

        runtime["token_estimate"] += _estimate_tokens(payload)
        if runtime["token_estimate"] > token_budget:
            raise RLMBridgeError(
                f"token_budget_exceeded:{runtime['token_estimate']}>{token_budget}"
            )

    def _parse_recursive_output(
        self,
        raw_output: Any,
    ) -> tuple[str, str, list[str], Optional[float], dict[str, Any]]:
        parsed = _normalize_to_dict(raw_output)

        final_candidate = (
            parsed.get("FINAL")
            or parsed.get("final")
            or parsed.get("final_answer")
            or parsed.get("answer")
            or parsed.get("FINAL_VAR")
            or parsed.get("final_var")
            or parsed
        )

        final_dict = _normalize_to_dict(final_candidate)

        verdict = str(
            final_dict.get("verdict")
            or final_dict.get("status")
            or parsed.get("verdict")
            or parsed.get("status")
            or "NOT_MENTIONED"
        )
        reasoning = str(
            final_dict.get("reasoning")
            or parsed.get("reasoning")
            or "No reasoning returned by recursive bridge."
        )

        evidence_raw = final_dict.get("evidence")
        if evidence_raw is None:
            evidence_raw = parsed.get("evidence")
        evidence = _to_string_list(evidence_raw)

        confidence = _to_float(final_dict.get("confidence"))
        if confidence is None:
            confidence = _to_float(parsed.get("confidence"))

        return verdict, reasoning, evidence, confidence, parsed

    def _push_event(
        self,
        trajectory: list[TrajectoryEvent],
        action: str,
        input_summary: str,
        output_summary: str,
        model: Optional[str] = None,
        latency_ms: Optional[int] = None,
        tokens: Optional[int] = None,
    ) -> None:
        self._step += 1
        trajectory.append(
            TrajectoryEvent(
                step=self._step,
                action=action,
                input_summary=input_summary,
                output_summary=output_summary,
                model=model,
                latency_ms=latency_ms,
                tokens=tokens,
            )
        )

    @staticmethod
    def _claim_preview(payload: dict[str, Any]) -> str:
        claim = payload.get("claim")
        if isinstance(claim, dict):
            parts = []
            for key in (
                "entity",
                "relationship",
                "target",
                "subject",
                "predicate",
                "object",
            ):
                value = claim.get(key)
                if value:
                    parts.append(str(value))
            text = " ".join(parts).strip()
            if text:
                return text
        return str(claim)[:200]


def _normalize_to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {"raw": value}
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError:
            return {"raw": value}
        if isinstance(loaded, dict):
            return loaded
        return {"raw": loaded}
    return {"raw": value}


def _to_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_str(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _estimate_tokens(value: Any) -> int:
    # Lightweight estimate to enforce guardrails without tokenizer dependency.
    text = (
        json.dumps(value, ensure_ascii=False, default=str)
        if not isinstance(value, str)
        else value
    )
    return max(1, len(text) // 4)


def _looks_complex_claim(text: str) -> bool:
    lowered = text.lower()
    complexity_markers = [" và ", ";", ",", " nhưng ", " trong khi ", " đồng thời "]
    return any(token in lowered for token in complexity_markers) or len(lowered) > 140
