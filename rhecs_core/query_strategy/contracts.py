from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class QueryStrategy(str, Enum):
    DIRECT_LLM = "direct_llm"
    RLM_RECURSIVE = "rlm_recursive"


@dataclass(frozen=True)
class QueryBudget:
    max_recursion: int = 2
    call_budget: int = 10
    token_budget: int = 5000
    timeout_budget_ms: int = 30000


@dataclass
class QueryRequest:
    original_sentence: str
    claim: dict[str, Any]
    strategy: QueryStrategy = QueryStrategy.DIRECT_LLM
    claim_id: Optional[str] = None
    context: Optional[dict[str, Any]] = None
    budget: QueryBudget = field(default_factory=QueryBudget)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DirectLLMRequest:
    original_sentence: str
    claim: dict[str, Any]
    claim_id: Optional[str] = None
    context: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RLMRecursiveRequest:
    original_sentence: str
    claim: dict[str, Any]
    claim_id: Optional[str] = None
    context: Optional[dict[str, Any]] = None
    budget: QueryBudget = field(default_factory=QueryBudget)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryEvent:
    step: int
    action: str
    input_summary: str
    output_summary: str
    model: Optional[str] = None
    latency_ms: Optional[int] = None
    tokens: Optional[int] = None


@dataclass
class DirectLLMResponse:
    verdict: str
    reasoning: str
    evidence: list[str]
    confidence: Optional[float] = None
    model_used: Optional[str] = None
    fallback_count: int = 0
    raw: Optional[dict[str, Any]] = None


@dataclass
class RLMRecursiveResponse:
    verdict: str
    reasoning: str
    evidence: list[str]
    confidence: Optional[float] = None
    trajectory: list[TrajectoryEvent] = field(default_factory=list)
    subcall_count: int = 0
    max_depth_reached: int = 0
    raw: Optional[dict[str, Any]] = None


@dataclass
class QueryResponse:
    strategy_used: QueryStrategy
    verdict: str
    reasoning: str
    evidence: list[str]
    confidence: Optional[float] = None
    trajectory: list[TrajectoryEvent] = field(default_factory=list)
    model_used: Optional[str] = None
    fallback_count: int = 0
    degraded: bool = False
    error: Optional[str] = None
    raw: Optional[dict[str, Any]] = None

    @classmethod
    def from_direct(cls, payload: DirectLLMResponse) -> "QueryResponse":
        return cls(
            strategy_used=QueryStrategy.DIRECT_LLM,
            verdict=payload.verdict,
            reasoning=payload.reasoning,
            evidence=payload.evidence,
            confidence=payload.confidence,
            model_used=payload.model_used,
            fallback_count=payload.fallback_count,
            raw=payload.raw,
        )

    @classmethod
    def from_recursive(cls, payload: RLMRecursiveResponse) -> "QueryResponse":
        return cls(
            strategy_used=QueryStrategy.RLM_RECURSIVE,
            verdict=payload.verdict,
            reasoning=payload.reasoning,
            evidence=payload.evidence,
            confidence=payload.confidence,
            trajectory=payload.trajectory,
            raw=payload.raw,
        )
