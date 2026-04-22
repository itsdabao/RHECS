import os
from dataclasses import dataclass

from rhecs_core.runtime.contracts import VerificationStrategy


@dataclass(frozen=True)
class RuntimeConfig:
    default_strategy: VerificationStrategy = VerificationStrategy.DIRECT_LLM
    max_recursion: int = 2
    call_budget: int = 10
    token_budget: int = 5000
    timeout_budget: int = 30000

    def to_dict(self) -> dict[str, int | str]:
        return {
            "default_strategy": self.default_strategy.value,
            "max_recursion": self.max_recursion,
            "call_budget": self.call_budget,
            "token_budget": self.token_budget,
            "timeout_budget": self.timeout_budget,
        }

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        strategy_raw = (
            (
                os.getenv("RHECS_VERIFICATION_STRATEGY")
                or os.getenv("RHECS_DEFAULT_STRATEGY")
                or VerificationStrategy.DIRECT_LLM.value
            )
            .strip()
            .lower()
        )

        strategy_map = {
            VerificationStrategy.DIRECT_LLM.value: VerificationStrategy.DIRECT_LLM,
            VerificationStrategy.RLM_RECURSIVE.value: VerificationStrategy.RLM_RECURSIVE,
        }
        default_strategy = strategy_map.get(
            strategy_raw, VerificationStrategy.DIRECT_LLM
        )

        max_recursion = _read_positive_int_env(
            "RHECS_ROUTER_MAX_RECURSION",
            fallback_env="RHECS_MAX_RECURSION",
            default=2,
        )
        call_budget = _read_positive_int_env(
            "RHECS_ROUTER_MAX_CALLS",
            fallback_env="RHECS_CALL_BUDGET",
            default=10,
        )
        token_budget = _read_positive_int_env(
            "RHECS_ROUTER_MAX_TOKENS",
            fallback_env="RHECS_TOKEN_BUDGET",
            default=5000,
        )
        timeout_budget = _read_positive_int_env(
            "RHECS_ROUTER_TIMEOUT_MS",
            fallback_env="RHECS_TIMEOUT_BUDGET_MS",
            default=30000,
        )

        return cls(
            default_strategy=default_strategy,
            max_recursion=max_recursion,
            call_budget=call_budget,
            token_budget=token_budget,
            timeout_budget=timeout_budget,
        )


def _read_positive_int_env(primary_env: str, fallback_env: str, default: int) -> int:
    raw = os.getenv(primary_env)
    if raw is None:
        raw = os.getenv(fallback_env)
    if raw is None:
        return default

    value = int(raw.strip())
    if value <= 0:
        raise ValueError(f"{primary_env} must be > 0")
    return value
