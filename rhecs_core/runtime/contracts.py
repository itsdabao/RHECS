from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RequestRuntimeState(str, Enum):
    RECEIVED = "RECEIVED"
    CLAIMS_EXTRACTED = "CLAIMS_EXTRACTED"
    VERIFICATION_IN_PROGRESS = "VERIFICATION_IN_PROGRESS"
    VERIFICATION_DONE = "VERIFICATION_DONE"
    RESTORATION_IN_PROGRESS = "RESTORATION_IN_PROGRESS"
    RESTORATION_DONE = "RESTORATION_DONE"
    FINALIZED = "FINALIZED"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


class ClaimRuntimeState(str, Enum):
    CLAIM_CREATED = "CLAIM_CREATED"
    PLAN_GENERATED = "PLAN_GENERATED"
    SANDBOX_EXECUTED = "SANDBOX_EXECUTED"
    VERDICT_ASSIGNED = "VERDICT_ASSIGNED"
    PATCH_GENERATED = "PATCH_GENERATED"
    PATCH_APPLIED = "PATCH_APPLIED"
    CLAIM_FAILED = "CLAIM_FAILED"


class RuntimeErrorType(str, Enum):
    PROVIDER_ERROR = "provider_error"
    POLICY_ERROR = "policy_error"
    EXECUTION_ERROR = "execution_error"
    DATA_ERROR = "data_error"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class RuntimeTransition:
    entity_type: str
    entity_id: str
    from_state: Optional[str]
    to_state: str
    stage: str
    reason: Optional[str] = None
    timestamp: str = field(default_factory=_utc_now_iso)


@dataclass
class RuntimeErrorInfo:
    error_type: RuntimeErrorType
    stage: str
    message: str
    claim_id: Optional[str] = None
    retryable: bool = False
    timestamp: str = field(default_factory=_utc_now_iso)
