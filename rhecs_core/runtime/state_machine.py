from rhecs_core.runtime.contracts import ClaimRuntimeState, RequestRuntimeState


class InvalidRuntimeTransition(ValueError):
    pass


_REQUEST_TRANSITIONS: dict[RequestRuntimeState, set[RequestRuntimeState]] = {
    RequestRuntimeState.RECEIVED: {
        RequestRuntimeState.CLAIMS_EXTRACTED,
        RequestRuntimeState.FAILED,
    },
    RequestRuntimeState.CLAIMS_EXTRACTED: {
        RequestRuntimeState.VERIFICATION_IN_PROGRESS,
        RequestRuntimeState.FAILED,
    },
    RequestRuntimeState.VERIFICATION_IN_PROGRESS: {
        RequestRuntimeState.VERIFICATION_DONE,
        RequestRuntimeState.DEGRADED,
        RequestRuntimeState.FAILED,
    },
    RequestRuntimeState.VERIFICATION_DONE: {
        RequestRuntimeState.RESTORATION_IN_PROGRESS,
        RequestRuntimeState.DEGRADED,
        RequestRuntimeState.FAILED,
    },
    RequestRuntimeState.RESTORATION_IN_PROGRESS: {
        RequestRuntimeState.RESTORATION_DONE,
        RequestRuntimeState.DEGRADED,
        RequestRuntimeState.FAILED,
    },
    RequestRuntimeState.RESTORATION_DONE: {
        RequestRuntimeState.FINALIZED,
        RequestRuntimeState.DEGRADED,
        RequestRuntimeState.FAILED,
    },
    RequestRuntimeState.FINALIZED: set(),
    RequestRuntimeState.DEGRADED: set(),
    RequestRuntimeState.FAILED: set(),
}


_CLAIM_TRANSITIONS: dict[ClaimRuntimeState, set[ClaimRuntimeState]] = {
    ClaimRuntimeState.CLAIM_CREATED: {
        ClaimRuntimeState.PLAN_GENERATED,
        ClaimRuntimeState.CLAIM_FAILED,
    },
    ClaimRuntimeState.PLAN_GENERATED: {
        ClaimRuntimeState.SANDBOX_EXECUTED,
        ClaimRuntimeState.CLAIM_FAILED,
    },
    ClaimRuntimeState.SANDBOX_EXECUTED: {
        ClaimRuntimeState.VERDICT_ASSIGNED,
        ClaimRuntimeState.CLAIM_FAILED,
    },
    ClaimRuntimeState.VERDICT_ASSIGNED: {
        ClaimRuntimeState.PATCH_GENERATED,
        ClaimRuntimeState.CLAIM_FAILED,
    },
    ClaimRuntimeState.PATCH_GENERATED: {
        ClaimRuntimeState.PATCH_APPLIED,
        ClaimRuntimeState.CLAIM_FAILED,
    },
    ClaimRuntimeState.PATCH_APPLIED: set(),
    ClaimRuntimeState.CLAIM_FAILED: set(),
}


def transition_request_state(
    current: RequestRuntimeState,
    target: RequestRuntimeState,
) -> RequestRuntimeState:
    if target in _REQUEST_TRANSITIONS[current]:
        return target
    raise InvalidRuntimeTransition(f"Invalid request transition: {current.value} -> {target.value}")



def transition_claim_state(
    current: ClaimRuntimeState,
    target: ClaimRuntimeState,
) -> ClaimRuntimeState:
    if target in _CLAIM_TRANSITIONS[current]:
        return target
    raise InvalidRuntimeTransition(f"Invalid claim transition: {current.value} -> {target.value}")
