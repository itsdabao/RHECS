from rhecs_core.runtime import (
    ClaimRuntimeState,
    InvalidRuntimeTransition,
    RequestRuntimeState,
    transition_claim_state,
    transition_request_state,
)


def test_request_state_valid_transition():
    assert (
        transition_request_state(
            RequestRuntimeState.RECEIVED,
            RequestRuntimeState.CLAIMS_EXTRACTED,
        )
        == RequestRuntimeState.CLAIMS_EXTRACTED
    )


def test_request_state_invalid_transition_raises():
    try:
        transition_request_state(RequestRuntimeState.RECEIVED, RequestRuntimeState.FINALIZED)
    except InvalidRuntimeTransition:
        return
    raise AssertionError("Expected InvalidRuntimeTransition")


def test_claim_state_valid_transition():
    assert (
        transition_claim_state(
            ClaimRuntimeState.CLAIM_CREATED,
            ClaimRuntimeState.PLAN_GENERATED,
        )
        == ClaimRuntimeState.PLAN_GENERATED
    )


def test_claim_state_invalid_transition_raises():
    try:
        transition_claim_state(ClaimRuntimeState.CLAIM_CREATED, ClaimRuntimeState.PATCH_APPLIED)
    except InvalidRuntimeTransition:
        return
    raise AssertionError("Expected InvalidRuntimeTransition")
