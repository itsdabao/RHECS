from rhecs_core.runtime import (
    ClaimRuntimeState,
    InvalidRuntimeEvent,
    InvalidRuntimeTransition,
    RequestRuntimeState,
    RuntimeEventType,
    transition_claim_state,
    transition_request_state,
    validate_request_event,
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
        transition_request_state(
            RequestRuntimeState.RECEIVED, RequestRuntimeState.FINALIZED
        )
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
        transition_claim_state(
            ClaimRuntimeState.CLAIM_CREATED, ClaimRuntimeState.PATCH_APPLIED
        )
    except InvalidRuntimeTransition:
        return
    raise AssertionError("Expected InvalidRuntimeTransition")


def test_request_event_valid_in_verification_state():
    assert (
        validate_request_event(
            RequestRuntimeState.VERIFICATION_IN_PROGRESS,
            RuntimeEventType.RLM_SUBCALL_STARTED,
        )
        == RuntimeEventType.RLM_SUBCALL_STARTED
    )


def test_request_event_finished_valid_in_verification_state():
    assert (
        validate_request_event(
            RequestRuntimeState.VERIFICATION_IN_PROGRESS,
            RuntimeEventType.RLM_SUBCALL_FINISHED,
        )
        == RuntimeEventType.RLM_SUBCALL_FINISHED
    )


def test_request_event_failed_valid_in_verification_state():
    assert (
        validate_request_event(
            RequestRuntimeState.VERIFICATION_IN_PROGRESS,
            RuntimeEventType.RLM_SUBCALL_FAILED,
        )
        == RuntimeEventType.RLM_SUBCALL_FAILED
    )


def test_request_event_invalid_outside_verification_state_raises():
    try:
        validate_request_event(
            RequestRuntimeState.CLAIMS_EXTRACTED,
            RuntimeEventType.RLM_SUBCALL_FAILED,
        )
    except InvalidRuntimeEvent:
        return
    raise AssertionError("Expected InvalidRuntimeEvent")
