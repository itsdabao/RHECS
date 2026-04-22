"""Tests for WS2-02: Sandbox Error Taxonomy."""

import pytest

from rhecs_core.verification.sandbox_errors import (
    RETRY_POLICY,
    SandboxError,
    SandboxErrorType,
    SandboxResult,
    SandboxTelemetry,
    classify_stderr,
)


class TestClassifyStderr:
    """Test stderr classification into error types."""

    def test_syntax_error(self):
        stderr = '  File "test.py", line 5\n    x = \nSyntaxError: invalid syntax'
        error_type, line = classify_stderr(stderr)
        assert error_type == SandboxErrorType.SYNTAX_ERROR
        assert line == 5

    def test_import_error(self):
        stderr = "ModuleNotFoundError: No module named 'nonexistent'"
        error_type, line = classify_stderr(stderr)
        assert error_type == SandboxErrorType.IMPORT_ERROR

    def test_name_error(self):
        stderr = "Traceback (most recent call last):\n  File \"test.py\", line 3\nNameError: name 'x' is not defined"
        error_type, line = classify_stderr(stderr)
        assert error_type == SandboxErrorType.RUNTIME_ERROR
        assert line == 3

    def test_type_error(self):
        stderr = "TypeError: unsupported operand type(s)"
        error_type, _ = classify_stderr(stderr)
        assert error_type == SandboxErrorType.RUNTIME_ERROR

    def test_traceback(self):
        stderr = "Traceback (most recent call last):\n  unknown error"
        error_type, _ = classify_stderr(stderr)
        assert error_type == SandboxErrorType.RUNTIME_ERROR

    def test_empty_stderr(self):
        error_type, _ = classify_stderr("")
        assert error_type == SandboxErrorType.UNKNOWN

    def test_unknown_text(self):
        error_type, _ = classify_stderr("something completely unexpected happened")
        assert error_type == SandboxErrorType.UNKNOWN

    def test_permission_error(self):
        stderr = "PermissionError: [Errno 13] Permission denied"
        error_type, _ = classify_stderr(stderr)
        assert error_type == SandboxErrorType.RUNTIME_ERROR


class TestRetryPolicy:
    """Test that retry policy is correct per error type."""

    def test_policy_violation_not_retryable(self):
        assert RETRY_POLICY[SandboxErrorType.POLICY_VIOLATION] is False

    def test_syntax_error_retryable(self):
        assert RETRY_POLICY[SandboxErrorType.SYNTAX_ERROR] is True

    def test_runtime_error_retryable(self):
        assert RETRY_POLICY[SandboxErrorType.RUNTIME_ERROR] is True

    def test_import_error_not_retryable(self):
        assert RETRY_POLICY[SandboxErrorType.IMPORT_ERROR] is False

    def test_timeout_not_retryable(self):
        assert RETRY_POLICY[SandboxErrorType.TIMEOUT] is False

    def test_output_parse_retryable(self):
        assert RETRY_POLICY[SandboxErrorType.OUTPUT_PARSE_ERROR] is True

    def test_unknown_not_retryable(self):
        assert RETRY_POLICY[SandboxErrorType.UNKNOWN] is False


class TestSandboxResult:
    """Test SandboxResult backward compatibility."""

    def test_success_result_get(self):
        result = SandboxResult(success=True, output={"evidence": ["test"]})
        assert result.get("success") is True
        assert result.get("output") == {"evidence": ["test"]}
        assert result.get("error") is None
        assert result.get("nonexistent", "default") == "default"

    def test_error_result_get(self):
        error = SandboxError(
            error_type=SandboxErrorType.TIMEOUT,
            message="Timed out",
            retryable=False,
        )
        result = SandboxResult(success=False, error=error)
        assert result.get("success") is False
        assert result.get("error") == "Timed out"

    def test_to_dict_success(self):
        result = SandboxResult(
            success=True, output={"k": "v"}, policy_checked=True, execution_time_ms=100
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["output"] == {"k": "v"}
        assert d["policy_checked"] is True
        assert "error" not in d

    def test_to_dict_error(self):
        error = SandboxError(
            error_type=SandboxErrorType.POLICY_VIOLATION,
            message="blocked",
            retryable=False,
        )
        result = SandboxResult(success=False, error=error, policy_checked=True)
        d = result.to_dict()
        assert d["success"] is False
        assert d["error"]["error_type"] == "policy_violation"
        assert d["error"]["retryable"] is False


class TestSandboxTelemetry:
    """Test telemetry counter tracking."""

    def test_record_success(self):
        telemetry = SandboxTelemetry()
        result = SandboxResult(success=True, output={})
        telemetry.record(result)
        assert telemetry.total_executions == 1
        assert telemetry.successes == 1

    def test_record_policy_block(self):
        telemetry = SandboxTelemetry()
        error = SandboxError(
            error_type=SandboxErrorType.POLICY_VIOLATION,
            message="blocked",
            retryable=False,
        )
        result = SandboxResult(success=False, error=error)
        telemetry.record(result)
        assert telemetry.total_executions == 1
        assert telemetry.policy_blocks == 1
        assert telemetry.successes == 0

    def test_record_timeout(self):
        telemetry = SandboxTelemetry()
        error = SandboxError(
            error_type=SandboxErrorType.TIMEOUT,
            message="timeout",
            retryable=False,
        )
        result = SandboxResult(success=False, error=error)
        telemetry.record(result)
        assert telemetry.timeouts == 1

    def test_to_dict(self):
        telemetry = SandboxTelemetry()
        result_ok = SandboxResult(success=True, output={})
        telemetry.record(result_ok)
        telemetry.record(result_ok)

        d = telemetry.to_dict()
        assert d["total_executions"] == 2
        assert d["successes"] == 2
        assert d["success_rate"] == 1.0

    def test_mixed_results(self):
        telemetry = SandboxTelemetry()
        telemetry.record(SandboxResult(success=True, output={}))
        telemetry.record(
            SandboxResult(
                success=False,
                error=SandboxError(
                    error_type=SandboxErrorType.SYNTAX_ERROR,
                    message="bad syntax",
                    retryable=True,
                ),
            )
        )
        telemetry.record(
            SandboxResult(
                success=False,
                error=SandboxError(
                    error_type=SandboxErrorType.TIMEOUT,
                    message="timeout",
                    retryable=False,
                ),
            )
        )

        assert telemetry.total_executions == 3
        assert telemetry.successes == 1
        assert telemetry.syntax_errors == 1
        assert telemetry.timeouts == 1
        d = telemetry.to_dict()
        assert abs(d["success_rate"] - 1 / 3) < 0.01


class TestSandboxErrorSerialization:
    def test_error_to_dict(self):
        error = SandboxError(
            error_type=SandboxErrorType.RUNTIME_ERROR,
            message="NameError: x is not defined",
            retryable=True,
            raw_stderr="Traceback...",
            line=42,
        )
        d = error.to_dict()
        assert d["error_type"] == "runtime_error"
        assert d["retryable"] is True
        assert d["line"] == 42
        assert d["raw_stderr"] == "Traceback..."
