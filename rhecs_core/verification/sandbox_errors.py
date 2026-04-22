"""
WS2-02: Structured Error Taxonomy for Sandbox Execution.

Provides a unified classification of sandbox errors with retry guidance
and telemetry support. Used by sandbox_manager and pipeline to make
intelligent retry decisions.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SandboxErrorType(str, Enum):
    """Fine-grained classification of sandbox execution errors."""

    # Pre-execution errors
    POLICY_VIOLATION = "policy_violation"
    SYNTAX_ERROR = "syntax_error"

    # Execution errors
    RUNTIME_ERROR = "runtime_error"
    IMPORT_ERROR = "import_error"
    TIMEOUT = "timeout"

    # Post-execution errors
    OUTPUT_PARSE_ERROR = "output_parse_error"

    # Unknown / catch-all
    UNKNOWN = "unknown"


# Retry policy per error type
RETRY_POLICY: dict[SandboxErrorType, bool] = {
    SandboxErrorType.POLICY_VIOLATION: False,  # Never retry — code is fundamentally unsafe
    SandboxErrorType.SYNTAX_ERROR: True,  # LLM can fix syntax on retry
    SandboxErrorType.RUNTIME_ERROR: True,  # LLM can fix logic errors on retry
    SandboxErrorType.IMPORT_ERROR: False,  # Missing dependency — retry won't help
    SandboxErrorType.TIMEOUT: False,  # Infinite loop — retry won't help
    SandboxErrorType.OUTPUT_PARSE_ERROR: True,  # LLM can fix JSON output on retry
    SandboxErrorType.UNKNOWN: False,  # Unknown errors are not retryable
}


@dataclass
class SandboxError:
    """Structured error from sandbox execution."""

    error_type: SandboxErrorType
    message: str
    retryable: bool
    raw_stderr: Optional[str] = None
    line: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "error_type": self.error_type.value,
            "message": self.message,
            "retryable": self.retryable,
            "raw_stderr": self.raw_stderr,
            "line": self.line,
        }


@dataclass
class SandboxResult:
    """Unified result from sandbox execution."""

    success: bool
    output: Optional[dict] = None
    error: Optional[SandboxError] = None
    policy_checked: bool = False
    execution_time_ms: Optional[int] = None

    def to_dict(self) -> dict:
        result: dict = {
            "success": self.success,
            "policy_checked": self.policy_checked,
        }
        if self.output is not None:
            result["output"] = self.output
        if self.error is not None:
            result["error"] = self.error.to_dict()
        if self.execution_time_ms is not None:
            result["execution_time_ms"] = self.execution_time_ms
        return result

    # Backward compatibility: support dict-like access for existing pipeline code
    def get(self, key: str, default=None):
        if key == "success":
            return self.success
        if key == "output":
            return self.output
        if key == "error":
            if self.error:
                return self.error.message
            return default
        return default


@dataclass
class SandboxTelemetry:
    """Counters for sandbox execution telemetry."""

    total_executions: int = 0
    policy_blocks: int = 0
    syntax_errors: int = 0
    runtime_errors: int = 0
    timeouts: int = 0
    parse_errors: int = 0
    successes: int = 0

    def record(self, result: SandboxResult) -> None:
        self.total_executions += 1
        if result.success:
            self.successes += 1
        elif result.error:
            counter_map = {
                SandboxErrorType.POLICY_VIOLATION: "policy_blocks",
                SandboxErrorType.SYNTAX_ERROR: "syntax_errors",
                SandboxErrorType.RUNTIME_ERROR: "runtime_errors",
                SandboxErrorType.IMPORT_ERROR: "runtime_errors",
                SandboxErrorType.TIMEOUT: "timeouts",
                SandboxErrorType.OUTPUT_PARSE_ERROR: "parse_errors",
            }
            attr = counter_map.get(result.error.error_type)
            if attr:
                setattr(self, attr, getattr(self, attr) + 1)

    def to_dict(self) -> dict:
        return {
            "total_executions": self.total_executions,
            "policy_blocks": self.policy_blocks,
            "syntax_errors": self.syntax_errors,
            "runtime_errors": self.runtime_errors,
            "timeouts": self.timeouts,
            "parse_errors": self.parse_errors,
            "successes": self.successes,
            "success_rate": (
                self.successes / self.total_executions
                if self.total_executions > 0
                else 0.0
            ),
        }


def classify_stderr(stderr: str) -> tuple[SandboxErrorType, Optional[int]]:
    """
    Classify subprocess stderr into a structured error type.
    Returns (error_type, line_number_if_available).
    """
    if not stderr:
        return SandboxErrorType.UNKNOWN, None

    lower = stderr.lower()

    # Syntax errors
    if "syntaxerror" in lower:
        line = _extract_line_number(stderr)
        return SandboxErrorType.SYNTAX_ERROR, line

    # Import errors
    if "importerror" in lower or "modulenotfounderror" in lower:
        return SandboxErrorType.IMPORT_ERROR, None

    # Common runtime errors
    runtime_markers = [
        "nameerror",
        "typeerror",
        "valueerror",
        "keyerror",
        "indexerror",
        "attributeerror",
        "zerodivisionerror",
        "filenotfounderror",
        "permissionerror",
        "oserror",
        "runtimeerror",
        "traceback",
    ]
    for marker in runtime_markers:
        if marker in lower:
            line = _extract_line_number(stderr)
            return SandboxErrorType.RUNTIME_ERROR, line

    return SandboxErrorType.UNKNOWN, None


def _extract_line_number(stderr: str) -> Optional[int]:
    """Try to extract a line number from a Python traceback."""
    import re

    # Match patterns like 'line 42' or 'File "...", line 42'
    match = re.search(r"line\s+(\d+)", stderr)
    if match:
        return int(match.group(1))
    return None
