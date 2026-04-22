"""
Sandbox Manager — Executes LLM-generated Python code in an isolated subprocess.

WS2 hardening changes:
- Integrated AST-based policy guard (WS2-01) before execution.
- Structured error taxonomy via SandboxResult/SandboxError (WS2-02).
- Configurable timeout via environment variable RHECS_SANDBOX_TIMEOUT.
- Telemetry counters for observability.
- Backward-compatible output: SandboxResult supports .get() for existing pipeline code.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Optional

from rhecs_core.verification.policy_guard import PolicyResult, check_policy
from rhecs_core.verification.sandbox_errors import (
    RETRY_POLICY,
    SandboxError,
    SandboxErrorType,
    SandboxResult,
    SandboxTelemetry,
    classify_stderr,
)

# Module-level telemetry singleton
_telemetry = SandboxTelemetry()

DEFAULT_TIMEOUT_SECONDS = 15


def get_telemetry() -> SandboxTelemetry:
    """Return the module-level telemetry counters."""
    return _telemetry


def _get_timeout() -> int:
    """Read timeout from env var, with a sensible default."""
    raw = os.environ.get("RHECS_SANDBOX_TIMEOUT")
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return DEFAULT_TIMEOUT_SECONDS


def execute_sandbox_code(code_string: str, tenant_id: str) -> SandboxResult:
    """
    Executes native Python code in a strict subprocess safely.
    Injects the TENANT_ID into OS environment variables.
    Enforces the Subprocess I/O constraint.

    Returns a SandboxResult with structured error info when execution fails.
    """
    started_at = time.perf_counter()

    # ── Step 1: Policy Guard (AST check before execution) ───────────
    policy_result: PolicyResult = check_policy(code_string)
    if not policy_result.allowed:
        error = SandboxError(
            error_type=SandboxErrorType.POLICY_VIOLATION,
            message=policy_result.summary(),
            retryable=RETRY_POLICY[SandboxErrorType.POLICY_VIOLATION],
        )
        result = SandboxResult(
            success=False,
            error=error,
            policy_checked=True,
            execution_time_ms=_elapsed_ms(started_at),
        )
        _telemetry.record(result)
        return result

    # ── Step 2: Write code to temp file ─────────────────────────────
    timeout_seconds = _get_timeout()

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sandbox.py", delete=False, encoding="utf-8"
    ) as temp_script:
        temp_script.write(code_string)
        script_path = temp_script.name

    # ── Step 3: Prepare isolated environment ────────────────────────
    env = os.environ.copy()
    env["TENANT_ID"] = tenant_id
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{project_root}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else project_root
    )

    try:
        # ── Step 4: Execute in subprocess ───────────────────────────
        # NOTE: In Production Linux, replace [sys.executable] with
        # ["sudo", "-u", "sandboxuser", "python"]
        proc_result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )

        # Clean up temp file
        os.remove(script_path)

        # ── Step 5: Handle non-zero exit (crash/error) ──────────────
        if proc_result.returncode != 0:
            error_type, error_line = classify_stderr(proc_result.stderr)
            error = SandboxError(
                error_type=error_type,
                message=proc_result.stderr.strip()
                or "Non-zero exit code with no stderr",
                retryable=RETRY_POLICY.get(error_type, False),
                raw_stderr=proc_result.stderr,
                line=error_line,
            )
            result = SandboxResult(
                success=False,
                error=error,
                policy_checked=True,
                execution_time_ms=_elapsed_ms(started_at),
            )
            _telemetry.record(result)
            return result

        # ── Step 6: Parse JSON stdout ───────────────────────────────
        try:
            output_data = json.loads(proc_result.stdout.strip())
            result = SandboxResult(
                success=True,
                output=output_data,
                policy_checked=True,
                execution_time_ms=_elapsed_ms(started_at),
            )
            _telemetry.record(result)
            return result
        except json.JSONDecodeError:
            error = SandboxError(
                error_type=SandboxErrorType.OUTPUT_PARSE_ERROR,
                message=(
                    f"Failed to parse JSON from stdout. "
                    f"The model must strictly only print valid JSON. "
                    f"Raw output received:\n{proc_result.stdout}"
                ),
                retryable=RETRY_POLICY[SandboxErrorType.OUTPUT_PARSE_ERROR],
                raw_stderr=proc_result.stdout,
            )
            result = SandboxResult(
                success=False,
                error=error,
                policy_checked=True,
                execution_time_ms=_elapsed_ms(started_at),
            )
            _telemetry.record(result)
            return result

    except subprocess.TimeoutExpired:
        # Kill hanging scripts
        _safe_remove(script_path)
        error = SandboxError(
            error_type=SandboxErrorType.TIMEOUT,
            message=f"Execution timed out after {timeout_seconds} seconds. Code likely entered an infinite loop.",
            retryable=RETRY_POLICY[SandboxErrorType.TIMEOUT],
        )
        result = SandboxResult(
            success=False,
            error=error,
            policy_checked=True,
            execution_time_ms=_elapsed_ms(started_at),
        )
        _telemetry.record(result)
        return result

    except Exception as exc:
        _safe_remove(script_path)
        error = SandboxError(
            error_type=SandboxErrorType.UNKNOWN,
            message=f"Unknown sandbox error: {exc}",
            retryable=RETRY_POLICY[SandboxErrorType.UNKNOWN],
        )
        result = SandboxResult(
            success=False,
            error=error,
            policy_checked=True,
            execution_time_ms=_elapsed_ms(started_at),
        )
        _telemetry.record(result)
        return result


def _elapsed_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)


def _safe_remove(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass
