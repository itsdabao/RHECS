"""Tests for WS2-01: Policy Guard."""

import pytest

from rhecs_core.verification.policy_guard import (
    PolicyResult,
    PolicyViolationType,
    check_policy,
)


class TestSafeScripts:
    """Scripts that SHOULD be allowed."""

    def test_basic_json_print(self):
        code = 'import json\nresult = {"key": "value"}\nprint(json.dumps(result))'
        result = check_policy(code)
        assert result.allowed, result.summary()

    def test_search_evidence_import(self):
        code = (
            "from rhecs_core.verification.sandbox_helpers import search_evidence\n"
            "import json\n"
            "evidence = search_evidence('test query')\n"
            "print(json.dumps({'evidence': evidence}))\n"
        )
        result = check_policy(code)
        assert result.allowed, result.summary()

    def test_math_operations(self):
        code = "import math\nimport json\nx = math.sqrt(16)\nprint(json.dumps({'result': x}))"
        result = check_policy(code)
        assert result.allowed, result.summary()

    def test_string_operations(self):
        code = "import re\nimport json\nm = re.search(r'hello', 'hello world')\nprint(json.dumps({'found': bool(m)}))"
        result = check_policy(code)
        assert result.allowed, result.summary()

    def test_collections_usage(self):
        code = "from collections import Counter\nimport json\nc = Counter([1,2,2,3])\nprint(json.dumps(dict(c)))"
        result = check_policy(code)
        assert result.allowed, result.summary()

    def test_os_environ_allowed(self):
        """os module is partially allowed (environ access is common in sandbox)."""
        code = "import os\nimport json\ntid = os.environ.get('TENANT_ID', '')\nprint(json.dumps({'tid': tid}))"
        result = check_policy(code)
        assert result.allowed, result.summary()


class TestDangerousImports:
    """Scripts with dangerous imports that MUST be blocked."""

    def test_subprocess_import(self):
        code = "import subprocess\nsubprocess.run(['ls'])"
        result = check_policy(code)
        assert not result.allowed
        assert any(
            v.violation_type == PolicyViolationType.DANGEROUS_IMPORT
            for v in result.violations
        )

    def test_socket_import(self):
        code = "import socket\ns = socket.socket()"
        result = check_policy(code)
        assert not result.allowed

    def test_shutil_import(self):
        code = "import shutil\nshutil.rmtree('/tmp')"
        result = check_policy(code)
        assert not result.allowed

    def test_requests_import(self):
        code = "import requests\nrequests.get('http://evil.com')"
        result = check_policy(code)
        assert not result.allowed

    def test_pickle_import(self):
        code = "import pickle"
        result = check_policy(code)
        assert not result.allowed

    def test_from_subprocess_import(self):
        code = "from subprocess import run\nrun(['ls'])"
        result = check_policy(code)
        assert not result.allowed

    def test_importlib_import(self):
        code = "import importlib\nimportlib.import_module('os')"
        result = check_policy(code)
        assert not result.allowed

    def test_ctypes_import(self):
        code = "import ctypes"
        result = check_policy(code)
        assert not result.allowed

    def test_unknown_module_blocked(self):
        """Modules not in allowlist should be blocked."""
        code = "import pandas"
        result = check_policy(code)
        assert not result.allowed


class TestDangerousCalls:
    """Scripts with dangerous function calls that MUST be blocked."""

    def test_exec_call(self):
        code = "exec('print(1)')"
        result = check_policy(code)
        assert not result.allowed
        assert any(
            v.violation_type == PolicyViolationType.DANGEROUS_CALL
            for v in result.violations
        )

    def test_eval_call(self):
        code = "eval('1+1')"
        result = check_policy(code)
        assert not result.allowed

    def test_open_call(self):
        code = "f = open('/etc/passwd', 'r')"
        result = check_policy(code)
        assert not result.allowed

    def test_compile_call(self):
        code = "compile('print(1)', '<string>', 'exec')"
        result = check_policy(code)
        assert not result.allowed

    def test_dunder_import(self):
        code = "__import__('os')"
        result = check_policy(code)
        assert not result.allowed


class TestDangerousAttributes:
    """Scripts with dangerous attribute access that MUST be blocked."""

    def test_os_system(self):
        code = "import os\nos.system('ls')"
        result = check_policy(code)
        assert not result.allowed
        assert any(
            v.violation_type == PolicyViolationType.DANGEROUS_ATTRIBUTE
            for v in result.violations
        )

    def test_os_popen(self):
        code = "import os\nos.popen('ls')"
        result = check_policy(code)
        assert not result.allowed

    def test_os_remove(self):
        code = "import os\nos.remove('/tmp/file')"
        result = check_policy(code)
        assert not result.allowed

    def test_sys_exit(self):
        code = "import sys\nsys.exit(0)"
        # sys is not in allowlist, so import itself is blocked
        result = check_policy(code)
        assert not result.allowed


class TestSyntaxErrors:
    """Invalid Python code should be caught at parse time."""

    def test_syntax_error(self):
        code = "def foo(\n  return 1"
        result = check_policy(code)
        assert not result.allowed
        assert result.violations[0].violation_type == PolicyViolationType.SYNTAX_ERROR

    def test_empty_code(self):
        """Empty code is technically valid Python."""
        result = check_policy("")
        assert result.allowed


class TestPolicySummary:
    def test_pass_summary(self):
        result = PolicyResult(allowed=True, violations=[])
        assert "PASS" in result.summary()

    def test_blocked_summary(self):
        code = "import subprocess"
        result = check_policy(code)
        assert "BLOCKED" in result.summary()
        assert "1 violation" in result.summary()


class TestMultipleViolations:
    """Scripts with multiple violations should report all of them."""

    def test_multiple_violations(self):
        code = (
            "import subprocess\n"
            "import socket\n"
            "exec('print(1)')\n"
            "open('/etc/passwd')\n"
        )
        result = check_policy(code)
        assert not result.allowed
        assert len(result.violations) >= 4  # at least 4 violations
