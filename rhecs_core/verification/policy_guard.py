"""
WS2-01: AST-based Policy Guard for Sandbox Execution.

Parses untrusted Python code via `ast` module and rejects scripts
containing dangerous imports, function calls, or attribute accesses
before they reach subprocess execution.

RLM reference applied:
- Pattern: rlm uses local exec with soft sandbox (rlm/environments/local_repl.py).
- Adaptation: RHECS adds a hard pre-exec AST guard layer that rlm does not have,
  because RHECS runs LLM-generated code in a subprocess with tenant isolation.
- No coupling to rlm private APIs.
"""

import ast
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class PolicyViolationType(str, Enum):
    """Classification of policy violations."""

    DANGEROUS_IMPORT = "dangerous_import"
    DANGEROUS_CALL = "dangerous_call"
    DANGEROUS_ATTRIBUTE = "dangerous_attribute"
    SYNTAX_ERROR = "syntax_error"
    AST_DEPTH_EXCEEDED = "ast_depth_exceeded"


@dataclass
class PolicyViolation:
    """A single policy violation found in submitted code."""

    violation_type: PolicyViolationType
    description: str
    line: Optional[int] = None
    col: Optional[int] = None


@dataclass
class PolicyResult:
    """Result of a policy guard check."""

    allowed: bool
    violations: list[PolicyViolation] = field(default_factory=list)

    def summary(self) -> str:
        if self.allowed:
            return "PASS: code complies with sandbox policy."
        details = "; ".join(
            f"[L{v.line}] {v.violation_type.value}: {v.description}"
            for v in self.violations
        )
        return f"BLOCKED: {len(self.violations)} violation(s) — {details}"


# ── Denylist: modules and functions that must never appear ──────────────

DENIED_MODULES: set[str] = {
    "subprocess",
    "shutil",
    "signal",
    "ctypes",
    "multiprocessing",
    "threading",
    "socket",
    "http",
    "urllib",
    "requests",
    "ftplib",
    "smtplib",
    "webbrowser",
    "code",
    "codeop",
    "compileall",
    "importlib",
    "pkgutil",
    "zipimport",
    "pickle",
    "shelve",
    "marshal",
    "tempfile",
}

DENIED_ATTRIBUTE_CHAINS: set[str] = {
    "os.system",
    "os.popen",
    "os.exec",
    "os.execl",
    "os.execle",
    "os.execlp",
    "os.execlpe",
    "os.execv",
    "os.execve",
    "os.execvp",
    "os.execvpe",
    "os.spawn",
    "os.spawnl",
    "os.spawnle",
    "os.spawnlp",
    "os.spawnlpe",
    "os.spawnv",
    "os.spawnve",
    "os.spawnvp",
    "os.spawnvpe",
    "os.fork",
    "os.forkpty",
    "os.kill",
    "os.killpg",
    "os.remove",
    "os.unlink",
    "os.rmdir",
    "os.rename",
    "os.makedirs",
    "os.mkdir",
    "os.chmod",
    "os.chown",
    "os.listdir",
    "os.walk",
    "os.scandir",
    "sys.exit",
    "sys._exit",
    "builtins.__import__",
}

DENIED_BUILTINS: set[str] = {
    "exec",
    "eval",
    "compile",
    "__import__",
    "globals",
    "locals",
    "breakpoint",
    "exit",
    "quit",
}

# ── Allowlist: modules and functions known safe ─────────────────────────

ALLOWED_MODULES: set[str] = {
    "json",
    "math",
    "re",
    "collections",
    "itertools",
    "functools",
    "operator",
    "string",
    "textwrap",
    "datetime",
    "copy",
    "typing",
    "dataclasses",
    "enum",
    "statistics",
    "decimal",
    "fractions",
    "rhecs_core",
    "rhecs_core.verification.sandbox_helpers",
}

MAX_AST_DEPTH = 50


# ── AST Visitor ─────────────────────────────────────────────────────────


class _PolicyVisitor(ast.NodeVisitor):
    """Walks the AST tree and collects policy violations."""

    def __init__(self) -> None:
        self.violations: list[PolicyViolation] = []
        self._depth = 0

    def _pos(self, node: ast.AST) -> dict:
        return {
            "line": getattr(node, "lineno", None),
            "col": getattr(node, "col_offset", None),
        }

    def _add(self, vtype: PolicyViolationType, desc: str, node: ast.AST) -> None:
        self.violations.append(
            PolicyViolation(
                violation_type=vtype,
                description=desc,
                **self._pos(node),
            )
        )

    def generic_visit(self, node: ast.AST) -> None:
        self._depth += 1
        if self._depth > MAX_AST_DEPTH:
            self.violations.append(
                PolicyViolation(
                    violation_type=PolicyViolationType.AST_DEPTH_EXCEEDED,
                    description=f"AST depth {self._depth} exceeds limit {MAX_AST_DEPTH}",
                )
            )
            self._depth -= 1
            return
        super().generic_visit(node)
        self._depth -= 1

    # ── Import checks ───────────────────────────────────────────────

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            top_module = alias.name.split(".")[0]
            if alias.name in DENIED_MODULES or top_module in DENIED_MODULES:
                self._add(
                    PolicyViolationType.DANGEROUS_IMPORT,
                    f"import '{alias.name}' is denied",
                    node,
                )
            elif top_module not in ALLOWED_MODULES and top_module != "os":
                # os is allowed partially (for os.environ only)
                self._add(
                    PolicyViolationType.DANGEROUS_IMPORT,
                    f"import '{alias.name}' is not in the allowlist",
                    node,
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        top_module = module.split(".")[0]

        if module in DENIED_MODULES or top_module in DENIED_MODULES:
            self._add(
                PolicyViolationType.DANGEROUS_IMPORT,
                f"from '{module}' import is denied",
                node,
            )
        elif top_module not in ALLOWED_MODULES and top_module != "os":
            self._add(
                PolicyViolationType.DANGEROUS_IMPORT,
                f"from '{module}' import is not in the allowlist",
                node,
            )
        self.generic_visit(node)

    # ── Attribute access checks ─────────────────────────────────────

    def visit_Attribute(self, node: ast.Attribute) -> None:
        chain = _resolve_attribute_chain(node)
        if chain:
            for denied in DENIED_ATTRIBUTE_CHAINS:
                if chain == denied or chain.startswith(denied + "."):
                    self._add(
                        PolicyViolationType.DANGEROUS_ATTRIBUTE,
                        f"attribute access '{chain}' is denied",
                        node,
                    )
                    break
        self.generic_visit(node)

    # ── Call checks (builtins) ──────────────────────────────────────

    def visit_Call(self, node: ast.Call) -> None:
        func_name = _resolve_call_name(node)
        if func_name in DENIED_BUILTINS:
            self._add(
                PolicyViolationType.DANGEROUS_CALL,
                f"call to '{func_name}' is denied",
                node,
            )

        # Check open() — only allow with known safe modes
        if func_name == "open":
            self._add(
                PolicyViolationType.DANGEROUS_CALL,
                "open() is denied in sandbox; use search_evidence() for data access",
                node,
            )

        self.generic_visit(node)


# ── Helpers ─────────────────────────────────────────────────────────────


def _resolve_attribute_chain(node: ast.Attribute) -> Optional[str]:
    """Resolve a.b.c attribute chain to a dotted string."""
    parts: list[str] = [node.attr]
    current = node.value
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


def _resolve_call_name(node: ast.Call) -> Optional[str]:
    """Resolve the function name from a Call node."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return _resolve_attribute_chain(node.func)
    return None


# ── Public API ──────────────────────────────────────────────────────────


def check_policy(code: str) -> PolicyResult:
    """
    Parse and validate Python code against the sandbox execution policy.

    Returns a PolicyResult indicating whether the code is allowed to execute
    and any violations found.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return PolicyResult(
            allowed=False,
            violations=[
                PolicyViolation(
                    violation_type=PolicyViolationType.SYNTAX_ERROR,
                    description=f"SyntaxError: {exc.msg}",
                    line=exc.lineno,
                    col=exc.offset,
                )
            ],
        )

    visitor = _PolicyVisitor()
    visitor.visit(tree)

    return PolicyResult(
        allowed=len(visitor.violations) == 0,
        violations=visitor.violations,
    )
