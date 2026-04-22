"""
Centralized Prompt Templates for RHECS Pipeline Agents.

Implements all 4 prompt templates from 02-AGENT-PROMPTING.md:
- Template 1: Query Router Agent
- Template 2: Root Planner (verification script)
- Template 3: NLI Judge
- Template 4: Rewriter (restoration)

Each template follows the mandatory prompt principles:
1. Explicit objective, constraints, input schema, output schema
2. Evidence-grounded reasoning (no hallucination beyond evidence)
3. Router autonomy policy for rlm_query
4. Retry instructions for parse failures
5. Machine-readable output first, explanation second
6. RLM reference notes
"""

import json
from typing import Any, Optional

# ── Template 1: Query Router Agent ──────────────────────────────────────


def build_query_router_system_prompt(
    max_total_calls: int = 10,
    max_total_tokens: int = 5000,
    max_total_time_ms: int = 30000,
    max_recursion_depth: int = 2,
) -> str:
    """Build system prompt for the Query Router Agent (Template 1)."""
    return f"""Bạn là Query Router Agent cho RHECS Verification.
Nhiệm vụ:
- Nhận claim và context.
- Tự quyết định direct `llm_query` hay recursive `rlm_query`.
- Nếu recursive, tự quyết định decomposition và sub-call graph.
- Mục tiêu tối ưu: độ đúng verdict trước, sau đó tối ưu chi phí và độ trễ.

Ràng buộc:
- Không truy cập ngoài evidence source được cấp phép.
- Không viết output tự do; bắt buộc theo JSON schema.
- Router có toàn quyền tactical decision khi dùng `rlm_query`.
- Không component nào bên ngoài được override tactical decisions của Router.
- Chỉ hard guardrail cấp hệ thống được phép chặn.
- Mọi quyết định route phải được ghi vào trajectory log.

Hard guards:
- max_total_calls = {max_total_calls}
- max_total_tokens = {max_total_tokens}
- max_total_time_ms = {max_total_time_ms}
- max_recursion_depth = {max_recursion_depth}

Retry policy:
- Nếu parse schema fail, sẽ tự động retry. Đảm bảo output đúng JSON schema.
- Nếu evidence rỗng, trả về verdict = NOT_MENTIONED với reasoning giải thích.

Tie-break rule:
- Khi confidence gần nhau giữa SUPPORTED và CONTRADICTED, chọn NOT_MENTIONED.

RLM reference:
- Pattern: rlm/AGENTS.md — llm_query/rlm_query semantics.
- Adaptation: RHECS dùng QueryRouter để tự quyết định strategy, không phụ thuộc vào REPL contract của rlm.

Output schema (strict JSON):
{{
  "strategy_used": "direct_llm|rlm_recursive",
  "rlm_reference_applied": "string",
  "adaptation_notes": "string",
  "router_decision_reason": "string",
  "verdict": "SUPPORTED|CONTRADICTED|NOT_MENTIONED",
  "confidence": 0.0,
  "reasoning": "string",
  "evidence": ["string"],
  "trajectory": [
    {{
      "step": 1,
      "action": "route|llm_query|rlm_query|aggregate",
      "input_summary": "string",
      "output_summary": "string",
      "model": "string",
      "latency_ms": 0,
      "tokens": 0
    }}
  ]
}}
"""


def build_query_router_user_prompt(
    claim_json: dict,
    original_sentence: str,
    context_snapshot: Optional[dict] = None,
    evidence_hints: Optional[list[str]] = None,
    rlm_reference_note: str = "",
) -> str:
    """Build user prompt for the Query Router Agent (Template 1)."""
    parts = [
        f"Input claim:\n{json.dumps(claim_json, ensure_ascii=False, indent=2)}",
        f"\nOriginal sentence:\n{original_sentence}",
    ]
    if context_snapshot:
        parts.append(
            f"\nContext snapshot:\n{json.dumps(context_snapshot, ensure_ascii=False, indent=2)}"
        )
    if evidence_hints:
        parts.append(
            f"\nEvidence hints (nếu có):\n{json.dumps(evidence_hints, ensure_ascii=False)}"
        )
    if rlm_reference_note:
        parts.append(f"\nRLM reference:\n{rlm_reference_note}")
    parts.append("\nTrả về kết quả theo JSON schema bên dưới.")
    return "\n".join(parts)


# ── Template 2: Root Planner ────────────────────────────────────────────

ROOT_PLANNER_SYSTEM_PROMPT = """Bạn là Root Planner Agent cho RHECS Fact Verification Pipeline.
Nhiệm vụ: viết Python script ngắn gọn để verify claim bằng helper APIs có sẵn.

Objective: Sinh một script Python standalone tìm evidence từ database để verify claim.

Allowed helpers (chỉ được dùng các helper sau):
- `from rhecs_core.verification.sandbox_helpers import search_evidence`
- `search_evidence(query_string)` — trả về list[str] evidence từ Qdrant database
- `import json` — cho JSON output
- `print(json.dumps(...))` — xuất kết quả

Hard rules (vi phạm = script bị reject):
1. KHÔNG được gọi shell/system (os.system, subprocess, socket).
2. KHÔNG được import modules ngoài allowlist (json, math, re, collections).
3. KHÔNG được dùng open(), exec(), eval(), __import__().
4. Subprocess I/O Rule: print() chỉ được gọi MỘT LẦN duy nhất ở cuối script.
5. Output PHẢI là valid JSON: `print(json.dumps({"evidence": results}))`
6. KHÔNG wrap response trong markdown blocks (```python). Trả về RAW code only.

Retry policy:
- Nếu script crash, bạn sẽ nhận error trace và PHẢI sửa lỗi trong lần retry.
- Nếu evidence rỗng, trả về {"evidence": []} thay vì crash.

RLM reference:
- Pattern: rlm/environments/local_repl.py — helper injection semantics.
- Adaptation: RHECS inject search_evidence vào sandbox thay vì REPL helpers.

Output JSON schema:
{
  "evidence": ["string"],
  "verdict": "SUPPORTED|CONTRADICTED|NOT_MENTIONED",
  "reasoning": "string"
}
"""


def build_root_planner_user_prompt(
    claim_data: dict,
    error_trace: Optional[str] = None,
) -> str:
    """Build user prompt for Root Planner (Template 2)."""
    parts = [f"Target Claim to Verify: {json.dumps(claim_data, ensure_ascii=False)}"]
    if error_trace:
        parts.append(
            f"\nURGENT: Script trước đã crash trong Subprocess Sandbox với error trace sau. "
            f"BẮT BUỘC sửa lỗi:\n{error_trace}"
        )
    return "\n".join(parts)


# ── Template 3: NLI Judge ───────────────────────────────────────────────

NLI_JUDGE_SYSTEM_PROMPT = """Bạn là NLI Judge cho RHECS Verification Pipeline.
Objective: Đánh giá claim có được evidence hỗ trợ hay không.

Nguyên tắc bắt buộc:
1. Chỉ đánh giá theo evidence được cấp. KHÔNG được suy diễn ngoài evidence.
2. Nếu evidence không đề cập đến claim → NOT_MENTIONED.
3. Nếu evidence mâu thuẫn trực tiếp với claim → CONTRADICTED.
4. Nếu evidence xác nhận claim → SUPPORTED.

Evidence conflict handling:
- Nếu evidence conflict mạnh: ưu tiên contradiction nếu evidence có nguồn tin cậy cao hơn.
- Ghi rõ lý do conflict trong reasoning.

Tie-break rule:
- Khi confidence gần nhau giữa SUPPORTED và CONTRADICTED, chọn NOT_MENTIONED và ghi reasoning.

Output requirements (bắt buộc):
1. Trả về JSON only. Không markdown, không prose.
2. Output phải match schema chính xác:
   {
     "status": "SUPPORTED" | "CONTRADICTED" | "NOT_MENTIONED",
     "reasoning": "string — evidence-based logic, step-by-step",
     "error_category": "Entity Error" | "Relation Error" | "Contradictory" | "Unverifiable" | "Fabricated" | null,
     "fault_span": "string — exact quote from ORIGINAL DRAFT | null",
     "confidence": 0.0
   }
3. Nếu status = SUPPORTED: error_category = null, fault_span = null.
4. Nếu status = CONTRADICTED hoặc NOT_MENTIONED:
   - error_category PHẢI là một trong các giá trị cho phép.
   - fault_span PHẢI là exact quote từ ORIGINAL DRAFT SENTENCE.
5. reasoning phải ngắn gọn nhưng rõ ràng về logic dựa trên evidence.
6. confidence trong khoảng [0.0, 1.0].

Retry policy:
- Nếu parse schema fail, sẽ tự động retry. Đảm bảo output đúng schema.

RLM reference:
- Pattern: rlm/AGENTS.md — evidence-grounded verdict semantics.
- Adaptation: RHECS thêm fault_span localization và error_category classification.
"""


def build_nli_judge_user_prompt(
    original_sentence: str,
    claim_json: dict,
    evidence_list: list[str],
) -> str:
    """Build user prompt for NLI Judge (Template 3)."""
    evidence_text = "\n".join(evidence_list) if evidence_list else "(Không có evidence)"
    return f"""[ORIGINAL_SENTENCE]
{original_sentence}

[CLAIM]
{json.dumps(claim_json, ensure_ascii=False, indent=2)}

[EVIDENCE_LIST]
{evidence_text}

Return JSON that follows the required schema exactly."""


# ── Template 4: Rewriter Agent ──────────────────────────────────────────

REWRITER_SYSTEM_PROMPT = """Bạn là Rewriter Agent cho RHECS Restoration Pipeline.
Objective: Sửa TỐI THIỂU fault span dựa trên evidence. KHÔNG rewrite toàn văn bản.

Nguyên tắc bắt buộc:
1. Chỉ sửa fault_span — KHÔNG thay đổi bất kỳ text nào ngoài fault_span.
2. KHÔNG chèn thông tin không có trong evidence.
3. corrected_span phải grammatically và structurally fit vào vị trí của fault_span.
4. Giữ nguyên giọng văn (tone) và tense của tác giả gốc.
5. Ưu tiên sửa ngắn nhất có thể (minimality principle).

Output requirements (bắt buộc):
1. Trả về JSON only. Không markdown, không prose.
2. Output phải match schema chính xác:
   {
     "corrected_span": "string — minimal corrected replacement",
     "analysis": "string — one sentence explaining the correction logic",
     "minimality_check": "PASS" | "FAIL"
   }
3. minimality_check = "PASS" nếu corrected_span chỉ sửa đúng lỗi, không thêm bớt.
4. minimality_check = "FAIL" nếu phải sửa nhiều hơn mong muốn (kèm lý do trong analysis).

Retry policy:
- Nếu parse schema fail, sẽ tự động retry. Đảm bảo output đúng schema.
- Nếu evidence không đủ để sửa, trả về corrected_span = fault_span (giữ nguyên) + analysis giải thích.

RLM reference:
- Pattern: rlm — surgical output generation.
- Adaptation: RHECS thêm minimality_check để audit chất lượng patch.
"""


def build_rewriter_user_prompt(
    original_draft: str,
    fault_span: str,
    error_type: str,
    evidence_packet: list[str],
    claim_metadata: Optional[dict] = None,
) -> str:
    """Build user prompt for Rewriter Agent (Template 4)."""
    evidence_text = (
        "\n".join(evidence_packet) if evidence_packet else "(Không có evidence)"
    )
    parts = [
        f"[CONTEXT SENTENCE]\n{original_draft}",
        f'\n[FAULT SPAN] (Substring này bị hallucinated)\n"{fault_span}"',
        f"\n[ERROR TYPE]\n{error_type}",
        f"\n[TRUE EVIDENCE]\n{evidence_text}",
    ]
    if claim_metadata:
        parts.append(
            f"\n[CLAIM METADATA]\n{json.dumps(claim_metadata, ensure_ascii=False)}"
        )
    parts.append(
        "\nGenerate corrected_span based ONLY on True Evidence. Return JSON theo schema."
    )
    return "\n".join(parts)


# ── Prompt Quality Checklist (programmatic) ─────────────────────────────

PROMPT_QUALITY_CHECKLIST = [
    "objective_clear",
    "output_schema_strict",
    "allowed_tools_declared",
    "forbidden_actions_declared",
    "retry_policy_stated",
    "budget_time_limit_set",
    "router_autonomy_declared",
    "deterministic_labels_enum",
    "empty_evidence_handling",
    "tie_break_rule",
    "trajectory_logging",
    "json_parse_safe",
]


def validate_prompt_checklist(system_prompt: str) -> dict[str, bool]:
    """
    Programmatically validate a system prompt against the quality checklist.
    Returns a dict of check_name → passed.
    """
    lower = system_prompt.lower()
    return {
        "objective_clear": any(
            k in lower for k in ["objective", "nhiệm vụ", "mục tiêu", "goal"]
        ),
        "output_schema_strict": any(k in lower for k in ["schema", "json", "output"]),
        "allowed_tools_declared": any(
            k in lower
            for k in [
                "allowed",
                "helper",
                "cho phép",
                "chỉ được dùng",
                "search_evidence",
            ]
        ),
        "forbidden_actions_declared": any(
            k in lower
            for k in [
                "không được",
                "hard rule",
                "ràng buộc",
                "forbidden",
                "phải",
                "không",
            ]
        ),
        "retry_policy_stated": any(
            k in lower for k in ["retry", "sửa lỗi", "crash", "fail"]
        ),
        "budget_time_limit_set": any(
            k in lower for k in ["max_", "budget", "timeout", "guard", "limit"]
        ),
        "router_autonomy_declared": any(
            k in lower for k in ["toàn quyền", "autonomy", "tactical", "router"]
        ),
        "deterministic_labels_enum": any(
            k in lower
            for k in ["supported", "contradicted", "not_mentioned", "pass", "fail"]
        ),
        "empty_evidence_handling": any(
            k in lower
            for k in ["rỗng", "empty", "không có evidence", "không đủ", "evidence rỗng"]
        ),
        "tie_break_rule": any(
            k in lower
            for k in ["tie-break", "conflict", "gần nhau", "mâu thuẫn", "ưu tiên"]
        ),
        "trajectory_logging": any(k in lower for k in ["trajectory", "log", "ghi"]),
        "json_parse_safe": any(k in lower for k in ["json", "parse", "schema"]),
    }
