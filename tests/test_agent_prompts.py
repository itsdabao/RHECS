"""
Tests for agent prompt templates and quality checklist.

Validates all 4 prompt templates from 02-AGENT-PROMPTING.md
against the quality checklist and structural requirements.
"""

import pytest

from rhecs_core.prompts import (
    NLI_JUDGE_SYSTEM_PROMPT,
    PROMPT_QUALITY_CHECKLIST,
    REWRITER_SYSTEM_PROMPT,
    ROOT_PLANNER_SYSTEM_PROMPT,
    build_nli_judge_user_prompt,
    build_query_router_system_prompt,
    build_query_router_user_prompt,
    build_rewriter_user_prompt,
    build_root_planner_user_prompt,
    validate_prompt_checklist,
)


class TestQueryRouterPrompt:
    """Template 1: Query Router Agent."""

    def test_system_prompt_contains_hard_guards(self):
        prompt = build_query_router_system_prompt(
            max_total_calls=10,
            max_total_tokens=5000,
            max_total_time_ms=30000,
            max_recursion_depth=2,
        )
        assert "max_total_calls = 10" in prompt
        assert "max_total_tokens = 5000" in prompt
        assert "max_total_time_ms = 30000" in prompt
        assert "max_recursion_depth = 2" in prompt

    def test_system_prompt_contains_autonomy_policy(self):
        prompt = build_query_router_system_prompt()
        assert "toàn quyền" in prompt
        assert "tactical decision" in prompt

    def test_system_prompt_contains_json_schema(self):
        prompt = build_query_router_system_prompt()
        assert "strategy_used" in prompt
        assert "verdict" in prompt
        assert "trajectory" in prompt

    def test_user_prompt_includes_claim(self):
        prompt = build_query_router_user_prompt(
            claim_json={"entity": "Eiffel", "relationship": "built", "target": "tower"},
            original_sentence="Eiffel built the tower",
        )
        assert "Eiffel" in prompt
        assert "tower" in prompt

    def test_user_prompt_with_evidence_hints(self):
        prompt = build_query_router_user_prompt(
            claim_json={"entity": "X"},
            original_sentence="test",
            evidence_hints=["hint1", "hint2"],
        )
        assert "hint1" in prompt
        assert "hint2" in prompt


class TestRootPlannerPrompt:
    """Template 2: Root Planner."""

    def test_system_prompt_has_allowlist(self):
        assert "search_evidence" in ROOT_PLANNER_SYSTEM_PROMPT
        assert "import json" in ROOT_PLANNER_SYSTEM_PROMPT

    def test_system_prompt_has_denylist(self):
        assert "os.system" in ROOT_PLANNER_SYSTEM_PROMPT
        assert "subprocess" in ROOT_PLANNER_SYSTEM_PROMPT
        assert "socket" in ROOT_PLANNER_SYSTEM_PROMPT

    def test_system_prompt_has_io_rule(self):
        assert "MỘT LẦN" in ROOT_PLANNER_SYSTEM_PROMPT
        assert "json.dumps" in ROOT_PLANNER_SYSTEM_PROMPT

    def test_system_prompt_has_retry_policy(self):
        lower = ROOT_PLANNER_SYSTEM_PROMPT.lower()
        assert "retry" in lower or "sửa lỗi" in lower

    def test_user_prompt_basic(self):
        prompt = build_root_planner_user_prompt(
            claim_data={"entity": "test", "relationship": "is", "target": "working"},
        )
        assert "test" in prompt
        assert "working" in prompt

    def test_user_prompt_with_error_trace(self):
        prompt = build_root_planner_user_prompt(
            claim_data={"entity": "X"},
            error_trace="NameError: name 'foo' is not defined",
        )
        assert "NameError" in prompt
        assert "URGENT" in prompt or "BẮT BUỘC" in prompt


class TestNLIJudgePrompt:
    """Template 3: NLI Judge."""

    def test_system_prompt_has_evidence_grounding(self):
        assert "evidence" in NLI_JUDGE_SYSTEM_PROMPT.lower()
        assert (
            "suy diễn" in NLI_JUDGE_SYSTEM_PROMPT.lower()
            or "KHÔNG được suy diễn" in NLI_JUDGE_SYSTEM_PROMPT
        )

    def test_system_prompt_has_status_labels(self):
        assert "SUPPORTED" in NLI_JUDGE_SYSTEM_PROMPT
        assert "CONTRADICTED" in NLI_JUDGE_SYSTEM_PROMPT
        assert "NOT_MENTIONED" in NLI_JUDGE_SYSTEM_PROMPT

    def test_system_prompt_has_output_schema(self):
        assert "status" in NLI_JUDGE_SYSTEM_PROMPT
        assert "reasoning" in NLI_JUDGE_SYSTEM_PROMPT
        assert "fault_span" in NLI_JUDGE_SYSTEM_PROMPT
        assert "confidence" in NLI_JUDGE_SYSTEM_PROMPT

    def test_system_prompt_has_tie_break(self):
        assert (
            "gần nhau" in NLI_JUDGE_SYSTEM_PROMPT
            or "tie-break" in NLI_JUDGE_SYSTEM_PROMPT.lower()
        )

    def test_system_prompt_has_conflict_handling(self):
        assert "conflict" in NLI_JUDGE_SYSTEM_PROMPT.lower()

    def test_user_prompt_structure(self):
        prompt = build_nli_judge_user_prompt(
            original_sentence="Eiffel đã xây tháp",
            claim_json={"entity": "Eiffel", "relationship": "xây", "target": "tháp"},
            evidence_list=["Gustave Eiffel designed the tower"],
        )
        assert "[ORIGINAL_SENTENCE]" in prompt
        assert "[CLAIM]" in prompt
        assert "[EVIDENCE_LIST]" in prompt
        assert "Eiffel" in prompt

    def test_user_prompt_empty_evidence(self):
        prompt = build_nli_judge_user_prompt(
            original_sentence="test",
            claim_json={"entity": "X"},
            evidence_list=[],
        )
        assert "Không có evidence" in prompt


class TestRewriterPrompt:
    """Template 4: Rewriter Agent."""

    def test_system_prompt_has_minimality(self):
        assert (
            "TỐI THIỂU" in REWRITER_SYSTEM_PROMPT
            or "minimal" in REWRITER_SYSTEM_PROMPT.lower()
        )

    def test_system_prompt_no_full_rewrite(self):
        assert (
            "KHÔNG rewrite toàn" in REWRITER_SYSTEM_PROMPT
            or "không rewrite" in REWRITER_SYSTEM_PROMPT.lower()
        )

    def test_system_prompt_evidence_only(self):
        assert "KHÔNG chèn thông tin không có trong evidence" in REWRITER_SYSTEM_PROMPT

    def test_system_prompt_has_schema(self):
        assert "corrected_span" in REWRITER_SYSTEM_PROMPT
        assert "analysis" in REWRITER_SYSTEM_PROMPT
        assert "minimality_check" in REWRITER_SYSTEM_PROMPT

    def test_user_prompt_structure(self):
        prompt = build_rewriter_user_prompt(
            original_draft="Eiffel đã xây tháp năm 1900",
            fault_span="năm 1900",
            error_type="Entity Error",
            evidence_packet=["Eiffel Tower was completed in 1889"],
        )
        assert "[CONTEXT SENTENCE]" in prompt
        assert "[FAULT SPAN]" in prompt
        assert "[ERROR TYPE]" in prompt
        assert "[TRUE EVIDENCE]" in prompt
        assert "năm 1900" in prompt

    def test_user_prompt_with_metadata(self):
        prompt = build_rewriter_user_prompt(
            original_draft="test",
            fault_span="span",
            error_type="error",
            evidence_packet=["ev"],
            claim_metadata={"time": "2024"},
        )
        assert "[CLAIM METADATA]" in prompt
        assert "2024" in prompt


class TestPromptQualityChecklist:
    """Validate all prompts pass the quality checklist."""

    def test_query_router_passes_checklist(self):
        prompt = build_query_router_system_prompt()
        results = validate_prompt_checklist(prompt)
        # Must pass at least 10 of 12 checks
        passed = sum(1 for v in results.values() if v)
        assert passed >= 10, f"Only {passed}/12 checks passed: {results}"

    def test_root_planner_passes_checklist(self):
        results = validate_prompt_checklist(ROOT_PLANNER_SYSTEM_PROMPT)
        passed = sum(1 for v in results.values() if v)
        assert passed >= 8, f"Only {passed}/12 checks passed: {results}"

    def test_nli_judge_passes_checklist(self):
        results = validate_prompt_checklist(NLI_JUDGE_SYSTEM_PROMPT)
        passed = sum(1 for v in results.values() if v)
        assert passed >= 9, f"Only {passed}/12 checks passed: {results}"

    def test_rewriter_passes_checklist(self):
        results = validate_prompt_checklist(REWRITER_SYSTEM_PROMPT)
        passed = sum(1 for v in results.values() if v)
        assert passed >= 7, f"Only {passed}/12 checks passed: {results}"

    def test_checklist_has_all_items(self):
        assert len(PROMPT_QUALITY_CHECKLIST) == 12
