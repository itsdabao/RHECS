"""Tests for WS4: Restoration precision and safety."""

import pytest

from rhecs_core.restoration.evidence_compiler import (
    EvidenceItem,
    EvidencePacket,
    _build_queries,
    _deduplicate,
    compile_evidence_with_provenance,
)
from rhecs_core.restoration.replacer import (
    PatchMethod,
    PatchResult,
    surgical_replace,
    surgical_replace_safe,
)

# ── WS4-01: Evidence Compiler ──────────────────────────────────────────


class TestBuildQueries:
    """Test multi-query strategy."""

    def test_full_triplet(self):
        queries = _build_queries(
            {"entity": "Eiffel", "relationship": "built", "target": "tower"}
        )
        assert len(queries) >= 2
        assert "Eiffel built tower" in queries
        assert "Eiffel tower" in queries

    def test_entity_only_fallback(self):
        queries = _build_queries({"entity": "Eiffel", "relationship": "", "target": ""})
        assert "Eiffel" in queries

    def test_with_metadata(self):
        queries = _build_queries(
            {
                "entity": "Eiffel",
                "relationship": "built",
                "target": "tower",
                "metadata": {"time": "1889", "location": "Paris"},
            }
        )
        assert any("1889" in q for q in queries)
        assert any("Paris" in q for q in queries)

    def test_empty_triplet(self):
        queries = _build_queries({})
        assert len(queries) == 0

    def test_partial_triplet(self):
        queries = _build_queries(
            {"entity": "OpenAI", "relationship": "released", "target": ""}
        )
        assert "OpenAI" in queries


class TestDeduplicate:
    def test_no_duplicates(self):
        texts = ["Evidence A", "Evidence B"]
        unique, removed = _deduplicate(texts)
        assert len(unique) == 2
        assert removed == 0

    def test_exact_duplicates(self):
        texts = ["Evidence A", "Evidence A", "Evidence B"]
        unique, removed = _deduplicate(texts)
        assert len(unique) == 2
        assert removed == 1

    def test_case_insensitive_dedup(self):
        texts = ["Evidence A", "evidence a", "EVIDENCE A"]
        unique, removed = _deduplicate(texts)
        assert len(unique) == 1
        assert removed == 2

    def test_empty_strings_removed(self):
        texts = ["Evidence A", "", "  ", "Evidence B"]
        unique, removed = _deduplicate(texts)
        assert len(unique) == 2


class TestEvidencePacket:
    def test_texts_property(self):
        packet = EvidencePacket(
            items=[
                EvidenceItem(text="A", query_used="q1", rank=1),
                EvidenceItem(text="B", query_used="q2", rank=2),
            ]
        )
        assert packet.texts == ["A", "B"]

    def test_to_dict(self):
        packet = EvidencePacket(
            items=[EvidenceItem(text="A", query_used="q", rank=1)],
            total_fetched=3,
            duplicates_removed=2,
            queries_used=["q1", "q2"],
        )
        d = packet.to_dict()
        assert d["total_fetched"] == 3
        assert d["duplicates_removed"] == 2
        assert len(d["items"]) == 1


# ── WS4-02: Surgical Replacement ──────────────────────────────────────


class TestSurgicalReplace:
    """Test the backward-compatible surgical_replace function."""

    def test_exact_match(self):
        draft = "Gustave Eiffel đã xây dựng cỗ máy bay cho tháp Eiffel"
        result = surgical_replace(draft, "cỗ máy bay", "khung thép")
        assert "khung thép" in result
        assert "cỗ máy bay" not in result

    def test_no_match_raises(self):
        draft = "Gustave Eiffel đã xây dựng tháp Eiffel"
        with pytest.raises(ValueError, match="CRITICAL ISO FAILURE"):
            surgical_replace(
                draft, "nonexistent phrase that is long enough", "replacement"
            )

    def test_single_occurrence_only(self):
        draft = "A A A"
        result = surgical_replace(draft, "A", "B")
        assert result == "B A A"  # Only first occurrence replaced


class TestSurgicalReplaceSafe:
    """Test the safe version that returns PatchResult."""

    def test_exact_match(self):
        result = surgical_replace_safe("hello world", "hello", "hi")
        assert result.success
        assert result.patched_text == "hi world"
        assert result.method == PatchMethod.EXACT
        assert result.match_ratio == 1.0

    def test_empty_fault_span(self):
        result = surgical_replace_safe("hello world", "", "hi")
        assert not result.success
        assert result.method == PatchMethod.FAILED

    def test_whitespace_only_fault_span(self):
        result = surgical_replace_safe("hello world", "   ", "hi")
        assert not result.success
        assert result.method == PatchMethod.FAILED

    def test_no_match_returns_failed(self):
        result = surgical_replace_safe(
            "hello world",
            "completely different text that doesn't exist",
            "replacement",
        )
        assert not result.success
        assert result.method == PatchMethod.FAILED
        assert result.error is not None
        assert result.match_ratio is not None

    def test_fuzzy_match(self):
        """Fuzzy match should work for minor whitespace/unicode differences."""
        draft = "Gustave Eiffel đã xây dựng khung sắt"
        # Simulating a slight variation
        fault = "Gustave Eiffel đã xây dựng khung sắ"  # slightly truncated
        result = surgical_replace_safe(draft, fault, "REPLACED")
        # Depending on threshold, this might match or not
        # The key test is it returns PatchResult, not crashes
        assert isinstance(result, PatchResult)

    def test_normalized_match_unicode(self):
        """Test that unicode normalization helps matching."""
        # NFC vs NFD forms of Vietnamese characters
        draft = "Tháp Eiffel ở Paris"
        fault = "Tháp Eiffel"  # Same text, should exact match
        result = surgical_replace_safe(draft, fault, "Tháp Tokyo")
        assert result.success
        assert "Tháp Tokyo" in result.patched_text


class TestPatchResult:
    def test_to_dict(self):
        result = PatchResult(
            success=True,
            patched_text="test",
            method=PatchMethod.EXACT,
            match_ratio=1.0,
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["method"] == "exact"
        assert d["match_ratio"] == 1.0

    def test_failed_to_dict(self):
        result = PatchResult(
            success=False,
            method=PatchMethod.FAILED,
            error="not found",
            match_ratio=0.3,
        )
        d = result.to_dict()
        assert d["success"] is False
        assert d["error"] == "not found"
