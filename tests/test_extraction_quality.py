"""Tests for WS3: Extraction quality and unresolved handling."""

import pytest

from rhecs_core.extraction.extractor import (
    AtomicClaim,
    ClaimList,
    ClaimMetadata,
    ClaimStatus,
)


class TestClaimStatus:
    """WS3-01: Test claim status enum and normalization."""

    def test_resolved_status(self):
        claim = AtomicClaim(
            entity="Eiffel",
            relationship="built",
            target="tower",
            metadata=ClaimMetadata(),
            status=ClaimStatus.RESOLVED,
        )
        assert claim.is_resolved
        assert not claim.is_unresolved

    def test_unresolved_status(self):
        claim = AtomicClaim(
            entity="Eiffel",
            relationship="built",
            target="tower",
            metadata=ClaimMetadata(),
            status=ClaimStatus.UNRESOLVED_AMBIGUITY,
        )
        assert claim.is_unresolved
        assert not claim.is_resolved

    def test_status_from_string_resolved(self):
        data = {
            "entity": "X",
            "relationship": "Y",
            "target": "Z",
            "metadata": {},
            "status": "resolved",
        }
        claim = AtomicClaim.model_validate(data)
        assert claim.status == ClaimStatus.RESOLVED

    def test_status_from_string_unresolved(self):
        data = {
            "entity": "X",
            "relationship": "Y",
            "target": "Z",
            "metadata": {},
            "status": "unresolved_ambiguity",
        }
        claim = AtomicClaim.model_validate(data)
        assert claim.status == ClaimStatus.UNRESOLVED_AMBIGUITY

    def test_invalid_status_defaults_to_unresolved(self):
        """WS3-01: Unknown status values should default to unresolved_ambiguity."""
        data = {
            "entity": "X",
            "relationship": "Y",
            "target": "Z",
            "metadata": {},
            "status": "garbage_value",
        }
        claim = AtomicClaim.model_validate(data)
        assert claim.status == ClaimStatus.UNRESOLVED_AMBIGUITY

    def test_missing_status_defaults_to_resolved(self):
        """Default status when missing is resolved (backward compat)."""
        data = {
            "entity": "X",
            "relationship": "Y",
            "target": "Z",
            "metadata": {},
        }
        claim = AtomicClaim.model_validate(data)
        assert claim.status == ClaimStatus.RESOLVED


class TestClaimListFiltering:
    """WS3-01: Test claim list filtering by status."""

    def _make_claim(self, status: ClaimStatus) -> AtomicClaim:
        return AtomicClaim(
            entity="E",
            relationship="R",
            target="T",
            metadata=ClaimMetadata(),
            status=status,
        )

    def test_resolved_filter(self):
        cl = ClaimList(
            claims=[
                self._make_claim(ClaimStatus.RESOLVED),
                self._make_claim(ClaimStatus.UNRESOLVED_AMBIGUITY),
                self._make_claim(ClaimStatus.RESOLVED),
            ]
        )
        assert len(cl.resolved_claims) == 2
        assert len(cl.unresolved_claims) == 1
        assert cl.unresolved_count == 1

    def test_all_resolved(self):
        cl = ClaimList(
            claims=[
                self._make_claim(ClaimStatus.RESOLVED),
            ]
        )
        assert len(cl.resolved_claims) == 1
        assert len(cl.unresolved_claims) == 0

    def test_empty_claims(self):
        cl = ClaimList(claims=[])
        assert cl.resolved_claims == []
        assert cl.unresolved_claims == []
        assert cl.unresolved_count == 0


class TestClaimMetadataValidation:
    """WS3-02: Test metadata schema stability."""

    def test_normal_metadata(self):
        meta = ClaimMetadata(time="2024", location="Paris", condition=None)
        assert meta.time == "2024"
        assert meta.location == "Paris"
        assert meta.condition is None

    def test_unknown_fields_stripped(self):
        """WS3-02: Extra fields from LLM should be stripped without crashing."""
        data = {
            "time": "2024",
            "location": "Paris",
            "extra_field": "should_be_stripped",
            "another": 42,
        }
        meta = ClaimMetadata.model_validate(data)
        assert meta.time == "2024"
        assert meta.location == "Paris"
        assert not hasattr(meta, "extra_field")
        assert not hasattr(meta, "another")

    def test_empty_metadata(self):
        meta = ClaimMetadata.model_validate({})
        assert meta.time is None
        assert meta.location is None
        assert meta.condition is None

    def test_null_values(self):
        meta = ClaimMetadata(time=None, location=None, condition=None)
        assert meta.time is None
