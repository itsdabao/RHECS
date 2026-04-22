"""
WS4-02: Robust Surgical Replacement.

Improvements over PoC:
- Configurable fuzzy match threshold via environment variable
- Returns structured PatchResult instead of raising ValueError on failure
- Multiple matching strategies: exact → normalized → fuzzy
- Detailed logging for debugging
"""

import difflib
import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger("rhecs_core.restoration")

DEFAULT_FUZZY_THRESHOLD = 0.8


class PatchMethod(str, Enum):
    """How the fault_span was located."""

    EXACT = "exact"
    NORMALIZED = "normalized"
    FUZZY = "fuzzy"
    FAILED = "failed"


@dataclass
class PatchResult:
    """Structured result from surgical replacement."""

    success: bool
    patched_text: Optional[str] = None
    method: PatchMethod = PatchMethod.FAILED
    match_ratio: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "method": self.method.value,
            "match_ratio": self.match_ratio,
            "error": self.error,
        }


def _get_fuzzy_threshold() -> float:
    """Read fuzzy threshold from env var."""
    raw = os.environ.get("RHECS_FUZZY_THRESHOLD")
    if raw:
        try:
            value = float(raw)
            if 0 < value <= 1:
                return value
        except ValueError:
            pass
    return DEFAULT_FUZZY_THRESHOLD


def _normalize_whitespace(text: str) -> str:
    """Collapse all whitespace variants to single space."""
    return re.sub(r"\s+", " ", text).strip()


def _normalize_unicode(text: str) -> str:
    """Normalize unicode to NFC form for consistent comparison."""
    return unicodedata.normalize("NFC", text)


def _full_normalize(text: str) -> str:
    """Apply both unicode and whitespace normalization."""
    return _normalize_whitespace(_normalize_unicode(text))


def surgical_replace(draft: str, fault_span: str, corrected_span: str) -> str:
    """
    Isolated Span Overwrite (ISO) methodology.
    Replaces a specific hallucinated string within the draft document safely.

    WS4-02: Enhanced with multiple matching strategies.
    Raises ValueError only as a last resort for backward compatibility with pipeline.
    """
    result = surgical_replace_safe(draft, fault_span, corrected_span)
    if result.success:
        return result.patched_text
    raise ValueError(
        f"CRITICAL ISO FAILURE: Failed to locate fault_span '{fault_span}' "
        f"reliably within original draft. Method attempted: {result.method.value}. "
        f"Match ratio: {result.match_ratio}"
    )


def surgical_replace_safe(
    draft: str, fault_span: str, corrected_span: str
) -> PatchResult:
    """
    Safe version that returns PatchResult instead of raising.
    Tries multiple matching strategies in order:
    1. Exact match
    2. Normalized match (whitespace + unicode normalization)
    3. Fuzzy match (difflib with configurable threshold)
    """
    if not fault_span or not fault_span.strip():
        return PatchResult(
            success=False,
            method=PatchMethod.FAILED,
            error="fault_span is empty",
        )

    # Strategy 1: Exact match
    if fault_span in draft:
        patched = draft.replace(fault_span, corrected_span, 1)
        logger.debug("Patch applied via exact match")
        return PatchResult(
            success=True,
            patched_text=patched,
            method=PatchMethod.EXACT,
            match_ratio=1.0,
        )

    # Strategy 2: Normalized match (whitespace + unicode)
    normalized_draft = _full_normalize(draft)
    normalized_span = _full_normalize(fault_span)

    if normalized_span and normalized_span in normalized_draft:
        # Find the original position by searching character by character
        pos = _find_normalized_position(draft, fault_span)
        if pos is not None:
            start, end = pos
            patched = draft[:start] + corrected_span + draft[end:]
            logger.debug(
                "Patch applied via normalized match at position %d-%d", start, end
            )
            return PatchResult(
                success=True,
                patched_text=patched,
                method=PatchMethod.NORMALIZED,
                match_ratio=1.0,
            )

    # Strategy 3: Fuzzy match
    threshold = _get_fuzzy_threshold()
    matcher = difflib.SequenceMatcher(None, draft, fault_span)
    match = matcher.find_longest_match(0, len(draft), 0, len(fault_span))

    if len(fault_span) > 0:
        ratio = match.size / len(fault_span)
    else:
        ratio = 0.0

    if ratio >= threshold and match.size > max(len(fault_span) * threshold, 3):
        patched = draft[: match.a] + corrected_span + draft[match.a + match.size :]
        logger.debug(
            "Patch applied via fuzzy match (ratio=%.2f, threshold=%.2f)",
            ratio,
            threshold,
        )
        return PatchResult(
            success=True,
            patched_text=patched,
            method=PatchMethod.FUZZY,
            match_ratio=ratio,
        )

    # All strategies failed
    logger.warning(
        "All patch strategies failed for fault_span '%s' (best fuzzy ratio=%.2f, threshold=%.2f)",
        fault_span[:100],
        ratio,
        threshold,
    )
    return PatchResult(
        success=False,
        method=PatchMethod.FAILED,
        match_ratio=ratio,
        error=(
            f"Could not locate fault_span reliably. "
            f"Best fuzzy ratio: {ratio:.2f}, required: {threshold:.2f}"
        ),
    )


def _find_normalized_position(draft: str, fault_span: str) -> Optional[tuple[int, int]]:
    """
    Find the start and end position in the original draft text
    that corresponds to the normalized fault_span.
    """
    normalized_span = _full_normalize(fault_span)
    span_len = len(normalized_span)

    # Sliding window over normalized draft
    for start in range(len(draft)):
        for end in range(start + 1, min(start + len(fault_span) * 2, len(draft) + 1)):
            candidate = _full_normalize(draft[start:end])
            if candidate == normalized_span:
                return (start, end)

    return None
