"""
WS4-01: Evidence Compiler with quality controls.

Improvements over PoC:
- Multiple query strategies (entity+relation+target, entity-only, relationship-focused)
- Deduplication of evidence strings
- Provenance tracking for each evidence item
- Configurable top_k via environment variable
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from rhecs_core.verification.sandbox_helpers import search_evidence

logger = logging.getLogger("rhecs_core.restoration")

DEFAULT_EVIDENCE_TOP_K = 3


@dataclass
class EvidenceItem:
    """A single piece of evidence with provenance."""

    text: str
    query_used: str
    rank: int
    source: str = "qdrant"

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "query_used": self.query_used,
            "rank": self.rank,
            "source": self.source,
        }


@dataclass
class EvidencePacket:
    """Structured evidence result with provenance and dedup info."""

    items: list[EvidenceItem] = field(default_factory=list)
    total_fetched: int = 0
    duplicates_removed: int = 0
    queries_used: list[str] = field(default_factory=list)

    @property
    def texts(self) -> list[str]:
        """Return just the text strings for backward compatibility."""
        return [item.text for item in self.items]

    def to_dict(self) -> dict:
        return {
            "items": [item.to_dict() for item in self.items],
            "total_fetched": self.total_fetched,
            "duplicates_removed": self.duplicates_removed,
            "queries_used": self.queries_used,
        }


def _get_top_k() -> int:
    """Read evidence top_k from env var."""
    raw = os.environ.get("RHECS_EVIDENCE_TOP_K")
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return DEFAULT_EVIDENCE_TOP_K


def _build_queries(claim_triplet: dict) -> list[str]:
    """
    Build multiple search queries from a claim triplet for better recall.
    Returns queries in priority order.
    """
    entity = str(claim_triplet.get("entity", "")).strip()
    relationship = str(claim_triplet.get("relationship", "")).strip()
    target = str(claim_triplet.get("target", "")).strip()

    queries: list[str] = []

    # Primary: full triplet
    full_query = f"{entity} {relationship} {target}".strip()
    if full_query:
        queries.append(full_query)

    # Secondary: entity + target (skip relationship for broader match)
    if entity and target:
        queries.append(f"{entity} {target}")

    # Tertiary: entity-only for maximum recall
    if entity and entity not in queries:
        queries.append(entity)

    # Include metadata if available
    metadata = claim_triplet.get("metadata")
    if isinstance(metadata, dict):
        for key in ("time", "location"):
            value = metadata.get(key)
            if value and isinstance(value, str) and value.strip():
                meta_query = f"{entity} {value.strip()}"
                if meta_query not in queries:
                    queries.append(meta_query)

    return queries


def _deduplicate(texts: list[str]) -> tuple[list[str], int]:
    """Remove duplicate evidence strings, preserving order. Returns (deduped, count_removed)."""
    seen: set[str] = set()
    unique: list[str] = []
    for text in texts:
        normalized = text.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(text)
    return unique, len(texts) - len(unique)


def compile_evidence(claim_triplet: dict) -> list[str]:
    """
    Acts as the Evidence Compiler module for Phase 4.
    Re-fetches exact context relevant to the missing/failed claim using the Dual-Mode Qdrant filter.

    WS4-01 improvements:
    - Multiple query strategies for better recall
    - Deduplication
    - Provenance logging

    Returns list[str] for backward compatibility with pipeline.
    """
    packet = compile_evidence_with_provenance(claim_triplet)
    return packet.texts


def compile_evidence_with_provenance(claim_triplet: dict) -> EvidencePacket:
    """
    Full evidence compilation with provenance tracking.
    Use this when you need audit trail information.
    """
    top_k = _get_top_k()
    queries = _build_queries(claim_triplet)
    packet = EvidencePacket(queries_used=queries)

    all_texts: list[str] = []

    for query in queries:
        try:
            results = search_evidence(query=query, top_k=top_k)
            packet.total_fetched += len(results)
            all_texts.extend(results)
        except Exception as exc:
            logger.warning("Evidence search failed for query '%s': %s", query, exc)

    # Deduplicate
    unique_texts, dupes = _deduplicate(all_texts)
    packet.duplicates_removed = dupes

    # Build items with provenance
    for rank, text in enumerate(unique_texts, start=1):
        # Determine which query produced this result
        query_source = queries[0] if queries else "unknown"
        for q in queries:
            if q.lower() in text.lower() or any(
                word in text.lower() for word in q.lower().split()[:2]
            ):
                query_source = q
                break

        packet.items.append(
            EvidenceItem(
                text=text,
                query_used=query_source,
                rank=rank,
            )
        )

    if not packet.items:
        logger.warning(
            "No evidence found for claim: entity=%s, relationship=%s, target=%s",
            claim_triplet.get("entity"),
            claim_triplet.get("relationship"),
            claim_triplet.get("target"),
        )

    return packet
