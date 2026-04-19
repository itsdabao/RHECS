from rhecs_core.verification.sandbox_helpers import search_evidence

def compile_evidence(claim_triplet: dict) -> list[str]:
    """
    Acts as the Evidence Compiler module for Phase 4.
    Re-fetches exact context relevant to the missing/failed claim using the Dual-Mode Qdrant filter.
    """
    # Simply form a semantic string out of the triplet
    query_str = f"{claim_triplet.get('entity', '')} {claim_triplet.get('relationship', '')} {claim_triplet.get('target', '')}"
    
    # We leverage the locked search logic from the Sandbox
    evidence_list = search_evidence(query=query_str, top_k=2)
    return evidence_list
