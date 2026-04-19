import difflib

def surgical_replace(draft: str, fault_span: str, corrected_span: str) -> str:
    """
    Isolated Span Overwrite (ISO) methodology.
    Replaces a specific hallucinated string within the draft document safely.
    It attempts exact match first, falling back to difflib.SequenceMatcher to catch minor whitespace or formatting gaps.
    """
    if fault_span in draft:
        return draft.replace(fault_span, corrected_span, 1) # Only replace the first occurrence safely
        
    # Fallback to fuzzy match (whitespace or hidden unicode characters can break direct `in` matching when interacting with LLM APIs)
    matcher = difflib.SequenceMatcher(None, draft, fault_span)
    match = matcher.find_longest_match(0, len(draft), 0, len(fault_span))
    
    # 80% tolerance threshold
    if match.size > max(len(fault_span) * 0.8, 3): 
        return draft[:match.a] + corrected_span + draft[match.a + match.size:]
        
    # Failsafe abort. Better to not patch at all than to patch in the wrong place.
    raise ValueError(f"CRITICAL ISO FAILURE: Failed to locate fault_span '{fault_span}' reliably within original draft.")
