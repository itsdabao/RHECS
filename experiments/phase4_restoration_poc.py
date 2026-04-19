import asyncio
import os
import json
from restoration.evidence_compiler import compile_evidence
from restoration.rewriter import fix_claim
from restoration.replacer import surgical_replace

async def restore_claim(draft_sentence: str, fault_span: str, error_type: str, claim_triplet: dict):
    """
    Async coroutine handling the isolated pipeline for a single Hallucinated Claim.
    Runs the Evidence Compiler synchronously within an isolated thread, then fires the GPU-bound API.
    """
    print(f"[Restoration] 🔍 Compiling evidence for: '{fault_span}'...")
    evidence = await asyncio.to_thread(compile_evidence, claim_triplet)
    
    if not evidence:
        print(f"[Restoration] ⚠️ No evidence found for '{fault_span}'. Cannot safely patch.")
        return None
        
    print(f"[Restoration] 🛠️ Rewriting ISO Patch for '{fault_span}' via Gemma-4-26b...")
    repair_instruction = await fix_claim(draft_sentence, fault_span, error_type, evidence)
    
    return {
        "fault_span": fault_span,
        "corrected_span": repair_instruction.corrected_span,
        "analysis": repair_instruction.analysis
    }

async def run_phase4(mock_error_map: dict):
    print("Initialize Phase 4: Surgical Restoration...")
    
    # 1. Dispatch independent claims to concurrent async workers
    tasks = []
    original_draft = mock_error_map["raw_draft"]
    
    for err in mock_error_map["errors"]:
        tasks.append(
            restore_claim(original_draft, err["fault_span"], err["error_type"], err["claim_triplet"])
        )
        
    # 2. Wait for all API operations (Golden Balance Architecture)
    repair_results = await asyncio.gather(*tasks)
    
    # 3. Synchronous Sequential Overwrite (Reduces Race Conditions on String State!)
    restored_draft = original_draft
    print("\n--- Applying ISO Patches ---")
    for patch in repair_results:
        if patch:
            print(f"🔧 Applying Fix: '{patch['fault_span']}' -> '{patch['corrected_span']}'")
            print(f"   [LLM Analysis]: {patch['analysis']}")
            try:
                restored_draft = surgical_replace(restored_draft, patch['fault_span'], patch['corrected_span'])
            except ValueError as e:
                print(e)
                
    print("\n================ FINAL RESTORED DRAFT ================")
    print(restored_draft)
    print("======================================================")

if __name__ == "__main__":
    # Mock output generated directly from Phase 3 Verification & Localization error mappings
    mock_phase3_output = {
        "raw_draft": "In 2023, OpenAI releases ChatGPT-4, and its safety protocol has been worsened, says Sam Altman. Gustave Eiffel established his reputation by building the external frame for the Statue of Liberty.",
        "errors": [
            {
                "fault_span": "worsened",
                "error_type": "Contradictory",
                "claim_triplet": {"entity": "OpenAI ChatGPT-4 safety protocol", "relationship": "has been", "target": "worsened"}
            },
            {
                "fault_span": "external frame",
                "error_type": "Entity Error",
                "claim_triplet": {"entity": "Gustave Eiffel", "relationship": "built", "target": "external frame for the Statue of Liberty"}
            }
        ]
    }
    
    asyncio.run(run_phase4(mock_phase3_output))
