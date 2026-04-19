import json
import asyncio
from verification.sandbox_helpers import setup_mock_qdrant
from verification.sandbox_manager import execute_sandbox_code
from verification.root_planner import generate_verification_script
from verification.nli_judge import judge_evidence

from typing import Optional

# Hard limit for preventing infinite sandbox LLM retry loops
MAX_RETRIES = 2

async def verify_claim(original_sentence: str, claim: dict, tenant_id: Optional[str] = None):
    print(f"\n--- Verifying Claim: {claim['entity']} -> {claim['relationship']} -> {claim['target']} ---")
    
    error_trace = None
    for attempt in range(MAX_RETRIES):
        print(f"[{claim['entity']}][Sandbox Attempt {attempt + 1}/{MAX_RETRIES}]")
        
        # 1. Root Planner generates script (sync OS call wrapped in to_thread)
        print(f"[{claim['entity']}] Root Planner generating verification Python script...")
        script_code = await asyncio.to_thread(generate_verification_script, claim, error_trace)
        print(f"[{claim['entity']}] Generated Code:\n{script_code}")
        
        # 2. Execute safely in Subprocess (sync blocking call wrapped in to_thread)
        print(f"[{claim['entity']}] Executing in OS-Isolated Sandbox...")
        sandbox_result = await asyncio.to_thread(execute_sandbox_code, script_code, tenant_id)
        
        if sandbox_result["success"]:
            # Sandbox passed perfectly
            evidence_data = sandbox_result["output"]
            print(f"[{claim['entity']}] Sandbox Raw Evidence JSON Extract:\n{evidence_data}")
            
            # 3. Context-Aware Sub-LLM Judging (Merged Reasoning API Call)
            print(f"[{claim['entity']}] Sending Context to NLI Judge (Gemini Async)...")
            evidence_list = evidence_data.get("evidence", evidence_data)
            
            verdict = await judge_evidence(original_sentence, claim, evidence_list)
            return verdict
            
        else:
            # Sandbox failed, caught traceback constraint
            error_trace = sandbox_result["error"]
            print(f"[{claim['entity']}] Sandbox Crashed. Error Trace:\n{error_trace}")
            print(f"[{claim['entity']}] Feeding trace back to Root Planner for Recursive Repair...")
            
    # If we exit the loop, we hit the MAX_RETRIES hard stop
    print(f"[{claim['entity']}] MAX_RETRIES Exceeded. Yielding Unverified Failure.")
    from verification.nli_judge import VerificationResult, NLIStatus, ErrorCategory
    return VerificationResult(
        status=NLIStatus.NOT_MENTIONED, 
        reasoning=f"Sandbox crashed repeatedly across {MAX_RETRIES} attempts. Final error: {error_trace}",
        error_category=ErrorCategory.UNVERIFIABLE,
        fault_span=None
    )

async def run_phase3():
    print("Initializing Unprivileged local Vector Database...")
    setup_mock_qdrant("tenant_123")
    
    # Simulating a payload pipeline extraction from Phase 2
    mock_payload = {
        "id": "claim_ext_01",
        "raw_draft": "Gustave Eiffel was a French civil engineer. He designed the Eiffel Tower, which is located in Paris. He also built the internal frame for the Statue of Liberty.",
        "claims": [
             {
                 "entity": "Gustave Eiffel",
                 "relationship": "built",
                 "target": "internal frame for the Statue of Liberty",
                 "status": "resolved"
             },
             {
                 "entity": "Eiffel Tower",
                 "relationship": "located in",
                 "target": "Paris",
                 "status": "resolved"
             }
        ]
    }
    
    print(f"Starting parallel verification of {len(mock_payload['claims'])} claims...")
    
    # Fire off verification tasks in parallel
    tasks = [
        verify_claim(mock_payload["raw_draft"], claim, tenant_id="tenant_123")
        for claim in mock_payload["claims"]
    ]
    
    results = await asyncio.gather(*tasks)
    
    print("\n=== FINAL TRIBUNAL VERDICTS (ERROR MAP) ===")
    error_map = []
    for claim, verdict in zip(mock_payload["claims"], results):
        claim_result = {
            "claim": claim,
            "verification": verdict.model_dump()
        }
        error_map.append(claim_result)
        print(json.dumps(claim_result, indent=2))
        
    # Later this error_map will be passed to Module 04

if __name__ == "__main__":
    asyncio.run(run_phase3())
