import os
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel
from rhecs_core.llm.model_router import generate_content_with_fallback

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
google_client = genai.Client(api_key=api_key)

class PlannerOutput(BaseModel):
    python_script: str

def generate_verification_script(claim_data: dict, error_trace: str = None) -> str:
    system_prompt = """You are a highly secure Root Planner Agent for a Fact Verification Pipeline.
Your strictly bound task is to write a standalone Python script to search a database to find evidence verifying a given claim.

2. Strict Architectural Constraints:
3. 1. You MUST import the search helper exactly like this: `from rhecs_core.verification.sandbox_helpers import search_evidence`
2. You must call `search_evidence(query_string)` with a highly relevant semantic search string based on the claim.
3. Subprocess I/O Rule: You must NOT use `print()` arbitrarily. You are only allowed to print EXACTLY ONCE at the very end of your script. The output MUST be strictly valid JSON (e.g., `import json; print(json.dumps({"evidence": results}))`).
4. Do not wrap your response in markdown blocks like ```python. Return RAW code only.
"""

    user_prompt = f"Target Claim to Verify: {claim_data}\n"
    if error_trace:
        user_prompt += f"\nURGENT: Your previous script crashed inside the Subprocess Sandbox with this exact error trace. You MUST fix it:\\n{error_trace}\\n"
        
    response = generate_content_with_fallback(
        client=google_client,
        contents=user_prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=PlannerOutput,
        ),
        env_var="PLANNER_MODEL_CANDIDATES",
    )
    return PlannerOutput.model_validate_json(response.text).python_script
