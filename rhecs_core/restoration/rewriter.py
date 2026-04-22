"""
Rewriter Agent — generates surgical patches for hallucinated spans.

Prompt template: Template 4 from 02-AGENT-PROMPTING.md.
Uses centralized prompts from rhecs_core.prompts.
"""

import os
from typing import Optional

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from rhecs_core.llm.model_router import generate_content_with_fallback_async
from rhecs_core.prompts import REWRITER_SYSTEM_PROMPT, build_rewriter_user_prompt

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
google_client = genai.Client(api_key=api_key)


class RepairInstruction(BaseModel):
    analysis: str = Field(
        description="One brief sentence explaining the correction logic."
    )
    corrected_span: str = Field(
        description="The precise and minimal corrected string that will neatly drop into the original sentence."
    )
    minimality_check: Optional[str] = Field(
        default="PASS",
        description="PASS if corrected_span only fixes the error, FAIL if broader changes were needed.",
    )


@retry(stop=stop_after_attempt(8), wait=wait_exponential(multiplier=2, min=3, max=30))
async def fix_claim(
    sentence: str,
    fault_span: str,
    error_type: str,
    evidence: list[str],
    claim_metadata: Optional[dict] = None,
) -> RepairInstruction:
    """
    Generate a surgical patch for a hallucinated span using Template 4 (Rewriter).

    Prompt features from 02-AGENT-PROMPTING.md:
    - Minimal correction only (no full rewrite)
    - Evidence-only corrections (no external facts)
    - Minimality check audit
    - Grammatical/structural fit
    """
    user_prompt = build_rewriter_user_prompt(
        original_draft=sentence,
        fault_span=fault_span,
        error_type=error_type,
        evidence_packet=evidence,
        claim_metadata=claim_metadata,
    )

    response = await generate_content_with_fallback_async(
        client=google_client,
        contents=user_prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=REWRITER_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=RepairInstruction,
        ),
        env_var="REWRITER_MODEL_CANDIDATES",
    )
    return RepairInstruction.model_validate_json(response.text)
