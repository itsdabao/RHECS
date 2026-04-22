"""
NLI Judge Agent — evaluates claims against evidence.

Prompt template: Template 3 from 02-AGENT-PROMPTING.md.
Uses centralized prompts from rhecs_core.prompts.
"""

import os
from enum import Enum
from typing import Optional

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from rhecs_core.llm.model_router import generate_content_with_fallback_async
from rhecs_core.prompts import NLI_JUDGE_SYSTEM_PROMPT, build_nli_judge_user_prompt

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
google_client = genai.Client(api_key=api_key)


class NLIStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    NOT_MENTIONED = "NOT_MENTIONED"


class ErrorCategory(str, Enum):
    ENTITY_ERROR = "Entity Error"
    RELATION_ERROR = "Relation Error"
    CONTRADICTORY = "Contradictory"
    UNVERIFIABLE = "Unverifiable"
    FABRICATED = "Fabricated"


class VerificationResult(BaseModel):
    status: NLIStatus
    reasoning: str = Field(
        description="Step-by-step logic checking the claim against evidence"
    )
    error_category: Optional[ErrorCategory] = Field(
        default=None, description="Must be provided if status is CONTRADICTED"
    )
    fault_span: Optional[str] = Field(
        default=None,
        description="Exact phrase from the original sentence where the fault occurs",
    )
    confidence: Optional[float] = Field(
        default=None, description="Confidence score between 0.0 and 1.0"
    )


@retry(stop=stop_after_attempt(8), wait=wait_exponential(multiplier=2, min=3, max=30))
async def judge_evidence(
    original_sentence: str, triplet: dict, evidence: list
) -> VerificationResult:
    """
    Judge a claim against evidence using Template 3 (NLI Judge).

    Prompt features from 02-AGENT-PROMPTING.md:
    - Evidence-grounded reasoning only
    - Tie-break rule (near-equal confidence → NOT_MENTIONED)
    - Evidence conflict handling
    - Confidence scoring
    """
    evidence_list = evidence if isinstance(evidence, list) else [str(evidence)]

    user_prompt = build_nli_judge_user_prompt(
        original_sentence=original_sentence,
        claim_json=triplet,
        evidence_list=evidence_list,
    )

    response = await generate_content_with_fallback_async(
        client=google_client,
        contents=user_prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=NLI_JUDGE_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=VerificationResult,
        ),
        env_var="JUDGE_MODEL_CANDIDATES",
    )
    return VerificationResult.model_validate_json(response.text)
