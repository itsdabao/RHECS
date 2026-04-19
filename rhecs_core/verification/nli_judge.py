import os
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from rhecs_core.llm.model_router import generate_content_with_fallback_async

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
    reasoning: str = Field(description="Step-by-step logic checking the claim against evidence")
    error_category: Optional[ErrorCategory] = Field(default=None, description="Must be provided if status is CONTRADICTED")
    fault_span: Optional[str] = Field(default=None, description="Exact phrase from the original sentence where the fault occurs")

@retry(stop=stop_after_attempt(8), wait=wait_exponential(multiplier=2, min=3, max=30))
async def judge_evidence(original_sentence: str, triplet: dict, evidence: list) -> VerificationResult:
    system_prompt = """You are a strict Context-Aware NLI Judge and Fault Localizer.
Your duty is to evaluate the Extracted Claim against Database Evidence with Merged Reasoning:
1. Provide a step-by-step 'reasoning' analysis.
2. Output a 'status' (SUPPORTED, CONTRADICTED, NOT_MENTIONED).
3. If CONTRADICTED or NOT_MENTIONED, identify the 'error_category'.
4. Most crucially, if there is a fault, provide the exact 'fault_span' directly quoted from the ORIGINAL DRAFT SENTENCE to localize the hallucination.
"""
    evidence_text = "\n".join(evidence) if isinstance(evidence, list) else str(evidence)
    
    user_prompt = f"""
[ORIGINAL DRAFT SENTENCE]
{original_sentence}

[EXTRACTED CLAIM TRIPLET]
{triplet}

[DATABASE EVIDENCE]
{evidence_text}

Provide your detailed reasoning, verdict, and if required, the exact fault span.
"""

    response = await generate_content_with_fallback_async(
        client=google_client,
        contents=user_prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=VerificationResult,
        ),
        env_var="JUDGE_MODEL_CANDIDATES",
    )
    return VerificationResult.model_validate_json(response.text)
