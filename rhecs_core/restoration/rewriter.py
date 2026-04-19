import os
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from rhecs_core.llm.model_router import generate_content_with_fallback_async

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
google_client = genai.Client(api_key=api_key)

class RepairInstruction(BaseModel):
    analysis: str = Field(description="One brief sentence analyzing the grammatical tense and constraints of the surrounding context.")
    corrected_span: str = Field(description="The precise and minimal corrected string that will neatly drop into the original sentence.")

@retry(stop=stop_after_attempt(8), wait=wait_exponential(multiplier=2, min=3, max=30))
async def fix_claim(sentence: str, fault_span: str, error_type: str, evidence: list[str]) -> RepairInstruction:
    """
    Acts as the Rewriter Agent for Phase 4.
    Strictly outputs a surgical patch based on the compiled DB evidence.
     NEVER commands a full text rewrite to preserve the original author's voice and reduce latency.
    """
    evidence_text = "\n".join(evidence)
    
    system_prompt = """You are a highly analytical Restorer Agent for a RAG system.
Your goal is to surgically patch a HALLUCINATION within a sentence without rewriting the entirety of the original draft.
You are given the Context Sentence, the Exact Fault Span that is incorrect, the Error Type, and True Evidence.
Output a tiny `corrected_span` that structurally and grammatically fits into the exact hole left by `fault_span`.
"""

    user_prompt = f"""
[CONTEXT SENTENCE]
{sentence}

[FAULT SPAN] (This substring is hallucinated)
"{fault_span}"

[ERROR TYPE]
{error_type}

[TRUE EVIDENCE]
{evidence_text}

Generate the appropriate `corrected_span` based ONLY on the True Evidence.
"""

    response = await generate_content_with_fallback_async(
        client=google_client,
        contents=user_prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=RepairInstruction,
        ),
        env_var="REWRITER_MODEL_CANDIDATES",
    )
    return RepairInstruction.model_validate_json(response.text)
