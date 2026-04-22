"""
Root Planner Agent — generates verification scripts for sandbox execution.

Prompt template: Template 2 from 02-AGENT-PROMPTING.md.
Uses centralized prompts from rhecs_core.prompts.
"""

import os

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel

from rhecs_core.llm.model_router import generate_content_with_fallback
from rhecs_core.prompts import (
    ROOT_PLANNER_SYSTEM_PROMPT,
    build_root_planner_user_prompt,
)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
google_client = genai.Client(api_key=api_key)


class PlannerOutput(BaseModel):
    python_script: str


def generate_verification_script(claim_data: dict, error_trace: str = None) -> str:
    """
    Generate a Python verification script for a given claim.
    Uses Template 2 (Root Planner) from centralized prompts.
    """
    user_prompt = build_root_planner_user_prompt(
        claim_data=claim_data,
        error_trace=error_trace,
    )

    response = generate_content_with_fallback(
        client=google_client,
        contents=user_prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=ROOT_PLANNER_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=PlannerOutput,
        ),
        env_var="PLANNER_MODEL_CANDIDATES",
    )
    return PlannerOutput.model_validate_json(response.text).python_script
