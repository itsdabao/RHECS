import json
import os
from enum import Enum
from typing import List, Optional

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field, model_validator

from rhecs_core.llm.model_router import generate_content_with_fallback_async

# --- 1. SCHEMA DEFINITIONS ---


class ClaimStatus(str, Enum):
    """WS3-01: Explicit enum for claim resolution status."""

    RESOLVED = "resolved"
    UNRESOLVED_AMBIGUITY = "unresolved_ambiguity"


class ClaimMetadata(BaseModel):
    """WS3-02: Strict metadata schema with controlled extra fields."""

    time: Optional[str] = Field(default=None)
    location: Optional[str] = Field(default=None)
    condition: Optional[str] = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def _strip_unknown_fields(cls, values):
        """WS3-02: Log warning and strip unknown metadata fields to prevent schema drift."""
        if not isinstance(values, dict):
            return values
        known_fields = {"time", "location", "condition"}
        unknown = set(values.keys()) - known_fields
        if unknown:
            import logging

            logging.getLogger("rhecs_core.extraction").warning(
                "Metadata contained unknown fields %s — stripped for schema stability",
                unknown,
            )
        return {k: v for k, v in values.items() if k in known_fields}


class AtomicClaim(BaseModel):
    """
    A single atomic claim extracted from text.

    WS3-01: status field is now a proper enum, and unresolved claims
    are explicitly preserved and tagged for downstream handling.
    """

    entity: str
    relationship: str
    target: str
    metadata: ClaimMetadata
    status: ClaimStatus = Field(
        default=ClaimStatus.RESOLVED,
        description="Whether the claim's pronouns/references were fully resolved",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_status(cls, values):
        """WS3-01: Normalize status to valid enum, defaulting to unresolved_ambiguity if invalid."""
        if not isinstance(values, dict):
            return values
        raw_status = values.get("status", "resolved")
        if isinstance(raw_status, str):
            normalized = raw_status.strip().lower()
            valid_values = {s.value for s in ClaimStatus}
            if normalized not in valid_values:
                import logging

                logging.getLogger("rhecs_core.extraction").warning(
                    "Unknown claim status '%s' — defaulting to unresolved_ambiguity",
                    raw_status,
                )
                values["status"] = ClaimStatus.UNRESOLVED_AMBIGUITY.value
        return values

    @property
    def is_resolved(self) -> bool:
        return self.status == ClaimStatus.RESOLVED

    @property
    def is_unresolved(self) -> bool:
        return self.status == ClaimStatus.UNRESOLVED_AMBIGUITY


class ClaimList(BaseModel):
    claims: List[AtomicClaim]

    @property
    def resolved_claims(self) -> List[AtomicClaim]:
        """WS3-01: Filter to only resolved claims."""
        return [c for c in self.claims if c.is_resolved]

    @property
    def unresolved_claims(self) -> List[AtomicClaim]:
        """WS3-01: Filter to only unresolved claims."""
        return [c for c in self.claims if c.is_unresolved]

    @property
    def unresolved_count(self) -> int:
        return len(self.unresolved_claims)


# --- 2. EXTRACTION FUNCTION ---

from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
google_client = genai.Client(api_key=api_key)


@retry(stop=stop_after_attempt(8), wait=wait_exponential(multiplier=2, max=30))
async def extract_vietnamese_claims(text: str) -> ClaimList:
    system_prompt = """Bạn là chuyên gia trích xuất thông tin tiếng Việt.
Nhiệm vụ: đọc văn bản và trích xuất các khẳng định nguyên tử dưới dạng Entity-Relationship-Target.

Yeu cau bat buoc ve output:
1. Chi tra ve JSON hop le, khong markdown, khong giai thich them.
2. Top-level PHAI la object co duy nhat key "claims".
3. "claims" PHAI la list (co the rong).
4. Moi phan tu trong "claims" PHAI co du cac field:
   - "entity": string
   - "relationship": string
   - "target": string
   - "metadata": object voi cac key "time", "location", "condition" (gia tri string hoac null)
   - "status": string, chi duoc la "resolved" hoac "unresolved_ambiguity"
5. Toan bo noi dung entity/relationship/target/metadata phai giu nguyen tieng Viet theo ngu canh, KHONG tu dong dich sang ngon ngu khac.
6. Neu khong trich xuat duoc claim nao, tra ve {"claims": []}.
7. Neu co dai tu khong ro rang (vi du: "no", "ho", "day") ma KHONG the xac dinh duoc thuc the goc, dat status = "unresolved_ambiguity" thay vi bo qua claim do.
"""

    user_prompt = f"""
[VAN_BAN]
{text}

Hay tra ve JSON theo dung schema da yeu cau.
"""

    response = await generate_content_with_fallback_async(
        client=google_client,
        contents=user_prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0,
            response_mime_type="application/json",
            response_schema=ClaimList,
        ),
        env_var="EXTRACTOR_MODEL_CANDIDATES",
    )
    return ClaimList.model_validate_json(response.text)


# --- 3. CHẠY TEST TỪ MOCK DATA ---
def run_pipeline():
    print("Đang tải mock data...")
    with open("mock_payload.json", "r", encoding="utf-8") as f:
        payloads = json.load(f)

    for item in payloads:
        print(f"\n--- Đang xử lý ID: {item['id']} ---")
        print(f"Văn bản gốc: {item['draft_response']}")

        try:
            result = extract_vietnamese_claims(item["draft_response"])
            print("Kết quả JSON:")
            print(result.model_dump_json(indent=2))
        except Exception as e:
            print(f"Lỗi khi xử lý: {e}")


if __name__ == "__main__":
    run_pipeline()
