import json
import os
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field
from typing import List, Optional
from rhecs_core.llm.model_router import generate_content_with_fallback_async

# --- 1. ĐỊNH NGHĨA SCHEMA ---
class ClaimMetadata(BaseModel):
    time: Optional[str] = Field(default=None)
    location: Optional[str] = Field(default=None)
    condition: Optional[str] = Field(default=None)

class AtomicClaim(BaseModel):
    entity: str
    relationship: str
    target: str
    metadata: ClaimMetadata
    status: str

class ClaimList(BaseModel):
    claims: List[AtomicClaim]

from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
google_client = genai.Client(api_key=api_key)

@retry(stop=stop_after_attempt(8), wait=wait_exponential(multiplier=2, max=30))
async def extract_vietnamese_claims(text: str) -> ClaimList:
    system_prompt = """Bạn là chuyên gia trích xuất thông tin tiếng Việt. 
    Nhiệm vụ của bạn là đọc văn bản, phân giải đại từ (ví dụ: thay 'nó' bằng danh từ gốc) và trích xuất các khẳng định dưới dạng Entity-Relationship-Target.
    
    Quy tắc BẮT BUỘC:
    1. MỌI thông tin trích xuất (entity, relationship, target, metadata) PHẢI giữ nguyên ngôn ngữ tiếng Việt. Tuyệt đối KHÔNG tự động dịch sang tiếng Anh.
    2. Nếu đại từ KHÔNG THỂ xác định rõ ràng ngữ cảnh (ví dụ: 'Anh ấy' không biết là ai), gán status là 'unresolved_ambiguity', ngược lại gán 'resolved'."""
    
    response = await generate_content_with_fallback_async(
        client=google_client,
        contents=text,
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
            result = extract_vietnamese_claims(item['draft_response'])
            print("Kết quả JSON:")
            print(result.model_dump_json(indent=2))
        except Exception as e:
            print(f"Lỗi khi xử lý: {e}")

if __name__ == "__main__":
    run_pipeline()