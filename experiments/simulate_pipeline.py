import asyncio
import json
import os
import sys

from dotenv import load_dotenv

# Tải API keys từ file .env TRƯỚC KHI load các module có sử dụng Google SDK
load_dotenv()

# Đẩy thư mục root của dự án (CS431) vào PYTHONPATH để Python nhận diện được package rhecs_core
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rhecs_core.pipeline import RHECSPipeline
from rhecs_core.verification.sandbox_helpers import setup_mock_qdrant

async def simulate_e2e_pipeline():
    print("==================================================")
    print("🚀 BẮT ĐẦU MÔ PHỎNG END-TO-END RHECS PIPELINE")
    print("==================================================\n")
    
    # 1. Khởi tạo Database giả lập cho bài kiểm tra
    print("[Simulation] Setup Mock Qdrant Database (Tenant: tenant_123)...")
    setup_mock_qdrant("tenant_123")

    if not os.getenv("GEMINI_MODEL_CANDIDATES"):
        os.environ["GEMINI_MODEL_CANDIDATES"] = "gemini-3-flash-preview,gemini-2.5-flash,gemini-2.0-flash,gemini-flash-latest"

    print(f"[Simulation] Model fallback chain: {os.getenv('GEMINI_MODEL_CANDIDATES')}")
    
    # 2. Định nghĩa Sample 
    # Câu 1: Chứa một lỗi Ảo giác (Hallucination) về Gustave Eiffel
    # Câu 2: Chứa sự thật hoàn toàn đúng (Không bị lỗi)
    dirty_draft = "Năm 1889, Gustave Eiffel đã xây dựng cỗ máy bay cho tháp Eiffel ở Paris. Ông cũng thiết kế bộ khung bên ngoài cho tượng Nữ thần Tự Do."
    
    print("\n[Simulation] VĂN BẢN ĐẦU VÀO (Có ảo giác):")
    print(f"\"{dirty_draft}\"\n")
    
    # 3. Khởi chạy Pipeline Engine
    engine = RHECSPipeline(tenant_id="tenant_123")
    
    try:
        # Pipeline sẽ tự động văng các print() log mô phỏng các Phase 2 -> 3 -> 4
        result = await engine.process_document(dirty_draft)
        
        # 4. Hiển thị Output 
        print("\n==================================================")
        print("🎯 KẾT QUẢ CUỐI CÙNG SAU KHI SỬA LỖI (RESTORED TEXT)")
        print("==================================================")
        print(f"\"{result['restored_text']}\"\n")
        
        print("📊 METRICS THỐNG KÊ:")
        print(json.dumps(result['metrics'], indent=2))
        
        print("\n📋 AUDIT TRAIL (Nhật ký Phẫu thuật chi tiết):")
        print(json.dumps(result['audit_trail'], indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"\n❌ Pipeline thất bại do cấu hình. Error: {e}")
        import traceback
        if hasattr(e, 'last_attempt'):
            traceback.print_exception(type(e.last_attempt.exception()), e.last_attempt.exception(), e.last_attempt.exception().__traceback__)
        else:
            traceback.print_exc()
        print("Bạn đã chắc chắn có GEMINI_API_KEY trong file .env và đã cài đặt thư viện 'google-genai' chưa?")

if __name__ == "__main__":
    asyncio.run(simulate_e2e_pipeline())
