import os
from fastapi import FastAPI, BackgroundTasks, HTTPException, Body, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Import các service của chúng ta
from rag_service import ingest_document, generate_rag_response, qdrant_client, QDRANT_COLLECTION_NAME
from chatwoot_service import send_chatwoot_reply

# Load các biến môi trường
load_dotenv()

# Khởi tạo FastAPI App với Metadata đầy đủ (tốt cho SEO/Swagger API Docs)
app = FastAPI(
    title="Simple RAG Chatwoot Base System",
    description="Hệ thống RAG cơ bản tích hợp Chatwoot phục vụ học tập và nghiên cứu flow RAG end-to-end.",
    version="1.0.0"
)


# --- PYDANTIC MODELS FOR REQUEST VALIDATION ---

class IngestRequest(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Công ty TNHH Vibe Code có trụ sở tại 123 Đường Láng, Hà Nội. Giờ làm việc từ 8:00 sáng đến 17:30 chiều, từ thứ Hai đến thứ Sáu hàng tuần. Số hotline hỗ trợ khách hàng là 1900-xxxx. Chính sách đổi trả hàng áp dụng trong vòng 7 ngày kể từ ngày nhận hàng với điều kiện sản phẩm còn nguyên tem mác."
            }
        }


# --- UTILITIES ---

async def process_rag_and_reply_chatwoot(account_id: int, conversation_id: int, query: str):
    """
    Hàm xử lý RAG bất đồng bộ chạy dưới nền (Background Task).
    
    TẠI SAO CẦN DÙNG BACKGROUND TASKS?
    - Webhook của Chatwoot (và hầu hết các nền tảng khác) yêu cầu phản hồi HTTP 200 OK cực kỳ nhanh (thường dưới 2-3 giây).
    - Quá trình RAG (Embedding câu hỏi -> Vector Search -> Ghép Prompt -> LLM Sinh đáp án) có thể tốn từ 2 đến 7 giây tùy tốc độ mạng và OpenAI API.
    - Nếu xử lý đồng bộ trực tiếp trong Webhook API, Chatwoot sẽ bị timeout và tự động gửi lại webhook nhiều lần (gây ra vòng lặp vô hạn và spam tin nhắn).
    - Sử dụng FastAPI `BackgroundTasks` giúp trả về HTTP 200 OK ngay lập tức cho Chatwoot, rồi chạy ngầm RAG và gửi phản hồi sau.
    """
    try:
        # Bước 1: Thực hiện luồng RAG để lấy câu trả lời từ LLM
        answer = generate_rag_response(query)
        
        # Bước 2: Gửi câu trả lời ngược lại Chatwoot qua API
        await send_chatwoot_reply(account_id, conversation_id, answer)
    except Exception as e:
        print(f"[Background Task] Đã xảy ra lỗi khi xử lý RAG & gửi phản hồi: {str(e)}")


# --- API ENDPOINTS ---

@app.get("/health", summary="Kiểm tra trạng thái hệ thống và kết nối DB")
async def health_check():
    """
    Kiểm tra trạng thái hoạt động của Server FastAPI và kết nối tới Vector DB Qdrant.
    """
    health_status = {
        "status": "healthy",
        "openai_api": "configured" if os.getenv("OPENAI_API_KEY") else "missing",
        "chatwoot_api": "configured" if os.getenv("CHATWOOT_API_KEY") else "missing",
        "qdrant": "disconnected"
    }
    
    try:
        # Thử lấy thông tin collection để kiểm tra kết nối Qdrant
        qdrant_client.collection_exists(QDRANT_COLLECTION_NAME)
        health_status["qdrant"] = "connected"
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["qdrant_error"] = str(e)
        
    if health_status["status"] == "healthy":
        return JSONResponse(content=health_status, status_code=status.HTTP_200_OK)
    return JSONResponse(content=health_status, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)


@app.post("/ingest", summary="Nhập tài liệu tri thức vào Vector DB")
async def ingest_data(payload: IngestRequest):
    """
    Endpoint nhận tài liệu dạng văn bản dài từ admin, thực hiện chunking, 
    embedding và lưu trữ các vector đại diện vào Qdrant.
    """
    if not payload.text or not payload.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Nội dung văn bản (text) không được để trống."
        )
        
    try:
        # Gọi rag_service để ingest tài liệu
        chunks_created = ingest_document(payload.text)
        
        return {
            "status": "success",
            "message": "Nạp tài liệu tri thức thành công!",
            "chunks_created": chunks_created
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi hệ thống khi nạp tài liệu: {str(e)}"
        )


@app.post("/chatwoot/webhook", summary="Nhận webhook sự kiện từ Chatwoot")
async def chatwoot_webhook(background_tasks: BackgroundTasks, payload: dict = Body(...)):
    """
    Endpoint tiếp nhận webhook từ Chatwoot khi có tin nhắn mới.
    
    LUỒNG XỬ LÝ WEBHOOK:
    1. Kiểm tra sự kiện: chỉ xử lý sự kiện tạo tin nhắn (`message_created`).
    2. Kiểm tra nguồn tin nhắn: chỉ xử lý tin nhắn đi vào (`incoming` - từ khách hàng). 
       BỎ QUA tin nhắn đi ra (`outgoing` - từ bot/agent) và tin nhắn nội bộ (private) để tránh lặp vô hạn.
    3. Trích xuất: ID tài khoản Chatwoot, ID cuộc hội thoại, và nội dung câu hỏi.
    4. Kích hoạt Background Task để thực hiện luồng RAG và gửi phản hồi mà không bắt Chatwoot chờ đợi.
    """
    # 1. Trích xuất các thông tin cốt lõi từ webhook payload
    event = payload.get("event")
    message_type = payload.get("message_type")
    is_private = payload.get("private", False)
    content = payload.get("content")
    
    print(f"[Webhook] Nhận sự kiện: {event} | Kiểu tin: {message_type} | Private: {is_private}")
    
    # 2. Kiểm tra điều kiện để AI xử lý tin nhắn
    # - Sự kiện phải là tạo tin nhắn mới
    # - Tin nhắn phải từ khách hàng gửi đến (incoming)
    # - Không phải là ghi chú nội bộ (private)
    if event == "message_created" and message_type == "incoming" and not is_private:
        # Trích xuất thông tin định danh
        account_info = payload.get("account", {})
        account_id = account_info.get("id")
        
        conversation_info = payload.get("conversation", {})
        conversation_id = conversation_info.get("id")
        
        # Nếu payload định dạng dẹt (tùy phiên bản Chatwoot)
        if not account_id:
            account_id = payload.get("account_id")
        if not conversation_id:
            conversation_id = payload.get("conversation_id")
            
        # Kiểm tra tính hợp lệ của dữ liệu nhận được
        if not account_id or not conversation_id:
            print("[Webhook] Lỗi: Không trích xuất được account_id hoặc conversation_id.")
            return {"status": "ignored", "reason": "Missing identifiers"}
            
        if not content or not content.strip():
            print("[Webhook] Bỏ qua: Tin nhắn không chứa văn bản (có thể là hình ảnh hoặc file đính kèm).")
            return {"status": "ignored", "reason": "Empty message content"}
            
        # 3. Kích hoạt Background Task để xử lý RAG & gửi trả lời
        print(f"[Webhook] Nhận tin nhắn hợp lệ từ Conversation #{conversation_id}. Đang lên lịch xử lý ngầm...")
        background_tasks.add_task(
            process_rag_and_reply_chatwoot,
            account_id=account_id,
            conversation_id=conversation_id,
            query=content.strip()
        )
        
        return {"status": "processing", "message": "RAG task dispatched successfully"}
        
    else:
        # Bỏ qua các sự kiện không liên quan (ví dụ tin nhắn bot gửi đi, đổi trạng thái phòng chat, v.v.)
        reason = "Event or message type not targeted"
        if message_type == "outgoing":
            reason = "Ignored outgoing message to prevent infinite reply loop"
        elif is_private:
            reason = "Ignored private note"
            
        print(f"[Webhook] Bỏ qua sự kiện. Lý do: {reason}")
        return {"status": "ignored", "reason": reason}


# Khởi chạy trực tiếp bằng python main.py để thuận tiện chạy thử
if __name__ == "__main__":
    import uvicorn
    # Đọc cấu hình Port từ .env
    port = int(os.getenv("PORT", 8000))
    print(f"--- Đang khởi động FastAPI Server tại port {port} ---")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
