import os
from fastapi import FastAPI, BackgroundTasks, HTTPException, Body, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Import các service của chúng ta
from rag_service import (
    async_ingest_document,
    async_ingest_product,
    async_generate_rag_response,
    async_qdrant_client,
    QDRANT_COLLECTION_NAME
)
from chatwoot_service import send_chatwoot_reply

# Load các biến môi trường
load_dotenv()

# Khởi tạo FastAPI App với Metadata đầy đủ (tốt cho SEO/Swagger API Docs)
app = FastAPI(
    title="Supplement Vector Data Product RAG System",
    description="Hệ thống Vector Data Product & RAG chuyên dụng cho Thực phẩm chức năng tích hợp Chatwoot.",
    version="1.0.0"
)


# --- PYDANTIC MODELS FOR REQUEST VALIDATION ---

class IngestRequest(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Công ty TNHH Vibe Code có trụ sở tại 123 Đường Láng, Hà Nội. Giờ làm việc từ 8:00 sáng đến 17:30 chiều, từ thứ Hai đến thứ Sáu hàng tuần."
            }
        }


class ProductIngestRequest(BaseModel):
    product_id: str
    name: str
    descriptionShort: str
    description: str
    ingredient: str
    type: str

    class Config:
        json_schema_extra = {
            "example": {
                "product_id": "sp_alipas_01",
                "name": "Sâm Alipas",
                "descriptionShort": "Tăng cường sinh lực nam giới, kéo dài sung mãn.",
                "description": "Sâm Alipas mới chứa các tinh chất quý hỗ trợ tăng cường testosterone nội sinh, làm chậm quá trình mãn dục nam và hỗ trợ cải thiện rối loạn cương dương.",
                "ingredient": "Mật nhân (Eurycoma Longifolia), Tinh chất thông biển Pháp, chiết xuất hàu đại dương.",
                "type": "Sinh lý nam"
            }
        }


# --- UTILITIES ---

async def process_rag_and_reply_chatwoot(account_id: int, conversation_id: int, query: str):
    """
    Hàm xử lý RAG bất đồng bộ chạy dưới nền (Background Task).
    """
    try:
        # Bước 1: Thực hiện luồng RAG để lấy câu trả lời từ LLM bất đồng bộ
        answer = await async_generate_rag_response(query)
        
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
        # Thử lấy thông tin collection để kiểm tra kết nối Qdrant bất đồng bộ
        await async_qdrant_client.collection_exists(QDRANT_COLLECTION_NAME)
        health_status["qdrant"] = "connected"
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["qdrant_error"] = str(e)
        
    if health_status["status"] == "healthy":
        return JSONResponse(content=health_status, status_code=status.HTTP_200_OK)
    return JSONResponse(content=health_status, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)


@app.post("/ingest", summary="Nhập tài liệu tri thức chung vào Vector DB")
async def ingest_data(payload: IngestRequest):
    """
    Endpoint nhận tài liệu dạng văn bản dài từ admin, thực hiện chunking, 
    embedding và lưu trữ các vector đại diện vào Qdrant bất đồng bộ.
    """
    if not payload.text or not payload.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Nội dung văn bản (text) không được để trống."
        )
        
    try:
        # Gọi rag_service bất đồng bộ để ingest tài liệu
        chunks_created = await async_ingest_document(payload.text)
        
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


@app.post("/ingest/product", summary="Nạp hoặc Cập nhật 1 sản phẩm TPCN có cấu trúc")
async def ingest_single_product(payload: ProductIngestRequest):
    """
    API endpoint nhận thông tin 1 sản phẩm từ Admin CMS để nạp/cập nhật vector vào Qdrant.
    Được thiết kế theo mô hình 'Delete-then-Insert' để chống trùng lặp dữ liệu.
    """
    try:
        chunks_created = await async_ingest_product(
            product_id=payload.product_id.strip(),
            name=payload.name.strip(),
            description_short=payload.descriptionShort.strip(),
            description=payload.description.strip(),
            ingredient=payload.ingredient.strip(),
            product_type=payload.type.strip()
        )
        return {
            "status": "success",
            "message": f"Nạp/Cập nhật sản phẩm '{payload.name}' thành công!",
            "chunks_created": chunks_created
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi hệ thống khi nạp sản phẩm '{payload.name}': {str(e)}"
        )


@app.post("/ingest/products/bulk", summary="Nạp hàng loạt nhiều sản phẩm TPCN (Bulk Import)")
async def ingest_bulk_products(payload: list[ProductIngestRequest]):
    """
    API endpoint nhận danh sách nhiều sản phẩm để nạp hàng loạt (phục vụ import 49 sản phẩm ban đầu từ Admin).
    """
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Danh sách sản phẩm nạp không được để trống."
        )
        
    try:
        total_chunks = 0
        success_products = []
        
        # Gọi lần lượt nạp bất đồng bộ từng sản phẩm
        for prod in payload:
            chunks_created = await async_ingest_product(
                product_id=prod.product_id.strip(),
                name=prod.name.strip(),
                description_short=prod.descriptionShort.strip(),
                description=prod.description.strip(),
                ingredient=prod.ingredient.strip(),
                product_type=prod.type.strip()
            )
            total_chunks += chunks_created
            success_products.append(prod.name)
            
        return {
            "status": "success",
            "message": f"Nạp hàng loạt thành công {len(payload)} sản phẩm!",
            "total_chunks_created": total_chunks,
            "products_imported": success_products
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi hệ thống khi nạp hàng loạt sản phẩm: {str(e)}"
        )


@app.post("/chatwoot/webhook", summary="Nhận webhook sự kiện từ Chatwoot")
async def chatwoot_webhook(background_tasks: BackgroundTasks, payload: dict = Body(...)):
    """
    Endpoint tiếp nhận webhook từ Chatwoot khi có tin nhắn mới.
    """
    event = payload.get("event")
    message_type = payload.get("message_type")
    is_private = payload.get("private", False)
    content = payload.get("content")
    
    print(f"[Webhook] Nhận sự kiện: {event} | Kiểu tin: {message_type} | Private: {is_private}")
    
    if event == "message_created" and message_type == "incoming" and not is_private:
        account_info = payload.get("account", {})
        account_id = account_info.get("id")
        
        conversation_info = payload.get("conversation", {})
        conversation_id = conversation_info.get("id")
        
        if not account_id:
            account_id = payload.get("account_id")
        if not conversation_id:
            conversation_id = payload.get("conversation_id")
            
        if not account_id or not conversation_id:
            print("[Webhook] Lỗi: Không trích xuất được account_id hoặc conversation_id.")
            return {"status": "ignored", "reason": "Missing identifiers"}
            
        if not content or not content.strip():
            print("[Webhook] Bỏ qua: Tin nhắn không chứa văn bản.")
            return {"status": "ignored", "reason": "Empty message content"}
            
        print(f"[Webhook] Nhận tin nhắn từ Conversation #{conversation_id}. Đang lên lịch xử lý ngầm...")
        background_tasks.add_task(
            process_rag_and_reply_chatwoot,
            account_id=account_id,
            conversation_id=conversation_id,
            query=content.strip()
        )
        
        return {"status": "processing", "message": "RAG task dispatched successfully"}
        
    else:
        reason = "Event or message type not targeted"
        if message_type == "outgoing":
            reason = "Ignored outgoing message to prevent infinite reply loop"
        elif is_private:
            reason = "Ignored private note"
            
        print(f"[Webhook] Bỏ qua sự kiện. Lý do: {reason}")
        return {"status": "ignored", "reason": reason}


# Khởi chạy uvicorn
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"--- Đang khởi động FastAPI Server tại port {port} ---")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

