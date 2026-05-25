import os
import httpx
from dotenv import load_dotenv

# Load các biến môi trường từ file .env
load_dotenv()

CHATWOOT_URL = os.getenv("CHATWOOT_URL", "")
CHATWOOT_API_KEY = os.getenv("CHATWOOT_API_KEY", "")


async def send_chatwoot_reply(account_id: int, conversation_id: int, content: str) -> bool:
    """
    Gửi câu trả lời từ AI Server ngược lại cho khách hàng thông qua Chatwoot API.
    
    CƠ CHẾ HOẠT ĐỘNG:
    - Khi server xử lý xong RAG và có câu trả lời từ LLM, ta cần đẩy câu trả lời này về giao diện chat của khách hàng.
    - Chatwoot cung cấp API để gửi tin nhắn mới vào một cuộc hội thoại (Conversation) cụ thể.
    - URL API dạng: {CHATWOOT_URL}/api/v1/accounts/{account_id}/conversations/{conversation_id}/messages
    - Headers: Cần truyền `api_access_token` để xác thực quyền gửi tin (đây có thể là Access Token của Agent hoặc Bot).
    - Body: 
        + `content`: Nội dung tin nhắn trả lời.
        + `message_type`: Kiểu tin nhắn. Ở đây ta chọn "outgoing" (tin nhắn đi từ phía bot/tổng đài viên gửi tới khách hàng).
    """
    if not CHATWOOT_URL or not CHATWOOT_API_KEY:
        print("[Chatwoot] Cảnh báo: Thiếu cấu hình CHATWOOT_URL hoặc CHATWOOT_API_KEY trong file .env.")
        return False

    # Định dạng chuẩn URL bằng cách xóa dấu gạch chéo '/' thừa ở cuối (nếu có)
    base_url = CHATWOOT_URL.rstrip("/")
    url = f"{base_url}/api/v1/accounts/{account_id}/conversations/{conversation_id}/messages"

    # Định nghĩa headers xác thực cho Chatwoot
    headers = {
        "api_access_token": CHATWOOT_API_KEY,
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    # Nội dung gửi đi
    payload = {
        "content": content,
        "message_type": "outgoing"  # 'outgoing' đại diện cho tin nhắn từ Bot/Agent gửi cho Khách hàng
    }

    print(f"[Chatwoot] Đang gửi phản hồi tới Conversation #{conversation_id} (Account #{account_id})...")

    # Sử dụng httpx gửi request bất đồng bộ (async) để không block luồng xử lý chính của FastAPI
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, headers=headers, timeout=10.0)
            
            # Chatwoot trả về HTTP 200 hoặc 201 khi tạo tin nhắn thành công
            if response.status_code in [200, 201]:
                print(f"[Chatwoot] Gửi phản hồi thành công!")
                return True
            else:
                print(f"[Chatwoot] Gửi phản hồi thất bại. HTTP Status: {response.status_code}")
                print(f"[Chatwoot] Chi tiết phản hồi lỗi: {response.text}")
                return False
        except Exception as e:
            print(f"[Chatwoot] Lỗi kết nối tới Chatwoot API: {str(e)}")
            return False
