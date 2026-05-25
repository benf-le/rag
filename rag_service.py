import os
import uuid
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Load các biến môi trường từ file .env
load_dotenv()

# Cấu hình từ file .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "knowledge_base")

# Khởi tạo OpenAI Client
# Thư viện OpenAI tự động đọc OPENAI_API_KEY từ biến môi trường,
# nhưng ta truyền tường minh để code rõ ràng và dễ học.
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Khởi tạo Qdrant Client
# - PHÂN BIỆT CLOUD & LOCAL:
#   + Chạy Qdrant Cloud (Platform Qdrant): Truyền cả QDRANT_URL (dạng https://...) và QDRANT_API_KEY.
#   + Chạy Qdrant Local bằng Docker: Chỉ cần truyền QDRANT_URL (http://localhost:6333) và bỏ trống API Key.
if QDRANT_API_KEY:
    print(f"[Qdrant] Đang kết nối tới Qdrant Cloud tại: {QDRANT_URL}")
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    print(f"[Qdrant] Đang kết nối tới Qdrant Local tại: {QDRANT_URL}")
    qdrant_client = QdrantClient(url=QDRANT_URL)


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """
    Chia nhỏ văn bản dài thành các đoạn (chunks) có kích thước cố định kèm độ gối đầu (overlap).
    
    TẠI SAO CẦN CHUNKING?
    1. Giới hạn ngữ cảnh (Context Window): Các mô hình LLM có giới hạn số lượng token đầu vào. Ta không thể gửi cả cuốn sách cho LLM được.
    2. Tìm kiếm chính xác hơn: Việc chia nhỏ giúp tìm đúng đoạn thông tin chứa câu trả lời, thay vì tìm kiếm trên một file tài liệu khổng lồ khiến kết quả bị loãng.
    3. Tránh mất thông tin: Độ gối đầu (overlap) giúp giữ ngữ cảnh liền mạch giữa các đoạn liền kề, tránh trường hợp câu văn quan trọng bị cắt đôi ở ranh giới phân mảnh.
    """
    chunks = []
    start = 0
    # Loại bỏ khoảng trắng thừa ở hai đầu
    text = text.strip()
    
    # Thực hiện cắt chuỗi đơn giản bằng số ký tự (character-based slicing)
    # Đây là phương án đơn giản, trực quan và dễ hiểu nhất để học tập.
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - chunk_overlap)
        
    return chunks


def get_embedding(text: str) -> list[float]:
    """
    Chuyển đổi một đoạn văn bản thành vector biểu diễn ngữ nghĩa (embedding).
    Sử dụng model: text-embedding-3-large (Mặc định trả về vector 3072 chiều).
    
    TẠI SAO CẦN EMBEDDING?
    - Máy tính và Vector Database không thể hiểu từ ngữ một cách trực tiếp như con người.
    - Embedding chuyển văn bản thành một chuỗi số (vector). Các từ/câu có nghĩa tương đồng nhau
      sẽ được ánh xạ thành các vector nằm gần nhau trong không gian đa chiều (Semantic Space).
    - Ví dụ: Vector của "Hà Nội" sẽ nằm gần vector của "Thủ đô Việt Nam" hơn là vector của "Trái chuối".
    """
    response = openai_client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding


def init_collection():
    """
    Khởi tạo collection trong Qdrant nếu chưa tồn tại.
    - Kích thước vector: 3072 (đặc trưng của model text-embedding-3-large).
    - Khoảng cách Metric sử dụng là Cosine Similarity (phổ biến nhất cho so khớp ngữ nghĩa văn bản).
    
    VECTOR DATABASE HOẠT ĐỘNG THẾ NÀO?
    - Vector DB lưu trữ các vectors cùng với metadata (ở đây là đoạn văn bản gốc).
    - Nó được tối ưu hóa để thực hiện các phép toán khoảng cách hình học cực nhanh trên hàng triệu vectors.
    - Khi ta tìm kiếm, nó không tìm từ khóa (như Ctrl+F hay SQL LIKE), mà nó tìm các vector có hướng gần trùng nhau nhất.
    """
    # Kiểm tra xem collection đã tồn tại chưa
    if not qdrant_client.collection_exists(collection_name=QDRANT_COLLECTION_NAME):
        print(f"[Qdrant] Collection '{QDRANT_COLLECTION_NAME}' chưa tồn tại. Tiến hành khởi tạo...")
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=3072,  # Kích thước vector 3072 chiều (text-embedding-3-large)
                distance=models.Distance.COSINE  # Sử dụng so khớp Cosine Similarity
            )
        )
        print(f"[Qdrant] Đã tạo thành công collection: {QDRANT_COLLECTION_NAME}")


def ingest_document(text: str) -> int:
    """
    Nhập tài liệu tri thức vào hệ thống RAG:
    Quy trình:
    1. Chia nhỏ văn bản dài thành các chunks.
    2. Với mỗi chunk, gọi OpenAI Embedding API để lấy vector 3072 chiều.
    3. Tạo cấu trúc Point (gồm ID, vector và payload chứa văn bản gốc).
    4. Lưu toàn bộ các Points vào Vector Database Qdrant.
    """
    # Đảm bảo collection đã được tạo
    init_collection()
    
    # Bước 1: Chia nhỏ văn bản
    chunks = chunk_text(text)
    print(f"[Ingest] Đã phân mảnh tài liệu thành {len(chunks)} đoạn nhỏ.")
    
    points = []
    # Bước 2: Tạo embedding cho từng đoạn
    for i, chunk in enumerate(chunks):
        print(f"[Ingest] Đang tạo embedding cho chunk {i+1}/{len(chunks)}...")
        vector = get_embedding(chunk)
        point_id = str(uuid.uuid4()) # Sinh ID duy nhất cho mỗi vector
        
        # Thêm Point vào danh sách
        points.append(
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload={"text": chunk} # Lưu văn bản gốc để hiển thị/làm ngữ cảnh sau này
            )
        )
    
    # Bước 3: Đẩy dữ liệu vào Qdrant
    print(f"[Qdrant] Đang lưu {len(points)} vectors vào database...")
    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        points=points
    )
    print("[Qdrant] Lưu thành công!")
    return len(chunks)


def search_similar_chunks(query: str, limit: int = 3) -> list[str]:
    """
    Tìm kiếm các đoạn văn bản có nghĩa gần nhất với câu hỏi:
    Quy trình:
    1. Sinh vector embedding cho câu hỏi của User.
    2. Sử dụng Qdrant Search để thực hiện Similarity Search (Tìm top vectors gần nhất).
    3. Trích xuất và trả về nội dung text gốc trong payload.
    """
    init_collection()
    
    # Bước 1: Sinh vector cho câu hỏi
    query_vector = get_embedding(query)
    
    # Bước 2: Truy vấn Qdrant
    search_result = qdrant_client.search(
        collection_name=QDRANT_COLLECTION_NAME,
        query_vector=query_vector,
        limit=limit
    )
    
    # Bước 3: Lấy ra text từ payload
    contexts = []
    for hit in search_result:
        contexts.append(hit.payload["text"])
        print(f"[Search] Tìm thấy chunk tương đồng (Score: {hit.score:.4f}): '{hit.payload['text'][:60]}...'")
        
    return contexts


def generate_rag_response(query: str) -> str:
    """
    Thực hiện luồng RAG (Retrieval-Augmented Generation) hoàn chỉnh:
    1. TÌM KIẾM (Retrieval): Gọi Vector DB tìm các ngữ cảnh liên quan nhất với câu hỏi.
    2. GHÉP PROMPT (Augmentation): Đưa các đoạn ngữ cảnh tìm được vào System Prompt mẫu (Hỗ trợ đa ngôn ngữ).
    3. SINH ĐÁP ÁN (Generation): Gửi Prompt hoàn chỉnh cho LLM (GPT-4o-mini) để sinh câu trả lời chuẩn xác.
    """
    print(f"[RAG] Bắt đầu xử lý câu hỏi: '{query}'")
    
    # 1. TÌM KIẾM
    contexts = search_similar_chunks(query, limit=3)
    
    if not contexts:
        context_text = "Không tìm thấy thông tin nào trong kho tri thức của hệ thống."
    else:
        context_text = "\n---\n".join(contexts)
        
    # 2. GHÉP PROMPT (Đa ngôn ngữ)
    # Prompt chỉ đạo LLM trả lời bằng chính ngôn ngữ mà người dùng hỏi
    system_prompt = (
        "Bạn là một Trợ lý ảo hỗ trợ khách hàng chuyên nghiệp, lịch sự và chu đáo.\n"
        "Nhiệm vụ của bạn là trả lời câu hỏi của khách hàng dựa trên NGUỒN NGỮ CẢNH (Context) được cung cấp dưới đây.\n"
        "Hãy tuân thủ nghiêm ngặt các nguyên tắc sau:\n"
        "1. ĐA NGÔN NGỮ (Multi-language): Hãy trả lời câu hỏi bằng CHÍNH NGÔN NGỮ mà khách hàng đã dùng để hỏi.\n"
        "   - Nếu khách hàng hỏi bằng tiếng Việt -> trả lời bằng tiếng Việt.\n"
        "   - Nếu khách hàng hỏi bằng tiếng Anh (English) -> dịch thông tin liên quan từ Context và trả lời bằng tiếng Anh.\n"
        "   - Nếu khách hàng hỏi bằng tiếng Nhật (Japanese) -> trả lời bằng tiếng Nhật.\n"
        "2. CHỈ sử dụng thông tin trong Nguồn Ngữ Cảnh để trả lời. Không được tự ý bịa đặt hoặc suy diễn thông tin ngoài.\n"
        "3. Trả lời ngắn gọn, súc tích và trực tiếp giải quyết câu hỏi của khách hàng.\n"
        "4. Nếu trong Nguồn Ngữ Cảnh KHÔNG chứa thông tin nào để trả lời câu hỏi, hãy phản hồi lịch sự bằng chính ngôn ngữ của người dùng:\n"
        "   - Ví dụ (Tiếng Việt): \"Dạ, xin lỗi anh/chị, hiện tại em chưa có thông tin về vấn đề này. Em sẽ ghi nhận để cập nhật sớm nhất ạ.\"\n"
        "   - Ví dụ (Tiếng Anh): \"I am sorry, but I do not have information about this issue right now. I will record it to update as soon as possible.\"\n\n"
        f"--- BẮT ĐẦU NGUỒN NGỮ CẢNH ---\n{context_text}\n--- KẾT THÚC NGUỒN NGỮ CẢNH ---"
    )
    
    # 3. SINH ĐÁP ÁN
    print("[LLM] Đang gọi OpenAI GPT-4o-mini...")
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0.3 # Giữ nhiệt độ thấp để LLM trả lời nghiêm túc, bám sát context, tránh bị ảo tưởng (hallucination)
    )
    
    answer = response.choices[0].message.content
    print("[RAG] Đã sinh ra câu trả lời thành công.")
    return answer
