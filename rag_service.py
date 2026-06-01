import os
import uuid
from dotenv import load_dotenv
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
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

# Khởi tạo OpenAI Client bất đồng bộ
async_openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Khởi tạo Qdrant Client bất đồng bộ
if QDRANT_API_KEY:
    print(f"[Qdrant] Đang kết nối async tới Qdrant Cloud tại: {QDRANT_URL}")
    async_qdrant_client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    print(f"[Qdrant] Đang kết nối async tới Qdrant Local tại: {QDRANT_URL}")
    async_qdrant_client = AsyncQdrantClient(url=QDRANT_URL)


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> list[str]:
    """
    Chia nhỏ văn bản dài thành các đoạn (chunks) có kích thước cố định kèm độ gối đầu (overlap).
    Sử dụng cơ chế cắt chuỗi đơn giản theo số ký tự.
    """
    chunks = []
    start = 0
    text = text.strip()
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - chunk_overlap)
        
    return chunks


async def async_get_embedding(text: str) -> list[float]:
    """
    Chuyển đổi một đoạn văn bản thành vector biểu diễn ngữ nghĩa (embedding) bất đồng bộ.
    """
    response = await async_openai_client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding


async def async_get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Tạo embedding cho danh sách các đoạn văn bản cùng một lúc (Batching).
    """
    if not texts:
        return []
    response = await async_openai_client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL
    )
    return [data.embedding for data in response.data]


async def async_init_collection():
    """
    Khởi tạo collection trong Qdrant bất đồng bộ nếu chưa tồn tại.
    - Kích thước vector: 3072 (text-embedding-3-large).
    - Sử dụng Cosine Similarity.
    """
    exists = await async_qdrant_client.collection_exists(collection_name=QDRANT_COLLECTION_NAME)
    if not exists:
        print(f"[Qdrant] Collection '{QDRANT_COLLECTION_NAME}' chưa tồn tại. Tiến hành khởi tạo...")
        await async_qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=3072,
                distance=models.Distance.COSINE
            )
        )
        print(f"[Qdrant] Đã tạo thành công collection: {QDRANT_COLLECTION_NAME}")


async def async_delete_product_vectors(product_id: str):
    """
    Xóa tất cả các point có payload.product_id trùng khớp với product_id để tránh trùng lặp/rác dữ liệu.
    """
    await async_init_collection()
    print(f"[Qdrant] Đang xóa toàn bộ các vector cũ của product_id: '{product_id}'...")
    await async_qdrant_client.delete(
        collection_name=QDRANT_COLLECTION_NAME,
        points_selector=models.Filter(
            must=[
                models.FieldCondition(
                    key="product_id",
                    match=models.MatchValue(value=product_id)
                )
            ]
        )
    )
    print(f"[Qdrant] Đã dọn dẹp xong vector cũ cho product_id: '{product_id}'.")


async def async_ingest_product(
    product_id: str,
    name: str,
    description_short: str,
    description: str,
    ingredient: str,
    product_type: str
) -> int:
    """
    Nạp dữ liệu sản phẩm có cấu trúc từ Admin vào Qdrant (Xóa trước, Nạp sau).
    """
    # 1. Khởi tạo collection nếu chưa có
    await async_init_collection()
    
    # 2. Thực hiện xóa toàn bộ vector cũ của product_id này để tránh rác dữ liệu
    await async_delete_product_vectors(product_id)
    
    # 3. Ráp dữ liệu thành Structured Markdown để giữ ngữ cảnh đầy đủ cho mỗi chunk
    full_structured_text = (
        f"Sản phẩm: {name}\n"
        f"Phân loại: {product_type}\n"
        f"Mô tả ngắn: {description_short}\n"
        f"Thành phần chính: {ingredient}\n"
        f"Chi tiết sản phẩm: {description}"
    )
    
    # 4. Phân mảnh (chunking) thông minh
    chunks = chunk_text(full_structured_text, chunk_size=1000, chunk_overlap=100)
    print(f"[Ingest] Đã phân mảnh sản phẩm '{name}' thành {len(chunks)} đoạn.")
    
    # 5. Tạo Embeddings hàng loạt (Batching) cho tất cả các chunk cùng lúc
    print(f"[Ingest] Đang sinh embedding hàng loạt cho {len(chunks)} chunks...")
    vectors = await async_get_embeddings_batch(chunks)
    
    # 6. Tạo PointStruct để upsert vào Qdrant kèm Metadata phong phú
    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        point_id = str(uuid.uuid4())
        points.append(
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "text": chunk,
                    "product_id": product_id,
                    "product_name": name,
                    "product_type": product_type,
                    "chunk_index": i
                }
            )
        )
        
    # 7. Upsert hàng loạt vào Qdrant
    print(f"[Qdrant] Đang lưu batch {len(points)} vectors cho sản phẩm '{name}'...")
    await async_qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        points=points
    )
    print(f"[Qdrant] Đồng bộ thành công sản phẩm '{name}'!")
    return len(chunks)


async def async_ingest_document(text: str) -> int:
    """
    Nhập tài liệu tri thức văn bản thô vào Qdrant bất đồng bộ.
    """
    await async_init_collection()
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
    print(f"[Ingest] Đã phân mảnh tài liệu thành {len(chunks)} đoạn nhỏ.")
    
    vectors = await async_get_embeddings_batch(chunks)
    
    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        point_id = str(uuid.uuid4())
        points.append(
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "text": chunk,
                    "product_id": "general_knowledge",
                    "product_name": "Tài liệu chung"
                }
            )
        )
        
    await async_qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        points=points
    )
    print(f"[Qdrant] Lưu thành công tài liệu tri thức chung!")
    return len(chunks)


async def async_search_similar_chunks(query: str, limit: int = 3) -> list[str]:
    """
    Tìm kiếm các đoạn văn bản có nghĩa gần nhất với câu hỏi bất đồng bộ.
    """
    await async_init_collection()
    
    # Bước 1: Sinh vector cho câu hỏi
    query_vector = await async_get_embedding(query)
    
    # Bước 2: Truy vấn Qdrant
    search_result = await async_qdrant_client.search(
        collection_name=QDRANT_COLLECTION_NAME,
        query_vector=query_vector,
        limit=limit
    )
    
    # Bước 3: Lấy ra text từ payload
    contexts = []
    for hit in search_result:
        contexts.append(hit.payload["text"])
        print(f"[Search] Tìm thấy chunk tương đồng (Score: {hit.score:.4f}): '{hit.payload.get('text', '')[:60]}...'")
        
    return contexts


async def async_generate_rag_response(query: str) -> str:
    """
    Thực hiện luồng RAG hoàn chỉnh bất đồng bộ.
    """
    print(f"[RAG] Bắt đầu xử lý câu hỏi: '{query}'")
    
    # 1. TÌM KIẾM
    contexts = await async_search_similar_chunks(query, limit=3)
    
    if not contexts:
        context_text = "Không tìm thấy thông tin nào trong kho tri thức của hệ thống."
    else:
        context_text = "\n---\n".join(contexts)
        
    # 2. GHÉP PROMPT (Đa ngôn ngữ)
    system_prompt = (
        "Bạn là một Trợ lý ảo hỗ trợ khách hàng chuyên nghiệp, lịch sự và chu đáo về Thực phẩm chức năng.\n"
        "Nhiệm vụ của bạn là trả lời câu hỏi của khách hàng dựa trên NGUỒN NGỮ CẢNH (Context) được cung cấp dưới đây.\n"
        "Hãy tuân thủ nghiêm ngặt các nguyên tắc sau:\n"
        "1. ĐA NGÔN NGỮ (Multi-language): Hãy trả lời câu hỏi bằng CHÍNH NGÔN NGỮ mà khách hàng đã dùng để hỏi.\n"
        "   - Nếu khách hàng hỏi bằng tiếng Việt -> trả lời bằng tiếng Việt.\n"
        "   - Nếu khách hàng hỏi bằng tiếng Anh -> dịch thông tin liên quan từ Context và trả lời bằng tiếng Anh.\n"
        "2. CHỈ sử dụng thông tin trong Nguồn Ngữ Cảnh để trả lời. Không được tự ý bịa đặt hoặc suy diễn thông tin ngoài.\n"
        "3. Trả lời ngắn gọn, súc tích và trực tiếp giải quyết câu hỏi của khách hàng.\n"
        "4. Nếu trong Nguồn Ngữ Cảnh KHÔNG chứa thông tin nào để trả lời câu hỏi, hãy phản hồi lịch sự bằng chính ngôn ngữ của người dùng:\n"
        "   - Ví dụ (Tiếng Việt): \"Dạ, xin lỗi anh/chị, hiện tại em chưa có thông tin về vấn đề này. Em sẽ ghi nhận để cập nhật sớm nhất ạ.\"\n"
        "   - Ví dụ (Tiếng Anh): \"I am sorry, but I do not have information about this issue right now. I will record it to update as soon as possible.\"\n\n"
        f"--- BẮT ĐẦU NGUỒN NGỮ CẢNH ---\n{context_text}\n--- KẾT THÚC NGUỒN NGỮ CẢNH ---"
    )
    
    # 3. SINH ĐÁP ÁN
    print("[LLM] Đang gọi OpenAI GPT-4o-mini...")
    response = await async_openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0.0 # Giữ nhiệt độ bằng 0.0 để tránh bịa đặt thông tin y tế/TPCN
    )
    
    answer = response.choices[0].message.content
    print("[RAG] Đã sinh ra câu trả lời thành công.")
    return answer

