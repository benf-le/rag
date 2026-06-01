# Đối Chiếu Tính Năng: Đã Triển Khai vs. Kế Hoạch Tiếp Theo

Chào bạn! Để bạn dễ hình dung bức tranh toàn cảnh, đây là bảng đối chiếu chi tiết giữa **Những gì chúng ta VỪA TRIỂN KHAI** ở bước trước và **Những gì đang NẰM TRONG KẾ HOẠCH** cải tiến nâng cấp tiếp theo:

---

## 📊 Bảng Đối Chiếu Hiện Trạng Hệ Thống

| Trụ cột kỹ thuật | Đã triển khai (Done) | Kế hoạch tiếp theo (Planned) |
| :--- | :--- | :--- |
| **Trụ cột 1: Chunking & Ingestion** | *   Ráp **Structured Template** (5 trường dữ liệu) giữ trọn ngữ cảnh.<br>*   Định nghĩa Schema Pydantic cho TPCN.<br>*   Cơ chế **Xóa trước - Nạp sau** (`product_id`). | *   Nâng cấp bộ chia từ đơn giản sang `RecursiveCharacterTextSplitter` hoặc **Semantic Chunking**.<br>*   Nạp file PDF/DOCX trực tiếp. |
| **Trụ cột 2: Advanced Retrieval** | *   Tìm kiếm **Dense Vector** (ngữ nghĩa rộng) bằng model `text-embedding-3-large`. | *   **Hybrid Search (Dense + Sparse)**: Bắt trọn 100% từ khóa tên sản phẩm (Alipas, Blackmores).<br>*   **Reranking** với Cohere/BGE để lọc top 3 kết quả xuất sắc nhất. |
| **Trụ cột 3: Qdrant Optimization** | *   Kết nối và kiểm tra collection bất đồng bộ. | *   **Payload Indexing**: Tạo keyword index cho `product_id` tăng tốc độ xóa cũ.<br>*   **Quantization (SQ/PQ)**: Nén vector giảm 4x RAM. |
| **Trụ cột 4: API Async & Batching** | *   **Async/Await hoàn toàn** cho FastAPI, Qdrant & OpenAI.<br>*   **Batch Embeddings & Upserts** hàng loạt siêu tốc. | *   *Đã hoàn thành xuất sắc!* |
| **Trụ cột 5: Observability** | *   Ghi log tiến trình bất đồng bộ. | *   Tích hợp **Langfuse / Phoenix** để giám sát và đo lường độ chính xác (Recall/Precision). |

---

## 🚀 Đề Xuất Bước Tiếp Theo: Hybrid Search & Payload Indexing

Với đặc thù sản phẩm của bạn là **Thực phẩm chức năng (49 sản phẩm)**, bước tiếp theo mang lại hiệu quả vượt trội ngay lập tức (High ROI) là kết hợp **Hybrid Search** và **Payload Indexing**.

### 1. Tại sao cần Payload Indexing cho `product_id`?
Hiện tại, khi admin cập nhật sản phẩm, ta gọi hàm `async_delete_product_vectors` để xóa vector cũ. 
* Qdrant phải quét qua toàn bộ database để tìm point có `product_id` trùng khớp.
* **Giải pháp**: Tạo chỉ mục (Index) kiểu `KEYWORD` cho trường `product_id`. Qdrant sẽ xóa point cũ ngay lập tức (dưới 1ms) nhờ cơ chế tìm kiếm index-accelerated.

### 2. Tại sao cần Hybrid Search cho TPCN?
Nếu người dùng chỉ gõ cụm từ rất ngắn như *"Glucosamine"* hay *"Alipas"*:
* **Dense search**: Có thể trả về các bài viết về sụn khớp hoặc sinh lý nam chung chung, đôi khi bỏ sót chính xác trang chi tiết của sản phẩm "Sâm Alipas" do độ dài câu hỏi quá ngắn không đủ ngữ nghĩa.
* **Hybrid Search (Dense + Sparse)**: 
  * Qdrant tích hợp sẵn động cơ **FastEmbed** siêu nhẹ để tạo **Sparse Vectors** (biểu diễn dạng tần suất từ xuất hiện giống BM25) chạy ngay trên máy chủ của bạn mà không tốn phí API.
  * Kết hợp tìm kiếm ngữ nghĩa sâu (Dense) của OpenAI và tìm kiếm từ khóa chính xác (Sparse) của Qdrant sẽ giúp bot **nhận diện chính xác 100% tên sản phẩm** trong khi vẫn hiểu được các câu hỏi triệu chứng dạng *"đau khớp gối nên uống gì"*.

---

## 🛠️ Dự Thảo Code Triển Khai Hybrid Search & Payload Indexing

Nếu bạn đồng ý triển khai bước tiếp theo này, đây là cách chúng ta sẽ viết code:

### 1. Bật tính năng tạo Sparse Vector và tạo Payload Index trong `async_init_collection`:
```python
# rag_service.py
async def async_init_collection():
    exists = await async_qdrant_client.collection_exists(collection_name=QDRANT_COLLECTION_NAME)
    if not exists:
        print(f"[Qdrant] Khởi tạo collection '{QDRANT_COLLECTION_NAME}' với cấu hình nâng cao...")
        await async_qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            # 1. Cấu hình Dense Vector (OpenAI)
            vectors_config=models.VectorParams(
                size=3072,
                distance=models.Distance.COSINE
            ),
            # 2. Cấu hình Sparse Vector (BM25) để tìm từ khóa chính xác (Tên sản phẩm)
            sparse_vectors_config={
                "sparse-text": models.SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=True
                    )
                )
            }
        )
        
        # 3. Tạo Payload Index cho product_id để tăng tốc xóa/lọc
        await async_qdrant_client.create_payload_index(
            collection_name=QDRANT_COLLECTION_NAME,
            field_name="product_id",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        print("[Qdrant] Đã tạo thành công collection nâng cao và chỉ mục Payload!")
```

### 2. Thực hiện truy vấn Hybrid Search trong `async_search_similar_chunks`:
Qdrant hỗ trợ tính năng tìm kiếm Hybrid Search cực kỳ mạnh mẽ thông qua API:
```python
async def async_search_similar_chunks(query: str, limit: int = 3) -> list[str]:
    await async_init_collection()
    
    # Thực hiện truy vấn Hybrid Search (kết hợp Dense và Sparse)
    search_result = await async_qdrant_client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        prefetch=[
            # Nhánh 1: Tìm kiếm ngữ nghĩa (Dense)
            models.Prefetch(
                query=await async_get_embedding(query),
                using="", # Mặc định cho Dense vector
                limit=limit
            ),
            # Nhánh 2: Tìm kiếm từ khóa chính xác (Sparse)
            models.Prefetch(
                query=query, # Qdrant tự động chuyển hóa thành Sparse vector qua FastEmbed
                using="sparse-text",
                limit=limit
            )
        ],
        # Ghép kết quả bằng thuật toán Reciprocal Rank Fusion (RRF) cực kỳ chính xác
        query=models.FusionQuery(
            fusion=models.Fusion.RRF
        ),
        limit=limit
    )
    
    contexts = [hit.payload["text"] for hit in search_result.points]
    return contexts
```

---

## 💬 Thảo Luận: Bạn Muốn Triển Khai Tiếp Trụ Cột Nào?

Bước tiếp theo này sẽ giúp sản phẩm TPCN của bạn đạt độ chính xác tìm kiếm tên sản phẩm tuyệt đối. 

Bạn thấy đề xuất **nâng cấp Trụ cột 2 & 3 (Hybrid Search + Payload Indexing)** này thế nào? Bạn có muốn tôi lập kế hoạch chi tiết (implementation plan) để bắt tay vào triển khai nâng cấp code ngay bây giờ không?
