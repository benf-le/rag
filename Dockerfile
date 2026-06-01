# Sử dụng image Python slim chính thức để giảm dung lượng image tối đa
FROM python:3.11-slim as builder

# Đặt thư mục làm việc trong container
WORKDIR /app

# Thiết lập các biến môi trường để tối ưu hóa Python trong Docker
# - PYTHONDONTWRITEBYTECODE: Không ghi các file .pyc ra đĩa
# - PYTHONUNBUFFERED: Đảm bảo log của Python được in ra stdout/stderr lập tức mà không bị buffer
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Cài đặt các công cụ biên dịch nếu cần thiết (slim không có sẵn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt trước để tận dụng Docker layer cache cho dependencies
COPY requirements.txt .

# Cài đặt các thư viện Python vào thư mục cục bộ user
RUN pip install --no-cache-dir --user -r requirements.txt


# --- STAGE 2: RUNTIME ---
FROM python:3.11-slim as runner

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/root/.local/bin:$PATH \
    PORT=8000

# Copy cài đặt Python từ builder stage
COPY --from=builder /root/.local /root/.local

# Copy toàn bộ mã nguồn vào thư mục làm việc
COPY . .

# Expose port chạy của ứng dụng FastAPI
EXPOSE 8000

# Healthcheck giúp Docker và các orchestrator giám sát trạng thái ứng dụng qua endpoint /health
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

# Lệnh khởi chạy ứng dụng sử dụng uvicorn
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
