# Build stage
FROM python:3.12-slim AS builder

WORKDIR /app

COPY src/embeddings/requirements.txt .
RUN pip install --no-cache-dir --target=/app/deps -r requirements.txt

# Runtime stage
FROM python:3.12-slim

WORKDIR /app

# Copy only installed packages
COPY --from=builder /app/deps /usr/local/lib/python3.12/site-packages/

# Copy source code
COPY src/client.py src/client.py
COPY src/embeddings/ src/embeddings/
RUN touch src/__init__.py

CMD ["python", "-m", "src.embeddings.generate"]
