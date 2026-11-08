FROM python:3.12-slim

WORKDIR /app

ARG EMBEDDING_PROVIDER=remote
ARG EMBEDDING_MODEL_NAME=
ARG EMBEDDING_REVISION=
# The runtime filesystem is read-only, so a local model is downloaded at build time.
ENV HF_HOME=/opt/huggingface

COPY requirements.txt requirements-local.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN if [ "$EMBEDDING_PROVIDER" = "local" ]; then \
      test -n "$EMBEDDING_MODEL_NAME" || { echo "EMBEDDING_MODEL_NAME is required" >&2; exit 1; }; \
      pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-local.txt && \
      python -c "import sys; from sentence_transformers import SentenceTransformer; SentenceTransformer(sys.argv[1], revision=sys.argv[2] or None, device='cpu')" \
        "$EMBEDDING_MODEL_NAME" "$EMBEDDING_REVISION"; \
    fi
ENV HF_HUB_OFFLINE=1

COPY . .
RUN useradd --uid 10001 --create-home appuser
USER appuser
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
