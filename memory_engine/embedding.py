import asyncio
import os
import hashlib
import json
import threading
from types import SimpleNamespace
from openai import AsyncOpenAI

# Must match halfvec(1536) in postgres/init.sql.
EMBEDDING_DIM = 1536

_embedding_client = None


def embedding_provider() -> str:
    provider = os.getenv("EMBEDDING_PROVIDER", "remote").strip().lower()
    if provider not in {"remote", "local"}:
        raise ValueError("EMBEDDING_PROVIDER must be 'remote' or 'local'")
    return provider


def embedding_space() -> str:
    """Identify the configured vector space without persisting provider URLs."""
    if embedding_provider() == "local":
        source = "local"
    else:
        source = os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    identity = [
        source,
        os.getenv("EMBEDDING_MODEL_NAME"),
        os.getenv("EMBEDDING_REVISION", ""),
        EMBEDDING_DIM,
    ]
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


class LocalEmbeddings:
    """In-process CPU sentence-transformers model with the embeddings.create shape.

    Smaller models are zero-padded to EMBEDDING_DIM. Padding preserves cosine
    similarity, so the halfvec(1536) columns and HNSW indexes are unchanged.
    """

    def __init__(self, model_name: str, revision: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "EMBEDDING_PROVIDER=local needs memory_engine/requirements-local.txt; "
                "rebuild the image with EMBEDDING_PROVIDER=local"
            ) from error
        self._model = SentenceTransformer(model_name, revision=revision, device="cpu")
        self._dim = self._model.get_sentence_embedding_dimension()
        if self._dim > EMBEDDING_DIM:
            raise ValueError(
                f"{model_name} produces {self._dim}-dim vectors; the schema stores {EMBEDDING_DIM}"
            )
        # Fast tokenizers are not safe for concurrent use from several threads.
        self._lock = threading.Lock()

    def _encode(self, texts):
        with self._lock:
            return self._model.encode(
                texts,
                batch_size=32,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

    async def create(self, input, **kwargs):
        texts = input if isinstance(input, list) else [input]
        vectors = await asyncio.to_thread(self._encode, texts)
        padding = [0.0] * (EMBEDDING_DIM - self._dim)
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=i, embedding=vector.tolist() + padding)
                for i, vector in enumerate(vectors)
            ]
        )


def get_embedding_client():
    global _embedding_client
    if _embedding_client is None:
        if embedding_provider() == "local":
            model_name = os.getenv("EMBEDDING_MODEL_NAME")
            if not model_name:
                raise ValueError("EMBEDDING_PROVIDER=local requires EMBEDDING_MODEL_NAME")
            _embedding_client = SimpleNamespace(
                embeddings=LocalEmbeddings(
                    model_name, os.getenv("EMBEDDING_REVISION") or None
                )
            )
        else:
            _embedding_client = AsyncOpenAI(
                base_url=os.getenv("EMBEDDING_BASE_URL"),
                api_key=os.getenv("EMBEDDING_API_KEY"),
            )
    return _embedding_client
