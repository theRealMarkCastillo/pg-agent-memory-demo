import os
import hashlib
import json
from openai import AsyncOpenAI

_embedding_client = None


def embedding_space() -> str:
    """Identify the configured vector space without persisting provider URLs."""
    identity = [
        os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        os.getenv("EMBEDDING_MODEL_NAME"),
        os.getenv("EMBEDDING_REVISION", ""),
        1536,
    ]
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


def get_embedding_client() -> AsyncOpenAI:
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = AsyncOpenAI(
            base_url=os.getenv("EMBEDDING_BASE_URL"),
            api_key=os.getenv("EMBEDDING_API_KEY"),
        )
    return _embedding_client
