import os
from openai import AsyncOpenAI

_embedding_client = None


def get_embedding_client() -> AsyncOpenAI:
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = AsyncOpenAI(
            base_url=os.getenv("EMBEDDING_BASE_URL"),
            api_key=os.getenv("EMBEDDING_API_KEY"),
        )
    return _embedding_client
