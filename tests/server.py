"""Isolated test engine. Never use this embedding implementation in deployment."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "memory_engine"))
from main import app as app
from routers import companion, developer, enterprise, task_agent
from fake_embeddings import FakeEmbeddings

for module in (companion, developer, enterprise, task_agent):
    module.get_embedding_client = lambda: SimpleNamespace(embeddings=FakeEmbeddings())
