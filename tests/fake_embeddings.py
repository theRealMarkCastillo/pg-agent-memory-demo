"""Deterministic lexical vectors for logic regressions, NOT a semantic-quality model."""

import hashlib
import math
import re
from types import SimpleNamespace

GROUPS = [
    "access control security policy permissions",
    "contractor contractors nda terms agreement confidentiality",
    "remote remotely work guidelines",
    "hiking hikes trail trails outdoors outdoor nature",
    "photo photos photography photographic",
    "embed embedding embeddings embeds conversion converts vector vectors semantic semantically",
    "scrape scraped scraping prices pricing product products ecommerce commerce competitor",
    "financial finance earnings sec filings pnl",
    "user users lookup retrieve retrieves retrieval find finds",
    "config configuration environment settings",
    "legal contract clauses",
    "pet pets cat cats",
    "live lives residence location tokyo",
    "allergic allergy peanuts",
    "coffee drink drinks",
]
ALIASES = {word: words.split()[0] for words in GROUPS for word in words.split()}
STOP = set(
    "a an the of to for and or in on is are from with using into by given raw how do i my what me about at only all may purposes logged audited quarterly".split()
)


def vector(text):
    values = [0.0] * 1536
    for word in set(re.findall(r"\w+", text.lower())) - STOP:
        word = ALIASES.get(word, word)
        idx = int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big") % 1536
        values[idx] += 1
    if not any(values):
        values[0] = 1
    norm = math.sqrt(sum(x * x for x in values))
    return [x / norm for x in values]


class FakeEmbeddings:
    async def create(self, input, **kwargs):
        values = input if isinstance(input, list) else [input]
        if any("[EMBEDDING_FAILURE]" in t for t in values):
            from openai import APIConnectionError
            import httpx

            raise APIConnectionError(
                request=httpx.Request("POST", "http://test.invalid/embeddings")
            )
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=i, embedding=vector(t))
                for i, t in enumerate(values)
            ]
        )
