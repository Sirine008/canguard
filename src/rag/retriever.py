"""Hybrid retrieval over the knowledge base.

Retrieval is semantic (MiniLM embeddings, cosine), with one exact-match
boost: any DTC code written explicitly in the question (e.g. "P0171") is
looked up directly and pinned to the top of the results. Sibling codes such
as P0171/P0172/P0174 have near-identical descriptions, so pure semantic
ranking often puts a sibling first when the user already named the code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    EMBEDDING_MODEL,
    RAG_TOP_K,
)


@dataclass
class RetrievedDoc:
    """One retrieval hit with its cosine similarity."""

    doc_id: str
    text: str
    metadata: dict
    similarity: float


@lru_cache(maxsize=1)
def _embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


@lru_cache(maxsize=1)
def _collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(CHROMA_COLLECTION)


_DTC_PATTERN = re.compile(r"\b([PBCU]\d{4})\b", re.IGNORECASE)


def _exact_code_docs(query: str) -> list[RetrievedDoc]:
    """Direct lookups for DTC codes written explicitly in the question."""
    codes = [c.upper() for c in _DTC_PATTERN.findall(query)]
    if not codes:
        return []
    result = _collection().get(
        ids=[f"dtc-{code}" for code in codes],
        include=["documents", "metadatas"],
    )
    return [
        RetrievedDoc(doc_id=doc_id, text=text, metadata=metadata,
                     similarity=1.0)  # exact mention of the code
        for doc_id, text, metadata in zip(
            result["ids"], result["documents"], result["metadatas"])
    ]


def retrieve(query: str, top_k: int = RAG_TOP_K) -> list[RetrievedDoc]:
    """Exact code mentions first, then semantic hits up to ``top_k``."""
    docs = _exact_code_docs(query)
    seen = {d.doc_id for d in docs}

    embedding = _embedder().encode(
        [query], normalize_embeddings=True
    )[0].tolist()
    result = _collection().query(
        query_embeddings=[embedding], n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    for doc_id, text, metadata, distance in zip(
        result["ids"][0], result["documents"][0],
        result["metadatas"][0], result["distances"][0],
    ):
        if doc_id in seen:
            continue
        docs.append(RetrievedDoc(
            doc_id=doc_id, text=text, metadata=metadata,
            similarity=1.0 - float(distance),  # cosine distance -> similarity
        ))
    return docs[:top_k]
