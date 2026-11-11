"""Build the vector knowledge base for the diagnostic assistant.

Indexes two kinds of documents into a persistent ChromaDB collection:

* every DTC entry from ``knowledge_base/dtc_codes.json`` (one document per
  code: description + causes + checks),
* every ``.md`` / ``.txt`` / ``.pdf`` file in ``knowledge_base/docs/``,
  split into paragraph chunks.

Embeddings: ``all-MiniLM-L6-v2`` (CPU, ~90 MB). Run::

    python -m src.rag.build_kb
"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    EMBEDDING_MODEL,
    KNOWLEDGE_BASE_DIR,
)

CHUNK_TARGET_CHARS = 800


def dtc_documents() -> tuple[list[str], list[str], list[dict]]:
    """One retrievable document per DTC code."""
    with open(KNOWLEDGE_BASE_DIR / "dtc_codes.json", encoding="utf-8") as f:
        data = json.load(f)
    ids, texts, metadatas = [], [], []
    for entry in data["codes"]:
        text = (
            f"DTC {entry['code']} ({entry['system']}): {entry['description']}.\n"
            f"Possible causes: {'; '.join(entry['possible_causes'])}.\n"
            f"Recommended checks: {'; '.join(entry['recommended_checks'])}."
        )
        ids.append(f"dtc-{entry['code']}")
        texts.append(text)
        metadatas.append({
            "type": "dtc", "code": entry["code"], "system": entry["system"],
            "source": "SAE J2012 / ISO 15031-6 generic definitions",
        })
    return ids, texts, metadatas


def _read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="replace")


def chunk_text(text: str, target: int = CHUNK_TARGET_CHARS) -> list[str]:
    """Greedy paragraph packing into ~target-char chunks."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) > target:
            chunks.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        chunks.append(current)
    return chunks


def doc_documents() -> tuple[list[str], list[str], list[dict]]:
    """Chunked documents from ``knowledge_base/docs/``."""
    ids, texts, metadatas = [], [], []
    docs_dir = KNOWLEDGE_BASE_DIR / "docs"
    for path in sorted(docs_dir.iterdir()):
        if path.suffix.lower() not in (".md", ".txt", ".pdf"):
            continue
        for i, chunk in enumerate(chunk_text(_read_text(path))):
            ids.append(f"doc-{path.stem}-{i}")
            texts.append(chunk)
            metadatas.append({
                "type": "doc", "source": path.name, "chunk": i,
            })
    return ids, texts, metadatas


def build() -> int:
    """(Re)build the persistent collection; returns the document count."""
    ids, texts, metadatas = dtc_documents()
    doc_ids, doc_texts, doc_metadatas = doc_documents()
    ids += doc_ids
    texts += doc_texts
    metadatas += doc_metadatas

    print(f"Embedding {len(texts)} documents with {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    embeddings = model.encode(
        texts, batch_size=64, show_progress_bar=False,
        normalize_embeddings=True,
    )

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(
        CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    collection.add(ids=ids, documents=texts, metadatas=metadatas,
                   embeddings=embeddings.tolist())
    print(f"Collection '{CHROMA_COLLECTION}' built: "
          f"{collection.count()} documents -> {CHROMA_DIR}")
    return collection.count()


if __name__ == "__main__":
    build()
