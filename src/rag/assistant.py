"""DTC diagnostic assistant: retrieval-grounded generation with Ollama.

The assistant answers **only** from retrieved knowledge-base context and
cites the DTC codes it used. If nothing sufficiently similar is retrieved,
it answers "not in knowledge base" without calling the LLM. If Ollama is
not reachable, it degrades gracefully to returning the retrieved context
verbatim (flagged in the response), so the API and dashboard keep working.

Quick manual test::

    python -m src.rag.assistant "What does P0171 mean and what should I check?"
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import httpx
import ollama

from src.config import OLLAMA_HOST, OLLAMA_MODEL, RAG_MIN_SIMILARITY, RAG_TOP_K
from src.rag.retriever import RetrievedDoc, retrieve

NOT_IN_KB = ("I can't answer that from the knowledge base. It covers generic "
             "OBD-II diagnostic trouble codes (SAE J2012) and CAN bus / "
             "OBD-II reference notes.")

SYSTEM_PROMPT = """You are a vehicle diagnostic assistant. Answer the user's \
question using ONLY the reference context provided below. Rules:
- If the context does not contain the answer, reply exactly: "Not in \
knowledge base."
- When you use a DTC entry, cite its code (e.g. P0171) in the answer.
- Be concise and practical: meaning first, then likely causes, then checks \
in a sensible order.
- Never invent codes, causes or specifications that are not in the context.
- Generic codes only; for manufacturer-specific codes say OEM documentation \
is required."""


@dataclass
class AssistantResponse:
    """Answer plus the sources that grounded it."""

    answer: str
    sources: list[dict] = field(default_factory=list)
    grounded: bool = True          # False when retrieval was too weak
    llm_used: bool = True          # False when Ollama was unreachable


def ollama_available() -> bool:
    """True if an Ollama server responds at the configured host."""
    try:
        response = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=2.0)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def _format_context(docs: list[RetrievedDoc]) -> str:
    blocks = []
    for doc in docs:
        label = doc.metadata.get("code") or doc.metadata.get("source", doc.doc_id)
        blocks.append(f"[{label}]\n{doc.text}")
    return "\n\n---\n\n".join(blocks)


def _sources_payload(docs: list[RetrievedDoc]) -> list[dict]:
    return [
        {
            "id": doc.doc_id,
            "code": doc.metadata.get("code"),
            "source": doc.metadata.get("source"),
            "similarity": round(doc.similarity, 3),
        }
        for doc in docs
    ]


def answer(question: str, top_k: int = RAG_TOP_K) -> AssistantResponse:
    """Retrieve, then generate an answer grounded in the retrieved context."""
    docs = retrieve(question, top_k=top_k)
    if not docs or docs[0].similarity < RAG_MIN_SIMILARITY:
        return AssistantResponse(
            answer=NOT_IN_KB, sources=_sources_payload(docs[:2]),
            grounded=False, llm_used=False,
        )

    context = _format_context(docs)
    if not ollama_available():
        summary = "\n\n".join(
            f"[{d.metadata.get('code') or d.metadata.get('source')}] {d.text}"
            for d in docs[:3]
        )
        return AssistantResponse(
            answer=("(LLM offline — showing the matching knowledge-base "
                    f"entries instead)\n\n{summary}"),
            sources=_sources_payload(docs), llm_used=False,
        )

    client = ollama.Client(host=OLLAMA_HOST)
    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Reference context:\n\n{context}\n\n---\n\n"
                f"Question: {question}")},
        ],
        options={"temperature": 0.2, "num_predict": 400},
    )
    return AssistantResponse(
        answer=response["message"]["content"].strip(),
        sources=_sources_payload(docs),
    )


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What does P0420 mean?"
    result = answer(question)
    print(f"Q: {question}\n")
    print(result.answer)
    print("\nSources:", ", ".join(
        s["code"] or s["source"] or s["id"] for s in result.sources))
