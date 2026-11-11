"""Retrieval evaluation for the diagnostic assistant.

18 questions with expected DTC codes (or expected refusal). Reports
hit-rate@1/@3/@5 (expected code among the top-k retrieved documents) plus,
when Ollama is reachable, a grounding check on a subset: does the generated
answer cite the expected code?

Results -> ``reports/rag_eval.json``. Run::

    python -m src.rag.evaluate_rag
"""

from __future__ import annotations

import json

from src.config import REPORTS_DIR
from src.rag.assistant import answer, ollama_available
from src.rag.retriever import retrieve

# (question, expected DTC code or None when the right behavior is a refusal)
EVAL_SET: list[tuple[str, str | None]] = [
    ("What does P0171 mean and what should I check first?", "P0171"),
    ("My scanner shows system too rich on bank 1", "P0172"),
    ("Catalyst efficiency below threshold bank 1 — is my catalytic converter dead?", "P0420"),
    ("Random multiple cylinder misfire detected", "P0300"),
    ("Cylinder 3 misfire — how do I find out if it's the coil?", "P0303"),
    ("Code says EVAP small leak detected", "P0442"),
    ("Large EVAP leak right after refueling, what's the most likely cause?", "P0455"),
    ("Mass air flow sensor circuit malfunction", "P0100"),
    ("Engine coolant temperature circuit high input", "P0118"),
    ("Coolant temperature stays below thermostat regulating temperature", "P0128"),
    ("Crankshaft position sensor circuit — car cranks but won't start", "P0335"),
    ("Insufficient EGR flow code", "P0401"),
    ("What is P0700 and why is there no other engine code?", "P0700"),
    ("Lost communication with the ECM, where do I start?", "U0100"),
    ("What does control module bus off mean on CAN?", "U0073"),
    ("Left front wheel speed sensor circuit fault with ABS light on", "C0035"),
    ("How do I check CAN bus termination resistance?", "U0001"),
    # Out-of-scope: the assistant should refuse rather than invent an answer
    ("What is the torque spec for the cylinder head bolts on a 2015 Golf?", None),
]

GROUNDING_SUBSET = 6  # generated-answer checks when Ollama is up


def main() -> dict:
    hits = {1: 0, 3: 0, 5: 0}
    refusals_correct = 0
    refusal_questions = 0
    details = []

    for question, expected in EVAL_SET:
        docs = retrieve(question, top_k=5)
        retrieved_codes = [d.metadata.get("code") for d in docs]
        row: dict = {"question": question, "expected": expected,
                     "retrieved": retrieved_codes,
                     "top_similarity": round(docs[0].similarity, 3)}
        if expected is None:
            refusal_questions += 1
            response = answer(question)
            row["refused"] = not response.grounded
            refusals_correct += int(not response.grounded)
        else:
            for k in hits:
                if expected in retrieved_codes[:k]:
                    hits[k] += 1
            row["rank"] = (retrieved_codes.index(expected) + 1
                           if expected in retrieved_codes else None)
        details.append(row)

    n_code_questions = sum(1 for _, e in EVAL_SET if e is not None)
    results: dict = {
        "n_questions": len(EVAL_SET),
        "retrieval_hit_rate": {
            f"@{k}": round(h / n_code_questions, 3) for k, h in hits.items()
        },
        "refusal_correct": f"{refusals_correct}/{refusal_questions}",
        "details": details,
    }

    if ollama_available():
        grounded_ok = 0
        checked = []
        for question, expected in EVAL_SET[:GROUNDING_SUBSET]:
            response = answer(question)
            cited = expected is not None and expected in response.answer
            grounded_ok += int(cited)
            checked.append({"question": question, "expected": expected,
                            "code_cited": cited})
        results["generation_check"] = {
            "n": GROUNDING_SUBSET,
            "expected_code_cited": grounded_ok,
            "details": checked,
        }
    else:
        results["generation_check"] = "skipped (Ollama not reachable)"

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "rag_eval.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps({k: v for k, v in results.items() if k != "details"},
                     indent=2))
    print(f"full results -> {REPORTS_DIR / 'rag_eval.json'}")
    return results


if __name__ == "__main__":
    main()
