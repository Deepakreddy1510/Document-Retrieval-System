from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pdf_rag.generation import ABSTAIN_MESSAGE  # noqa: E402
from pdf_rag.pipeline import RAGPipeline  # noqa: E402


def load_examples(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run deterministic generation checks and save answers for manual review."
    )
    parser.add_argument("--dataset", type=Path, default=ROOT / "evaluation" / "dataset.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "evaluation" / "generation_results.json")
    args = parser.parse_args()

    pipeline = RAGPipeline()
    pipeline.initialize()
    examples = load_examples(args.dataset)

    rows = []
    citation_checks = []
    refusal_checks = []

    for example in examples:
        result = pipeline.answer(example["question"])
        answerable = bool(example.get("answerable", True))
        abstained = result.answer.strip() == ABSTAIN_MESSAGE
        has_citation = "[S" in result.answer

        if answerable:
            citation_checks.append(1.0 if has_citation and not abstained else 0.0)
        else:
            refusal_checks.append(1.0 if abstained else 0.0)

        rows.append(
            {
                "question": example["question"],
                "answerable": answerable,
                "reference_answer": example.get("reference_answer"),
                "generated_answer": result.answer,
                "retrieved_sources": [
                    {"document": s.filename, "page": s.page_number, "label": s.label}
                    for s in result.sources
                ],
                "has_source_citation": has_citation,
                "abstained": abstained,
                "manual_review": {
                    "faithful_to_context": None,
                    "answers_question": None,
                    "citation_supported": None,
                    "notes": "",
                },
            }
        )

    report = {
        "deterministic_metrics": {
            "citation_presence_rate_on_answerable": (
                sum(citation_checks) / len(citation_checks) if citation_checks else None
            ),
            "refusal_accuracy_on_unanswerable": (
                sum(refusal_checks) / len(refusal_checks) if refusal_checks else None
            ),
        },
        "cases": rows,
        "note": (
            "Faithfulness and factual correctness are intentionally left for manual audit in v1 "
            "rather than pretending an LLM judge is ground truth."
        ),
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["deterministic_metrics"], indent=2))
    print(f"Manual-review report: {args.out}")


if __name__ == "__main__":
    main()
