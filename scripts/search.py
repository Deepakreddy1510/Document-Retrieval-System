from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pdf_rag.pipeline import RAGPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect hybrid + reranked evidence without calling the LLM."
    )
    parser.add_argument("question")
    args = parser.parse_args()

    pipeline = RAGPipeline()
    pipeline.initialize()
    results = pipeline.retrieve(args.question)

    if not results:
        print("No chunks retrieved.")
        return

    for rank, result in enumerate(results, start=1):
        preview = " ".join(result.text.split())[:500]
        print(f"\n#{rank} {result.filename} — page {result.page_number}")
        print(
            "scores: "
            f"dense={result.dense_score} "
            f"lexical={result.lexical_score} "
            f"rrf={result.rrf_score} "
            f"rerank={result.rerank_score}"
        )
        print(preview)


if __name__ == "__main__":
    main()
