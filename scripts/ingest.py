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
    parser = argparse.ArgumentParser(description="Index one or more text-based PDF files.")
    parser.add_argument("pdfs", type=Path, nargs="+")
    args = parser.parse_args()

    pipeline = RAGPipeline()
    pipeline.initialize()

    for path in args.pdfs:
        if path.suffix.lower() != ".pdf":
            print(f"SKIP {path}: not a PDF")
            continue
        result = pipeline.ingest_pdf(filename=path.name, pdf_bytes=path.read_bytes())
        print(
            f"{result.status.upper():15} {result.filename} | "
            f"pages={result.page_count} chunks={result.chunk_count}"
        )


if __name__ == "__main__":
    main()
