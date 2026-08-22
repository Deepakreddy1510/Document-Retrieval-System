from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pdf_rag.config import settings  # noqa: E402
from pdf_rag.database import Database  # noqa: E402
from pdf_rag.embeddings import EmbeddingModel  # noqa: E402
from pdf_rag.evaluation import score_retrieval  # noqa: E402
from pdf_rag.reranker import Reranker  # noqa: E402
from pdf_rag.retrieval import HybridRetriever  # noqa: E402


def load_examples(path: Path) -> list[dict]:
    examples = []

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    return examples


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def evaluate(dataset_path: Path, k: int) -> dict:
    database = Database(settings.database_url)

    embedder = EmbeddingModel(
        settings.embedding_model,
        settings.embedding_dim,
        settings.embedding_query_prefix,
    )

    retriever = HybridRetriever(
        database,
        embedder,
        dense_k=settings.dense_k,
        lexical_k=settings.lexical_k,
        fused_k=settings.fused_k,
        rrf_k=settings.rrf_k,
    )

    reranker = Reranker(settings.reranker_model)

    examples = [
        x
        for x in load_examples(dataset_path)
        if x.get("answerable", True)
    ]

    if not examples:
        raise ValueError("Dataset has no answerable retrieval examples.")

    # ---------------------------------------------------------
    # Warm up embedding model, database path, and CrossEncoder.
    # This prevents the first benchmark query from carrying most
    # of the model-loading / PyTorch initialization cost.
    # ---------------------------------------------------------
    warmup_question = examples[0]["question"]

    warmup_dense = retriever.dense(
        warmup_question,
        limit=max(k, settings.dense_k),
    )

    retriever.lexical(
        warmup_question,
        limit=max(k, settings.lexical_k),
    )

    reranker.rerank(
        warmup_question,
        warmup_dense,
        top_k=max(k, settings.final_k),
    )

    aggregate: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    cases = []

    for example in examples:
        question = example["question"]
        expected = example["expected_sources"]

        configurations = {}
        timings = {}

        # -----------------------------------------------------
        # 1. Dense retrieval
        # -----------------------------------------------------
        started = time.perf_counter()

        dense = retriever.dense(
            question,
            limit=max(k, settings.dense_k),
        )

        dense_time = (time.perf_counter() - started) * 1000

        configurations["dense"] = dense
        timings["dense"] = dense_time

        # -----------------------------------------------------
        # 2. Lexical retrieval
        # -----------------------------------------------------
        started = time.perf_counter()

        lexical = retriever.lexical(
            question,
            limit=max(k, settings.lexical_k),
        )

        lexical_time = (time.perf_counter() - started) * 1000

        configurations["lexical"] = lexical
        timings["lexical"] = lexical_time

        # -----------------------------------------------------
        # 3. Hybrid retrieval: dense + lexical + RRF
        # -----------------------------------------------------
        started = time.perf_counter()

        hybrid = retriever.hybrid(question)

        hybrid_time = (time.perf_counter() - started) * 1000

        configurations["hybrid"] = hybrid
        timings["hybrid"] = hybrid_time

        # -----------------------------------------------------
        # 4. Dense + CrossEncoder reranking
        # -----------------------------------------------------
        started = time.perf_counter()

        dense_reranked = reranker.rerank(
            question,
            dense,
            top_k=max(k, settings.final_k),
        )

        dense_rerank_only_time = (
            time.perf_counter() - started
        ) * 1000

        configurations["dense_reranked"] = dense_reranked

        # End-to-end latency:
        # dense retrieval + reranking
        timings["dense_reranked"] = (
            dense_time + dense_rerank_only_time
        )

        # -----------------------------------------------------
        # 5. Hybrid + CrossEncoder reranking
        # -----------------------------------------------------
        started = time.perf_counter()

        hybrid_reranked = reranker.rerank(
            question,
            hybrid,
            top_k=max(k, settings.final_k),
        )

        hybrid_rerank_only_time = (
            time.perf_counter() - started
        ) * 1000

        configurations["hybrid_reranked"] = hybrid_reranked

        # End-to-end latency:
        # hybrid retrieval + reranking
        timings["hybrid_reranked"] = (
            hybrid_time + hybrid_rerank_only_time
        )

        # -----------------------------------------------------
        # Score every configuration
        # -----------------------------------------------------
        case = {
            "id": example.get("id"),
            "question": question,
            "configurations": {},
        }

        for name, results in configurations.items():
            metrics = score_retrieval(
                results,
                expected,
                k=k,
            )

            aggregate[name]["hit_at_k"].append(
                metrics.hit_at_k
            )
            aggregate[name]["precision_at_k"].append(
                metrics.precision_at_k
            )
            aggregate[name]["recall_at_k"].append(
                metrics.recall_at_k
            )
            aggregate[name]["mrr"].append(
                metrics.reciprocal_rank
            )
            aggregate[name]["latency_ms"].append(
                timings[name]
            )

            case["configurations"][name] = {
                "metrics": metrics.__dict__,
                "latency_ms": round(timings[name], 2),
                "top_results": [
                    {
                        "document": r.filename,
                        "page": r.page_number,
                        "chunk_id": r.chunk_id,
                    }
                    for r in results[:k]
                ],
            }

        cases.append(case)

    # ---------------------------------------------------------
    # Aggregate results
    #
    # Retrieval metrics -> arithmetic mean across questions
    # Latency           -> median across questions
    # ---------------------------------------------------------
    summary = {}

    for name, fields in aggregate.items():
        summary[name] = {
            "hit_at_k": round(
                mean(fields["hit_at_k"]),
                4,
            ),
            "precision_at_k": round(
                mean(fields["precision_at_k"]),
                4,
            ),
            "recall_at_k": round(
                mean(fields["recall_at_k"]),
                4,
            ),
            "mrr": round(
                mean(fields["mrr"]),
                4,
            ),
            "latency_ms": round(
                median(fields["latency_ms"]),
                4,
            ),
        }

    return {
        "k": k,
        "examples": len(examples),
        "summary": summary,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare PDF RAG retrieval configurations."
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "evaluation" / "dataset.jsonl",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "evaluation" / "retrieval_results.json",
    )

    args = parser.parse_args()

    report = evaluate(
        args.dataset,
        args.k,
    )

    args.out.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(
        f"Evaluated {report['examples']} "
        f"answerable questions at k={report['k']}\n"
    )

    print("Latency shown below is median end-to-end latency.\n")

    for name, metrics in report["summary"].items():
        print(
            f"{name:20} "
            f"Hit@K={metrics['hit_at_k']:.3f}  "
            f"Precision@K={metrics['precision_at_k']:.3f}  "
            f"Recall@K={metrics['recall_at_k']:.3f}  "
            f"MRR={metrics['mrr']:.3f}  "
            f"Latency={metrics['latency_ms']:.1f} ms"
        )

    print(f"\nDetailed report: {args.out}")


if __name__ == "__main__":
    main()