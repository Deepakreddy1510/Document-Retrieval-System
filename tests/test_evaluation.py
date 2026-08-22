from pdf_rag.evaluation import score_retrieval
from pdf_rag.models import SearchResult


def make_result(page):
    return SearchResult(
        chunk_id=page,
        document_id=1,
        filename="paper.pdf",
        page_number=page,
        chunk_index=0,
        text="text",
    )


def test_mrr_and_hit_at_k():
    results = [make_result(1), make_result(8), make_result(10)]
    expected = [{"document": "paper.pdf", "pages": [8]}]
    metrics = score_retrieval(results, expected, k=3)

    assert metrics.hit_at_k == 1.0
    assert metrics.recall_at_k == 1.0
    assert metrics.reciprocal_rank == 0.5
    assert metrics.precision_at_k == 1 / 3
