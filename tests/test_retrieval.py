from pdf_rag.models import SearchResult
from pdf_rag.fusion import reciprocal_rank_fusion


def result(chunk_id, dense=None, lexical=None):
    return SearchResult(
        chunk_id=chunk_id,
        document_id=1,
        filename="doc.pdf",
        page_number=chunk_id,
        chunk_index=0,
        text=f"chunk {chunk_id}",
        dense_score=dense,
        lexical_score=lexical,
    )


def test_rrf_rewards_results_present_in_both_rankings():
    dense = [result(1, dense=0.9), result(2, dense=0.8)]
    lexical = [result(2, lexical=1.2), result(3, lexical=0.7)]

    fused = reciprocal_rank_fusion([dense, lexical], rrf_k=60, limit=3)

    assert fused[0].chunk_id == 2
    assert fused[0].dense_score == 0.8
    assert fused[0].lexical_score == 1.2
    assert fused[0].rrf_score is not None
