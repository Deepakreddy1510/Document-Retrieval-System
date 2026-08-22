from __future__ import annotations

from .models import SearchResult, Source


def build_context(
    results: list[SearchResult],
    *,
    max_chars: int = 20000,
) -> tuple[str, list[Source]]:
    """Build source-labelled LLM context and the matching citation metadata."""
    blocks: list[str] = []
    sources: list[Source] = []
    used_chars = 0

    for index, result in enumerate(results, start=1):
        label = f"S{index}"
        header = f"[{label}] Document: {result.filename} | Page: {result.page_number}\n"
        block = header + result.text.strip()

        remaining = max_chars - used_chars
        if remaining <= len(header):
            break
        if len(block) > remaining:
            block = block[:remaining].rstrip()
        if not block.strip():
            continue

        blocks.append(block)
        used_chars += len(block) + 2
        sources.append(
            Source(
                label=label,
                chunk_id=result.chunk_id,
                filename=result.filename,
                page_number=result.page_number,
                text=result.text,
                score=result.rerank_score,
            )
        )

        if used_chars >= max_chars:
            break

    return "\n\n".join(blocks), sources
