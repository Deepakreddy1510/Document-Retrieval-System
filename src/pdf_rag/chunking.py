from __future__ import annotations

from typing import Any, Protocol, Sequence

from .models import Chunk, PageText


class TokenizerLike(Protocol):
    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> Sequence[Any]: ...

    def decode(
        self,
        token_ids: Sequence[Any],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str: ...


def chunk_pages(
    pages: list[PageText],
    tokenizer: TokenizerLike,
    chunk_size: int = 450,
    overlap: int = 60,
) -> list[Chunk]:
    """Create page-bounded, token-aware chunks with overlap.

    Keeping chunks within one PDF page makes page citations deterministic and easy
    to explain. Cross-page chunking can be tested later as an experiment.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    step = chunk_size - overlap
    chunks: list[Chunk] = []

    for page in pages:
        if not page.text.strip():
            continue
        token_ids = list(
            tokenizer.encode(
                page.text,
                add_special_tokens=False,
            )
        )
        page_chunk_index = 0

        for start in range(0, len(token_ids), step):
            piece = token_ids[start : start + chunk_size]
            if not piece:
                break

            text = tokenizer.decode(
                piece,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
            if text:
                chunks.append(
                    Chunk(
                        page_number=page.page_number,
                        chunk_index=page_chunk_index,
                        text=text,
                    )
                )
                page_chunk_index += 1

            if start + chunk_size >= len(token_ids):
                break

    return chunks
