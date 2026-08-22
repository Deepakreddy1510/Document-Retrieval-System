from pdf_rag.chunking import chunk_pages
from pdf_rag.models import PageText


class WhitespaceTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True):
        return " ".join(token_ids)


def test_chunking_preserves_page_and_overlap():
    page = PageText(page_number=3, text="one two three four five six seven eight nine ten")
    chunks = chunk_pages([page], WhitespaceTokenizer(), chunk_size=5, overlap=2)

    assert [c.page_number for c in chunks] == [3, 3, 3]
    assert chunks[0].text == "one two three four five"
    assert chunks[1].text == "four five six seven eight"
    assert chunks[2].text == "seven eight nine ten"


def test_invalid_overlap_rejected():
    page = PageText(page_number=1, text="hello world")
    try:
        chunk_pages([page], WhitespaceTokenizer(), chunk_size=5, overlap=5)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
