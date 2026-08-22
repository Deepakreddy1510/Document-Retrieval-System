import pymupdf

from pdf_rag.pdf_processor import clean_text, extract_pdf_bytes


def test_clean_text_joins_hyphenated_line_wrap():
    assert clean_text("regu-\nlarization") == "regularization"


def test_extract_pdf_keeps_page_numbers():
    doc = pymupdf.open()
    first = doc.new_page()
    first.insert_text((72, 72), "First page")
    second = doc.new_page()
    second.insert_text((72, 72), "Second page")
    data = doc.tobytes()
    doc.close()

    pages = extract_pdf_bytes(data)
    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert "First page" in pages[0].text
    assert pages[1].page_number == 2
