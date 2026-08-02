"""PDF text extraction with per-page tracking for accurate source citations."""
from dataclasses import dataclass
from typing import List

try:
    import fitz  # PyMuPDF
    _BACKEND = "pymupdf"
except ImportError:
    from pypdf import PdfReader
    _BACKEND = "pypdf"


@dataclass
class PageText:
    page_number: int
    text: str


def extract_pages_from_pdf(uploaded_file) -> List[PageText]:
    """Extract text from a PDF, page by page, so chunks can cite exact pages.

    Args:
        uploaded_file: A file-like object (e.g. Streamlit's UploadedFile).

    Returns:
        List of PageText, one entry per non-empty page.

    Raises:
        ValueError: if no extractable text is found (e.g. a scanned PDF
            with no OCR layer).
    """
    uploaded_file.seek(0)
    pages: List[PageText] = []

    if _BACKEND == "pymupdf":
        data = uploaded_file.read()
        doc = fitz.open(stream=data, filetype="pdf")
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append(PageText(page_number=i, text=text))
        doc.close()
    else:
        reader = PdfReader(uploaded_file)
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(PageText(page_number=i, text=text))

    if not pages:
        raise ValueError(
            "No extractable text found in this PDF. It may be a scanned "
            "image without an OCR layer — try a text-based PDF instead."
        )
    return pages


def extract_text_from_pdf(uploaded_file) -> str:
    """Backward-compatible helper: returns the full document as one string."""
    pages = extract_pages_from_pdf(uploaded_file)
    return "\n\n".join(p.text for p in pages)
