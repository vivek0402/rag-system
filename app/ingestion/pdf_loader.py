# app/ingestion/pdf_loader.py

import logging
from pathlib import Path
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def load_pdf(file_path: str | Path) -> list[dict]:
    """
    Extract text from a PDF file, page by page.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of dicts, one per page:
        [{"page": 1, "text": "...", "source": "file.pdf"}, ...]
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if file_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {file_path.suffix}")

    pages = []

    try:
        doc = fitz.open(file_path)
        logger.info(f"Opened PDF: {file_path.name} | Pages: {len(doc)}")

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text().strip()

            if not text:
                logger.warning(f"Page {page_num} has no extractable text - possibly scanned.")
                continue

            pages.append({
                "page": page_num,
                "text": text,
                "source": file_path.name
            })

        doc.close()

    except Exception as e:
        logger.error(f"Failed to process PDF {file_path.name}: {e}")
        raise

    logger.info(f"Extracted {len(pages)} pages with text from {file_path.name}")
    return pages