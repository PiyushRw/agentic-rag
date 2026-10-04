import re
import unicodedata

from backend.ingestion.loader import PageText


def clean_text(text: str) -> str:
    # NFKC turns ligatures like "ﬁ" into "fi" and normalizes odd Unicode spaces
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00ad", "")                  # invisible soft hyphens
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)       # "inter-\nnational" -> "international"
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)       # single newline -> space (PDF hard wraps)
    text = re.sub(r"[ \t]+", " ", text)                # collapse runs of spaces
    text = re.sub(r"\n{3,}", "\n\n", text)             # max one blank line
    return text.strip()


def clean_pages(pages: list[PageText]) -> list[PageText]:
    cleaned = []
    for page in pages:
        text = clean_text(page.text)
        if text:
            cleaned.append(PageText(page_number=page.page_number, text=text))
    return cleaned