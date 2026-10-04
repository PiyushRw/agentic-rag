from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config.settings import settings
from backend.ingestion.loader import PageText


@dataclass
class ChunkData:
    chunk_index: int          # position within the document (0, 1, 2...)
    page_number: int | None
    text: str


def chunk_pages(pages: list[PageText]) -> list[ChunkData]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],  # prefer paragraph, then sentence, then word breaks
    )
    chunks: list[ChunkData] = []
    for page in pages:
        for piece in splitter.split_text(page.text):
            chunks.append(
                ChunkData(chunk_index=len(chunks), page_number=page.page_number, text=piece)
            )
    return chunks