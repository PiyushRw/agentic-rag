from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question must not be blank.")
        return value


class SourceOut(BaseModel):
    chunk_id: int
    document_id: int
    filename: str
    page_number: int | None
    snippet: str
    score: float | None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceOut]


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)  # lets Pydantic read SQLAlchemy objects

    id: int
    filename: str
    num_chunks: int
    created_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentOut]