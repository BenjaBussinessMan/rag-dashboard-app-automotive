from typing import Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=5, ge=1, le=10)
    model: str | None = None


class Source(BaseModel):
    n: int
    doc_id: str
    fuente: str | None = None
    pagina: int | None = None
    doc_type: str | None = None
    score: float
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    grounded: bool
    refused: bool
    retried: bool
    problems: list[str]
    filters: dict
    sources: list[Source]
    model: str
    seconds: float


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    k: int = Field(default=5, ge=1, le=20)
    mode: Literal["bm25", "dense", "hybrid"] = "hybrid"
    understand: bool = True
    rerank: bool = False
    filters: dict | None = None


class SearchResponse(BaseModel):
    query: str
    hits: list[Source]
    parsed_filters: dict
