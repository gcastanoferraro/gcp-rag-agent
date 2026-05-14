from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Pregunta sobre GCP")
    thread_id: Optional[str] = Field(
        default="default",
        description="ID de sesión para memoria conversacional"
    )


class IngestRequest(BaseModel):
    urls: list[str] = Field(
        ...,
        min_length=1,
        description="Lista de URLs de documentación GCP a ingestar"
    )


class Source(BaseModel):
    title: str
    url: str
    page: Optional[int] = None
    relevance_score: float


class RelatedDoc(BaseModel):
    title: str
    url: str
    reason: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    related: list[RelatedDoc]
    thread_id: str


class IngestResponse(BaseModel):
    ingested: int
    collection: str
    message: str


class HealthResponse(BaseModel):
    status: str
    weaviate: str
    collection: str