from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.config import get_settings
from app.schemas import (
    QueryRequest, QueryResponse,
    IngestRequest, IngestResponse,
    HealthResponse, Source, RelatedDoc,
)
from app.agent import agent
from app.weaviate_client import get_client, check_health

settings = get_settings()

app = FastAPI(
    title="GCP Docs RAG Agent",
    description="Agente de IA que responde preguntas sobre Google Cloud Platform usando RAG",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    client = get_client()
    try:
        status = check_health(client)
    finally:
        client.close()
    return HealthResponse(
        status="ok",
        weaviate=status["weaviate"],
        collection=status["collection"],
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    logger.info(f"Query: '{request.question}' | thread: {request.thread_id}")
    try:
        result = agent.query(
            question=request.question,
            thread_id=request.thread_id,
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    return QueryResponse(
        answer=result["answer"],
        sources=[Source(**s) for s in result["sources"] if s.get("url")],
        related=[RelatedDoc(**r) for r in result["related"] if r.get("url")],
        thread_id=result["thread_id"],
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    logger.info(f"Ingest: {len(request.urls)} URLs")
    try:
        from ingest.ingest import run_ingest
        ingested = run_ingest(request.urls)
    except Exception as e:
        logger.error(f"Error ingest: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    return IngestResponse(
        ingested=ingested,
        collection=settings.weaviate_collection,
        message=f"Se ingrestaron {ingested} fragmentos exitosamente",
    )