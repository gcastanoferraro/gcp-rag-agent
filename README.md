# GCP Docs RAG Agent 🤖

Agente de IA que responde preguntas técnicas sobre **Google Cloud Platform** usando RAG (Retrieval-Augmented Generation). El sistema busca en documentación oficial de GCP y genera respuestas precisas con fuentes citadas.

## 🏗️ Arquitectura

```
Usuario → FastAPI → LangGraph Agent → RAG Search (Weaviate) → LLM (Groq/Llama)
                         ↓
                   3 herramientas:
                   • search_gcp_docs       → búsqueda híbrida en Weaviate
                   • get_doc_metadata      → metadata de fuentes
                   • get_related_docs      → documentación relacionada
```

**Stack:**
- **LLM:** Llama 3.3 70B via Groq (gratuito)
- **Embeddings:** Voyage AI — `voyage-3` (gratuito)
- **Vector DB:** Weaviate 1.27 (local via Docker)
- **Orquestación:** LangGraph con memoria por sesión (thread_id)
- **API:** FastAPI con documentación automática en `/docs`

## 🚀 Instalación

### Requisitos
- Python 3.11
- Docker Desktop
- API keys gratuitas: [Groq](https://console.groq.com) · [Voyage AI](https://dash.voyageai.com)

### Setup

```bash
# 1. Cloná el repositorio
git clone https://github.com/gcastanoferraro/gcp-rag-agent
cd gcp-rag-agent

# 2. Instalá dependencias
pip install -r requirements.txt

# 3. Configurá las variables de entorno
cp .env.example .env
# Editá .env con tus API keys

# 4. Levantá Weaviate
docker-compose up -d

# 5. Levantá el servidor
py -3.11 -m uvicorn app.main:app --reload --port 8000
```

### Ingestión de documentación GCP

```python
from ingest.ingest import run_ingest

urls = [
    "https://cloud.google.com/bigquery/docs/introduction",
    "https://cloud.google.com/storage/docs/introduction",
    "https://cloud.google.com/run/docs/overview/what-is-cloud-run",
]

run_ingest(urls)
```

## 📡 Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/health` | Estado del sistema y Weaviate |
| POST | `/query` | Hacer una pregunta sobre GCP |
| POST | `/ingest` | Cargar URLs de documentación GCP |

Documentación interactiva disponible en `http://localhost:8000/docs`

## 💬 Ejemplo de uso

**Request:**
```json
POST /query
{
  "question": "¿Cómo configuro un bucket de Cloud Storage con acceso público?",
  "thread_id": "user-123"
}
```

**Response:**
```json
{
  "answer": "Para configurar un bucket de Cloud Storage con acceso público...",
  "sources": [
    {
      "title": "Cloud Storage overview",
      "url": "https://cloud.google.com/storage/docs/introduction",
      "page": 0,
      "relevance_score": 0.999
    }
  ],
  "related": [
    {
      "title": "IAM and access control",
      "url": "https://cloud.google.com/storage/docs/access-control",
      "reason": "Documentación relacionada"
    }
  ],
  "thread_id": "user-123"
}
```

## 🧠 Cómo funciona

1. **Ingestión:** Las páginas de GCP se descargan, parten en fragmentos de 1500 caracteres con 150 de overlap, y se convierten en vectores con Voyage AI. Los vectores se guardan en Weaviate con HNSW.

2. **Query:** El agente LangGraph recibe la pregunta, decide qué herramientas usar (ReAct loop), ejecuta una búsqueda híbrida (semántica + keywords BM25) en Weaviate, y pasa los fragmentos relevantes al LLM como contexto.

3. **Respuesta:** El LLM genera una respuesta basada únicamente en la documentación recuperada, cita las fuentes y sugiere documentación relacionada.

## 📁 Estructura del proyecto

```
gcp-rag-agent/
├── app/
│   ├── main.py              # FastAPI — endpoints
│   ├── agent.py             # LangGraph — orquestación del agente
│   ├── tools.py             # Herramientas del agente
│   ├── weaviate_client.py   # Conexión y queries a Weaviate
│   ├── schemas.py           # Modelos Pydantic
│   └── config.py            # Configuración centralizada
├── ingest/
│   └── ingest.py            # Pipeline de ingestión de documentos
├── docker-compose.yml       # Weaviate local
├── requirements.txt
└── .env.example
```

## 🔧 Variables de entorno

| Variable | Descripción | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | API key de Groq | — |
| `GROQ_MODEL` | Modelo de Groq | `llama-3.3-70b-versatile` |
| `VOYAGE_API_KEY` | API key de Voyage AI | — |
| `VOYAGE_MODEL` | Modelo de embeddings | `voyage-3` |
| `WEAVIATE_URL` | URL de Weaviate | `http://localhost:8082` |
| `CHUNK_SIZE` | Tamaño de fragmentos | `1500` |
| `CHUNK_OVERLAP` | Overlap entre fragmentos | `150` |
| `RETRIEVAL_TOP_K` | Fragmentos a recuperar | `5` |

## 👤 Autor

**Gonzalo Castaño Ferraro**
[LinkedIn](https://www.linkedin.com/in/gonzalo-casta%C3%B1o-ferraro-594361222/) · GCP Professional Data Engineer · GCP Associate Cloud Engineer
