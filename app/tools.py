import voyageai
from langchain_core.tools import tool
from loguru import logger
from app.config import get_settings
from app.weaviate_client import get_client, hybrid_search, get_related_docs

settings = get_settings()
voyage_client = voyageai.Client(api_key=settings.voyage_api_key)


def get_embedding(text: str) -> list[float]:
    """Genera un embedding con Voyage AI."""
    result = voyage_client.embed(
        [text],
        model=settings.voyage_model,
        input_type="query",
    )
    return result.embeddings[0]


@tool
def search_gcp_docs(query: str) -> str:
    """
    Busca en la documentación oficial de GCP.
    Usá esta herramienta cuando necesites información técnica sobre
    cualquier servicio o producto de Google Cloud Platform.

    Args:
        query: pregunta o término a buscar en la documentación GCP

    Returns:
        Fragmentos relevantes de la documentación con sus fuentes
    """
    logger.info(f"Tool search_gcp_docs llamada con: '{query}'")

    query_vector = get_embedding(query)
    weaviate_client = get_client()

    try:
        chunks = hybrid_search(
            client=weaviate_client,
            query_vector=query_vector,
            query_text=query,
            top_k=settings.retrieval_top_k,
        )
    finally:
        weaviate_client.close()

    if not chunks:
        return "No encontré documentación relevante para esa consulta."

    # Formateamos los resultados para que el LLM los pueda leer
    output = []
    for i, chunk in enumerate(chunks, 1):
        output.append(
            f"[Fuente {i}]\n"
            f"Título: {chunk['title']}\n"
            f"URL: {chunk['url']}\n"
            f"Página: {chunk['page']}\n"
            f"Score: {chunk['relevance_score']:.3f}\n"
            f"Contenido:\n{chunk['content']}\n"
        )

    return "\n---\n".join(output)


@tool
def get_doc_metadata(url: str) -> str:
    """
    Obtiene metadata de un documento GCP dado su URL.
    Usá esta herramienta para obtener más información sobre
    una fuente específica antes de citarla en la respuesta.

    Args:
        url: URL del documento GCP

    Returns:
        Metadata del documento incluyendo título y descripción
    """
    logger.info(f"Tool get_doc_metadata llamada con: '{url}'")

    weaviate_client = get_client()
    try:
        collection = weaviate_client.collections.get(settings.weaviate_collection)
        results = collection.query.fetch_objects(
            filters=collection.query.filter.by_property("url").equal(url),
            limit=1,
        )
    finally:
        weaviate_client.close()

    if not results.objects:
        return f"No encontré metadata para la URL: {url}"

    obj = results.objects[0]
    return (
        f"Título: {obj.properties['title']}\n"
        f"URL: {obj.properties['url']}\n"
        f"Fragmentos indexados: disponibles en la colección"
    )


@tool
def get_related_documentation(query: str, exclude_urls: str = "") -> str:
    """
    Sugiere documentación de GCP relacionada con el tema consultado.
    Usá esta herramienta al final de cada respuesta para recomendar
    documentación adicional relevante al usuario.

    Args:
        query: tema o pregunta del usuario
        exclude_urls: URLs separadas por coma que ya se usaron como fuente

    Returns:
        Lista de documentos relacionados con título y URL
    """
    logger.info(f"Tool get_related_documentation llamada con: '{query}'")

    query_vector = get_embedding(query)
    exclude_list = [u.strip() for u in exclude_urls.split(",") if u.strip()]

    weaviate_client = get_client()
    try:
        related = get_related_docs(
            client=weaviate_client,
            query_vector=query_vector,
            exclude_urls=exclude_list,
            top_k=3,
        )
    finally:
        weaviate_client.close()

    if not related:
        return "No encontré documentación relacionada adicional."

    output = []
    for doc in related:
        output.append(f"- {doc['title']}\n  {doc['url']}")

    return "\n".join(output)


# Lista de herramientas disponibles para el agente
TOOLS = [search_gcp_docs, get_doc_metadata, get_related_documentation]