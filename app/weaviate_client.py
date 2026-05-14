import weaviate
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.classes.query import MetadataQuery, HybridFusion
from loguru import logger
from app.config import get_settings

settings = get_settings()


def get_client():
    client = weaviate.connect_to_local(host="localhost", port=8082)
    return client


def create_collection(client):
    collection_name = settings.weaviate_collection
    if client.collections.exists(collection_name):
        return
    client.collections.create(
        name=collection_name,
        vectorizer_config=Configure.Vectorizer.none(),
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE
        ),
        properties=[
            Property(name="content",  data_type=DataType.TEXT),
            Property(name="title",    data_type=DataType.TEXT),
            Property(name="url",      data_type=DataType.TEXT),
            Property(name="page",     data_type=DataType.INT),
            Property(name="chunk_id", data_type=DataType.INT),
        ],
    )
    logger.info(f"Colección '{collection_name}' creada")


def insert_chunks(client, chunks, vectors):
    collection = client.collections.get(settings.weaviate_collection)
    inserted = 0
    with collection.batch.dynamic() as batch:
        for chunk, vector in zip(chunks, vectors):
            batch.add_object(
                properties={
                    "content":  chunk["content"],
                    "title":    chunk["title"],
                    "url":      chunk["url"],
                    "page":     chunk.get("page", 0),
                    "chunk_id": chunk["chunk_id"],
                },
                vector=vector,
            )
            inserted += 1
    return inserted


def hybrid_search(client, query_vector, query_text, top_k=5, alpha=0.5):
    collection = client.collections.get(settings.weaviate_collection)
    results = collection.query.hybrid(
        query=query_text,
        vector=query_vector,
        alpha=alpha,
        limit=top_k,
        fusion_type=HybridFusion.RELATIVE_SCORE,
        return_metadata=MetadataQuery(score=True),
    )
    chunks = []
    for obj in results.objects:
        chunks.append({
            "content":         obj.properties["content"],
            "title":           obj.properties["title"],
            "url":             obj.properties["url"],
            "page":            obj.properties.get("page", 0),
            "chunk_id":        obj.properties.get("chunk_id", 0),
            "relevance_score": obj.metadata.score or 0.0,
        })
    return chunks


def get_related_docs(client, query_vector, exclude_urls, top_k=3):
    collection = client.collections.get(settings.weaviate_collection)
    results = collection.query.near_vector(
        near_vector=query_vector,
        limit=top_k + len(exclude_urls),
        return_metadata=MetadataQuery(distance=True),
    )
    related = []
    seen_urls = set(exclude_urls)
    for obj in results.objects:
        url = obj.properties["url"]
        if url in seen_urls:
            continue
        seen_urls.add(url)
        related.append({"title": obj.properties["title"], "url": url})
        if len(related) >= top_k:
            break
    return related


def check_health(client):
    try:
        is_ready = client.is_ready()
        collection_exists = client.collections.exists(settings.weaviate_collection)
        return {
            "weaviate":   "ok" if is_ready else "error",
            "collection": "ok" if collection_exists else "missing",
        }
    except Exception as e:
        return {"weaviate": f"error: {str(e)}", "collection": "unknown"}