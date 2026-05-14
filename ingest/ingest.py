import voyageai
import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from app.config import get_settings
from app.weaviate_client import get_client, create_collection, insert_chunks

settings = get_settings()
voyage_client = voyageai.Client(api_key=settings.voyage_api_key)


def load_url(url):
    logger.info(f"Cargando: {url}")
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    title = ""
    if soup.find("h1"):
        title = soup.find("h1").get_text(strip=True)
    elif soup.find("title"):
        title = soup.find("title").get_text(strip=True)
    content_tag = (
        soup.find("article") or
        soup.find("main") or
        soup.find("div", {"class": "devsite-article-body"}) or
        soup.find("body")
    )
    for tag in content_tag.find_all(["script", "style", "nav", "footer"]):
        tag.decompose()
    content = content_tag.get_text(separator="\n", strip=True)
    logger.info(f"Cargado: '{title}' — {len(content)} chars")
    return {"title": title, "content": content, "url": url}


def split_document(doc):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    fragments = splitter.split_text(doc["content"])
    chunks = []
    for i, fragment in enumerate(fragments):
        chunks.append({
            "content":  fragment,
            "title":    doc["title"],
            "url":      doc["url"],
            "page":     0,
            "chunk_id": i,
        })
    logger.info(f"'{doc['title']}' → {len(chunks)} fragmentos")
    return chunks


def generate_embeddings(chunks):
    texts = [chunk["content"] for chunk in chunks]
    all_vectors = []
    batch_size = 128
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        logger.info(f"Embeddings batch {i//batch_size + 1} — {len(batch)} textos")
        result = voyage_client.embed(
            batch,
            model=settings.voyage_model,
            input_type="document",
        )
        all_vectors.extend(result.embeddings)
    logger.info(f"Generados {len(all_vectors)} embeddings")
    return all_vectors


def run_ingest(urls):
    logger.info(f"Iniciando ingest de {len(urls)} URLs")
    client = get_client()
    try:
        create_collection(client)
        total_inserted = 0
        for url in urls:
            try:
                doc = load_url(url)
                chunks = split_document(doc)
                if not chunks:
                    continue
                vectors = generate_embeddings(chunks)
                inserted = insert_chunks(client, chunks, vectors)
                total_inserted += inserted
            except Exception as e:
                logger.error(f"Error en {url}: {e}")
                continue
        logger.info(f"Ingest completado — {total_inserted} fragmentos")
        return total_inserted
    finally:
        client.close()