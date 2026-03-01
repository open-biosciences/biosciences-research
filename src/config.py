# config.py

"""
Configuration module for Biosciences RAG system.

Provides cached getter functions for LLM, embeddings, and Qdrant client.
Configuration is read from environment variables with sensible defaults.
"""

import os
from functools import lru_cache
from typing import List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv

load_dotenv(override=not os.getenv("CI"))

# ---------- Configuration from environment variables ----------
QDRANT_URL = os.getenv("QDRANT_URL", os.getenv("QDRANT_API_URL"))                 # prefer this if set
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")   # fallback path
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY   = os.getenv("COHERE_API_KEY")
HF_TOKEN         = os.getenv("HF_TOKEN")

LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

COLLECTION_NAME= os.getenv("QDRANT_COLLECTION")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
# ----------------------------------------

@lru_cache(maxsize=1)
def get_llm():
    """
    Get cached LLM instance.

    Returns:
        ChatOpenAI instance with temperature=0 for deterministic outputs
    """
    return ChatOpenAI(model=OPENAI_MODEL_NAME, temperature=0)


@lru_cache(maxsize=1)
def get_embeddings():
    """
    Get cached embeddings instance.

    Returns:
        OpenAIEmbeddings instance
    """
    return OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def get_qdrant():
    """
    URL-first convention. If QDRANT_URL is set, use it.
    Otherwise fall back to host/port. Only pass api_key if provided.
    
    Get cached Qdrant client instance.

    Returns:
        QdrantClient connected to configured URL or host/port
    """
    kwargs = {}
    if QDRANT_API_KEY:          # avoid passing empty key (breaks docker default)
        kwargs["api_key"] = QDRANT_API_KEY

    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL, **kwargs)
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, **kwargs)


def get_collection_name() -> str:
    """
    Get configured collection name.

    Returns:
        Collection name string
    """
    return COLLECTION_NAME


def create_vector_store(
    documents: List[Document],
    collection_name: str = None,
    recreate_collection: bool = False
) -> QdrantVectorStore:
    """
    Create and populate Qdrant vector store.

    This factory function handles:
    - Creating Qdrant collection if it doesn't exist
    - Optionally recreating collection if it does exist
    - Populating vector store with documents

    Args:
        documents: List of Document objects to add to vector store
        collection_name: Override default collection name (optional)
        recreate_collection: If True, delete existing collection first (default: False)

    Returns:
        Populated QdrantVectorStore instance

    Example:
        >>> from src.utils import load_documents_from_huggingface
        >>> from src.config import create_vector_store
        >>> documents = load_documents_from_huggingface()
        >>> vector_store = create_vector_store(documents, recreate_collection=True)
    """
    client = get_qdrant()
    embeddings = get_embeddings()
    collection = collection_name or get_collection_name()

    # Check if collection exists
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if collection in collection_names and recreate_collection:
        client.delete_collection(collection)
        collection_names.remove(collection)

    # Create collection if needed
    if collection not in collection_names:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

    # Build vector store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection,
        embedding=embeddings,
    )

    # Add documents if collection is new or recreated
    if collection not in collection_names or recreate_collection:
        vector_store.add_documents(documents=documents)

    return vector_store
