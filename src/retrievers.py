# retrievers.py

"""
Retriever factory functions for GDELT RAG system.

This module provides factory functions to create retrievers. Retrievers cannot
be instantiated at module level because they require documents and vector stores
that must be loaded first.
"""

from typing import Dict, List
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_qdrant import QdrantVectorStore
from src.config import COHERE_RERANK_MODEL


def create_retrievers(
    documents: List[Document],
    vector_store: QdrantVectorStore,
    k: int = 5,
) -> Dict[str, object]:
    """
    Create all retriever instances.

    This factory function creates 4 different retrieval strategies:
    1. Naive: Dense vector search using embeddings
    2. BM25: Sparse keyword matching
    3. Ensemble: Hybrid combination of dense + sparse
    4. Cohere Rerank: Contextual compression with reranking

    Args:
        documents: List of Document objects (required for BM25)
        vector_store: Populated QdrantVectorStore instance
        k: Number of documents to retrieve (default: 5)

    Returns:
        Dictionary mapping retriever names to retriever instances.
        Keys: 'naive', 'bm25', 'ensemble', 'cohere_rerank'

    Example:
        >>> from src.utils import load_documents_from_huggingface
        >>> from src.config import create_vector_store
        >>> from src.retrievers import create_retrievers
        >>>
        >>> documents = load_documents_from_huggingface()
        >>> vector_store = create_vector_store(documents, recreate_collection=True)
        >>> retrievers = create_retrievers(documents, vector_store)
        >>>
        >>> # Use individual retrievers
        >>> naive_docs = retrievers['naive'].invoke("What is GDELT?")
        >>> bm25_docs = retrievers['bm25'].invoke("What is GDELT?")

    Notes:
        - All retrievers return up to k documents
        - BM25 operates on in-memory document collection
        - Ensemble combines dense and sparse with 50/50 weighting
        - Cohere rerank retrieves 20 documents then reranks to top k
    """
    # Naive: Dense vector search using embeddings
    naive_retriever = vector_store.as_retriever(search_kwargs={"k": k})

    # BM25: Sparse keyword matching over in-memory docs
    bm25_retriever = BM25Retriever.from_documents(documents, k=k)

    # Ensemble: Hybrid search (dense + sparse)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[naive_retriever, bm25_retriever], weights=[0.5, 0.5]
    )

    # Cohere Rerank: Contextual compression
    # First retrieve a wider set (20 docs), then rerank to top k
    wide_retriever = vector_store.as_retriever(search_kwargs={"k": max(20, k)})
    reranker = CohereRerank(model=COHERE_RERANK_MODEL)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=wide_retriever,
        # Note: search_kwargs not valid here - k controlled by base_retriever
    )

    return {
        "naive": naive_retriever,
        "bm25": bm25_retriever,
        "ensemble": ensemble_retriever,
        "cohere_rerank": compression_retriever,
    }
