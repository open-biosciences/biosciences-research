"""LangGraph Server entry points for Biosciences Research RAG.

Exposes all 4 retrieval strategies as separate graphs. The expensive
initialisation (HuggingFace download, Qdrant connection, BM25 index)
happens exactly once via a shared ``@lru_cache`` builder.
"""

from functools import lru_cache
from typing import Dict

from src.config import create_vector_store, get_llm
from src.graph import build_all_graphs
from src.retrievers import create_retrievers
from src.utils import load_documents_from_huggingface


@lru_cache(maxsize=1)
def _build_graphs() -> Dict[str, object]:
    """Build all 4 RAG graphs once and cache them."""
    docs = load_documents_from_huggingface()
    vs = create_vector_store(docs, recreate_collection=False)
    rets = create_retrievers(docs, vs, k=5)
    return build_all_graphs(rets, llm=get_llm())


def get_naive():
    """LangGraph Server entry point — naive (dense vector) retrieval."""
    return _build_graphs()["naive"]


def get_bm25():
    """LangGraph Server entry point — BM25 (sparse keyword) retrieval."""
    return _build_graphs()["bm25"]


def get_ensemble():
    """LangGraph Server entry point — ensemble (hybrid) retrieval."""
    return _build_graphs()["ensemble"]


def get_cohere_rerank():
    """LangGraph Server entry point — Cohere rerank retrieval."""
    return _build_graphs()["cohere_rerank"]
