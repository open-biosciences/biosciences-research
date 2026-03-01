# __init__.py

"""
GDELT RAG System

A production-grade RAG (Retrieval-Augmented Generation) system for querying
GDELT (Global Database of Events, Language, and Tone) knowledge graphs.

This package provides:
- Configuration management (config)
- Document loading utilities (utils)
- Retriever factory functions (retrievers)
- LangGraph workflow builders (graph)
- State schema definitions (state)
- Prompt templates (prompts)

Example usage:
    >>> from src.utils import load_documents_from_huggingface
    >>> from src.config import create_vector_store
    >>> from src.retrievers import create_retrievers
    >>> from src.graph import build_all_graphs
    >>>
    >>> # Load data
    >>> documents = load_documents_from_huggingface()
    >>>
    >>> # Create vector store
    >>> vector_store = create_vector_store(documents, recreate_collection=True)
    >>>
    >>> # Create retrievers
    >>> retrievers = create_retrievers(documents, vector_store)
    >>>
    >>> # Build LangGraph workflows
    >>> graphs = build_all_graphs(retrievers)
    >>>
    >>> # Query the system
    >>> result = graphs['naive'].invoke({"question": "What is GDELT?"})
    >>> print(result['response'])
"""

__version__ = "0.1.0"
__all__ = ["config", "graph", "prompts", "retrievers", "state", "utils"]
