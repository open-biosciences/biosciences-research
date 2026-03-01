# state.py

"""
Benchmark adapter shape for the RAGAS evaluation harness.

The RAG agent itself uses ``RAGState`` (defined in ``src/graph.py``), which
extends LangChain's ``AgentState`` with a ``context`` field.

This module documents the *dict shape* returned by
``invoke_for_benchmark()`` — the adapter that translates between the
agent's message-based interface and the flat dict the evaluation scripts
expect.

Benchmark dict shape::

    {
        "question": str,       # The input question
        "response": str,       # The generated answer text
        "context": List[Document],  # Retrieved documents
    }

The legacy ``State`` TypedDict is retained below for backward
compatibility with ``scripts/run_full_evaluation.py``.
"""

from typing import List

from typing_extensions import TypedDict

from langchain_core.documents import Document


class State(TypedDict):
    """Legacy TypedDict used by run_full_evaluation.py's StateGraph pipeline."""

    question: str
    context: List[Document]
    response: str
