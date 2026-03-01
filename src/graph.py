# graph.py

"""
LangGraph workflow factory for GDELT RAG system.

This module provides factory functions to create LangGraph workflows. Graphs
cannot be instantiated at module level because they depend on retrievers that
must be created first.
"""

from typing import Dict, List
from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate

from src.state import State
from src.prompts import BASELINE_PROMPT
from src.config import get_llm


def build_graph(retriever, llm=None, prompt_template: str = None):
    """
    Build a compiled LangGraph pipeline for a single retriever.

    This creates a simple two-node graph:
    START → retrieve → generate → END

    The retrieve node fetches relevant documents, and the generate node
    produces an answer based on those documents.

    Args:
        retriever: Retriever instance to use for document retrieval
        llm: ChatOpenAI instance (defaults to get_llm() if None)
        prompt_template: RAG prompt template string (defaults to BASELINE_PROMPT)

    Returns:
        Compiled StateGraph that can be invoked with {"question": "..."}

    Example:
        >>> from src.utils import load_documents_from_huggingface
        >>> from src.config import create_vector_store
        >>> from src.retrievers import create_retrievers
        >>> from src.graph import build_graph
        >>>
        >>> documents = load_documents_from_huggingface()
        >>> vector_store = create_vector_store(documents)
        >>> retrievers = create_retrievers(documents, vector_store)
        >>> graph = build_graph(retrievers['naive'])
        >>>
        >>> result = graph.invoke({"question": "What is GDELT?"})
        >>> print(result['response'])

    Notes:
        - Node functions return partial state updates (dict)
        - LangGraph automatically merges updates into state
        - This follows LangGraph best practices for state management
    """
    if llm is None:
        llm = get_llm()

    if prompt_template is None:
        prompt_template = BASELINE_PROMPT

    rag_prompt = ChatPromptTemplate.from_template(prompt_template)

    # Define node functions (return partial state updates)
    def retrieve(state: State) -> dict:
        """
        Retrieve relevant documents for the question.

        Args:
            state: Current state with 'question' key

        Returns:
            Dict with 'context' key containing List[Document]
        """
        docs: List[Document] = retriever.invoke(state["question"])
        return {"context": docs}

    def generate(state: State) -> dict:
        """
        Generate answer from retrieved context.

        Args:
            state: Current state with 'question' and 'context' keys

        Returns:
            Dict with 'response' key containing answer string
        """
        docs_content = "\n\n".join(d.page_content for d in state.get("context", []))
        msgs = rag_prompt.format_messages(
            question=state["question"], context=docs_content
        )
        response = llm.invoke(msgs)
        return {"response": response.content}

    # Build graph
    graph = StateGraph(State)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


def build_all_graphs(retrievers: Dict[str, object], llm=None) -> Dict[str, object]:
    """
    Build compiled graphs for all retrievers.

    Convenience function to create a graph for each retriever in the
    retrievers dictionary.

    Args:
        retrievers: Dictionary of retriever instances from create_retrievers()
        llm: Optional ChatOpenAI instance (shared across all graphs)

    Returns:
        Dictionary mapping retriever names to compiled graphs.
        Same keys as input retrievers dict.

    Example:
        >>> from src.utils import load_documents_from_huggingface
        >>> from src.config import create_vector_store
        >>> from src.retrievers import create_retrievers
        >>> from src.graph import build_all_graphs
        >>>
        >>> documents = load_documents_from_huggingface()
        >>> vector_store = create_vector_store(documents)
        >>> retrievers = create_retrievers(documents, vector_store)
        >>> graphs = build_all_graphs(retrievers)
        >>>
        >>> # All graphs ready to use
        >>> result_naive = graphs['naive'].invoke({"question": "What is GDELT?"})
        >>> result_bm25 = graphs['bm25'].invoke({"question": "What is GDELT?"})
        >>> result_ensemble = graphs['ensemble'].invoke({"question": "What is GDELT?"})
        >>> result_rerank = graphs['cohere_rerank'].invoke({"question": "What is GDELT?"})
    """
    return {name: build_graph(ret, llm) for name, ret in retrievers.items()}
