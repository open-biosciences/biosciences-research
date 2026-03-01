# graph.py

"""
LangChain agent factory for GDELT RAG system.

This module provides factory functions to create LangChain agents with
retrieval middleware. Uses create_agent + dynamic_prompt from LangChain 1.0
instead of manual StateGraph construction.
"""

from typing import Dict, List

from typing_extensions import NotRequired

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain_core.documents import Document

from src.config import get_llm
from src.prompts import SYSTEM_PROMPT


class RAGState(AgentState):
    """Agent state extended with retrieved context for RAGAS compatibility.

    Inherits ``messages`` from AgentState. The ``context`` field stores
    retrieved documents so the evaluation harness can extract them after
    invocation.
    """

    context: NotRequired[List[Document]]


def build_graph(retriever, llm=None, prompt_template: str = None):
    """
    Build a LangChain agent with retrieval middleware.

    Uses ``create_agent`` with a ``dynamic_prompt`` middleware that performs
    retrieval and injects the results into the system prompt before the
    model call.

    Args:
        retriever: Retriever instance to use for document retrieval
        llm: Chat model instance (defaults to get_llm() if None)
        prompt_template: System prompt string (defaults to SYSTEM_PROMPT).
            Must *not* contain ``{question}`` or ``{context}`` placeholders;
            context is appended by the middleware and the question arrives
            via the message.

    Returns:
        CompiledStateGraph agent that can be invoked with
        ``{"messages": [{"role": "user", "content": "..."}]}``

    Example:
        >>> from src.graph import build_graph, invoke_for_benchmark
        >>> from src.retrievers import create_retrievers
        >>> from src.config import create_vector_store
        >>> from src.utils import load_documents_from_huggingface
        >>>
        >>> documents = load_documents_from_huggingface()
        >>> vector_store = create_vector_store(documents)
        >>> retrievers = create_retrievers(documents, vector_store)
        >>> agent = build_graph(retrievers['naive'])
        >>>
        >>> result = invoke_for_benchmark(agent, "What is GDELT?")
        >>> print(result['response'])
    """
    if llm is None:
        llm = get_llm()

    if prompt_template is None:
        prompt_template = SYSTEM_PROMPT

    @dynamic_prompt
    def retrieve_and_inject(request: ModelRequest) -> str:
        """Retrieve docs for the latest user message and build system prompt."""
        last_msg = request.state["messages"][-1]
        query = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        docs = retriever.invoke(query)
        request.state["context"] = docs
        docs_content = "\n\n".join(d.page_content for d in docs)
        return f"{prompt_template}\n\nContext:\n{docs_content}"

    return create_agent(
        llm, tools=[], middleware=[retrieve_and_inject], state_schema=RAGState
    )


def build_all_graphs(retrievers: Dict[str, object], llm=None) -> Dict[str, object]:
    """
    Build agents for all retrievers.

    Convenience function to create an agent for each retriever in the
    retrievers dictionary.

    Args:
        retrievers: Dictionary of retriever instances from create_retrievers()
        llm: Optional chat model instance (shared across all agents)

    Returns:
        Dictionary mapping retriever names to compiled agents.
        Same keys as input retrievers dict.

    Example:
        >>> graphs = build_all_graphs(retrievers)
        >>> result = invoke_for_benchmark(graphs['naive'], "What is GDELT?")
    """
    return {name: build_graph(ret, llm) for name, ret in retrievers.items()}


def invoke_for_benchmark(agent, question: str) -> dict:
    """
    Invoke agent and return RAGAS-compatible dict.

    Translates between the agent's message-based interface and the
    dict-based interface expected by the evaluation harness.

    Args:
        agent: Compiled agent from build_graph()
        question: The question to ask

    Returns:
        Dict with keys: ``question``, ``response``, ``context``
    """
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return {
        "question": question,
        "response": result["messages"][-1].content,
        "context": result.get("context", []),
    }
