# prompts.py

"""
Prompt constants for the RAG pipeline.

``SYSTEM_PROMPT`` is the middleware-compatible instruction string used by
``create_agent`` + ``dynamic_prompt``.  It contains *no* placeholders —
the middleware appends retrieved context, and the user question arrives
via the message.

``BASELINE_PROMPT`` is the legacy ChatPromptTemplate format string with
``{question}`` and ``{context}`` placeholders.  It is retained for
backward compatibility with ``scripts/run_full_evaluation.py``.
"""

# Middleware-compatible system prompt (no placeholders)
SYSTEM_PROMPT = """\
You are a helpful assistant who answers questions based on provided context. \
You must only use the provided context, and cannot use your own knowledge."""

# Legacy prompt template for run_full_evaluation.py (ChatPromptTemplate format)
BASELINE_PROMPT = """\
You are a helpful assistant who answers questions based on provided context. You must only use the provided context, and cannot use your own knowledge.

### Question
{question}

### Context
{context}
"""
