# Source Modules (src/)

Core library implementing the RAG system using factory pattern for clean, testable, modular code.

## Module Organization

```
src/
├── config.py           # Cached singletons (LLM, embeddings, Qdrant client)
├── state.py            # TypedDict schema for LangGraph state
├── prompts.py          # RAG prompt templates
├── retrievers.py       # Factory: create_retrievers()
├── graph.py            # Factory: build_graph(), build_all_graphs()
└── utils/
    ├── __init__.py     # Package exports
    ├── loaders.py      # HuggingFace dataset loaders
    └── manifest.py     # RUN_MANIFEST.json generation
```

## Factory Pattern Philosophy

**Why factories?** Retrievers and graphs depend on runtime data (documents, vector stores) that doesn't exist at module import time.

**Anti-Pattern** (module-level initialization):
```python
# ❌ THIS BREAKS - documents don't exist yet!
documents = load_documents_from_huggingface()
retriever = vectorstore.as_retriever()
```

**Correct Pattern** (factory function):
```python
# ✅ Creates retriever at runtime with actual data
def create_retrievers(documents, vector_store, k=5):
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return {"naive": retriever}
```

## Quick Start: Adding a New Retriever

```python
# 1. Edit src/retrievers.py
def create_retrievers(documents, vector_store, k=5):
    # ... existing retrievers ...
    your_retriever = YourRetrieverClass(vectorstore=vector_store, k=k)
    return {
        "naive": naive,
        "bm25": bm25,
        "ensemble": ensemble,
        "cohere_rerank": compression,
        "your_method": your_retriever,  # <-- Add here
    }

# 2. Validate
make validate  # Must pass 100%

# 3. Evaluate
make eval  # Automatically includes your new retriever
```

The system automatically evaluates your new retriever and includes results in `comparative_ragas_results.csv`.

## Module Reference

### config.py - Configuration Singletons

```python
from functools import lru_cache

@lru_cache(maxsize=1)
def get_llm():
    """Returns cached ChatOpenAI instance (gpt-4.1-mini, temperature=0)"""
    return ChatOpenAI(model="gpt-4.1-mini", temperature=0)

@lru_cache(maxsize=1)
def get_embeddings():
    """Returns cached OpenAIEmbeddings instance"""
    return OpenAIEmbeddings(model="text-embedding-3-small")

def create_vector_store(documents, collection_name="gdelt_rag", recreate_collection=False):
    """Factory for Qdrant vector store (NOT cached - allows multiple collections)"""
    # Creates new collection or reuses existing
```

**Why `@lru_cache`?** Ensures singleton behavior (one LLM instance per process).

### state.py - LangGraph State Schema

```python
from typing import List, TypedDict
from langchain_core.documents import Document

class State(TypedDict):
    """State schema for RAG graph"""
    question: str                  # User question
    context: List[Document]        # Retrieved documents
    response: str                  # Generated answer
```

### retrievers.py - Retriever Factory

```python
def create_retrievers(documents, vector_store, k=5):
    """Create all 4 retrieval strategies"""
    # 1. Naive: Dense vector search
    naive = vector_store.as_retriever(search_kwargs={"k": k})

    # 2. BM25: Sparse keyword matching
    bm25 = BM25Retriever.from_documents(documents, k=k)

    # 3. Ensemble: Hybrid (50% dense + 50% sparse)
    ensemble = EnsembleRetriever(retrievers=[naive, bm25], weights=[0.5, 0.5])

    # 4. Cohere Rerank: Retrieve 20 → rerank to top k
    wide_retriever = vector_store.as_retriever(search_kwargs={"k": max(20, k)})
    compression = ContextualCompressionRetriever(
        base_compressor=CohereRerank(model="rerank-v3.5"),
        base_retriever=wide_retriever
    )

    return {"naive": naive, "bm25": bm25, "ensemble": ensemble, "cohere_rerank": compression}
```

### graph.py - LangGraph Workflow Factory

```python
def build_graph(retriever, llm=None):
    """Build a LangGraph workflow for a single retriever"""
    if llm is None:
        llm = get_llm()

    def retrieve(state):
        docs = retriever.invoke(state["question"])
        return {"context": docs}  # Partial state update

    def generate(state):
        docs_content = "\n\n".join(d.page_content for d in state["context"])
        messages = BASELINE_PROMPT.format_messages(question=state["question"], context=docs_content)
        response = llm.invoke(messages)
        return {"response": response.content}  # Partial state update

    graph = StateGraph(State)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()
```

**Why Partial State Updates?** Node functions return dicts, not complete states. LangGraph automatically merges updates.

### utils/loaders.py - HuggingFace Loaders

```python
def load_documents_from_huggingface(
    repo_id="open-biosciences/biosciences-sources",
    split="train",
    revision=None
):
    """Load source documents from HuggingFace"""
    dataset = load_dataset(repo_id, split=split, revision=revision)
    documents = [Document(page_content=item["page_content"], metadata=item["metadata"]) for item in dataset]
    return documents
```

## Testing & Validation

```bash
# Application validation (must pass 100%)
make validate

# Manual testing
python -c "
from src.utils import load_documents_from_huggingface
from src.config import create_vector_store
from src.retrievers import create_retrievers
from src.graph import build_all_graphs

documents = load_documents_from_huggingface()
vector_store = create_vector_store(documents)
retrievers = create_retrievers(documents, vector_store)
graphs = build_all_graphs(retrievers)

result = graphs['cohere_rerank'].invoke({'question': 'What is GDELT?'})
print(result['response'])
"
```

## Best Practices

### DO:
✅ Use factories (`create_retrievers`, `build_graph`) instead of module-level instances
✅ Use `@lru_cache` for singletons (LLM, embeddings)
✅ Return partial state updates from LangGraph nodes
✅ Validate with `make validate` before committing

### DON'T:
❌ Create retrievers/graphs at module import time
❌ Hardcode API keys (use environment variables)
❌ Return complete states from LangGraph nodes
❌ Skip validation before deployment

## Further Reading

- **[CLAUDE.md](../CLAUDE.md)** - Complete technical reference
- **[scripts/README.md](../scripts/README.md)** - Evaluation scripts guide
- **[data/README.md](../data/README.md)** - Data flow documentation
