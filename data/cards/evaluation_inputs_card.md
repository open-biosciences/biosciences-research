---
license: apache-2.0
task_categories:
- question-answering
- text-retrieval
tags:
- rag
- ragas
- evaluation
- biosciences
- retrieval-comparison
- benchmark
pretty_name: Biosciences RAG Evaluation Inputs
size_categories:
- n<1K
---

# Biosciences RAG Evaluation Inputs

## Dataset Description

This dataset contains RAG inference outputs from 4 retrieval strategies evaluated on 12 biosciences research questions. Each retriever was tested on the same golden testset, producing 48 total records with retrieved contexts and LLM-generated answers ready for RAGAS evaluation.

### Dataset Summary

- **Total Examples**: 48 records (12 questions x 4 retrievers)
- **Retrievers Compared**:
  1. **Naive** — Dense vector similarity search (k=5)
  2. **BM25** — Sparse keyword-based retrieval
  3. **Ensemble** — 50% dense + 50% sparse hybrid
  4. **Cohere Rerank** — Dense retrieval (k=20) + rerank-v3.5 compression to top-5
- **Questions Per Retriever**: 12 from golden testset
- **Domain**: Biosciences — biomedical NLP, knowledge graphs, protein networks

### Data Fields

- `retriever` (string): Retriever strategy (`naive` | `bm25` | `ensemble` | `cohere_rerank`)
- `user_input` (string): The question or query
- `retrieved_contexts` (list[string]): Document chunks retrieved by the retriever
- `reference_contexts` (list[string]): Ground truth context passages containing the answer
- `response` (string): LLM-generated answer using retrieved contexts
- `reference` (string): Ground truth answer from golden testset
- `synthesizer_name` (string): RAGAS synthesizer that generated the test question

### Retriever Strategies

**Naive**:
- Dense vector similarity search using OpenAI text-embedding-3-small
- Top-k=5 documents returned

**BM25**:
- Sparse keyword-based retrieval with statistical term frequency scoring
- No semantic understanding — pure lexical matching

**Ensemble**:
- Hybrid approach: 50% weight naive + 50% weight BM25
- Balances semantic understanding with keyword matching

**Cohere Rerank**:
- Two-stage pipeline: dense retrieval (k=20 candidates) then Cohere rerank-v3.5 compression to top-5
- Most sophisticated strategy tested

### Data Splits

This dataset contains a single split with all 48 records from all 4 retrievers.

### Intended Use

- Benchmarking RAG retrieval strategies on biosciences documentation
- Comparing dense, sparse, hybrid, and reranking approaches
- Analyzing retrieval quality across different biomedical query types
- Input to RAGAS evaluation (see biosciences-evaluation-metrics for scored results)

### Evaluation Methodology

1. Load 140 source documents from `open-biosciences/biosciences-sources`
2. Create Qdrant vector store with text-embedding-3-small embeddings
3. Build 4 retriever strategies
4. Execute 12 queries per retriever via LangGraph workflows
5. Generate answers using gpt-4.1-mini with retrieved contexts
6. Save inference outputs for RAGAS evaluation

### Licensing

This dataset is released under the Apache 2.0 license.

### Related Datasets

- **Evaluation Metrics**: `open-biosciences/biosciences-evaluation-metrics` (RAGAS scores for these records)
- **Golden Testset**: `open-biosciences/biosciences-golden-testset` (ground truth QA pairs)
- **Source Documents**: `open-biosciences/biosciences-sources` (knowledge base)
