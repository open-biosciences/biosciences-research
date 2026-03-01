---
license: apache-2.0
task_categories:
- question-answering
- text-retrieval
tags:
- rag
- ragas
- evaluation
- metrics
- biosciences
- retrieval-comparison
pretty_name: Biosciences RAG Evaluation Metrics
size_categories:
- n<1K
---

# Biosciences RAG Evaluation Metrics

## Dataset Description

This dataset contains detailed RAGAS evaluation results with per-question metric scores for 4 retrieval strategies tested on the biosciences RAG system. Each record includes the full evaluation context (question, contexts, response) plus 4 RAGAS metric scores.

### Dataset Summary

- **Total Examples**: 48 records (12 questions x 4 retrievers)
- **Retrievers Evaluated**: Naive, BM25, Ensemble, Cohere Rerank
- **Metrics Per Record**: 4 RAGAS metrics
- **Questions Per Retriever**: 12 from golden testset
- **RAGAS Version**: 0.2.10

### Data Fields

- `retriever` (string): Retriever strategy (`naive` | `bm25` | `ensemble` | `cohere_rerank`)
- `user_input` (string): The question or query
- `retrieved_contexts` (list[string]): Document chunks retrieved by the retriever
- `reference_contexts` (list[string]): Ground truth context passages
- `response` (string): LLM-generated answer
- `reference` (string): Ground truth answer
- `faithfulness` (float): Score 0-1, measures if answer is grounded in retrieved contexts
- `answer_relevancy` (float): Score 0-1, measures if answer addresses the question
- `context_precision` (float): Score 0-1, measures if relevant contexts are ranked higher
- `context_recall` (float): Score 0-1, measures if ground truth information was retrieved

### RAGAS Metrics Explained

**Faithfulness** (Higher is Better):
- Evaluates if the generated answer is factually grounded in retrieved contexts
- Detects hallucinations and unsupported claims
- Score of 1.0 means every claim in the answer is supported by contexts

**Answer Relevancy** (Higher is Better):
- Measures how well the answer addresses the specific question
- Penalizes generic or off-topic responses
- Score of 1.0 means answer is perfectly relevant to question

**Context Precision** (Higher is Better):
- Evaluates retrieval ranking quality
- Measures if relevant contexts appear earlier in results
- Score of 1.0 means all relevant contexts ranked at top

**Context Recall** (Higher is Better):
- Measures if ground truth information was successfully retrieved
- Evaluates retrieval coverage and completeness
- Score of 1.0 means all reference contexts were retrieved

### Aggregate Performance Results

| Retriever | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Average |
|-----------|-------------|------------------|-------------------|----------------|---------|
| Ensemble | 0.9497 | 0.9680 | 0.7658 | 0.9603 | 91.09% |
| Cohere Rerank | 0.9044 | 0.9657 | 0.9167 | 0.8393 | 90.65% |
| BM25 | 0.8601 | 0.9645 | 0.8354 | 0.9325 | 89.81% |
| Naive | 0.9258 | 0.8889 | 0.6831 | 0.7773 | 81.88% |

**Key Insights**:
- Ensemble achieves best overall average (91.09%), +11.3% over naive baseline
- Cohere Rerank leads in context precision (0.9167) — best ranking quality
- Ensemble leads in context recall (0.9603) — best coverage
- Naive has highest faithfulness (0.9258) but worst retrieval quality
- BM25 surprisingly competitive, outperforming naive by 8 percentage points

### Data Splits

This dataset contains a single split with all 48 evaluation records.

### Evaluation Configuration

**Models**:
- LLM: gpt-4.1-mini (temperature=0)
- Embeddings: text-embedding-3-small
- Reranker: rerank-v3.5 (Cohere)
- RAGAS: v0.2.10

**Infrastructure**:
- Vector Store: Qdrant (localhost:6333)
- Chunk Strategy: Page-level (140 documents from 10 research papers)

### Use Cases

- Analyze which retrieval strategy works best for specific biomedical question types
- Study correlation between retrieval quality and answer quality
- Compare new retrieval strategies against these 4 baselines
- Debug retrieval failures using retrieved_contexts field

### Licensing

This dataset is released under the Apache 2.0 license.

### Related Datasets

- **Evaluation Inputs**: `open-biosciences/biosciences-evaluation-inputs` (same records without metric scores)
- **Golden Testset**: `open-biosciences/biosciences-golden-testset` (ground truth QA pairs)
- **Source Documents**: `open-biosciences/biosciences-sources` (knowledge base)
