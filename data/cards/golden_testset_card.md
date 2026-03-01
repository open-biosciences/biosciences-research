---
license: apache-2.0
task_categories:
- question-answering
- text-generation
tags:
- rag
- evaluation
- ragas
- biosciences
- biomedical-nlp
pretty_name: Biosciences RAG Golden Test Set
size_categories:
- n<1K
---

# Biosciences RAG Golden Test Set

## Dataset Description

This dataset contains 12 question-answering pairs for evaluating RAG systems on biomedical research topics. The QA pairs were synthetically generated using the RAGAS framework from 140 source documents spanning knowledge graphs, LLM applications in biomedicine, protein interaction databases, and gene-to-phenotype mapping.

### Dataset Summary

- **Total Examples**: 12 QA pairs
- **Purpose**: RAG system evaluation ground truth
- **Framework**: RAGAS v0.2.10 (synthetic test data generation)
- **Domain**: Biosciences — biomedical NLP, knowledge graphs, protein networks

### Data Fields

- `user_input` (string): The question or query
- `reference_contexts` (list[string]): Ground truth context passages that contain the answer
- `reference` (string): Ground truth answer
- `synthesizer_name` (string): RAGAS synthesizer used to generate the example:
  - `single_hop_specifc_query_synthesizer`: Single-hop specific queries
  - `multi_hop_specific_query_synthesizer`: Multi-hop specific queries
  - `multi_hop_abstract_query_synthesizer`: Multi-hop abstract queries

### Example Topics

The dataset includes questions about:
- Hybrid LLM-Knowledge Graph frameworks for biomedical QA
- Drug candidate discovery using integrative knowledge hubs
- Pan-cancer question answering with knowledge graphs
- Gene-to-phenotype mapping with LLMs
- Protein interaction databases and network analysis

### Data Splits

This dataset contains a single split with all 12 evaluation examples.

### Intended Use

This dataset is intended for:
- Evaluating RAG systems on biosciences research queries
- Benchmarking retrieval quality using RAGAS metrics:
  - Faithfulness
  - Answer Relevancy
  - Context Precision
  - Context Recall

### Licensing

This dataset is released under the Apache 2.0 license.

### Dataset Creation

Created using RAGAS synthetic test data generation as part of the Open Biosciences RAG evaluation pipeline (`make ingest` step). Three persona types generate diverse question styles across the 10 source papers.

### Related Datasets

- **Source Documents**: `open-biosciences/biosciences-sources` (140 page-level chunks)
- **Evaluation Inputs**: `open-biosciences/biosciences-evaluation-inputs` (RAG outputs)
- **Evaluation Metrics**: `open-biosciences/biosciences-evaluation-metrics` (RAGAS scores)
