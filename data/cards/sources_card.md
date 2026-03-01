---
license: apache-2.0
task_categories:
- text-retrieval
- question-answering
tags:
- rag
- biosciences
- knowledge-graphs
- retrieval
- biomedical-nlp
pretty_name: Biosciences RAG Source Documents
size_categories:
- n<1K
---

# Biosciences RAG Source Documents

## Dataset Description

This dataset contains 140 page-level document chunks extracted from 10 biomedical research papers. The documents form the knowledge base for a Retrieval-Augmented Generation (RAG) system focused on biosciences topics including knowledge graphs, LLM applications in biomedicine, and protein interaction databases.

### Dataset Summary

- **Total Documents**: 140 pages from 10 research papers
- **Domain**: Biomedical NLP, knowledge graphs, RAG in life sciences
- **Format**: PDF pages with extracted text and metadata
- **Use Case**: RAG knowledge base for biosciences-related queries

### Source Papers

| Pages | Topic |
|-------|-------|
| 31 | Biomedical preprint (bioRxiv 2025.08.01.668022) |
| 24 | Evaluating LLMs for Gene-to-Phenotype Mapping |
| 18 | Biomedical NLP (arXiv 2405.04819) |
| 16 | A Hybrid LLM-Knowledge Graph Framework for Biomedical QA |
| 12 | Knowledge Graph-Enhanced LLM for Pan-Cancer QA |
| 11 | Biomedical NLP (arXiv 2406.18626) |
| 11 | Improving LLM Applications in Biomedicine with RAG (systematic review) |
| 8 | The STRING Database in 2025: Protein Networks with Directionality |
| 5 | STRING database (Nucleic Acids Research, gkv1277) |
| 4 | Computational biology (btad570) |

### Data Fields

- `page_content` (string): Extracted text content from the PDF page
- `metadata` (dict): Document metadata including:
  - `title`: Paper title
  - `author`: Paper authors
  - `page`: Page number
  - `total_pages`: Total pages in source document
  - `source`: Original file path
  - `format`: Document format (PDF)
  - `producer`, `creator`: PDF metadata
  - `keywords`, `subject`: Paper keywords and subject (where available)

### Data Splits

This dataset contains a single split with all 140 documents.

### Licensing

This dataset is released under the Apache 2.0 license.

### Dataset Creation

Created as part of the Open Biosciences RAG evaluation pipeline. Source PDFs are ingested via `make ingest` which runs `scripts/ingest_raw_pdfs.py` with LlamaIndex transformations (HeadlineSplitter, SummaryExtractor, EmbeddingExtractor, ThemesExtractor, NERExtractor).

### Related Datasets

- **Golden Testset**: `open-biosciences/biosciences-golden-testset` (12 QA pairs generated from these sources)
- **Evaluation Inputs**: `open-biosciences/biosciences-evaluation-inputs` (RAG outputs)
- **Evaluation Metrics**: `open-biosciences/biosciences-evaluation-metrics` (RAGAS scores)
