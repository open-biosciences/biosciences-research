# CLAUDE.md — biosciences-research

## Purpose

This repo serves two complementary purposes:

1. **GDELT RAG Evaluation Pipeline** — Working code that benchmarks 4 retrieval strategies (naive, bm25, ensemble, cohere_rerank) against RAGAS metrics using 10 biosciences PDFs as source material
2. **Competency Questions & Prior Art Research** — 15 structured CQs, Paul Zamora's 12 oncology questions, and API pattern documentation that define what the Open Biosciences platform should answer

Owned by the **Research Workflows Engineer** (Agent #6).

## Quick Start

```bash
uv sync                          # Install dependencies
cp .env.example .env             # Configure (edit with real API keys)
make validate                    # 23-point environment check
make eval                        # RAGAS evaluation (~20-30 min, ~$5-6)
make deliverables                # Generate CSV from Parquet
make help                        # All available commands
```

## Directory Structure

```
biosciences-research/
├── src/                         # Core RAG library (factory pattern)
│   ├── config.py                #   Cached singletons (LLM, embeddings, Qdrant)
│   ├── state.py                 #   TypedDict schema for LangGraph state
│   ├── prompts.py               #   RAG prompt templates
│   ├── retrievers.py            #   Factory: create_retrievers()
│   ├── graph.py                 #   Factory: build_graph(), build_all_graphs()
│   └── utils/                   #   HuggingFace loaders, manifest generation
├── scripts/                     # Evaluation, ingestion, publishing scripts
├── data/
│   ├── raw/                     #   10 source PDFs
│   ├── interim/                 #   Extracted sources + golden testset (38 docs, 12 QA pairs)
│   ├── processed/               #   RAGAS evaluation outputs (Parquet)
│   └── cards/                   #   HuggingFace dataset card templates
├── deliverables/                # Derived CSV reports (regenerable via make deliverables)
├── docs/                        # CQ catalogs, prior art, research context
├── templates/                   # HuggingFace dataset card template
├── tests/                       # Test package (placeholder — validation via make validate)
├── Makefile                     # All commands
├── langgraph.json               # LangGraph config (entry: app.graph_app:get_app)
└── pyproject.toml               # Package: biosciences-research, Python >=3.11
```

## RAG Evaluation Pipeline

**Data flow:** `data/raw/ (10 PDFs) → data/interim/ (sources + golden testset) → data/processed/ (RAGAS metrics) → deliverables/ (CSV)`

**4 retrieval strategies:**
- **naive** — Dense vector search (Qdrant)
- **bm25** — Sparse keyword matching
- **ensemble** — Hybrid (50% dense + 50% sparse)
- **cohere_rerank** — Retrieve 20 → Cohere rerank to top k

**4 RAGAS metrics:** faithfulness, answer_relevancy, context_precision, context_recall

**Architecture:** Factory pattern — retrievers and LangGraph workflows instantiated at runtime, not import time. Pipeline: `START → retrieve → generate → END`. Inference and evaluation are architecturally decoupled: `*_evaluation_inputs.parquet` is written immediately after RAG inference, before RAGAS scoring — if RAGAS fails mid-run, inference results are preserved. See `src/README.md` for module details.

## Competency Questions & Research

- **15 CQs** (CQ1–CQ15) in `docs/competency-questions-catalog.md` — structured research questions driving knowledge graph construction, spanning FOP mechanisms, Alzheimer's, MAPK signaling, synthetic lethality, drug safety, and clinical trials
- **12 oncology CQs** from Paul Zamora in `docs/competency-questions-paul.md` — Doxorubicin toxicity, tumor microenvironment, NSCLC synthetic lethality
- **Prior art & API patterns** in `docs/prior-art-api-patterns.md` — positions work within STRING, TRAPI, BioThings landscape
- Each CQ documents: question, key entities with CURIEs, workflow steps, target `group_id` for Graphiti persistence
- CQs use the Fuzzy-to-Fact protocol and `lifesciences-graph-builder` skill
- Future: CQ catalogs and results to be managed as HuggingFace datasets

## Makefile Commands

**Data Preparation:**
- `make ingest` — Extract PDFs and generate golden testset (~5-10 min, ~$2-3)

**Development:**
- `make validate` — 23-point environment and module validation (100% pass required)
- `make eval` — Run RAGAS evaluation, reuse Qdrant collection
- `make eval recreate=true` — Force fresh Qdrant collection (adds ~5 min)
- `make deliverables` — Generate CSV files from Parquet data
- `make test` — Alias for validate
- `make notebook` — Start Jupyter notebook server

**Publishing (requires HF_TOKEN):**
- `make publish-interim` — Upload sources & golden testset to HuggingFace Hub
- `make publish-processed` — Upload evaluation results to HuggingFace Hub

**Environment:**
- `make env` — Show environment variable status

**Cleanup:**
- `make clean` — Clean Python cache and temp files
- `make clean-deliverables` — Clean derived CSVs (regenerable)
- `make clean-processed` — Clean processed data (requires re-eval)
- `make clean-all` — Full cleanup (cache + interim + processed + deliverables)

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENAI_API_KEY` | Yes | LLM (gpt-4.1-mini) and embeddings (text-embedding-3-small) |
| `COHERE_API_KEY` | Yes | Cohere rerank-v3.5 (cohere_rerank retriever fails without it) |
| `QDRANT_HOST` | Yes | Qdrant vector store host |
| `QDRANT_API_KEY` | Cloud only | Qdrant Cloud authentication (omit for local Docker) |
| `QDRANT_COLLECTION` | Yes | Collection name (default: `biosciences-data-sources`) |
| `HF_TOKEN` | Publishing | HuggingFace Hub uploads |
| `LANGSMITH_API_KEY` | Optional | LangSmith tracing |
| `LANGSMITH_PROJECT` | Optional | LangSmith project name |
| `OPENAI_MODEL_NAME` | Optional | Override default model (gpt-4.1-mini) |
| `EMBEDDING_MODEL_NAME` | Optional | Override default embeddings (text-embedding-3-small) |
| `COHERE_RERANK_MODEL` | Optional | Override default reranker (rerank-v3.5) |
| `HF_SOURCES_REV` | Optional | Pin HuggingFace sources dataset revision (e.g., `main@abc123`) for reproducible eval scores |
| `HF_GOLDEN_REV` | Optional | Pin HuggingFace golden testset revision for reproducible eval scores |

## Key Technologies

- **LangGraph** 0.6.7 — RAG pipeline orchestration (`START → retrieve → generate → END`)
- **LangChain** — Document loading, retriever abstractions, prompt templates
- **RAGAS** 0.2.10 — Retrieval-augmented generation evaluation metrics
- **Qdrant** — Vector store for dense retrieval
- **OpenAI** — gpt-4.1-mini (generation), text-embedding-3-small (embeddings)
- **Cohere** — rerank-v3.5 for contextual compression retrieval
- **HuggingFace Hub** — Dataset versioning and publishing
- **uv** — Package management, virtual environments

## Gotchas

- `PYTHONPATH=.` is required for direct script execution — Makefile handles this automatically
- `make eval` reuses the existing Qdrant collection by default; use `recreate=true` for fresh embeddings
- `langgraph.json` references `app.graph_app:get_app` — this is a placeholder entry point that doesn't exist yet
- `tests/` contains only `__init__.py`; validation is done via `make validate`, not pytest
- Cohere rerank retriever fails silently without `COHERE_API_KEY` set
- Full evaluation run costs ~$5-6 in API calls (OpenAI + Cohere)
- Script naming encodes idempotency: `run_*` = repeatable, `ingest_*` and `publish_*` = one-time operations

## Platform Context

- **Agent**: Research Workflows Engineer (#6)
- **Upstream**: `biosciences-mcp` (API tools), `biosciences-memory` (graph persistence)
- **Downstream**: `biosciences-evaluation` (quality metrics)
- **Conventions**: Python >=3.11, uv, hatchling, ruff, pyright, Pydantic v2
