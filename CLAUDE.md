# CLAUDE.md — biosciences-research

## Purpose

Research workflows, competency questions, and validation pipelines for the Open Biosciences platform. This repo is owned by the **Research Workflows Engineer** agent.

## Key Concepts

### Competency Questions

Competency questions (CQs) define what the platform should be able to answer. They drive knowledge graph construction and validation.

Example CQs:
- **CQ14**: "Given genes A and B reported as synthetic lethal, what drugs target their protein products, and are any in clinical trials?"
- **CQ15**: "What pathways are enriched among interaction partners of gene X?"

Each CQ maps to a specific traversal pattern across MCP servers.

### Graph-Builder Workflow

The `lifesciences-graph-builder` skill orchestrates multi-API traversals using the Fuzzy-to-Fact protocol:

1. **Resolve** — Fuzzy search to canonical CURIEs (HGNC, UniProt)
2. **Enrich** — Gather functional context (protein domains, GO terms)
3. **Expand** — Find interactions (STRING, BioGRID)
4. **Traverse** — Cross-database lookups (ChEMBL, ClinicalTrials.gov)
5. **Validate** — Cross-reference verification
6. **Persist** — Write to knowledge graph (Graphiti/Neo4j)

## Directory Structure (post-migration)

```
biosciences-research/
├── docs/
│   ├── competency-questions/
│   │   └── competency-questions-catalog.md
│   ├── research/               # Research outputs
│   └── speckit-standard-prompt.md
└── reference/                  # Reference materials
```

## Dependencies

- **Upstream**: `biosciences-mcp` (API tools), `biosciences-architecture` (schema compliance)
- **Downstream**: `biosciences-memory` (graph persistence)

## Conventions

- Competency questions use a numbered catalog (CQ01, CQ02, ...)
- Each CQ documents: question, required APIs, traversal pattern, expected output schema
- Validation reports compare agent outputs against known ground truth
- Python >=3.11, uv, hatchling, ruff, pyright
- Pydantic v2 for all models
- pytest with marker-based test organization

## Pre-Migration Source

Until Wave 4 migration: `/home/donbr/graphiti-org/lifesciences-research/docs/`
