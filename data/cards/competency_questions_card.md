---
license: mit
task_categories:
- question-answering
- text-generation
tags:
- biosciences
- knowledge-graph
- competency-questions
- biolink
- drug-discovery
- synthetic-lethality
pretty_name: Biosciences Competency Questions
size_categories:
- n<1K
---

# Biosciences Competency Questions

## Dataset Description

Structured competency questions (CQs) for benchmarking agentic biosciences research pipelines. Contains 27 questions spanning drug mechanisms, gene networks, synthetic lethality, clinical trials, and tumor microenvironment research — with CURIE-annotated entities, BioLink gold standard graphs, and Fuzzy-to-Fact workflow definitions.

### Dataset Summary

- **Total Questions**: 27 (15 structured catalog + 12 oncology prose)
- **Entities**: 48 CURIE-annotated entities across 12 CQs
- **Gold Graphs**: 15 entries (13 with BioLink JSON, 12 validated)
- **Domain**: Biosciences — drug discovery, gene networks, clinical trials
- **Sources**: DrugMechDB, DALK, STRING 2025, BioThings Explorer, Feng et al. 2022, Paul Zamora

### Splits

| Split | Rows | Description |
|-------|------|-------------|
| `questions` | 27 | All competency questions with metadata |
| `entities` | 48 | CURIE-annotated entities from catalog CQs |
| `gold_graphs` | 15 | BioLink gold standard graphs + validation status |

### Data Fields

#### `questions` split

- `cq_id` (string): Unique identifier (`cq1`–`cq15`, `paul-cq1`–`paul-cq12`)
- `category` (string): Topic category (e.g., `fop_mechanism`, `doxorubicin_toxicity`)
- `question` (string): Full question text
- `source` (string): Origin reference (e.g., `DrugMechDB`, `Paul Zamora`)
- `group_id` (string): Graphiti persistence namespace
- `gold_path` (string, nullable): Expected mechanism path
- `workflow_steps` (string, nullable): JSON-serialized list of API call steps
- `expected_template` (int, nullable): Graph template index (reserved)
- `has_curies` (bool): `true` for catalog CQs with entity tables
- `has_gold_graph` (bool): `true` if BioLink JSON is available

#### `entities` split

- `cq_id` (string): Parent question identifier
- `entity_name` (string): Human-readable name (e.g., `Palovarotene`)
- `curie` (string): Compact URI (e.g., `CHEMBL:2105648`)
- `role` (string): Entity role in the question context
- `biolink_type` (string): BioLink category (e.g., `biolink:SmallMolecule`)

#### `gold_graphs` split

- `cq_id` (string): Parent question identifier
- `nodes_json` (string, nullable): JSON-serialized BioLink nodes array
- `edges_json` (string, nullable): JSON-serialized BioLink edges array
- `validation_status` (string): `VALIDATED`, `PARTIAL`, or `PENDING`
- `validation_date` (string, nullable): ISO date of validation

### Loading Examples

```python
from datasets import load_dataset

ds = load_dataset("open-biosciences/biosciences-competency-questions")

# Access splits
questions = ds["questions"]
entities = ds["entities"]
gold_graphs = ds["gold_graphs"]

# Filter catalog CQs with gold graphs
with_gold = questions.filter(lambda x: x["has_gold_graph"])

# Get entities for a specific CQ
cq1_entities = entities.filter(lambda x: x["cq_id"] == "cq1")
```

### Intended Use

- Benchmarking agentic bio-research pipelines against gold standard knowledge graphs
- Evaluating Fuzzy-to-Fact protocol implementations
- Comparing RAG-based vs. agentic multi-hop reasoning on structured questions
- Defining evaluation contracts: the schema specifies what a correct answer looks like

### Citation

Sources include:
- **DrugMechDB** — Drug mechanism database (CQ1–CQ2)
- **Li et al. (2024)** — DALK: Dynamic Co-Augmentation of LLMs and KG (CQ3–CQ4)
- **Szklarczyk et al. (2025)** — STRING database with regulatory directionality (CQ5–CQ6)
- **Callaghan et al. (2023)** — BioThings Explorer federated KG (CQ7)
- **Feng et al. (2022)** — Synthetic lethality gene pairs, Sci. Adv. (CQ14)
- **Paul Zamora** — Oncology competency questions (paul-cq1–paul-cq12)

### Related Datasets

- **Source Documents**: `open-biosciences/biosciences-sources` (140 page-level chunks)
- **Golden Test Set**: `open-biosciences/biosciences-golden-testset` (12 QA pairs)
- **Evaluation Inputs**: `open-biosciences/biosciences-evaluation-inputs`
- **Evaluation Metrics**: `open-biosciences/biosciences-evaluation-metrics`

### Licensing

This dataset is released under the MIT license as part of the Open Biosciences platform.
