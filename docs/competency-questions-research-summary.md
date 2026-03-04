# Competency Questions in Biosciences Research: Research Summary

**Date**: 2026-03-02
**Context**: Research conducted by 3 parallel agents to inform the [biosciences-competency-questions-sample](https://huggingface.co/datasets/open-biosciences/biosciences-competency-questions-sample) HuggingFace dataset.

---

## 1. What Are Competency Questions?

Competency questions (CQs) are natural language questions that a knowledge graph or ontology must be able to answer. They function as **requirements specifications** for knowledge representation systems -- defining scope, driving schema design, and providing testable acceptance criteria.

The term was coined by **Gruninger & Fox (1995)** in their work on enterprise ontology engineering. Their core insight: ontological engineering should begin by defining the *problems* an ontology needs to solve, expressed as scenarios and questions. CQs do not themselves generate ontological commitments -- they are used to **evaluate** the commitments that have been made.

The **NeOn Methodology** (Suarez-Figueroa et al., 2012) extended this into a full lifecycle: CQs appear at requirements specification, conceptualization, formalization, and implementation stages.

## 2. CQ Type Taxonomy (Keet & Khan 2024)

The [ROCQS taxonomy](https://arxiv.org/abs/2412.13688) (Repository of Ontology Competency Questions, 438 annotated CQs) identifies five types:

| Type | Abbrev. | Purpose | Biosciences Example |
|------|---------|---------|---------------------|
| **Scoping** | SCQ | Demarcate what the KG covers | "What genes are implicated in Alzheimer's progression?" (cq3) |
| **Validating** | VCQ | Verify KG contains correct knowledge | "How do we validate the p53-MDM2-Nutlin axis?" (cq11, cq14) |
| **Foundational** | FCQ | Align entities to upper ontology categories | Implicit in BioLink typing across all CQs |
| **Relationship** | RCQ | Examine inter-entity relationship properties | "What TFs regulate BRCA1, and what does BRCA1 regulate?" (cq6) |
| **Metaproperty** | MpCQ | Classify entities by ontological characteristics | Not directly represented in current catalog |

Scoping and Validating CQs are the most relevant for applied biosciences KG construction. Foundational, Relationship, and Metaproperty CQs matter when aligning to BioLink or upper ontologies.

## 3. Four Roles CQs Play in Knowledge Graph Construction

| Role | Description |
|------|-------------|
| **Scoping** | Delineate what the KG must represent. "What drugs target amyloid-beta?" implies nodes for drugs and proteins, plus targeting edges. |
| **Validation** | Testable acceptance criteria. Each CQ should return correct answers; unanswerable CQs reveal gaps. |
| **Iteration** | Living documents that co-evolve with the KG. 63.5% of practitioners refine CQs iteratively (Bezerra et al. 2023). |
| **Interoperability** | Foundational CQs facilitate alignment between domain entities and upper-level ontologies. |

## 4. Reasoning Patterns

CQs encode distinct graph reasoning patterns. These map to both SPARQL query structures and NCATS Translator (TRAPI) query modes:

| Pattern | Hops | Description | TRAPI Mode | Our CQs |
|---------|------|-------------|------------|---------|
| **Single-hop lookup** | 1 | Direct entity-to-entity retrieval | Lookup | (subcomponents of most CQs) |
| **Multi-hop traversal** | 2-4 | Chain of typed edges | Inferred | cq1, cq2, cq4, cq8, cq11, cq14 |
| **Network expansion** | 1+fan | Anchor then expand to neighbors | Input_set | cq3, cq6 |
| **Directed traversal** | 2-3 | Follow regulatory direction (activation/inhibition) | Lookup | cq5 |
| **Federated multi-hop** | 4+ | Multi-hop across federated APIs | Inferred | cq7 |
| **Comparative analysis** | 2+ | Compare entities side-by-side | Lookup | cq9 |
| **Set difference** | 2+ | Elements in set A not in set B | Inferred | cq10 |
| **Aggregation** | 1+agg | Count, group, rank | -- | cq12 |
| **Ranking** | 2+agg | Order by computed metric | -- | cq13 |
| **Temporal analysis** | 2+time | Track progression over time | -- | cq15 |

The NCATS Translator supports four query modes: **Lookup** (single-hop), **Inferred** (multi-hop with scoring), **Pathfinder** (open-ended path discovery), and **Input_set** (shared-property discovery across batches).

## 5. Complexity Levels

Three frameworks define question complexity in biomedical QA:

### BioASQ Typology (standard benchmark)

| Level | Type | Description | Metric |
|-------|------|-------------|--------|
| L1 | Yes/No | Binary verification | Macro F1 |
| L2 | Factoid | Single entity answer | Mean Reciprocal Rank |
| L3 | List | Set of entities | Mean F1 |
| L4 | Summary | Synthesized text | ROUGE |

### Graph Structural Complexity (PrimeKGQA)

| Level | Subgraph | Description |
|-------|----------|-------------|
| Simple | 2-node (1-hop) | Direct relationship lookup |
| Moderate | 3-node (2-hop) | One intermediate entity |
| Complex | 4-node (3-hop) | Two intermediate entities |

### Applied Biosciences Complexity (synthesis)

| Complexity | Definition | CQ Examples |
|------------|-----------|-------------|
| **Factoid** | Single lookup, deterministic | (subcomponents) |
| **Relational** | 2-3 hop chain of API calls | cq1, cq4, cq5, cq11 |
| **Exploratory** | Fan-out expansion + filtering | cq2, cq3, cq6 |
| **Analytical** | Aggregation, comparison, gap analysis | cq9, cq10, cq12, cq13 |
| **Synthetic** | Multi-source integration requiring inference | cq7, cq14, cq15 |

## 6. CQ-to-Query Formalization

Wisniewski et al. (2019) analyzed 234 CQs formalized into 131 SPARQL-OWL queries:

- **106 distinct CQ linguistic patterns** reduced to 81 after normalization
- **46 distinct SPARQL query signatures** from 131 queries
- Top 10 query signatures cover **65.6%** of all queries
- The mapping is **many-to-many**: one CQ pattern can produce multiple valid SPARQL signatures

Key signal words:

| CQ Signal | Query Type | Reliability |
|-----------|-----------|-------------|
| "What/Which" at start | SELECT query | 100% |
| "Is/Are/Can/Does" at start | ASK (boolean) | 89% |
| "What types of" | Subclass hierarchy | 75% |

## 7. Existing Dataset Landscape

### Major Biomedical QA Benchmarks

| Dataset | Size | Key Feature |
|---------|------|-------------|
| [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) | 273K rows | Expert-labeled yes/no/maybe from abstracts |
| [BioASQ](https://participants-area.bioasq.org/datasets/) | 5K+ curated | Gold standard with factoid/list/yes-no/summary types |
| [MedQA-USMLE](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) | 11.5K | Clinical multiple-choice |
| [MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa) | 194K | Medical MCQ with explanations |
| [MMLU (medical)](https://huggingface.co/datasets/cais/mmlu) | ~3.8K medical | 6 biomedical subsets |

### Knowledge Graph QA Datasets

| Dataset | Size | Key Feature |
|---------|------|-------------|
| [PrimeKGQA](https://zenodo.org/records/13829395) | 84K pairs | SPARQL against PrimeKG (20 biomedical DBs) |
| [BioKGBench](https://huggingface.co/datasets/AutoLab-Westlake/BioKGBench-Dataset) | 8K rows | KG check, QA, claim verification |
| [MedReason](https://huggingface.co/datasets/UCSC-VLAA/MedReason) | 33K pairs | KG-derived multi-step reasoning chains |

### Ontology CQ Datasets

| Dataset | Size | Key Feature |
|---------|------|-------------|
| [CQ-to-SPARQL-OWL](https://data.mendeley.com/datasets/pp6hmfxgfg/1) | 234 CQs | Largest manual CQ-to-query gold standard |
| [ROCQS](https://github.com/zubeidaiscyber/ROCQS) | 438 CQs | Type-annotated (SCQ/VCQ/FCQ/RCQ/MpCQ) |

### Gap Our Dataset Fills

No existing dataset combines all of:

| Feature | Ours | PrimeKGQA | BioASQ | MedReason | CQ-to-SPARQL |
|---------|:----:|:---------:|:------:|:---------:|:------------:|
| Standardized CURIEs (HGNC, CHEMBL, MONDO) | Yes | No | Partial | No | No |
| BioLink-typed entities & predicates | Yes | No | No | No | OWL |
| Gold standard graph paths | Yes | SPARQL | Triples | Text | SPARQL |
| Executable API workflow steps | Yes | No | No | No | No |
| Multi-database federation | Yes | Single KG | PubMed | Single KG | Single ontology |
| Fuzzy-to-Fact discovery protocol | Yes | No | No | No | No |

## 8. Why CQs Matter for AI/LLM-Powered Research Agents

1. **Grounding LLM outputs**: KGARevion (Zitnik Lab, Harvard, 2024) showed KG integration via CQ-derived schemas improves medical QA accuracy by 5.2-10.4% across 15 models.

2. **Schema supervision**: Systems like NeOn-GPT use CQs to delineate knowledge scope, construct ontologies (TBox), then perform data population (ABox) under schema supervision. Without CQs, LLM-generated KGs lack coherent scope boundaries.

3. **Evaluation benchmarks**: Each CQ becomes a test case -- can the agent retrieve the right entities, traverse the right relationships, and synthesize the correct answer?

4. **Human-agent interface**: CQs are human-readable (unlike SPARQL) yet precise enough to drive automated workflows. They define the contract between researcher intent and agent execution.

5. **Continuous KG evolution**: Agentic Medical Knowledge Graphs (2025) use LLM agents to continuously generate and refine knowledge graphs. CQs provide the compass for this evolution.

## 9. Mapping the Open Biosciences CQ Catalog

| CQ | Domain | Reasoning | Complexity | Hops | BioASQ Type | TRAPI Mode |
|----|--------|-----------|-----------|------|-------------|------------|
| cq1 | Mechanism of Action | Multi-hop | Relational | 3 | Factoid/Summary | Inferred |
| cq2 | Drug Repurposing | Fan-out + filter | Exploratory | 4 | List | Inferred |
| cq3 | Gene Networks | Subgraph expansion | Exploratory | 4 | List/Summary | Input_set |
| cq4 | Therapeutic Targets | Multi-hop | Relational | 4 | List | Inferred |
| cq5 | Regulatory Cascade | Directed traversal | Relational | 3 | List | Lookup |
| cq6 | Regulatory Networks | Bidirectional | Exploratory | 3 | List | Input_set |
| cq7 | Federated Repurposing | Federated multi-hop | Synthetic | 5 | List | Inferred |
| cq8 | Synthetic Lethality | Multi-hop + validation | Analytical | 4 | Summary | Inferred |
| cq9 | Drug Safety | Comparative | Analytical | 3 | List/Summary | Lookup |
| cq10 | Orphan Drug Gap | Gap analysis | Analytical | 4 | List | Inferred |
| cq11 | Pathway Validation | Multi-hop | Relational | 3 | Summary | Lookup |
| cq12 | Health Emergencies | Aggregation | Analytical | 2 | List/Summary | -- |
| cq13 | Commercialization | Ranking | Analytical | 4 | List | -- |
| cq14 | SL Validation | Multi-source | Synthetic | 4 | Summary | Inferred |
| cq15 | CAR-T Regulatory | Temporal | Synthetic | 3 | Summary | -- |

## 10. Key References

1. **Gruninger, M. & Fox, M.S. (1995)**. "The Role of Competency Questions in Enterprise Engineering." *IFIP AICT*. [Springer](https://link.springer.com/chapter/10.1007/978-0-387-34847-6_3)

2. **Ren, Y. et al. (2014)**. "Towards Competency Question-Driven Ontology Authoring." *ESWC 2014*. [Springer](https://link.springer.com/chapter/10.1007/978-3-319-07443-6_50)

3. **Wisniewski, D. et al. (2019)**. "Analysis of Ontology Competency Questions and their Formalizations in SPARQL-OWL." *J. Web Semantics*, 59. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1570826819300617)

4. **Bezerra, C. et al. (2023)**. "Use of Competency Questions in Ontology Engineering: A Survey." *ER 2023*. [Springer](https://link.springer.com/chapter/10.1007/978-3-031-47262-6_3)

5. **Keet, C.M. & Khan, Z.C. (2024)**. "Discerning and Characterising Types of Competency Questions for Ontologies." *EKAW 2024*. [arXiv](https://arxiv.org/abs/2412.13688)

6. **Li, D. et al. (2024)**. "DALK: Dynamic Co-Augmentation of LLMs and KG." *EMNLP 2024 Findings*. [arXiv](https://arxiv.org/abs/2405.04819)

7. **Szklarczyk, D. et al. (2025)**. "STRING 2025: protein networks with directionality." *Nucleic Acids Res.* [Oxford](https://academic.oup.com/nar/article/53/D1/D1295/7416599)

8. **Callaghan, J. et al. (2023)**. "BioThings Explorer: federated KG of biomedical APIs." *Bioinformatics*. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10153288/)

9. **Feng et al. (2022)**. "Mapping the landscape of synthetic lethal interactions." *Sci. Adv.* 8, eabm6638. [Science](https://www.science.org/doi/10.1126/sciadv.abm6638)

10. **Fecho, K. et al. (2022)**. "Progress toward a universal biomedical data translator." *Clin. Transl. Sci.* [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9372428/)

---

## Dataset

**HuggingFace**: [open-biosciences/biosciences-competency-questions-sample](https://huggingface.co/datasets/open-biosciences/biosciences-competency-questions-sample)
- 15 CQs, 15 columns per row
- 10 reasoning types, 8 APIs, 2-5 hop range
- BioLink-typed entities with standardized CURIEs
- Executable workflow steps and gold standard graph paths
