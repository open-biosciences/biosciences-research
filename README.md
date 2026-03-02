# biosciences-research

Research workflows and outputs for the Open Biosciences platform — the place where research
questions are documented, executed, and their results curated.

Part of the [Open Biosciences](https://github.com/open-biosciences) multi-repo platform.

## Status

**Wave 4 (Validation) — In Progress.** Competency questions catalog and research docs have been migrated from the predecessor `lifesciences-research` repo (34 files in 4 directories, commit `13c64b0`). Prerequisites (Waves 2 and 3) are complete.

## What This Repo Contains

Content migrated from `lifesciences-research/docs/`:

### Competency Questions Catalog (CQ1–CQ15+)

Structured research questions that drive platform capabilities. Each entry documents:

- Research question statement
- Databases queried (with CURIE-resolved entity identifiers)
- Traversal pattern across MCP servers
- Expected output format
- Re-run instructions (reproducible)

Example research scenarios:

- **CQ8** — "What are the synthetic lethal partners for ARID1A in ovarian cancer?"
- **CQ2** — "What drugs target the ACVR1 pathway in fibrodysplasia ossificans progressiva?"
- **CQ14** — NSCLC drug repurposing via synthetic lethality pathways (TP53 partners from Feng et al. 2022)

### Research Outputs

Analysis artifacts, knowledge graph fragments, and validation reports produced by running
competency questions against the platform.

### Graph-Builder Workflow Outputs

Results from the `lifesciences-graph-builder` skill, which orchestrates multi-API traversals
using the Fuzzy-to-Fact protocol across biosciences-mcp servers.

### Reference Materials

Supporting documentation and background literature for research question domains.

## What Is NOT Here

SpecKit process artifacts (`specs/`, `.specify/`, SpecKit prompt docs) are **not** in this
repo. Per AGE-183 and AGE-184, those are governance artifacts and live in
[biosciences-program](https://github.com/open-biosciences/biosciences-program).

## Owner

Research Workflows Engineer (Agent 6)

## Dependencies

| Direction | Repo | Relationship |
|-----------|------|-------------|
| Upstream | [biosciences-mcp](https://github.com/open-biosciences/biosciences-mcp) | Graph-builder workflows call MCP tools for data retrieval |
| Upstream | [biosciences-memory](https://github.com/open-biosciences/biosciences-memory) | Research results persisted to the knowledge graph |

## Related Repos

- [biosciences-program](https://github.com/open-biosciences/biosciences-program) — ADRs, SpecKit artifacts, governance docs
- [biosciences-deepagents](https://github.com/open-biosciences/biosciences-deepagents) — LangGraph agents that execute research tasks
- [biosciences-evaluation](https://github.com/open-biosciences/biosciences-evaluation) — Quality metrics for research output accuracy
- [biosciences-program](https://github.com/open-biosciences/biosciences-program) — Migration tracking and cross-repo coordination

## License

MIT
