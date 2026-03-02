# CQ Plugin Design — Competency Question Discovery & Execution

**Date**: 2026-03-02
**Status**: Approved
**Plugin**: `open-biosciences-plugins/bio-research`
**Dataset**: [open-biosciences/biosciences-competency-questions-sample](https://huggingface.co/datasets/open-biosciences/biosciences-competency-questions-sample)

---

## Overview

Add two commands to the existing `bio-research` plugin for discovering and executing competency questions from the HuggingFace dataset. Commands follow the `ob-cq-` prefix convention.

## Commands

| Command | Skill | Purpose |
|---------|-------|---------|
| `/ob-cq-discover` | `cq-discover` | Browse, filter, and inspect CQs from the HuggingFace dataset |
| `/ob-cq-run` | `biosciences-cq-runner` | Load a CQ definition and execute it via graph-builder |

## Architecture

### cq-discover

Zero-dependency browsing of the 15 CQs. Three modes:

- **Summary table**: Show all CQs with ID, question, category, complexity, hops
- **Filtered list**: Filter by category, reasoning type, complexity, API
- **Detail view**: Full CQ definition with entities, edges, workflow steps, gold standard path

Uses DuckDB `hf://` protocol for direct dataset access without downloading. Falls back to local `docs/competency-questions-catalog.md` if HuggingFace is unreachable.

### biosciences-cq-runner

Structured entry point that loads a CQ definition and delegates execution to the existing `biosciences-graph-builder` skill's Fuzzy-to-Fact flow.

**Execution flow:**

```
/ob-cq-run cq1
    │
    LOAD ──────→ Load CQ definition from HuggingFace dataset
    │              (fallback: local competency-questions-catalog.md)
    PREFLIGHT ──→ Verify biosciences-mcp gateway is connected
    │
    EXECUTE ────→ Delegate each workflow_step to graph-builder
    │              (Fuzzy-to-Fact: Anchor → Enrich → Expand → Traverse)
    VALIDATE ───→ Compare results vs gold_standard_path + biolink_edges
    │
    PERSIST ────→ Write per-CQ results to local Cowork directory
    │              (.ob-cq/cq1/results.json, entities.json, edges.json)
    REPORT ─────→ Validation report (pass/fail per dimension)
    │
    PUBLISH ────→ (Optional, --publish) Push results to HuggingFace
```

**Key principle**: The runner is NOT a standalone execution engine. It loads the CQ's entities, workflow steps, and gold standard, then feeds them into graph-builder. Graph-builder does the actual API orchestration.

## Design Decisions

1. **Persistence is local files, not Graphiti.** Each CQ gets its own directory under `.ob-cq/` in the Cowork workspace. Graphiti is optional/long-term — not a required dependency.

2. **CQ runner delegates to graph-builder.** The runner provides structure (what to look up, what to validate against); graph-builder provides execution (MCP tool calls, Fuzzy-to-Fact protocol).

3. **No `.mcp.json` changes needed.** The biosciences-mcp gateway is already configured in the bio-research plugin. No graphiti-docker dependency.

4. **No create/generate capability.** The plugin discovers and runs existing CQs only. New CQs are authored manually in the catalog and published to the HuggingFace dataset.

5. **Command prefix is `ob-cq-`** for consistency with other bio-research commands (`ob-research`, `ob-report`, `ob-review`, `ob-publish`).

## Local Persistence Structure

```
.ob-cq/
├── cq1/
│   ├── results.json       # Full execution results
│   ├── entities.json      # Resolved entities with CURIEs
│   ├── edges.json         # Discovered BioLink edges
│   └── validation.json    # Gold standard comparison
├── cq2/
│   └── ...
└── summary.json           # Batch run summary (when using /ob-cq-run all)
```

## Files to Create/Update

| File | Action | Description |
|------|--------|-------------|
| `commands/ob-cq-discover.md` | Create (rename from `ob-cq.md`) | Command frontmatter pointing to cq-discover skill |
| `commands/ob-cq-run.md` | Create | Command frontmatter for CQ execution |
| `commands/ob-cq.md` | Delete | Replaced by `ob-cq-discover` and `ob-cq-run` |
| `skills/biosciences-cq-runner/SKILL.md` | Rewrite | Remove Graphiti dependency, add local persistence, reframe as graph-builder delegation |
| `skills/cq-discover/SKILL.md` | Review | Minor consistency updates if needed |

## Validation

Each CQ execution validates three dimensions:

- **Entity validation**: Did we resolve each `key_entity` to the expected CURIE?
- **Edge validation**: Did we discover each `biolink_edge`?
- **Path validation**: Does the discovered mechanism match `gold_standard_path`?

Results are scored per-dimension with an overall verdict (pass/partial/fail).

## Usage Examples

```bash
# Browse all CQs
/ob-cq-discover

# Filter by reasoning type
/ob-cq-discover multi_hop_traversal

# View a specific CQ in detail
/ob-cq-discover cq1

# Run a specific CQ
/ob-cq-run cq1

# Run with no local persistence (dry run)
/ob-cq-run cq1 --no-persist

# Run all CQs as regression suite
/ob-cq-run all

# Run and publish results to HuggingFace
/ob-cq-run cq1 --publish
```
