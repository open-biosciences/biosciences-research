# CQ Plugin Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `/ob-cq-discover` and `/ob-cq-run` commands to the bio-research plugin with local file persistence instead of Graphiti.

**Architecture:** Two commands in `open-biosciences-plugins/bio-research/commands/` backed by two existing skills (`biosciences-cq-discover` and `biosciences-cq-runner`). The CQ runner delegates execution to the `biosciences-graph-builder` skill and persists results to local `.ob-cq/` directory files. No Graphiti dependency.

**Tech Stack:** Markdown skill files (no Python code), HuggingFace datasets library, DuckDB `hf://` protocol, biosciences-mcp gateway.

**Design Doc:** `docs/plans/2026-03-02-cq-plugin-design.md`

---

### Task 1: Create `ob-cq-discover.md` command

**Files:**
- Create: `open-biosciences-plugins/bio-research/commands/ob-cq-discover.md`

**Step 1: Write the command file**

```markdown
---
description: Browse, filter, and inspect competency questions from the HuggingFace dataset
argument-hint: "[filter | cq_id | --analysis]"
---

# Bio-Research: Discover Competency Questions

> If you see unfamiliar placeholders or need to check which tools are connected, see [CONNECTORS.md](../CONNECTORS.md).

You are browsing the competency questions catalog using the `biosciences-cq-discover` skill. This command loads CQ definitions from the `open-biosciences/biosciences-competency-questions-sample` HuggingFace dataset and presents them in three modes: summary table, filtered list, or detailed single-CQ view.

## Usage

```
/ob-cq-discover $ARGUMENTS
```

Where `$ARGUMENTS` is one of:
- **(empty)** -- show all 15 CQs as a summary table
- **Filter term** -- filter by category, reasoning type, complexity, disease area, or API name
- **CQ ID** -- e.g., `cq1`, `cq14` -- show full details for a specific CQ
- **`--analysis`** -- show cross-CQ analytics (API coverage, reasoning distribution, complexity breakdown)

## Workflow

### Step 1: Parse Arguments

Determine the mode from `$ARGUMENTS`:
- No arguments → Mode 1 (Summary Table)
- Matches a `cq_id` pattern (e.g., `cq1`, `cq14`) → Mode 3 (Detail View)
- `--analysis` → Cross-CQ Analysis
- Anything else → Mode 2 (Filtered List)

### Step 2: Execute Query

Use the `biosciences-cq-discover` skill to load data from HuggingFace via DuckDB and render the appropriate output.

### Step 3: Offer Next Steps

After displaying results, suggest:
- Run `/ob-cq-run <cq_id>` to execute a specific CQ against live MCP servers
- Run `/ob-cq-discover <filter>` to narrow results
- Run `/ob-cq-discover <cq_id>` to see full details

## Tips

- **No API keys needed**: This command only reads the public HuggingFace dataset
- **Quick overview**: Run with no arguments to see all 15 CQs at a glance
- **Find by API**: Use an API name like `STRING` or `ChEMBL` to find CQs that use that API
- **Detail view**: Use a CQ ID to see entities, edges, workflow steps, and gold standard path
```

**Step 2: Verify file exists**

Run: `ls -la open-biosciences-plugins/bio-research/commands/ob-cq-discover.md`
Expected: File exists with non-zero size

---

### Task 2: Create `ob-cq-run.md` command

**Files:**
- Create: `open-biosciences-plugins/bio-research/commands/ob-cq-run.md`

**Step 1: Write the command file**

```markdown
---
description: Run and validate a competency question from the HuggingFace dataset against live MCP servers
argument-hint: "<cq_id | list | all> [--publish] [--no-persist]"
---

# Bio-Research: Run Competency Question

> If you see unfamiliar placeholders or need to check which tools are connected, see [CONNECTORS.md](../CONNECTORS.md).

You are executing a structured competency question (CQ) from the `open-biosciences/biosciences-competency-questions-sample` HuggingFace dataset using the `biosciences-cq-runner` skill. This command loads a CQ definition, delegates execution to the `biosciences-graph-builder` skill's Fuzzy-to-Fact protocol, validates results against the gold standard, persists results locally, and generates a validation report.

## Usage

```
/ob-cq-run $ARGUMENTS
```

Where `$ARGUMENTS` is one of:
- **CQ ID**: e.g., `cq1`, `cq14` -- run a specific competency question
- **`list`** -- show all available CQs and prompt for selection
- **`all`** -- batch-run all CQs with a summary table at the end

Optional flags (append after the CQ ID):
- **`--publish`** -- publish validation results to HuggingFace (requires HF_TOKEN)
- **`--no-persist`** -- skip local file persistence (dry-run testing)

## How It Works

```
  /ob-cq-run cq1
       |
  LOAD -----> Load CQ definition from HuggingFace dataset
       |        (fallback: local competency-questions-catalog.md)
  PREFLIGHT -> Verify MCP gateway connectivity
       |
  EXECUTE ---> Delegate workflow_steps to graph-builder
       |        (Fuzzy-to-Fact: Anchor → Enrich → Expand → Traverse)
  VALIDATE --> Compare results vs gold_standard_path + biolink_edges
       |        (entity matching, edge matching, path verdict)
  PERSIST ---> Save per-CQ results to local .ob-cq/ directory
       |        (skip if --no-persist)
  REPORT ----> Generate validation report (pass/fail per step)
       |
  PUBLISH ---> (Optional) Push results to HuggingFace
```

## Workflow

### Step 1: Parse Arguments

Parse `$ARGUMENTS` to determine:
- **Mode**: single CQ, list, or batch
- **CQ ID**: if single mode, extract the ID (e.g., `cq1`, `cq14`)
- **Flags**: `--publish`, `--no-persist`

If mode is `list`, display the CQ catalog table and prompt for selection.

### Step 2: Load CQ Definition

Load the CQ from the HuggingFace dataset:

```python
from datasets import load_dataset
ds = load_dataset("open-biosciences/biosciences-competency-questions-sample", split="train")
cq = [row for row in ds if row["cq_id"] == cq_id][0]
```

If the dataset load fails, fall back to parsing `biosciences-research/docs/competency-questions-catalog.md`.

### Step 3: Preflight Checks

Verify the execution environment:
1. Test biosciences-mcp gateway connectivity
2. Check HF_TOKEN (warn if missing and --publish requested)
3. Verify all tools referenced by this CQ's workflow steps are available

If the gateway is not connected, abort and suggest running `/start`.

### Step 4: Execute Workflow Steps

Delegate each step from the CQ's `workflow_steps` to the `biosciences-graph-builder` skill following the Fuzzy-to-Fact protocol. Show progress as each step executes:

- Use MCP tools as primary, curl as fallback
- Chain step outputs (CURIEs resolved in step N feed into step N+1)
- Handle failures gracefully (skip and continue, never abort)
- Respect API rate limits (1s for STRING, 0.5s for NCBI)

### Step 5: Validate Results

Compare execution results against the CQ's gold standard:
- **Entity validation**: Did we resolve each `key_entity` to the expected CURIE?
- **Edge validation**: Did we discover each `biolink_edge`?
- **Path validation**: Does the discovered mechanism match `gold_standard_path`?
- **Score**: Compute pass/fail per dimension and overall verdict

### Step 6: Persist Results Locally

Unless `--no-persist` was specified, write per-CQ results to local files:

```
.ob-cq/
├── {cq_id}/
│   ├── results.json       # Full execution results (accumulator + metadata)
│   ├── entities.json      # Resolved entities with CURIEs
│   ├── edges.json         # Discovered BioLink edges
│   └── validation.json    # Gold standard comparison + scores
└── summary.json           # Batch run summary (when using /ob-cq-run all)
```

### Step 7: Present Validation Report

Output the full validation report following the template in the `biosciences-cq-runner` skill.

### Step 8: Publish (Optional)

If `--publish` was specified and HF_TOKEN is set:
- Confirm with user before publishing
- Push validation results to `open-biosciences/biosciences-cq-validations`

### Step 9: Offer Next Steps

Suggest follow-up actions:
- Run `/ob-report` to format a full evidence-graded report from the validated graph
- Run `/ob-review` to evaluate the report against 10 quality dimensions
- Run `/ob-publish` to generate the full publication pipeline
- Run `/ob-cq-run all` to validate all CQs as a regression suite
- Re-run `/ob-cq-run {id}` after fixing any identified issues

## Tips

- **Run `/start` first**: Verify MCP servers are connected before running CQs
- **Start with a simple CQ**: `cq1` (FOP Mechanism) has 6 steps and is a good first test
- **Batch mode for regression**: `all` mode validates the entire platform against 15 gold standards
- **No-persist for testing**: Use `--no-persist` when you want to validate without writing local files
- **Catalog corrections**: The validation report identifies CURIE discrepancies between the catalog and live APIs -- use these to improve the dataset
- **ChEMBL flakiness**: If ChEMBL steps fail with 500 errors, that is a known API issue, not a platform bug. The runner falls back to Open Targets automatically.
```

**Step 2: Verify file exists**

Run: `ls -la open-biosciences-plugins/bio-research/commands/ob-cq-run.md`
Expected: File exists with non-zero size

---

### Task 3: Delete old `ob-cq.md` command

**Files:**
- Delete: `open-biosciences-plugins/bio-research/commands/ob-cq.md`

**Step 1: Remove the file**

Run: `rm open-biosciences-plugins/bio-research/commands/ob-cq.md`

**Step 2: Verify deletion**

Run: `ls open-biosciences-plugins/bio-research/commands/`
Expected: `ob-cq.md` is gone, `ob-cq-discover.md` and `ob-cq-run.md` are present

---

### Task 4: Rewrite `biosciences-cq-runner/SKILL.md`

**Files:**
- Modify: `open-biosciences-plugins/bio-research/skills/biosciences-cq-runner/SKILL.md`

This is the largest task. Key changes from the existing 896-line file:

1. **Frontmatter description**: Update to mention local persistence and graph-builder delegation
2. **Header**: Remove "persist to Graphiti" from the summary
3. **Prerequisites**: Remove Graphiti Docker from optional, add note about graph-builder skill
4. **Execution Flow diagram**: Replace step 5 "PERSIST to Graphiti" with "PERSIST locally"
5. **Step 5 section**: Complete rewrite — local file persistence to `.ob-cq/` instead of Graphiti
6. **Preflight checks**: Remove Graphiti Docker check
7. **Step 3 EXECUTE**: Add explicit note that execution delegates to graph-builder
8. **All `/ob-cq` references**: Change to `/ob-cq-run`
9. **See Also section**: Remove Graphiti references, add graph-builder delegation note
10. **Error handling**: Remove "Graphiti Unavailable" section, update "Graphiti Instance Selection" removal

The complete rewritten SKILL.md content is provided in the implementation (too large to inline here — see the actual file write in the execution step).

---

### Task 5: Update `biosciences-cq-discover/SKILL.md`

**Files:**
- Modify: `open-biosciences-plugins/bio-research/skills/biosciences-cq-discover/SKILL.md`

**Changes:**

1. Update "Next Steps" in Mode 3 detail view:
   - Change `/ob-research` reference to `/ob-cq-run`
   - Change `/cq-discover` references to `/ob-cq-discover`
   - Remove Graphiti query suggestion

2. Update "Relationship to Other Skills" table:
   - Change "Run `/ob-research [CQ question]`" to "Run `/ob-cq-run [cq_id]`"
   - Change "biosciences-graph-builder skill" reference
   - Remove Graphiti query row

3. Update "Example Invocations":
   - Change `/cq-discover` to `/ob-cq-discover` in all examples

4. Update error handling:
   - Change `/cq-discover` fallback suggestion to `/ob-cq-discover`

---

### Task 6: Commit all changes

**Files:**
- All files from Tasks 1-5

**Step 1: Stage files in open-biosciences-plugins**

Run from `open-biosciences-plugins/`:
```bash
git add bio-research/commands/ob-cq-discover.md bio-research/commands/ob-cq-run.md
git add bio-research/skills/biosciences-cq-runner/SKILL.md bio-research/skills/biosciences-cq-discover/SKILL.md
git rm bio-research/commands/ob-cq.md
```

**Step 2: Commit**

```bash
git commit -m "feat: add ob-cq-discover and ob-cq-run commands with local persistence

Split the single /ob-cq command into /ob-cq-discover (browse/filter CQs)
and /ob-cq-run (execute CQs via graph-builder with local file persistence).

Key changes:
- Persistence uses local .ob-cq/ directory instead of Graphiti
- CQ runner delegates execution to biosciences-graph-builder skill
- No Graphiti dependency required
- Command prefix follows ob-cq- convention

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

**Step 3: Stage and commit design doc + implementation plan in biosciences-research**

Run from `biosciences-research/`:
```bash
git add docs/plans/2026-03-02-cq-plugin-design.md docs/plans/2026-03-02-cq-plugin-implementation.md
git commit -m "docs: add CQ plugin design and implementation plan

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```
