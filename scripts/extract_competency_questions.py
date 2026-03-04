#!/usr/bin/env python3
"""
Extract competency questions from markdown source files into structured Parquet datasets.

Inputs:
  - docs/competency-questions-catalog.md (15 structured CQs)
  - docs/competency-questions-paul.md (12 prose CQs)
  - docs/competency-validations/*.md (validation files)

Outputs (to data/interim/):
  - cq_questions.parquet (27 rows: 15 catalog + 12 Paul)
  - cq_entities.parquet (~100 rows: all CURIE-annotated entities)
  - cq_gold_graphs.parquet (~15 rows: catalog CQs with validation status)
  - cq_manifest.json (SHA-256 checksums + provenance)
"""

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
DOCS_DIR = BASE_DIR / "docs"
CATALOG_PATH = DOCS_DIR / "competency-questions-catalog.md"
PAUL_PATH = DOCS_DIR / "competency-questions-paul.md"
VALIDATIONS_DIR = DOCS_DIR / "competency-validations"
OUTPUT_DIR = BASE_DIR / "data" / "interim"

# CURIE prefix → BioLink type
CURIE_TO_BIOLINK: dict[str, str] = {
    "CHEMBL": "biolink:SmallMolecule",
    "HGNC": "biolink:Gene",
    "MONDO": "biolink:Disease",
    "WP": "biolink:Pathway",
    "STRING": "biolink:Protein",
    "NCT": "biolink:ClinicalTrial",
    "ENSG": "biolink:Gene",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def hash_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def infer_biolink_type(curie: str) -> str:
    """Infer BioLink type from CURIE prefix."""
    prefix = curie.split(":")[0]
    return CURIE_TO_BIOLINK.get(prefix, "biolink:NamedThing")


def extract_biolink_from_text(text: str) -> dict | None:
    """Extract BioLink JSON (with nodes/edges) from fenced code blocks."""
    json_blocks = re.findall(r"```json\n(.*?)```", text, re.DOTALL)
    for block in json_blocks:
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        # Direct nodes/edges at top level
        if "nodes" in data and "edges" in data:
            return {"nodes": data["nodes"], "edges": data["edges"]}
        # Nested under "graph" key
        if "graph" in data and isinstance(data["graph"], dict):
            g = data["graph"]
            if "nodes" in g and "edges" in g:
                return {"nodes": g["nodes"], "edges": g["edges"]}
    return None


# ---------------------------------------------------------------------------
# Catalog parsing
# ---------------------------------------------------------------------------
def parse_quick_reference(text: str) -> dict[str, dict]:
    """Parse the Quick Reference table → {cq_id: {category, summary, group_id, source}}."""
    ref: dict[str, dict] = {}
    in_table = False
    for line in text.split("\n"):
        if "| cq# | Category |" in line:
            in_table = True
            continue
        if in_table and line.startswith("|---"):
            continue
        if in_table and line.startswith("| cq"):
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if len(parts) >= 5:
                cq_id = parts[0]
                ref[cq_id] = {
                    "category": parts[1].lower().replace(" ", "_").replace("-", "_"),
                    "summary": parts[2],
                    "group_id": parts[3].strip("`"),
                    "source": parts[4],
                }
        elif in_table and not line.strip().startswith("|"):
            break
    return ref


def split_catalog_sections(text: str) -> list[dict]:
    """Split catalog into sections by ## cq{N}: headers."""
    sections: list[dict] = []
    current: dict | None = None

    for line in text.split("\n"):
        m = re.match(r"^## (cq\d+):\s*(.+)$", line)
        if m:
            if current:
                sections.append(current)
            current = {"cq_id": m.group(1), "title": m.group(2).strip(), "lines": []}
        elif current is not None:
            current["lines"].append(line)

    if current:
        sections.append(current)
    return sections


def extract_question(lines: list[str]) -> str:
    for line in lines:
        if line.startswith("**Question**:"):
            m = re.search(r"\*(.+?)\*", line.replace("**Question**:", ""))
            return m.group(1) if m else line.replace("**Question**:", "").strip()
    return ""


def extract_entity_table(lines: list[str]) -> list[dict]:
    """Extract rows from | Entity | CURIE | Role | tables."""
    entities: list[dict] = []
    in_table = False
    for line in lines:
        if "| Entity | CURIE | Role |" in line:
            in_table = True
            continue
        if in_table and line.strip().startswith("|---"):
            continue
        if in_table and line.startswith("|"):
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if len(parts) >= 3 and parts[1]:
                entities.append(
                    {"entity_name": parts[0], "curie": parts[1], "role": parts[2]}
                )
        elif in_table:
            break
    return entities


def extract_gold_path(lines: list[str]) -> str | None:
    for i, line in enumerate(lines):
        if "**Gold Standard Path**:" in line:
            next_line = lines[i + 1] if i + 1 < len(lines) else ""
            return next_line.strip().strip("`")
    return None


def extract_workflow_steps(lines: list[str]) -> list[str]:
    steps: list[str] = []
    in_workflow = False
    for line in lines:
        if "**Workflow**:" in line:
            in_workflow = True
            continue
        if in_workflow:
            m = re.match(r"^\d+\.\s+\*\*.+?\*\*:\s*(.+)", line)
            if m:
                steps.append(m.group(1).strip())
            elif line.strip() == "" and steps:
                continue
            elif steps and not line.startswith(" ") and not re.match(r"^\d+\.", line):
                break
    return steps


def extract_validation_path(lines: list[str]) -> str | None:
    """Extract validation file path from **Validation**: line."""
    for line in lines:
        if line.startswith("**Validation**:"):
            m = re.search(r"`(.+?)`", line)
            return m.group(1) if m else None
    return None


# ---------------------------------------------------------------------------
# Paul CQ parsing
# ---------------------------------------------------------------------------
PAUL_CATEGORIES: dict[int, str] = {
    1: "doxorubicin_toxicity",
    2: "doxorubicin_toxicity",
    3: "doxorubicin_toxicity",
    4: "doxorubicin_toxicity",
    5: "tumor_microenvironment",
    6: "tumor_microenvironment",
    7: "tumor_microenvironment",
    8: "nsclc_synthetic_lethality",
    9: "nsclc_synthetic_lethality",
    10: "method_validation",
    11: "method_validation",
    12: "method_validation",
}


def parse_paul_questions(text: str) -> list[dict]:
    """Parse Paul Zamora's 12 prose CQs."""
    questions: list[dict] = []
    for m in re.finditer(r"^cq(\d+)\.\s+(.+)$", text, re.MULTILINE):
        num = int(m.group(1))
        questions.append(
            {
                "cq_id": f"paul-cq{num}",
                "category": PAUL_CATEGORIES.get(num, "general"),
                "question": m.group(2).strip(),
                "source": "Paul Zamora",
                "group_id": f"paul-cq{num}",
                "gold_path": None,
                "workflow_steps": None,
                "expected_template": None,
                "has_curies": False,
                "has_gold_graph": False,
            }
        )
    return questions


# ---------------------------------------------------------------------------
# Validation file parsing
# ---------------------------------------------------------------------------
def parse_validation_file(path: Path) -> dict:
    """Parse a validation file for status, date, and optional BioLink JSON."""
    text = path.read_text()
    result: dict = {"status": "PENDING", "date": None, "biolink": None}

    for line in text.split("\n"):
        if line.startswith("**Status**:"):
            raw = line.replace("**Status**:", "").strip()
            if "VALIDATED" in raw:
                result["status"] = "VALIDATED"
            elif "PARTIAL" in raw:
                result["status"] = "PARTIAL"
        m_date = re.search(r"\*\*(?:Validation )?Date\*\*:\s*(\d{4}-\d{2}-\d{2})", line)
        if m_date:
            result["date"] = m_date.group(1)

    biolink = extract_biolink_from_text(text)
    if biolink:
        result["biolink"] = biolink

    return result


def load_validations() -> dict[str, dict]:
    """Load all validation files keyed by cq_id (e.g. 'cq1')."""
    validations: dict[str, dict] = {}
    if not VALIDATIONS_DIR.exists():
        return validations
    for path in sorted(VALIDATIONS_DIR.glob("cq*.md")):
        m = re.match(r"^cq(\d+)", path.name)
        if m:
            cq_id = f"cq{m.group(1)}"
            validations[cq_id] = parse_validation_file(path)
    return validations


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------
def extract_all():
    """Extract everything and write Parquet + manifest."""
    print("=" * 60)
    print("Competency Questions → Structured Dataset")
    print("=" * 60)

    # --- Read sources ---
    catalog_text = CATALOG_PATH.read_text()
    paul_text = PAUL_PATH.read_text()

    quick_ref = parse_quick_reference(catalog_text)
    catalog_sections = split_catalog_sections(catalog_text)
    validations = load_validations()

    print(f"\nSources:")
    print(f"  Catalog CQs: {len(catalog_sections)}")
    print(f"  Paul CQs:    12 (expected)")
    print(f"  Validation files: {len(validations)}")

    # --- Build questions rows ---
    question_rows: list[dict] = []
    entity_rows: list[dict] = []
    gold_graph_rows: list[dict] = []

    for section in catalog_sections:
        cq_id = section["cq_id"]
        lines = section["lines"]
        ref = quick_ref.get(cq_id, {})

        question = extract_question(lines)
        entities = extract_entity_table(lines)
        gold_path = extract_gold_path(lines)
        workflow = extract_workflow_steps(lines)

        # BioLink JSON: check catalog first, then validation file
        biolink = extract_biolink_from_text("\n".join(lines))
        val = validations.get(cq_id, {})
        if not biolink and val.get("biolink"):
            biolink = val["biolink"]

        has_gold_graph = biolink is not None

        question_rows.append(
            {
                "cq_id": cq_id,
                "category": ref.get("category", ""),
                "question": question,
                "source": ref.get("source", ""),
                "group_id": ref.get("group_id", ""),
                "gold_path": gold_path,
                "workflow_steps": json.dumps(workflow) if workflow else None,
                "expected_template": None,
                "has_curies": True,
                "has_gold_graph": has_gold_graph,
            }
        )

        # Entity rows
        for ent in entities:
            entity_rows.append(
                {
                    "cq_id": cq_id,
                    "entity_name": ent["entity_name"],
                    "curie": ent["curie"],
                    "role": ent["role"],
                    "biolink_type": infer_biolink_type(ent["curie"]),
                }
            )

        # Gold graph row (one per catalog CQ)
        gold_graph_rows.append(
            {
                "cq_id": cq_id,
                "nodes_json": json.dumps(biolink["nodes"]) if biolink else None,
                "edges_json": json.dumps(biolink["edges"]) if biolink else None,
                "validation_status": val.get("status", "PENDING"),
                "validation_date": val.get("date"),
            }
        )

    # Add Paul's CQs
    paul_questions = parse_paul_questions(paul_text)
    question_rows.extend(paul_questions)

    # --- Create DataFrames ---
    df_questions = pd.DataFrame(question_rows)
    df_entities = pd.DataFrame(entity_rows)
    df_gold_graphs = pd.DataFrame(gold_graph_rows)

    print(f"\nExtracted:")
    print(f"  Questions:   {len(df_questions)} rows ({len(catalog_sections)} catalog + {len(paul_questions)} Paul)")
    print(f"  Entities:    {len(df_entities)} rows")
    print(f"  Gold graphs: {len(df_gold_graphs)} rows")

    gold_with_json = df_gold_graphs["nodes_json"].notna().sum()
    print(f"  Gold graphs with BioLink JSON: {gold_with_json}")

    validated = (df_gold_graphs["validation_status"] == "VALIDATED").sum()
    print(f"  Validated CQs: {validated}/{len(df_gold_graphs)}")

    # --- Write Parquet ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    q_path = OUTPUT_DIR / "cq_questions.parquet"
    e_path = OUTPUT_DIR / "cq_entities.parquet"
    g_path = OUTPUT_DIR / "cq_gold_graphs.parquet"

    df_questions.to_parquet(q_path, index=False)
    df_entities.to_parquet(e_path, index=False)
    df_gold_graphs.to_parquet(g_path, index=False)

    print(f"\nWritten to {OUTPUT_DIR}/:")
    print(f"  {q_path.name}  ({q_path.stat().st_size:,} bytes)")
    print(f"  {e_path.name}   ({e_path.stat().st_size:,} bytes)")
    print(f"  {g_path.name} ({g_path.stat().st_size:,} bytes)")

    # --- Write manifest ---
    manifest = {
        "id": f"cq_extraction_{uuid.uuid4()}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "catalog": str(CATALOG_PATH.relative_to(BASE_DIR)),
            "paul": str(PAUL_PATH.relative_to(BASE_DIR)),
            "validations_dir": str(VALIDATIONS_DIR.relative_to(BASE_DIR)),
            "validation_files": len(validations),
        },
        "artifacts": {
            "cq_questions": {
                "path": str(q_path.relative_to(BASE_DIR)),
                "rows": len(df_questions),
                "sha256": hash_file(q_path),
            },
            "cq_entities": {
                "path": str(e_path.relative_to(BASE_DIR)),
                "rows": len(df_entities),
                "sha256": hash_file(e_path),
            },
            "cq_gold_graphs": {
                "path": str(g_path.relative_to(BASE_DIR)),
                "rows": len(df_gold_graphs),
                "sha256": hash_file(g_path),
            },
        },
        "lineage": {
            "hf": {"pending_upload": True},
        },
    }

    manifest_path = OUTPUT_DIR / "cq_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  {manifest_path.name}")
    print("\nDone.")


if __name__ == "__main__":
    extract_all()
