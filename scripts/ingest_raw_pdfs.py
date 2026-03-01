# [markdown]
# # Standardized RAGAS Golden Testset Pipeline
# - Extract PDFs -> LangChain `Document`s
# - Sanitize metadata for Arrow/JSON
# - Persist SOURCES and GOLDEN TESTSET to `/data/interim` in JSONL, Parquet, HF-dataset-on-disk
# - Write a manifest with checksums & schema for provenance


# Imports & Env Hardening (safe in VS Code, Cursor, and Jupyter)
from __future__ import annotations

# Standard library imports
import hashlib
import importlib.metadata as im
import json
import os
import platform
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

# Progress bars & flaky UIs off (even if we're not uploading now, this keeps notebooks calm)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")

# Third-party imports
import pandas as pd
from datasets import Dataset
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# LangChain imports
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_core.documents import Document

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator

# Config (paths, models, knobs)


# Project structure
# --- 1) Repo root detection (replaces Path.cwd().parent) ---
def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(5):  # climb up to 5 levels
        if (cur / "pyproject.toml").exists() or (cur / ".git").exists():
            return cur
        cur = cur.parent
    return start.resolve()  # fallback


SCRIPT_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()
project_root = find_repo_root(SCRIPT_DIR)

raw_path = project_root / "data" / "raw"
interim_path = project_root / "data" / "interim"
processed_path = project_root / "data" / "processed"
for p in (raw_path, interim_path, processed_path):
    p.mkdir(parents=True, exist_ok=True)


# File names (interim)
SRC_JSONL = interim_path / "sources.docs.jsonl"
SRC_PARQUET = interim_path / "sources.docs.parquet"
SRC_HF_DISK = interim_path / "sources.hfds"

GT_JSONL = interim_path / "golden_testset.jsonl"
GT_PARQUET = interim_path / "golden_testset.parquet"
GT_HF_DISK = interim_path / "golden_testset.hfds"

MANIFEST_JSON = interim_path / "manifest.json"

# LLM/Embedding (use small defaults; swap as needed)
OPENAI_MODEL_NAME = "gpt-4.1-mini"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"

# RAGAS testset controls
TESTSET_SIZE = 10
MAX_DOCS = None  # set to an int to limit docs during prototyping

# Optional: seeded randomness for any local sampling (LLM remains non-deterministic)
RANDOM_SEED = 42


# Helpers


def ensure_jsonable(obj: Any) -> Any:
    """
    Make metadata JSON-serializable without losing information.
    Non-primitive scalars -> str; nested structures preserved.
    """
    if isinstance(obj, dict):
        return {str(k): ensure_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [ensure_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # Path, UUID, datetime, custom classes -> str
    return str(obj)


def docs_to_jsonl(docs: Iterable[Document], path: Path) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for d in docs:
            rec = {
                "page_content": d.page_content,
                "metadata": ensure_jsonable(d.metadata),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def docs_to_parquet(docs: Iterable[Document], path: Path) -> int:
    rows = [
        {"page_content": d.page_content, **ensure_jsonable(d.metadata)} for d in docs
    ]
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    return len(df)


def docs_to_hfds(docs: Iterable[Document], path: Path) -> int:
    rows = [
        {"page_content": d.page_content, "metadata": ensure_jsonable(d.metadata)}
        for d in docs
    ]
    ds = Dataset.from_list(rows)
    path.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(path))
    return len(ds)


def hash_file(path: Path, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(manifest_path: Path, payload: Dict[str, Any]) -> None:
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def summarize_columns_from_jsonl(path: Path, sample_n: int = 5) -> Dict[str, Any]:
    cols = set()
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            # top-level keys + metadata keys
            cols |= set(obj.keys())
            if "metadata" in obj and isinstance(obj["metadata"], dict):
                cols |= {f"metadata.{k}" for k in obj["metadata"].keys()}
            if i < sample_n:
                samples.append(obj)
    return {"columns": sorted(cols), "sample": samples}


# 1) Extract PDFs -> LangChain Documents

# IMPORTANT: keep metadata JSON-serializable
loader = DirectoryLoader(str(raw_path), glob="*.pdf", loader_cls=PyMuPDFLoader)
docs: List[Document] = loader.load()
if MAX_DOCS:
    docs = docs[:MAX_DOCS]

print(f"Loaded {len(docs)} documents from {raw_path}")


# 2) Persist SOURCE documents to interim storage (JSONL, Parquet, HF-dataset)

n_jsonl = docs_to_jsonl(docs, SRC_JSONL)
n_parquet = docs_to_parquet(docs, SRC_PARQUET)
n_hfds = docs_to_hfds(docs, SRC_HF_DISK)

print(f"SOURCES -> JSONL:   {n_jsonl}  ({SRC_JSONL})")
print(f"SOURCES -> Parquet: {n_parquet}  ({SRC_PARQUET})")
print(f"SOURCES -> HFDS:    {n_hfds}  ({SRC_HF_DISK})")


# --- RAGAS testset generation: 0.3.x-first, 0.2.x fallback ---


if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY not set")

TransientErr = (RateLimitError, APITimeoutError, APIStatusError, APIConnectionError)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type(TransientErr),
)
def build_testset(docs, size: int):
    # --- Standard approach for ragas 0.2.x (current: 0.2.10) ---
    # This is the ACTIVE code path as of ragas 0.2.10
    # NOTE:  for 0.3.x certain approaches as use of LangChainEmbeddingsWrapper and LangchainLLMWrapper are deprecated

    llm = LangchainLLMWrapper(
        ChatOpenAI(model="gpt-4.1-mini", temperature=0, timeout=60, max_retries=6)
    )
    emb = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small", timeout=60, max_retries=6)
    )
    gen = TestsetGenerator(llm=llm, embedding_model=emb)
    return gen.generate_with_langchain_docs(docs, testset_size=size)


golden_testset = build_testset(docs, TESTSET_SIZE)
print("Generated golden testset:", type(golden_testset))


# 4) Persist GOLDEN TESTSET to interim storage

# A) JSONL (native for RAG eval workflows)
golden_testset.to_jsonl(str(GT_JSONL))

# B) Parquet (for analytics / quick vector-store ingestion)
#    Convert via pandas to keep things simple and robust
hf_ds = golden_testset.to_hf_dataset()  # -> datasets.Dataset or DatasetDict
if hasattr(hf_ds, "to_pandas"):
    pdf = hf_ds.to_pandas()
else:
    # handle DatasetDict (unlikely for small sets)
    from pandas import concat

    pdf = concat([split.to_pandas() for split in hf_ds.values()], ignore_index=True)
pdf.to_parquet(GT_PARQUET, index=False)

# C) HF dataset on disk (fast rehydrate into `datasets`)
GT_HF_DISK.mkdir(parents=True, exist_ok=True)
golden_testset.to_hf_dataset().save_to_disk(str(GT_HF_DISK))

print(f"GOLDEN -> JSONL:   {GT_JSONL}")
print(f"GOLDEN -> Parquet: {GT_PARQUET}")
print(f"GOLDEN -> HFDS:    {GT_HF_DISK}")


# 5) Manifest & provenance (one place to verify & rehydrate quickly)

# --- 5) Enriched manifest (env + artifacts + basic metrics) ---


def _ver(pkg):
    try:
        return im.version(pkg)
    except:
        return None


manifest = {
    "id": f"ragas_pipeline_{uuid.uuid4()}",
    "generated_at": datetime.utcnow().isoformat() + "Z",
    "run": {"random_seed": RANDOM_SEED},
    "env": {
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "langchain": _ver("langchain"),
        "ragas": _ver("ragas"),
        "datasets": _ver("datasets"),
        "pyarrow": _ver("pyarrow"),
        "huggingface_hub": _ver("huggingface-hub"),
    },
    "params": {
        "OPENAI_MODEL_NAME": OPENAI_MODEL_NAME,
        "EMBEDDING_MODEL_NAME": EMBEDDING_MODEL_NAME,
        "TESTSET_SIZE": TESTSET_SIZE,
        "MAX_DOCS": MAX_DOCS,
    },
    "paths": {
        "sources": {
            "jsonl": str(SRC_JSONL),
            "parquet": str(SRC_PARQUET),
            "hfds": str(SRC_HF_DISK),
        },
        "golden_testset": {
            "jsonl": str(GT_JSONL),
            "parquet": str(GT_PARQUET),
            "hfds": str(GT_HF_DISK),
        },
    },
    "fingerprints": {
        "sources": {
            "jsonl_sha256": hash_file(SRC_JSONL),
            "parquet_sha256": hash_file(SRC_PARQUET),
        },
        "golden_testset": {
            "jsonl_sha256": hash_file(GT_JSONL),
            "parquet_sha256": hash_file(GT_PARQUET),
        },
    },
    "quick_schema": {
        "sources_jsonl": summarize_columns_from_jsonl(SRC_JSONL),
        "golden_jsonl": summarize_columns_from_jsonl(GT_JSONL),
    },
    "artifacts": {
        "sources": {
            "jsonl": {"path": str(SRC_JSONL), "bytes": SRC_JSONL.stat().st_size},
            "parquet": {"path": str(SRC_PARQUET), "bytes": SRC_PARQUET.stat().st_size},
            "hfds": {"path": str(SRC_HF_DISK)},
        },
        "golden_testset": {
            "jsonl": {"path": str(GT_JSONL), "bytes": GT_JSONL.stat().st_size},
            "parquet": {"path": str(GT_PARQUET), "bytes": GT_PARQUET.stat().st_size},
            "hfds": {"path": str(GT_HF_DISK)},
        },
    },
}
write_manifest(MANIFEST_JSON, manifest)
