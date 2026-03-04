#!/usr/bin/env python3
"""
Publish competency questions dataset to HuggingFace Hub.

Uploads 3 Parquet files as a DatasetDict with splits:
  - questions (27 rows)
  - entities (48 rows)
  - gold_graphs (15 rows)

Requires: HF_TOKEN environment variable.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, login
import pandas as pd

# Configuration
HF_OWNER = "open-biosciences"
DATASET_NAME = f"{HF_OWNER}/biosciences-competency-questions"

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "interim"
CARDS_DIR = BASE_DIR / "data" / "cards"
MANIFEST_PATH = DATA_DIR / "cq_manifest.json"


def load_card() -> str:
    """Load HuggingFace dataset card."""
    return (CARDS_DIR / "competency_questions_card.md").read_text()


def update_manifest(repo_id: str):
    """Update cq_manifest.json with HF repo lineage."""
    with MANIFEST_PATH.open("r") as f:
        manifest = json.load(f)

    manifest["lineage"]["hf"] = {
        "dataset_repo_id": repo_id,
        "pending_upload": False,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }

    with MANIFEST_PATH.open("w") as f:
        json.dump(manifest, f, indent=2)

    print("\nUpdated cq_manifest.json with HF repo lineage")


def main():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set")

    print("Logging in to HuggingFace...")
    login(token=hf_token)
    api = HfApi()

    # Load Parquet files into DatasetDict
    print(f"\nLoading Parquet files from {DATA_DIR}...")
    ds = DatasetDict(
        {
            "questions": Dataset.from_pandas(
                pd.read_parquet(DATA_DIR / "cq_questions.parquet")
            ),
            "entities": Dataset.from_pandas(
                pd.read_parquet(DATA_DIR / "cq_entities.parquet")
            ),
            "gold_graphs": Dataset.from_pandas(
                pd.read_parquet(DATA_DIR / "cq_gold_graphs.parquet")
            ),
        }
    )

    for name, split in ds.items():
        print(f"  {name}: {len(split)} rows")

    # Upload dataset
    print(f"\nUploading to {DATASET_NAME}...")
    ds.push_to_hub(DATASET_NAME, private=False, token=hf_token)

    # Upload dataset card
    print("Uploading dataset card...")
    api.upload_file(
        path_or_fileobj=load_card().encode(),
        path_in_repo="README.md",
        repo_id=DATASET_NAME,
        repo_type="dataset",
        token=hf_token,
    )

    print(f"\nDataset uploaded: https://huggingface.co/datasets/{DATASET_NAME}")

    # Update manifest
    update_manifest(DATASET_NAME)

    print("\nDone.")


if __name__ == "__main__":
    main()
