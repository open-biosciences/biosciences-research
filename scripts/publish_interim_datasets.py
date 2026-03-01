#!/usr/bin/env python3
"""
Upload GDELT RAG datasets to Hugging Face Hub.

This script:
1. Loads source documents and golden testset datasets from local storage
2. Creates dataset cards with metadata
3. Uploads datasets to Hugging Face Hub
4. Updates manifest.json with dataset repo IDs
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_from_disk
from huggingface_hub import HfApi, login

# Configuration
HF_USERNAME = "dwb2023"
HF_OWNER = "open-biosciences"
SOURCES_DATASET_NAME = f"{HF_OWNER}/biosciences-sources"
GOLDEN_TESTSET_NAME = f"{HF_OWNER}/biosciences-golden-testset"

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "interim"
MANIFEST_PATH = DATA_DIR / "manifest.json"
CARDS_DIR = BASE_DIR / "data" / "cards"

SOURCES_PATH = DATA_DIR / "sources.hfds"
GOLDEN_TESTSET_PATH = DATA_DIR / "golden_testset.hfds"


def load_card(name: str) -> str:
    """Load a HuggingFace dataset card from external markdown file."""
    return (CARDS_DIR / f"{name}.md").read_text()


def load_manifest():
    """Load manifest.json."""
    with open(MANIFEST_PATH, "r") as f:
        return json.load(f)


def update_manifest(sources_repo: str, golden_testset_repo: str):
    """Update manifest with dataset repo IDs."""
    manifest = load_manifest()

    # Create lineage structure if it doesn't exist
    if "lineage" not in manifest:
        manifest["lineage"] = {}

    if "hf" not in manifest["lineage"]:
        manifest["lineage"]["hf"] = {}

    # Update lineage section
    manifest["lineage"]["hf"]["dataset_repo_id"] = {
        "sources": sources_repo,
        "golden_testset": golden_testset_repo
    }
    manifest["lineage"]["hf"]["pending_upload"] = False
    manifest["lineage"]["hf"]["uploaded_at"] = datetime.now(timezone.utc).isoformat()

    # Write updated manifest
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✅ Updated manifest.json with dataset repo IDs")


def main():
    """Main upload function."""
    # Check for HF token
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set")

    # Login to Hugging Face
    print("🔐 Logging in to Hugging Face...")
    login(token=hf_token)

    # Initialize API
    api = HfApi()

    # Load datasets
    print(f"\n📂 Loading datasets from {DATA_DIR}...")
    sources_dataset = load_from_disk(str(SOURCES_PATH))
    golden_testset_dataset = load_from_disk(str(GOLDEN_TESTSET_PATH))

    print(f"   • Sources dataset: {len(sources_dataset)} documents")
    print(f"   • Golden testset: {len(golden_testset_dataset)} examples")

    # Upload sources dataset
    print(f"\n📤 Uploading sources dataset to {SOURCES_DATASET_NAME}...")
    sources_dataset.push_to_hub(
        SOURCES_DATASET_NAME,
        private=False,
        token=hf_token
    )

    # Create and upload sources dataset card
    print(f"   • Creating dataset card...")
    api.upload_file(
        path_or_fileobj=load_card("sources_card").encode(),
        path_in_repo="README.md",
        repo_id=SOURCES_DATASET_NAME,
        repo_type="dataset",
        token=hf_token
    )
    print(f"   ✅ Sources dataset uploaded successfully!")
    print(f"      View at: https://huggingface.co/datasets/{SOURCES_DATASET_NAME}")

    # Upload golden testset dataset
    print(f"\n📤 Uploading golden testset to {GOLDEN_TESTSET_NAME}...")
    golden_testset_dataset.push_to_hub(
        GOLDEN_TESTSET_NAME,
        private=False,
        token=hf_token
    )

    # Create and upload golden testset dataset card
    print(f"   • Creating dataset card...")
    api.upload_file(
        path_or_fileobj=load_card("golden_testset_card").encode(),
        path_in_repo="README.md",
        repo_id=GOLDEN_TESTSET_NAME,
        repo_type="dataset",
        token=hf_token
    )
    print(f"   ✅ Golden testset uploaded successfully!")
    print(f"      View at: https://huggingface.co/datasets/{GOLDEN_TESTSET_NAME}")

    # Update manifest
    print(f"\n📝 Updating manifest...")
    update_manifest(SOURCES_DATASET_NAME, GOLDEN_TESTSET_NAME)

    print("\n🎉 All datasets uploaded successfully!")
    print(f"\n📊 Dataset URLs:")
    print(f"   • Sources: https://huggingface.co/datasets/{SOURCES_DATASET_NAME}")
    print(f"   • Golden Testset: https://huggingface.co/datasets/{GOLDEN_TESTSET_NAME}")


if __name__ == "__main__":
    main()
