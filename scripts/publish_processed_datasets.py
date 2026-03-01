#!/usr/bin/env python3
"""
Upload GDELT RAG processed evaluation datasets to Hugging Face Hub.

This script:
1. Loads evaluation_dataset.csv files from all retrievers (baseline, naive, bm25, ensemble, cohere_rerank)
2. Loads detailed_results.csv files from all retrievers with RAGAS metric scores
3. Adds 'retriever' column to identify source retriever
4. Creates consolidated datasets with comprehensive metadata
5. Uploads to Hugging Face Hub with dataset cards

Datasets Created:
- open-biosciences/biosciences-evaluation-inputs: RAGAS input datasets (questions, contexts, responses)
- open-biosciences/biosciences-evaluation-metrics: RAGAS evaluation results with metric scores
"""

import os
from pathlib import Path

import pandas as pd
from datasets import Dataset
from huggingface_hub import HfApi, login

# Configuration
HF_USERNAME = "dwb2023"
HF_OWNER = "open-biosciences"
EVALUATION_DATASETS_NAME = f"{HF_OWNER}/biosciences-evaluation-inputs"
DETAILED_RESULTS_NAME = f"{HF_OWNER}/biosciences-evaluation-metrics"

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
CARDS_DIR = BASE_DIR / "data" / "cards"

# Retriever names (matches filenames)
RETRIEVERS = ["naive", "bm25", "ensemble", "cohere_rerank"]


def load_card(name: str) -> str:
    """Load a HuggingFace dataset card from external markdown file."""
    return (CARDS_DIR / f"{name}.md").read_text()


def load_parquet_with_retriever_column(
    file_path: Path, retriever_name: str
) -> pd.DataFrame:
    """Load Parquet and add retriever column."""
    df = pd.read_parquet(file_path)
    df.insert(0, "retriever", retriever_name)
    return df


def load_and_consolidate_datasets(pattern: str) -> pd.DataFrame:
    """Load and consolidate all Parquet files matching pattern with retriever column."""
    dfs = []

    for retriever in RETRIEVERS:
        file_path = DATA_DIR / f"{retriever}_{pattern}.parquet"

        if not file_path.exists():
            print(f"   Warning: {file_path.name} not found, skipping...")
            continue

        print(f"   • Loading {file_path.name}...")
        df = load_parquet_with_retriever_column(file_path, retriever)
        dfs.append(df)
        print(f"      Loaded {len(df)} rows from {retriever}")

    if not dfs:
        raise ValueError(
            f"No Parquet files found matching pattern: *_{pattern}.parquet"
        )

    consolidated = pd.concat(dfs, ignore_index=True)
    print(
        f"   ✅ Consolidated {len(consolidated)} total rows from {len(dfs)} retrievers"
    )
    return consolidated


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

    # ========================================
    # Dataset 1: Evaluation Datasets
    # ========================================
    print(f"\n📂 Loading evaluation datasets from {DATA_DIR}...")
    eval_df = load_and_consolidate_datasets("evaluation_inputs")

    print("\n🔄 Converting evaluation datasets to HuggingFace Dataset...")
    eval_dataset = Dataset.from_pandas(eval_df)
    print(f"   • Dataset size: {len(eval_dataset)} examples")
    print(f"   • Features: {list(eval_dataset.features.keys())}")

    print(f"\n📤 Uploading evaluation datasets to {EVALUATION_DATASETS_NAME}...")
    eval_dataset.push_to_hub(EVALUATION_DATASETS_NAME, private=False, token=hf_token)

    # Create and upload dataset card
    print("   • Creating dataset card...")
    api.upload_file(
        path_or_fileobj=load_card("evaluation_inputs_card").encode(),
        path_in_repo="README.md",
        repo_id=EVALUATION_DATASETS_NAME,
        repo_type="dataset",
        token=hf_token,
    )
    print("   ✅ Evaluation datasets uploaded successfully!")
    print(f"      View at: https://huggingface.co/datasets/{EVALUATION_DATASETS_NAME}")

    # ========================================
    # Dataset 2: Detailed Results
    # ========================================
    print(f"\n📂 Loading detailed results from {DATA_DIR}...")
    results_df = load_and_consolidate_datasets("evaluation_metrics")

    print("\n🔄 Converting detailed results to HuggingFace Dataset...")
    results_dataset = Dataset.from_pandas(results_df)
    print(f"   • Dataset size: {len(results_dataset)} examples")
    print(f"   • Features: {list(results_dataset.features.keys())}")

    print(f"\n📤 Uploading detailed results to {DETAILED_RESULTS_NAME}...")
    results_dataset.push_to_hub(DETAILED_RESULTS_NAME, private=False, token=hf_token)

    # Create and upload dataset card
    print("   • Creating dataset card...")
    api.upload_file(
        path_or_fileobj=load_card("evaluation_metrics_card").encode(),
        path_in_repo="README.md",
        repo_id=DETAILED_RESULTS_NAME,
        repo_type="dataset",
        token=hf_token,
    )
    print("   ✅ Detailed results uploaded successfully!")
    print(f"      View at: https://huggingface.co/datasets/{DETAILED_RESULTS_NAME}")

    # ========================================
    # Summary
    # ========================================
    print("\n🎉 All datasets uploaded successfully!")
    print("\n📊 Dataset URLs:")
    print(
        f"   • Evaluation Datasets: https://huggingface.co/datasets/{EVALUATION_DATASETS_NAME}"
    )
    print(
        f"   • Detailed Results: https://huggingface.co/datasets/{DETAILED_RESULTS_NAME}"
    )

    print("\n📈 Dataset Statistics:")
    print(
        f"   • Evaluation Datasets: {len(eval_dataset)} examples across {len(eval_df['retriever'].unique())} retrievers"
    )
    print(
        f"   • Detailed Results: {len(results_dataset)} examples with RAGAS metric scores"
    )

    print("\n✨ Next Steps:")
    print("   1. Verify datasets on HuggingFace Hub")
    print("   2. Update README.md with new dataset references")
    print("   3. Update docs/deliverables.md with dataset URLs")


if __name__ == "__main__":
    main()
