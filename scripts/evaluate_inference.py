#!/usr/bin/env python3
"""
Step 1: Inference Generation for Evaluation
Generates raw evaluation data separately from Ragas metric evaluation.
Saves to `data/processed/*_evaluation_inputs.parquet`
"""

import os
import sys
import copy
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    load_documents_from_huggingface,
    load_golden_testset_from_huggingface,
)
from src.config import create_vector_store, get_llm
from src.retrievers import create_retrievers
from src.graph import build_all_graphs

parser = argparse.ArgumentParser()
parser.add_argument("--recreate", type=str, default="false", choices=["true", "false"])
args = parser.parse_args()
RECREATE_COLLECTION = args.recreate.lower() == "true"

DATASET_SOURCES = "open-biosciences/biosciences-sources"
DATASET_GOLDEN = "open-biosciences/biosciences-golden-testset"
K = 5
OUT_DIR = Path(__file__).parent.parent / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

print("=" * 80)
print("INFERENCE GENERATION (Step 1 of 2)")
print("=" * 80)

print("\nLoading source documents...")
docs = load_documents_from_huggingface(DATASET_SOURCES, "train")

print("\nLoading golden testset...")
golden_ds = load_golden_testset_from_huggingface(DATASET_GOLDEN, "train")
golden_df = golden_ds.to_pandas()

print(f"\nCreating vector store (recreate={RECREATE_COLLECTION})...")
vs = create_vector_store(docs, recreate_collection=RECREATE_COLLECTION)

print(f"\nCreating retrievers (k={K})...")
retrievers = create_retrievers(docs, vs, k=K)

print("\nBuilding LangGraph workflows...")
graphs = build_all_graphs(retrievers, llm=get_llm())

print("\n" + "=" * 80)
print("RUNNING INFERENCE")
print("=" * 80)

datasets = {}
for name, graph in graphs.items():
    print(f"\n📊 Processing {name} retriever...")
    df = copy.deepcopy(golden_df)
    df["response"] = None
    df["retrieved_contexts"] = None

    for idx, row in df.iterrows():
        q = row["user_input"]
        result = graph.invoke({"question": q})
        df.at[idx, "response"] = result["response"]
        df.at[idx, "retrieved_contexts"] = [d.page_content for d in result["context"]]

    inference_file = OUT_DIR / f"{name}_evaluation_inputs.parquet"
    df.to_parquet(str(inference_file), compression="zstd", index=False)
    print(f"   💾 Saved inference results: {inference_file.name}")

print(f"\n✓ Inference completely generated! Inputs saved to {OUT_DIR}")
