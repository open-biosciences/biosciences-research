#!/usr/bin/env python3
"""
RAGAS Evaluation Harness

THIS IS THE SAME AS run_full_evaluation.py BUT USES src/ MODULES INSTEAD OF INLINE CODE.

What it does:
- Runs 12 test questions through 4 retrievers (naive, bm25, ensemble, cohere_rerank)
- Evaluates with RAGAS (Faithfulness, ResponseRelevancy, ContextPrecision, LLMContextRecall)
- Saves results to deliverables/evaluation_evidence/

Time: 20-30 minutes
Cost: ~$5-6 in OpenAI API calls
Results: Identical to run_full_evaluation.py

Only difference: Uses factory functions from src/ instead of duplicating code.

See scripts/README.md for details.

Usage:
    make eval
    # or
    export PYTHONPATH=.
    python scripts/run_eval_harness.py
"""

import os
import sys
import copy
import json
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings

from src.utils import (
    load_documents_from_huggingface,
    load_golden_testset_from_huggingface,
)
from src.config import create_vector_store, get_llm
from src.retrievers import create_retrievers
from src.graph import build_all_graphs, invoke_for_benchmark


# ==============================================================================
# COMMAND LINE ARGUMENTS
# ==============================================================================

parser = argparse.ArgumentParser(
    description="RAGAS Evaluation Harness - Same as run_full_evaluation.py but uses src/ modules"
)
parser.add_argument(
    "--recreate",
    type=str,
    default="false",
    choices=["true", "false"],
    help="Recreate Qdrant collection (true) or reuse existing (false). Default: false",
)
args = parser.parse_args()

# Convert string to boolean
RECREATE_COLLECTION = args.recreate.lower() == "true"


# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATASET_SOURCES = "open-biosciences/biosciences-sources"
DATASET_GOLDEN = "open-biosciences/biosciences-golden-testset"
K = 5  # Number of documents to retrieve (matches run_full_evaluation.py)
OUT_DIR = Path(__file__).parent.parent / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Disable HuggingFace progress bars
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


# ==============================================================================
# PRE-FLIGHT CHECKS
# ==============================================================================

print("=" * 80)
print("RAGAS EVALUATION HARNESS (Using src/ Modules)")
print("=" * 80)
print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("❌ OPENAI_API_KEY environment variable must be set")

print("✓ OpenAI API key configured")

cohere_key = os.getenv("COHERE_API_KEY")
if cohere_key:
    print("✓ Cohere API key configured (cohere_rerank will work)")
else:
    print("⚠ Cohere API key not set (cohere_rerank may fail)")

# Load ingestion manifest if available (for data lineage)
ingest_manifest_path = Path(__file__).parent.parent / "data/interim/manifest.json"
data_provenance = None
if ingest_manifest_path.exists():
    with open(ingest_manifest_path) as f:
        ingest_manifest = json.load(f)
    data_provenance = {
        "ingest_manifest_id": ingest_manifest["id"],
        "ingest_timestamp": ingest_manifest["generated_at"],
        "sources_sha256": ingest_manifest["fingerprints"]["sources"]["jsonl_sha256"],
        "golden_testset_sha256": ingest_manifest["fingerprints"]["golden_testset"][
            "jsonl_sha256"
        ],
        "source_pdfs_count": ingest_manifest["params"]["MAX_DOCS"] or "all",
        "ragas_testset_size": ingest_manifest["params"]["TESTSET_SIZE"],
    }
    print(f"✓ Linked to ingestion manifest: {ingest_manifest['id'][:8]}...")
else:
    print(f"⚠ No ingestion manifest found at {ingest_manifest_path}")


# ==============================================================================
# LOAD DATA
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 1: LOADING DATA")
print("=" * 80)

print(f"\nLoading source documents from {DATASET_SOURCES}...")
docs = load_documents_from_huggingface(DATASET_SOURCES, "train")
print(f"✓ Loaded {len(docs)} source documents")

print(f"\nLoading golden testset from {DATASET_GOLDEN}...")
golden_ds = load_golden_testset_from_huggingface(DATASET_GOLDEN, "train")
golden_df = golden_ds.to_pandas()
print(f"✓ Loaded {len(golden_df)} test examples")
print(f"  Columns: {golden_df.columns.tolist()}")


# ==============================================================================
# BUILD RAG STACK
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 2: BUILDING RAG STACK")
print("=" * 80)

print(f"\nCreating vector store (recreate={RECREATE_COLLECTION})...")
vs = create_vector_store(docs, recreate_collection=RECREATE_COLLECTION)
if RECREATE_COLLECTION:
    print("✓ Vector store recreated and populated")
else:
    print("✓ Vector store connected (reusing existing collection)")

print(f"\nCreating retrievers (k={K})...")
retrievers = create_retrievers(docs, vs, k=K)
print(f"✓ Created {len(retrievers)} retrievers: {list(retrievers.keys())}")

print("\nBuilding LangGraph workflows...")
graphs = build_all_graphs(retrievers, llm=get_llm())
print(f"✓ Built {len(graphs)} compiled graphs")


# ==============================================================================
# RUN INFERENCE ACROSS ALL RETRIEVERS
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 3: RUNNING INFERENCE")
print("=" * 80)
print(f"\nProcessing {len(golden_df)} questions × {len(graphs)} retrievers...")

datasets = {}
for name, graph in graphs.items():
    print(f"\n📊 Processing {name} retriever...")

    # Create deep copy of golden dataset
    df = copy.deepcopy(golden_df)

    # Initialize columns
    df["response"] = None
    df["retrieved_contexts"] = None

    # Run inference
    for idx, row in df.iterrows():
        q = row["user_input"]
        result = invoke_for_benchmark(graph, q)
        df.at[idx, "response"] = result["response"]
        df.at[idx, "retrieved_contexts"] = [d.page_content for d in result["context"]]

    print(f"   ✓ Processed {len(df)} questions")

    # Save inference results immediately (before RAGAS evaluation)
    inference_file = OUT_DIR / f"{name}_evaluation_inputs.parquet"
    df.to_parquet(str(inference_file), compression="zstd", index=False)
    print(f"   💾 Saved inference results: {inference_file.name}")

    datasets[name] = df

print(f"\n✓ All inference complete! Results saved to {OUT_DIR}")


# ==============================================================================
# RAGAS EVALUATION
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 4: RAGAS EVALUATION")
print("=" * 80)
print("\nMetrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall")

openai_client = OpenAI()
evaluator_llm = llm_factory("gpt-4.1-mini", client=openai_client)
evaluator_emb = RagasOpenAIEmbeddings(
    client=openai_client, model="text-embedding-3-small"
)
run_cfg = RunConfig(timeout=360)

results = {}
for name, df in datasets.items():
    print(f"\n🔍 Evaluating {name}...")

    # Create RAGAS evaluation dataset
    # NOTE: evaluation_inputs.parquet already saved in Step 3 (after inference)
    eval_ds = EvaluationDataset.from_pandas(df)

    # Run RAGAS evaluation (evaluate() is deprecated in 0.4 but still functional)
    res = evaluate(
        dataset=eval_ds,
        metrics=[
            Faithfulness(llm=evaluator_llm),
            AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_emb),
            ContextPrecision(llm=evaluator_llm, name="context_precision"),
            ContextRecall(llm=evaluator_llm),
        ],
        run_config=run_cfg,
    )
    results[name] = res
    print("   ✓ Evaluation complete")

    # Save detailed results with per-question scores
    det_file = OUT_DIR / f"{name}_evaluation_metrics.parquet"
    res.to_pandas().to_parquet(det_file, compression="zstd", index=False)
    print(f"   💾 Saved evaluation metrics: {det_file.name}")

print("\n✓ All evaluations complete!")


# ==============================================================================
# COMPARATIVE ANALYSIS
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 5: COMPARATIVE ANALYSIS")
print("=" * 80)

comp = []
for name, res in results.items():
    rdf = res.to_pandas()
    row = {
        "Retriever": name.replace("_", " ").title(),
        "Faithfulness": rdf["faithfulness"].mean(),
        "Answer Relevancy": rdf["answer_relevancy"].mean(),
        "Context Precision": rdf["context_precision"].mean(),
        "Context Recall": rdf["context_recall"].mean(),
    }
    row["Average"] = (
        row["Faithfulness"]
        + row["Answer Relevancy"]
        + row["Context Precision"]
        + row["Context Recall"]
    ) / 4
    comp.append(row)

comp_df = (
    pd.DataFrame(comp).sort_values("Average", ascending=False).reset_index(drop=True)
)

# Save comparative table
comp_parquet = OUT_DIR / "comparative_ragas_results.parquet"
comp_df.to_parquet(comp_parquet, compression="zstd", index=False)
print(f"\n💾 Saved comparative results: {comp_parquet.name}")

# Display results
print("\n" + "=" * 80)
print("COMPARATIVE RAGAS RESULTS")
print("=" * 80)
print()
print(comp_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print()

# Find winner
winner = comp_df.iloc[0]
baseline = (
    comp_df[comp_df["Retriever"] == "Naive"].iloc[0]
    if "Naive" in comp_df["Retriever"].values
    else None
)

if baseline is not None and winner["Retriever"] != "Naive":
    improvement = (
        (winner["Average"] - baseline["Average"]) / baseline["Average"]
    ) * 100
    print(
        f"🏆 Winner: {winner['Retriever']} with {winner['Average']:.2%} average score"
    )
    print(f"   Improvement over baseline: +{improvement:.1f}%")
else:
    print(
        f"🏆 Best performer: {winner['Retriever']} with {winner['Average']:.2%} average score"
    )


# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 80)
print("EVALUATION SUMMARY")
print("=" * 80)

print(f"""
End time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Test Set:
  - Dataset: {DATASET_GOLDEN}
  - Questions: {len(golden_df)}

Retrievers Evaluated:
  {", ".join(datasets.keys())}

Output Files ({OUT_DIR}):
  - Evaluation inputs: {len(datasets)} × *_evaluation_inputs.parquet (6 columns: RAG outputs)
  - Evaluation metrics: {len(datasets)} × *_evaluation_metrics.parquet (10 columns: RAG + RAGAS)
  - Comparative summary: comparative_ragas_results.parquet
  - Provenance manifest: RUN_MANIFEST.json

Metrics Computed:
  - Faithfulness (answer grounded in context)
  - Answer Relevancy (answer addresses question)
  - Context Precision (relevant contexts ranked higher)
  - Context Recall (ground truth coverage)

All results saved to: {OUT_DIR.absolute()}
""")

# ==============================================================================
# GENERATE RUN MANIFEST FOR REPRODUCIBILITY
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 6: GENERATING RUN MANIFEST")
print("=" * 80)

# Import from src.utils package
from src.utils import generate_run_manifest

manifest_path = OUT_DIR / "RUN_MANIFEST.json"
manifest = generate_run_manifest(
    output_path=manifest_path,
    evaluation_results=results,
    retrievers_config={name: {"graph": g, "k": K} for name, g in graphs.items()},
    data_provenance=data_provenance,
)

print(f"\n💾 Saved run manifest: {manifest_path.name}")
print(f"   ✓ RAGAS version: {manifest['ragas_version']}")
print(f"   ✓ Python version: {manifest['python_version']}")
print(f"   ✓ Retriever configs: {len(manifest['retrievers'])}")
print("   ✓ Evaluation settings captured")

print("\n" + "=" * 80)
print("✅ EVALUATION COMPLETE")
print("=" * 80)
