#!/usr/bin/env python3
"""
Task 5 & 7: Comprehensive RAG Evaluation with RAGAS

Evaluates multiple retrieval systems using RAGAS 0.4 metrics:
- Faithfulness: Answer groundedness in retrieved context
- Answer Relevancy: Answer relevance to the question (class: AnswerRelevancy)
- Context Precision: Relevant contexts ranked higher (class: ContextPrecision)
- Context Recall: Ground truth coverage (class: ContextRecall)

Note: RAGAS class names are capitalized (e.g., AnswerRelevancy), but output
DataFrame columns are lowercase (e.g., 'answer_relevancy', 'context_recall').

Retrievers:
1. Naive (baseline) - Dense vector search
2. BM25 - Sparse keyword matching
3. Cohere Rerank - Contextual compression
4. Ensemble - Hybrid (dense + sparse)

Dataset: open-biosciences/biosciences-golden-testset (12 QA pairs)

Following session08-ragas-rag-evals.py LangGraph pattern.
"""

# Imports
import copy
import json
import os
from pathlib import Path
from typing import List

import pandas as pd
from datasets import load_dataset

# LangChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever

# LangChain Integrations
from langchain_cohere import CohereRerank
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

# LangGraph
from langgraph.graph import START, StateGraph

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# RAGAS 0.4
from openai import OpenAI as OpenAIClient
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings

# Typing
from typing_extensions import TypedDict


# Helper Functions for RAGAS Schema Validation
def validate_and_normalize_ragas_schema(
    df: pd.DataFrame, retriever_name: str = "unknown"
) -> pd.DataFrame:
    """
    Ensure DataFrame matches RAGAS 0.4.x schema requirements.

    Handles different column naming conventions and validates required fields.
    Prevents silent breakage when running with different RAGAS versions.

    Args:
        df: DataFrame to validate and normalize
        retriever_name: Name of retriever for logging

    Returns:
        DataFrame with normalized column names

    Raises:
        ValueError: If required columns are missing after normalization
    """
    import ragas

    # Log versions (first time only)
    if retriever_name == "naive" or retriever_name == "unknown":
        print(f"   RAGAS version: {ragas.__version__}")

    # Expected RAGAS columns (0.4.x spec)
    expected_cols = {"user_input", "response", "retrieved_contexts", "reference"}

    # Rename mapping for common variations
    rename_map = {
        "question": "user_input",
        "answer": "response",
        "contexts": "retrieved_contexts",
        "ground_truth": "reference",
        "ground_truths": "reference",
        "reference_contexts": "reference",
    }

    # Apply renaming
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Validate required columns present
    actual_cols = set(df.columns)
    missing = expected_cols - actual_cols

    if missing:
        raise ValueError(
            f"[{retriever_name}] Missing required columns for RAGAS: {missing}\n"
            f"  Expected: {sorted(expected_cols)}\n"
            f"  Actual: {sorted(actual_cols)}"
        )

    # Validate data types
    if not all(isinstance(df.iloc[0][col], str) for col in ["user_input", "response"]):
        raise ValueError(
            f"[{retriever_name}] 'user_input' and 'response' must be strings"
        )

    if not isinstance(df.iloc[0]["retrieved_contexts"], list):
        raise ValueError(f"[{retriever_name}] 'retrieved_contexts' must be a list")

    print(
        f"   ✓ {retriever_name}: Schema validated [{len(df)} rows, {len(df.columns)} cols]"
    )

    return df


# Configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "biosciences-data-sources"

# Disable XetHub progress bars
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Check for API keys
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable must be set")

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
_openai_client = OpenAIClient()
evaluator_llm = llm_factory("gpt-4.1-mini", client=_openai_client)
evaluator_emb = RagasOpenAIEmbeddings(
    client=_openai_client, model="text-embedding-3-small"
)

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

print("=" * 80)
print("COMPREHENSIVE RAG EVALUATION WITH RAGAS (TASKS 5 & 7)")
print("=" * 80)

if data_provenance:
    print(f"✓ Linked to ingestion manifest: {ingest_manifest['id'][:8]}...")

# 1. Load Golden Testset
print("\n1. Loading golden testset from HuggingFace...")
golden_dataset = load_dataset(
    "open-biosciences/biosciences-golden-testset", split="train"
)
golden_df = golden_dataset.to_pandas()

print(f"   ✓ Loaded {len(golden_df)} test examples")
print(f"   Columns: {golden_df.columns.tolist()}")

# 2. Load Source Documents
print("\n2. Loading source documents from HuggingFace...")
sources_dataset = load_dataset("open-biosciences/biosciences-sources", split="train")

# Convert to LangChain Documents
documents = []
for item in sources_dataset:
    page_content = item.get("page_content", "")

    # Handle nested metadata structure
    if "metadata" in item and isinstance(item["metadata"], dict):
        metadata = item["metadata"]
    else:
        metadata = {k: v for k, v in item.items() if k != "page_content"}

    doc = Document(page_content=page_content, metadata=metadata)
    documents.append(doc)

print(f"   ✓ Loaded {len(documents)} source documents")

# 3. Create RAG System with Docker Compose Qdrant
print("\n3. Creating RAG system with Qdrant...")

# Initialize models (temperature=0 for determinism)

print(f"   Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Check if collection exists, recreate if needed
collections = qdrant_client.get_collections().collections
collection_names = [c.name for c in collections]

if COLLECTION_NAME in collection_names:
    print(f"   Deleting existing collection '{COLLECTION_NAME}'...")
    qdrant_client.delete_collection(COLLECTION_NAME)

# Create collection
print(f"   Creating collection '{COLLECTION_NAME}'...")
qdrant_client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

# Create vector store
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

# Add documents to vector store
print(f"   Adding {len(documents)} documents to vector store...")
vector_store.add_documents(documents=documents)


# 4. Create All Retrievers
print("\n4. Creating retrievers...")

# Baseline: Dense vector search (k=5)
baseline_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
print("   ✓ Naive (dense vector search, top-5)")

# BM25: Sparse keyword matching
bm25_retriever = BM25Retriever.from_documents(documents, k=5)
print("   ✓ BM25 (sparse keyword matching, top-5)")

# Cohere Rerank: Contextual compression (retrieve 20, rerank to 5)
baseline_retriever_20 = vector_store.as_retriever(search_kwargs={"k": 20})
compressor = CohereRerank(model="rerank-v3.5")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=baseline_retriever_20
)

# Ensemble: Hybrid search (dense + sparse)
ensemble_retriever = EnsembleRetriever(
    retrievers=[baseline_retriever, bm25_retriever], weights=[0.5, 0.5]
)
print("   ✓ Ensemble (hybrid: 50% dense + 50% sparse, top-5)")


# in prompts.py - Shared prompt template
BASELINE_PROMPT = """\
You are a helpful assistant who answers questions based on provided context. You must only use the provided context, and cannot use your own knowledge.

### Question
{question}

### Context
{context}
"""

# in src/graph.py
rag_prompt = ChatPromptTemplate.from_template(BASELINE_PROMPT)


# Shared State
class State(TypedDict):
    question: str
    context: List[Document]
    response: str


# Modular retriever functions (following session08 pattern)
def retrieve_baseline(state):
    """Naive dense vector search"""
    retrieved_docs = baseline_retriever.invoke(state["question"])
    return {"context": retrieved_docs}


def retrieve_bm25(state):
    """BM25 sparse keyword matching"""
    retrieved_docs = bm25_retriever.invoke(state["question"])
    return {"context": retrieved_docs}


def retrieve_reranked(state):
    """Cohere contextual compression with reranking"""
    retrieved_docs = compression_retriever.invoke(state["question"])
    return {"context": retrieved_docs}


def retrieve_ensemble(state):
    """Ensemble hybrid search (dense + sparse)"""
    retrieved_docs = ensemble_retriever.invoke(state["question"])
    return {"context": retrieved_docs}


# Shared generate function
def generate(state):
    """Generate answer from context"""
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = rag_prompt.format_messages(
        question=state["question"], context=docs_content
    )
    response = llm.invoke(messages)
    return {"response": response.content}


# Create LangGraphs for each retriever
print("   Creating LangGraphs...")
baseline_graph_builder = StateGraph(State).add_sequence([retrieve_baseline, generate])
baseline_graph_builder.add_edge(START, "retrieve_baseline")
baseline_graph = baseline_graph_builder.compile()

bm25_graph_builder = StateGraph(State).add_sequence([retrieve_bm25, generate])
bm25_graph_builder.add_edge(START, "retrieve_bm25")
bm25_graph = bm25_graph_builder.compile()

ensemble_graph_builder = StateGraph(State).add_sequence([retrieve_ensemble, generate])
ensemble_graph_builder.add_edge(START, "retrieve_ensemble")
ensemble_graph = ensemble_graph_builder.compile()

rerank_graph_builder = StateGraph(State).add_sequence([retrieve_reranked, generate])
rerank_graph_builder.add_edge(START, "retrieve_reranked")
rerank_graph = rerank_graph_builder.compile()

# Configure retrievers
retrievers_config = {
    "naive": baseline_graph,
    "bm25": bm25_graph,
    "ensemble": ensemble_graph,
    "cohere_rerank": rerank_graph,
}

# 6. Test One Retriever
print("\n6. Testing baseline retriever...")
test_question = golden_df.iloc[0]["user_input"]
print(f"   Q: {test_question[:80]}...")

test_result = baseline_graph.invoke({"question": test_question})
print(f"   A: {test_result['response'][:100]}...")
print(f"   Retrieved {len(test_result['context'])} contexts")

# 7. Populate Datasets for All Retrievers (session08 pattern)
print("\n7. Running all test questions through all retrievers...")
print(
    f"   Processing {len(golden_df)} questions × {len(retrievers_config)} retrievers..."
)

datasets = {}

# Create output directory for immediate persistence
output_dir = Path(__file__).parent.parent / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

for retriever_name, graph in retrievers_config.items():
    print(f"\n   Processing {retriever_name} retriever...")

    # Create deep copy of golden dataset (session08 line 108)
    datasets[retriever_name] = copy.deepcopy(golden_df)

    # Initialize columns FIRST (pandas requirement)
    datasets[retriever_name]["response"] = None
    datasets[retriever_name]["retrieved_contexts"] = None

    # Populate with RAG outputs (session08 lines 223-226)
    for idx, row in datasets[retriever_name].iterrows():
        question = row["user_input"]
        result = graph.invoke({"question": question})
        datasets[retriever_name].at[idx, "response"] = result["response"]
        datasets[retriever_name].at[idx, "retrieved_contexts"] = [
            doc.page_content for doc in result["context"]
        ]

    print(f"   ✓ {retriever_name}: {len(datasets[retriever_name])} questions processed")

    # Save inference results immediately (before RAGAS evaluation)
    inference_file = output_dir / f"{retriever_name}_evaluation_inputs.parquet"
    datasets[retriever_name].to_parquet(
        str(inference_file), compression="zstd", index=False
    )
    print(f"   💾 Saved inference results: {inference_file.name}")

print("\n   ✓ All retriever datasets populated and saved!")

# 8. Create RAGAS EvaluationDatasets
print("\n8. Creating RAGAS EvaluationDatasets...")

evaluation_datasets = {}
for retriever_name, dataset in datasets.items():
    # Create EvaluationDataset directly from populated dataset (no schema mutation)
    evaluation_datasets[retriever_name] = EvaluationDataset.from_pandas(dataset)
    print(f"   ✓ {retriever_name}: EvaluationDataset created")

# 9. Run RAGAS Evaluation for All Retrievers
print("\n9. Running RAGAS evaluation for all retrievers...")
print("   Metrics: Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall")
print(
    "   (Output columns: faithfulness, answer_relevancy, context_precision, context_recall)"
)

custom_run_config = RunConfig(timeout=360)
evaluation_results = {}

for retriever_name, eval_dataset in evaluation_datasets.items():
    print(f"\n   Evaluating {retriever_name}...")

    result = evaluate(
        dataset=eval_dataset,
        metrics=[
            Faithfulness(llm=evaluator_llm),
            AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_emb),
            ContextPrecision(llm=evaluator_llm, name="context_precision"),
            ContextRecall(llm=evaluator_llm),
        ],
        run_config=custom_run_config,
    )

    evaluation_results[retriever_name] = result
    print(f"   ✓ {retriever_name} evaluation complete")

    # NOTE: evaluation_inputs already saved in Step 7 (after inference)
    # Save evaluation metrics only
    detailed_file = output_dir / f"{retriever_name}_evaluation_metrics.parquet"
    result.to_pandas().to_parquet(detailed_file, compression="zstd", index=False)
    print(f"   💾 Saved: {detailed_file.name}")

print("\n   ✓ All RAGAS evaluations complete!")
print(f"   ✓ All results saved to: {output_dir}")

# 10. Generate Comparative Summary Table (Task 7 requirement)
print("\n10. Generating comparative summary table...")

try:
    comparison_data = []
    for retriever_name, result in evaluation_results.items():
        result_df = result.to_pandas()

        # RAGAS 0.4.x returns lowercase column names from metrics
        row = {
            "Retriever": retriever_name.replace("_", " ").title(),
            "Faithfulness": result_df["faithfulness"].mean(),
            "Answer Relevancy": result_df[
                "answer_relevancy"
            ].mean(),  # Fixed: was 'response_relevancy'
            "Context Precision": result_df["context_precision"].mean(),
            "Context Recall": result_df["context_recall"].mean(),
        }

        # Calculate average score
        row["Average"] = (
            row["Faithfulness"]
            + row["Answer Relevancy"]
            + row["Context Precision"]
            + row["Context Recall"]
        ) / 4

        comparison_data.append(row)

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)

    # Sort by average score (descending)
    comparison_df = comparison_df.sort_values("Average", ascending=False).reset_index(
        drop=True
    )

    # Save comparative summary table (Task 7 deliverable)
    comparison_parquet = output_dir / "comparative_ragas_results.parquet"
    comparison_df.to_parquet(comparison_parquet, compression="zstd", index=False)
    print(f"   ✓ Comparative results saved to {comparison_parquet}")

except Exception as e:
    print(f"   ⚠️  Comparison table generation failed: {e}")
    print("   ✓ Individual retriever results already saved - no data lost!")
    comparison_df = None  # Set to None so we can check later

# 11. Display Results
print("\n" + "=" * 80)
print("COMPARATIVE RAG EVALUATION RESULTS (TASKS 5 & 7)")
print("=" * 80)
print(f"\nVector DB: Qdrant ({QDRANT_HOST}:{QDRANT_PORT})")
print(f"Test Set: {len(golden_df)} questions")
print(f"Retrievers Evaluated: {len(retrievers_config)}\n")

if comparison_df is not None:
    print("\n📊 Comparative RAGAS Metrics:")
    print("-" * 80)
    print(comparison_df.to_string(index=False))
    print("-" * 80)

    # Calculate improvement over baseline
    baseline_score = comparison_df[
        comparison_df["Retriever"].str.contains("Naive", case=False)
    ]["Average"].values[0]
    print("\n📈 Improvement Over Baseline (Naive):")
    print("-" * 80)
    for _, row in comparison_df.iterrows():
        if not row["Retriever"].lower().startswith("naive"):
            improvement = ((row["Average"] - baseline_score) / baseline_score) * 100
            symbol = "⬆️" if improvement > 0 else "⬇️"
            print(f"{row['Retriever']:<20} {symbol} {improvement:+.2f}%")
else:
    print("\n⚠️  Comparison table generation failed, but individual results were saved.")

print("\n" + "=" * 80)

# 12. Summary of Saved Results
print("\n12. Summary of saved results...")
print(f"   ✓ All results saved to: {output_dir}")
print(f"   ✓ Individual retriever datasets: {len(evaluation_datasets)} files")
print(f"   ✓ Individual detailed results: {len(evaluation_results)} files")
if comparison_df is not None:
    print("   ✓ Comparative summary table: comparative_ragas_results.csv")
else:
    print("   ⚠️  Comparative summary table: FAILED (but individual results saved)")

# Generate markdown table for deliverables
if comparison_df is not None:
    print("\n📋 Markdown Table (for deliverables.md):")
    print("\n```")
    print(comparison_df.to_markdown(index=False, floatfmt=".4f"))
    print("```")
else:
    print("\n⚠️  Markdown table not generated (comparison failed)")

print("\n" + "=" * 80)
print("✅ COMPREHENSIVE EVALUATION COMPLETE!")
print("=" * 80)
print(f"\nOutputs saved to: {output_dir}")
print("   - comparative_ragas_results.parquet (summary table)")
print("   - [retriever]_evaluation_inputs.parquet (6 columns: RAG outputs)")
print("   - [retriever]_evaluation_metrics.parquet (10 columns: RAG + RAGAS metrics)")
print("   - RUN_MANIFEST.json (provenance and reproducibility)")

# 13. Generate Run Manifest for Reproducibility
print("\n13. Generating RUN_MANIFEST.json for reproducibility...")
from src.utils import generate_run_manifest

manifest_path = output_dir / "RUN_MANIFEST.json"
manifest = generate_run_manifest(
    manifest_path, evaluation_results, retrievers_config, data_provenance
)
print(f"   ✓ Manifest saved to {manifest_path}")
print(
    f"   ✓ Captured: RAGAS {manifest['ragas_version']}, Python {manifest['python_version']}"
)
print(
    f"   ✓ Contains: {len(manifest['retrievers'])} retriever configs + evaluation settings"
)
if data_provenance:
    print(
        f"   ✓ Data lineage: Linked to ingestion manifest {data_provenance['ingest_manifest_id'][:8]}..."
    )
