#!/usr/bin/env python3
"""
Generate RUN_MANIFEST.json for Reproducibility

Captures exact configuration of RAGAS evaluation runs including:
- Model versions and parameters
- Retriever configurations
- Evaluation settings
- Dependencies

This manifest enables exact reproduction of evaluation results.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Import actual runtime configuration
from src.config import (
    OPENAI_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    COLLECTION_NAME,
    QDRANT_HOST,
    QDRANT_PORT,
)


def generate_manifest(
    output_path: Path,
    evaluation_results: Optional[Dict[str, Any]] = None,
    retrievers_config: Optional[Dict[str, Any]] = None,
    data_provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate run manifest JSON for reproducibility.

    Args:
        output_path: Path to save manifest JSON
        evaluation_results: Optional dict of RAGAS evaluation results
        retrievers_config: Optional dict of retriever configurations
        data_provenance: Optional dict linking to ingestion manifest

    Returns:
        Dictionary containing the manifest
    """

    # Import here to get actual versions
    try:
        import ragas

        ragas_version = ragas.__version__
    except ImportError:
        ragas_version = "unknown"

    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    manifest = {
        "ragas_version": ragas_version,
        "python_version": python_version,
        "llm": {
            "model": OPENAI_MODEL_NAME,  # Dynamic from src.config
            "temperature": 0,
            "provider": "openai",
            "purpose": "RAG generation and RAGAS evaluation",
        },
        "embeddings": {
            "model": EMBEDDING_MODEL_NAME,  # Dynamic from src.config
            "dimensions": 1536,
            "provider": "openai",
            "purpose": "Document and query embeddings",
        },
        "retrievers": [
            {
                "name": "naive",
                "type": "dense_vector_search",
                "description": "Baseline dense vector search with OpenAI embeddings",
                "k": 5,
                "distance_metric": "cosine",
                "rerank": False,
            },
            {
                "name": "bm25",
                "type": "sparse_keyword",
                "description": "BM25 sparse keyword matching (lexical)",
                "k": 5,
                "rerank": False,
            },
            {
                "name": "cohere_rerank",
                "type": "contextual_compression",
                "description": "Cohere rerank-v3.5 with contextual compression",
                "initial_k": 20,
                "top_n": 3,
                "rerank_model": "rerank-v3.5",
                "rerank_provider": "cohere",
                "rerank": True,
            },
            {
                "name": "ensemble",
                "type": "hybrid",
                "description": "Ensemble combining dense vector + sparse keyword",
                "dense_k": 5,
                "sparse_k": 5,
                "weights": [0.5, 0.5],
                "components": ["naive_dense", "bm25_sparse"],
                "rerank": False,
            },
        ],
        "evaluation": {
            "golden_testset": "open-biosciences/biosciences-golden-testset",
            "golden_testset_size": 12,  # Note: Actual size, RAGAS may generate more than requested
            "source_dataset": "open-biosciences/biosciences-sources",
            "source_dataset_size": 38,
            "metrics": [
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
            ],
            "timeout_seconds": 360,
            "ragas_run_config": {"timeout": 360, "max_workers": 4},
        },
        "vector_store": {
            "type": "qdrant",
            "collection_name": COLLECTION_NAME,  # Dynamic from src.config
            "host": QDRANT_HOST,  # Dynamic from src.config
            "port": QDRANT_PORT,  # Dynamic from src.config
            "distance": "cosine",
            "vector_size": 1536,
        },
        "skipped": [],
        "notes": [
            "All LLM calls use temperature=0 for determinism",
            "Fine-tuned embeddings out of scope per instructor guidance",
            "Evaluation follows RAGAS 0.2.10 API patterns from session08",
        ],
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "generated_by": "scripts/generate_run_manifest.py",
    }

    # Add evaluation results summary if provided
    if evaluation_results:
        results_summary = {}
        for retriever_name, result in evaluation_results.items():
            try:
                # Extract scores from RAGAS result object
                df = result.to_pandas()
                results_summary[retriever_name] = {
                    "faithfulness": float(df["faithfulness"].mean()),
                    "answer_relevancy": float(df["answer_relevancy"].mean()),
                    "context_precision": float(df["context_precision"].mean()),
                    "context_recall": float(df["context_recall"].mean()),
                    "average": float(
                        df[
                            [
                                "faithfulness",
                                "answer_relevancy",
                                "context_precision",
                                "context_recall",
                            ]
                        ]
                        .mean()
                        .mean()
                    ),
                }
            except Exception as e:
                results_summary[retriever_name] = {"error": str(e)}

        manifest["results_summary"] = results_summary

    # Add data provenance if provided
    if data_provenance:
        manifest["data_provenance"] = data_provenance

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


if __name__ == "__main__":
    # Standalone execution
    output_path = (
        Path(__file__).parent.parent / "data" / "processed" / "RUN_MANIFEST.json"
    )

    manifest = generate_manifest(output_path)

    print("=" * 80)
    print("RUN_MANIFEST.json Generated")
    print("=" * 80)
    print(f"\nLocation: {output_path}")
    print(f"RAGAS version: {manifest['ragas_version']}")
    print(f"Python version: {manifest['python_version']}")
    print(f"Retrievers: {len(manifest['retrievers'])}")
    print("\nManifest contains full configuration for reproducibility.")
    print("=" * 80)
