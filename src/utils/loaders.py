# loaders.py

"""
Data loading utilities for GDELT RAG system.

This module provides helper functions for loading and processing documents
from HuggingFace datasets.
"""

from typing import List
from datasets import load_dataset
from langchain_core.documents import Document


def load_documents_from_huggingface(
    dataset_name: str = "open-biosciences/biosciences-sources",
    split: str = "train",
    revision: str = None,
) -> List[Document]:
    """
    Load documents from HuggingFace dataset and convert to LangChain Documents.

    This function handles the conversion from HuggingFace dataset format to
    LangChain Document objects, properly extracting page_content and metadata.

    Args:
        dataset_name: HuggingFace dataset identifier (default: "open-biosciences/biosciences-sources")
        split: Dataset split to load (default: "train")
        revision: Dataset revision/commit SHA to pin (default: None, uses HF_SOURCES_REV env var if set)

    Returns:
        List of LangChain Document objects with page_content and metadata

    Example:
        >>> from src.utils import load_documents_from_huggingface
        >>> documents = load_documents_from_huggingface()
        >>> print(f"Loaded {len(documents)} documents")

        >>> # Pin to specific revision for reproducibility
        >>> documents = load_documents_from_huggingface(revision="abc123")
        >>> # Or use environment variable: export HF_SOURCES_REV=abc123

    Notes:
        - Handles nested metadata structures automatically
        - Preserves all metadata fields from the HuggingFace dataset
        - Empty or missing page_content defaults to empty string
        - Revision pinning prevents dataset drift over time
    """
    import os

    # Use provided revision, or fall back to environment variable, or None (latest)
    effective_revision = revision or os.getenv("HF_SOURCES_REV")

    # Load dataset from HuggingFace
    sources_dataset = load_dataset(
        dataset_name, split=split, revision=effective_revision
    )

    # Convert to LangChain Documents
    documents = []
    for item in sources_dataset:
        # Extract page content
        page_content = item.get("page_content", "")

        # Handle nested metadata structure
        if "metadata" in item and isinstance(item["metadata"], dict):
            # If metadata is a dict, use it directly
            metadata = item["metadata"]
        else:
            # Otherwise, create metadata from all fields except page_content
            metadata = {k: v for k, v in item.items() if k != "page_content"}

        # Create LangChain Document
        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)

    return documents


def load_golden_testset_from_huggingface(
    dataset_name: str = "open-biosciences/biosciences-golden-testset",
    split: str = "train",
    revision: str = None,
):
    """
    Load golden testset from HuggingFace dataset.

    Args:
        dataset_name: HuggingFace dataset identifier (default: "open-biosciences/biosciences-golden-testset")
        split: Dataset split to load (default: "train")
        revision: Dataset revision/commit SHA to pin (default: None, uses HF_GOLDEN_REV env var if set)

    Returns:
        HuggingFace Dataset object

    Example:
        >>> from src.utils import load_golden_testset_from_huggingface
        >>> golden_dataset = load_golden_testset_from_huggingface()
        >>> golden_df = golden_dataset.to_pandas()
        >>> print(f"Loaded {len(golden_df)} test examples")

        >>> # Pin to specific revision for reproducibility
        >>> golden_dataset = load_golden_testset_from_huggingface(revision="abc123")
        >>> # Or use environment variable: export HF_GOLDEN_REV=abc123

    Notes:
        - Revision pinning ensures test set consistency across runs
        - Prevents score drift due to dataset updates
    """
    import os

    # Use provided revision, or fall back to environment variable, or None (latest)
    effective_revision = revision or os.getenv("HF_GOLDEN_REV")

    return load_dataset(dataset_name, split=split, revision=effective_revision)
