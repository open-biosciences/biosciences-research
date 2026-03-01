#!/usr/bin/env python3
"""
Generate human-friendly deliverables from machine-readable processed data.

Architecture Compliance:
- Reads: data/processed/*.parquet (machine format, Parquet-first)
- Writes: deliverables/evaluation_evidence/*.csv (human format)
- Enforces: deliverables/ is DERIVED ONLY, never a working sink

Usage:
    python scripts/generate_deliverables.py

Output:
    - deliverables/evaluation_evidence/*.csv (human-readable)
    - deliverables/evaluation_evidence/RUN_MANIFEST.json (copied)
"""

import json
import shutil
from pathlib import Path

import pandas as pd

# Paths
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
DELIVERABLES_DIR = BASE_DIR / "deliverables" / "evaluation_evidence"

# Retriever names
RETRIEVERS = ["naive", "bm25", "ensemble", "cohere_rerank"]


def convert_parquet_to_csv(parquet_pattern: str, csv_pattern: str):
    """Convert Parquet files to CSV for human readability."""
    count = 0

    for retriever in RETRIEVERS:
        parquet_file = PROCESSED_DIR / parquet_pattern.format(retriever=retriever)

        if not parquet_file.exists():
            print(f"   ⚠️  Skipping {parquet_file.name} (not found)")
            continue

        csv_file = DELIVERABLES_DIR / csv_pattern.format(retriever=retriever)

        # Read Parquet, write CSV
        df = pd.read_parquet(parquet_file)
        df.to_csv(csv_file, index=False)

        print(f"   ✓ {parquet_file.name} → {csv_file.name} ({len(df)} rows)")
        count += 1

    return count


def main():
    """Generate all deliverables from processed data."""
    print("📂 Generating deliverables from data/processed/...")

    # Ensure output directory exists
    DELIVERABLES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Convert evaluation inputs (Parquet → CSV)
    print("\n1️⃣  Converting evaluation inputs...")
    count = convert_parquet_to_csv(
        "{retriever}_evaluation_inputs.parquet",
        "{retriever}_evaluation_dataset.csv"
    )
    print(f"   ✅ Converted {count} evaluation input files")

    # 2. Convert evaluation metrics (Parquet → CSV)
    print("\n2️⃣  Converting evaluation metrics...")
    count = convert_parquet_to_csv(
        "{retriever}_evaluation_metrics.parquet",
        "{retriever}_detailed_results.csv"
    )
    print(f"   ✅ Converted {count} evaluation metric files")

    # 3. Convert comparative results (Parquet → CSV)
    print("\n3️⃣  Converting comparative results...")
    comp_parquet = PROCESSED_DIR / "comparative_ragas_results.parquet"
    comp_csv = DELIVERABLES_DIR / "comparative_ragas_results.csv"

    if comp_parquet.exists():
        df = pd.read_parquet(comp_parquet)
        df.to_csv(comp_csv, index=False)
        print(f"   ✓ {comp_parquet.name} → {comp_csv.name} ({len(df)} rows)")
        print(f"   ✅ Converted comparative results")
    else:
        print(f"   ⚠️  {comp_parquet.name} not found")

    # 4. Copy RUN_MANIFEST.json
    print("\n4️⃣  Copying manifest...")
    manifest_src = PROCESSED_DIR / "RUN_MANIFEST.json"
    manifest_dst = DELIVERABLES_DIR / "RUN_MANIFEST.json"

    if manifest_src.exists():
        shutil.copy2(manifest_src, manifest_dst)

        # Pretty-print manifest
        with open(manifest_src) as f:
            manifest = json.load(f)

        print(f"   ✓ RUN_MANIFEST.json copied")
        print(f"      - Timestamp: {manifest.get('generated_at', 'N/A')}")
        retriever_names = [r['name'] for r in manifest.get('retrievers', [])]
        print(f"      - Retrievers: {', '.join(retriever_names)}")
        print(f"   ✅ Manifest copied")
    else:
        print(f"   ⚠️  RUN_MANIFEST.json not found in data/processed/")

    # Summary
    print("\n" + "="*60)
    print("🎉 Deliverables generated successfully!")
    print(f"\n📊 Location: {DELIVERABLES_DIR}")
    print("\n✨ Next steps:")
    print("   1. Review CSV files in deliverables/evaluation_evidence/")
    print("   2. Commit changes to feature branch")
    print("   3. Test full pipeline before merging")


if __name__ == "__main__":
    main()
