#!/usr/bin/env python3
"""
Validate Manifest Files and SHA-256 Checksums

This script validates that:
1. Interim manifest exists and is valid JSON
2. SHA-256 checksums in manifest match actual files
3. All referenced files exist
4. Data provenance chain is intact

Usage:
    python scripts/validate_manifests.py
    # or
    PYTHONPATH=. python scripts/validate_manifests.py
"""

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Tuple


def find_repo_root(start_path: Path) -> Path:
    """Find repository root by looking for .git directory."""
    current = start_path.resolve()
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    return start_path


# Set up paths
SCRIPT_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()
PROJECT_ROOT = find_repo_root(SCRIPT_DIR)
INTERIM_MANIFEST = PROJECT_ROOT / "data/interim/manifest.json"
RUN_MANIFEST = PROJECT_ROOT / "data/processed/RUN_MANIFEST.json"


def hash_file(path: Path, algo: str = "sha256") -> str:
    """
    Compute hash of a file.

    Args:
        path: Path to file
        algo: Hash algorithm (default: sha256)

    Returns:
        Hexadecimal hash string
    """
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_interim_manifest() -> Tuple[bool, list]:
    """
    Validate interim manifest and its SHA-256 checksums.

    Returns:
        (success: bool, errors: list)
    """
    errors = []

    # Check if manifest exists
    if not INTERIM_MANIFEST.exists():
        errors.append(f"❌ Interim manifest not found: {INTERIM_MANIFEST}")
        return False, errors

    print(f"📄 Validating interim manifest: {INTERIM_MANIFEST.relative_to(PROJECT_ROOT)}")

    # Load manifest
    try:
        with open(INTERIM_MANIFEST) as f:
            manifest = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"❌ Invalid JSON in interim manifest: {e}")
        return False, errors

    # Check required fields
    required_fields = ["id", "generated_at", "fingerprints", "paths", "artifacts"]
    for field in required_fields:
        if field not in manifest:
            errors.append(f"❌ Missing required field: {field}")

    if errors:
        return False, errors

    # Validate SHA-256 checksums
    print("\n🔐 Validating SHA-256 checksums...")
    datasets = ["sources", "golden_testset"]
    formats = ["jsonl", "parquet"]

    validated_count = 0
    for dataset in datasets:
        if dataset not in manifest["fingerprints"]:
            errors.append(f"❌ Missing fingerprints for dataset: {dataset}")
            continue

        for fmt in formats:
            hash_key = f"{fmt}_sha256"
            if hash_key not in manifest["fingerprints"][dataset]:
                errors.append(f"❌ Missing {hash_key} for {dataset}")
                continue

            # Get file path
            path_key = fmt
            if path_key not in manifest["paths"][dataset]:
                errors.append(f"❌ Missing path for {dataset}.{fmt}")
                continue

            file_path = Path(manifest["paths"][dataset][path_key])

            # Make path relative to project root if it's absolute
            if file_path.is_absolute():
                # Try to find it relative to current project root
                relative_path = PROJECT_ROOT / "data/interim" / file_path.name
                if relative_path.exists():
                    file_path = relative_path

            # Check if file exists
            if not file_path.exists():
                errors.append(f"❌ File not found: {file_path}")
                continue

            # Compute current hash
            expected_hash = manifest["fingerprints"][dataset][hash_key]
            actual_hash = hash_file(file_path)

            # Compare
            if expected_hash == actual_hash:
                print(f"   ✓ {dataset}.{fmt}: hash matches ({actual_hash[:16]}...)")
                validated_count += 1
            else:
                errors.append(
                    f"❌ {dataset}.{fmt}: hash mismatch!\n"
                    f"      Expected: {expected_hash}\n"
                    f"      Actual:   {actual_hash}"
                )

    print(f"\n✅ Validated {validated_count}/{len(datasets) * len(formats)} checksums")

    return len(errors) == 0, errors


def validate_run_manifest() -> Tuple[bool, list]:
    """
    Validate RUN_MANIFEST and its provenance chain.

    Returns:
        (success: bool, errors: list)
    """
    errors = []

    # Check if manifest exists
    if not RUN_MANIFEST.exists():
        # Not an error if we haven't run evaluation yet
        print(f"ℹ️  RUN_MANIFEST not found (run 'make eval' to generate): {RUN_MANIFEST.relative_to(PROJECT_ROOT)}")
        return True, []

    print(f"\n📄 Validating RUN_MANIFEST: {RUN_MANIFEST.relative_to(PROJECT_ROOT)}")

    # Load manifest
    try:
        with open(RUN_MANIFEST) as f:
            run_manifest = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"❌ Invalid JSON in RUN_MANIFEST: {e}")
        return False, errors

    # Check provenance link
    if "data_provenance" in run_manifest:
        print("\n🔗 Validating provenance chain...")

        # Load interim manifest to compare
        if INTERIM_MANIFEST.exists():
            with open(INTERIM_MANIFEST) as f:
                interim_manifest = json.load(f)

            # Check ID match
            expected_id = interim_manifest.get("id")
            actual_id = run_manifest["data_provenance"].get("ingest_manifest_id")

            if expected_id == actual_id:
                print(f"   ✓ Provenance ID matches: {actual_id}")
            else:
                errors.append(
                    f"❌ Provenance ID mismatch!\n"
                    f"      Interim manifest: {expected_id}\n"
                    f"      RUN_MANIFEST provenance: {actual_id}"
                )

            # Check SHA-256 hashes match
            datasets = ["sources", "golden_testset"]
            for dataset in datasets:
                expected_hash = interim_manifest.get("fingerprints", {}).get(dataset, {}).get("jsonl_sha256")
                actual_hash = run_manifest["data_provenance"].get(f"{dataset}_sha256")

                if expected_hash == actual_hash:
                    print(f"   ✓ {dataset} hash matches: {actual_hash[:16]}...")
                else:
                    errors.append(
                        f"❌ {dataset} hash mismatch in provenance!\n"
                        f"      Interim manifest: {expected_hash}\n"
                        f"      RUN_MANIFEST: {actual_hash}"
                    )
        else:
            errors.append(f"❌ Cannot validate provenance: interim manifest not found")
    else:
        print("   ⚠️  No data_provenance field in RUN_MANIFEST")

    return len(errors) == 0, errors


def main():
    """Main validation workflow."""
    print("="*60)
    print("🔍 Manifest Validation")
    print("="*60)

    all_errors = []

    # Validate interim manifest
    interim_ok, interim_errors = validate_interim_manifest()
    all_errors.extend(interim_errors)

    # Validate RUN_MANIFEST
    run_ok, run_errors = validate_run_manifest()
    all_errors.extend(run_errors)

    # Summary
    print("\n" + "="*60)
    if all_errors:
        print(f"❌ Validation FAILED with {len(all_errors)} error(s):\n")
        for error in all_errors:
            print(f"   {error}")
        print("\n" + "="*60)
        return 1
    else:
        print("✅ All manifest validations passed!")
        print("="*60)
        return 0


if __name__ == "__main__":
    sys.exit(main())
