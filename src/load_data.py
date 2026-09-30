"""
load_data.py
============
Multi-Source Record Linkage Pipeline
--------------------------------------
Utility module to load all dataset files (train + test) into pandas DataFrames
and print key exploratory statistics.

Dataset location  (on this machine):
    ~/OneDrive/Desktop/AML/student_resource/dataset/
        train/
            train_source1.tsv       - Source-1 business entities (train)
            train_source2.tsv       - Source-2 business entities (train)
            train_source3.tsv       - Source-3 business entities (train)
            train_ground_truth.tsv  - Ground truth matches (source1 -> source2/3)
        test/
            test_source1.tsv        - Source-1 business entities (test)
            test_source2.tsv        - Source-2 business entities (test)
            test_source3.tsv        - Source-3 business entities (test)

Schema for source files  : entity_id | business_name | business_address | country
Schema for ground truth  : source1_entity_id | matched_entity_ids
"""

import os
import sys
import pandas as pd

# Force UTF-8 output so special characters in business names don't crash the
# console on Windows (which defaults to cp1252).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------

# Absolute path to the dataset root – adjust if your layout differs.
_DATASET_ROOT = os.environ.get(
    "AML_DATASET_ROOT",
    os.path.join(
        os.path.expanduser("~"),
        "Downloads", "6ab10eb3b23ba_student_resource",
        "student_resource", "dataset",
    ),
)

_TRAIN = os.path.join(_DATASET_ROOT, "train")
_TEST  = os.path.join(_DATASET_ROOT, "test")

# All file paths keyed by a short descriptive name
FILE_PATHS: dict[str, str] = {
    # ── training sources ──────────────────────────────────────────────────
    "train_source1":      os.path.join(_TRAIN, "train_source1.tsv"),
    "train_source2":      os.path.join(_TRAIN, "train_source2.tsv"),
    "train_source3":      os.path.join(_TRAIN, "train_source3.tsv"),
    "train_ground_truth": os.path.join(_TRAIN, "train_ground_truth.tsv"),
    # ── test sources ──────────────────────────────────────────────────────
    "test_source1":       os.path.join(_TEST, "test_source1.tsv"),
    "test_source2":       os.path.join(_TEST, "test_source2.tsv"),
    "test_source3":       os.path.join(_TEST, "test_source3.tsv"),
}


# ---------------------------------------------------------------------------
# Core loaders
# ---------------------------------------------------------------------------

def load_source(path: str, name: str = "") -> pd.DataFrame:
    """
    Load a source TSV file (source1 / source2 / source3) into a DataFrame.

    Parameters
    ----------
    path : str
        Absolute or relative path to the .tsv file.
    name : str, optional
        Human-readable label used only in error messages.

    Returns
    -------
    pd.DataFrame
        Columns: entity_id, business_name, business_address, country
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[{name}] File not found: {path}\n"
            "Please verify _DATASET_ROOT points to the correct directory."
        )

    df = pd.read_csv(
        path,
        sep="\t",               # tab-separated values
        dtype=str,              # keep every column as string (no int coercion)
        keep_default_na=False,  # treat empty fields as "" rather than NaN
    )
    return df


def load_ground_truth(path: str) -> pd.DataFrame:
    """
    Load the training ground truth file.

    Raw columns
    -----------
    source1_entity_id  – a single Source-1 entity ID
    matched_entity_ids – comma-separated list of matching Source-2/3 IDs

    Added column
    ------------
    match_count – number of matched IDs (0 when the cell is empty)

    Parameters
    ----------
    path : str
        Absolute or relative path to train_ground_truth.tsv.

    Returns
    -------
    pd.DataFrame
        Columns: source1_entity_id, matched_entity_ids, match_count
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ground_truth] File not found: {path}")

    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)

    # Count how many matched IDs each Source-1 entity has.
    # A cell looks like "S2-123,S2-456,S3-789".  Empty string => 0 matches.
    df["match_count"] = df["matched_entity_ids"].apply(
        lambda cell: len(cell.split(",")) if cell.strip() else 0
    )

    return df


def load_all() -> dict[str, pd.DataFrame]:
    """
    Load every dataset file and return a dict keyed by names in FILE_PATHS.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: train_source1, train_source2, train_source3,
              train_ground_truth, test_source1, test_source2, test_source3
    """
    dfs: dict[str, pd.DataFrame] = {}

    for key, path in FILE_PATHS.items():
        print(f"  Loading {key} ...")
        if key == "train_ground_truth":
            dfs[key] = load_ground_truth(path)
        else:
            dfs[key] = load_source(path, name=key)

    return dfs


# ---------------------------------------------------------------------------
# Exploratory statistics helpers
# ---------------------------------------------------------------------------

def print_basic_info(key: str, df: pd.DataFrame) -> None:
    """
    Print shape and column names for a single DataFrame.

    Parameters
    ----------
    key : str           Human-readable name of the dataset.
    df  : pd.DataFrame
    """
    print(f"\n{'-'*60}")
    print(f"  {key.upper()}")
    print(f"{'-'*60}")
    print(f"  Shape   : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  Columns : {list(df.columns)}")


def print_unique_countries(key: str, df: pd.DataFrame) -> None:
    """
    Print unique country values and row counts for a source DataFrame.

    Silently skips DataFrames that have no 'country' column (e.g. ground truth).

    Parameters
    ----------
    key : str
    df  : pd.DataFrame
    """
    if "country" not in df.columns:
        return

    country_counts = df["country"].value_counts(dropna=False)
    n_unique = df["country"].nunique(dropna=False)

    print(f"\n  Unique countries ({n_unique}) in {key}:")
    for country, count in country_counts.items():
        label = repr(country) if country == "" else str(country)
        print(f"    {label:<30} {count:>10,}")


def print_ground_truth_match_distribution(gt_df: pd.DataFrame) -> None:
    """
    Print the match-count distribution for Source-1 entities.

    Categories
    ----------
    0 matches   – Source-1 entity has no corresponding record in Source-2/3.
    1 match     – Exactly one corresponding record.
    2+ matches  – Two or more corresponding records (including a detailed
                  breakdown by exact count).

    Parameters
    ----------
    gt_df : pd.DataFrame
        Must contain the 'match_count' column added by load_ground_truth().
    """
    total = len(gt_df)

    zero_matches = (gt_df["match_count"] == 0).sum()
    one_match    = (gt_df["match_count"] == 1).sum()
    two_plus     = (gt_df["match_count"] >= 2).sum()

    print(f"\n{'-'*60}")
    print("  GROUND TRUTH - Match-count distribution for Source-1 entities")
    print(f"{'-'*60}")
    print(f"  Total Source-1 entities   : {total:>10,}")
    print(f"  Entities with 0 matches   : {zero_matches:>10,}  "
          f"({zero_matches / total * 100:.2f}%)")
    print(f"  Entities with 1 match     : {one_match:>10,}  "
          f"({one_match / total * 100:.2f}%)")
    print(f"  Entities with 2+ matches  : {two_plus:>10,}  "
          f"({two_plus / total * 100:.2f}%)")

    # Detailed per-count breakdown for the 2+ bucket
    if two_plus > 0:
        print("\n  Breakdown of 2+ bucket (exact match count -> row count):")
        breakdown = (
            gt_df[gt_df["match_count"] >= 2]["match_count"]
            .value_counts()
            .sort_index()
        )
        for cnt, grp in breakdown.items():
            print(f"    {cnt:>2} matches : {grp:>10,}")


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Load all dataset files and print exploratory statistics to stdout.
    Run directly with:  python src/load_data.py
    """
    print("=" * 60)
    print("  Multi-Source Record Linkage Pipeline")
    print("  Dataset Explorer")
    print("=" * 60)

    # ── 1. Load all files ──────────────────────────────────────────────────
    print("\n[1] Loading files ...")
    dfs = load_all()
    print("  All files loaded successfully.\n")

    # ── 2. Shape + columns for every file ─────────────────────────────────
    print("\n[2] Shape & Columns")
    for key, df in dfs.items():
        print_basic_info(key, df)

    # ── 3. Unique countries per source ─────────────────────────────────────
    source_keys = [k for k in dfs if k != "train_ground_truth"]
    print("\n\n[3] Unique Countries per Source")
    for key in source_keys:
        print_unique_countries(key, dfs[key])

    # ── 4. Ground-truth match-count distribution ───────────────────────────
    print("\n\n[4] Ground Truth Analysis")
    print_ground_truth_match_distribution(dfs["train_ground_truth"])

    print(f"\n{'='*60}")
    print("  Done.")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Run when executed directly: python src/load_data.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
