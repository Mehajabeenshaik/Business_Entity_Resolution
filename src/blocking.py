"""
blocking.py
===========
Amazon ML Challenge 2026 - Business Entity Resolution
------------------------------------------------------
Blocking / Candidate Generation module.

Problem
-------
Source 1 has 2.2M records; Source 2 + 3 together have ~10M records.
A full O(n*m) pairwise comparison is infeasible, so we use *blocking*:
for each Source-1 entity we generate a small set of candidate IDs from
Source-2/3 that are plausible matches, using cheap key-based lookup.

Strategy - four complementary blocking keys per record
-------------------------------------------------------
BK1  country + first 4 chars of name_clean
       -> "us_gene"   captures same-country, same-prefix records
BK2  country + first significant token of name_clean
       -> "us_general"  looser than BK1, catches prefix mismatches
BK3  first 5 chars of name_clean  (no country)
       -> "gener"   bridges records that differ only in country field
BK4  sorted two most significant tokens of name_clean
       -> "design_general"  catches word-order transpositions

All four keys are generated from NORMALIZED text (name_clean, country_clean)
so that "Inc.", "inc", "INC" all map to the same token after normalize.py runs.

Safeguard
---------
Very common blocking keys (e.g. a 4-char prefix shared by thousands of
generic company names) can make candidate lists explode.  Each key bucket
is capped at MAX_CANDIDATES_PER_KEY so that runaway keys are suppressed.

Output
------
output/candidate_pairs.tsv
    source1_entity_id <TAB> candidate_entity_ids (comma-separated)
    One row per Source-1 entity.  Empty string when no candidates found.
"""

import csv
import os
import sys
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

# Allow running as: python src/blocking.py (from project root)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.load_data import FILE_PATHS, load_ground_truth, load_source
from src.normalize import normalize_dataframe

# Force UTF-8 console output on Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Output file path (relative to project root)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

# Cap the number of entity IDs stored per blocking key.
# Keys with more hits than this are effectively stop-words (e.g., a 4-char
# prefix shared by thousands of generic names) and add noise without recall gain.
MAX_CANDIDATES_PER_KEY = 150


# ---------------------------------------------------------------------------
# Blocking key generation
# ---------------------------------------------------------------------------

def _first_significant_token(tokens: list[str], min_len: int = 2) -> str:
    """
    Return the first token that is at least min_len characters long.
    Falls back to the very first token if none meets the threshold.

    Parameters
    ----------
    tokens  : list[str]  pre-split tokens from name_clean
    min_len : int        minimum character length to be 'significant'

    Returns
    -------
    str  (empty string if tokens is empty)
    """
    for tok in tokens:
        if len(tok) >= min_len:
            return tok
    return tokens[0] if tokens else ""


def _sorted_two_tokens(tokens: list[str], min_len: int = 3) -> str:
    """
    Return a canonical string built from the two most significant tokens,
    sorted alphabetically to handle word-order variations.

    e.g. tokens = ["general", "design", "innovations"]
         -> sorted(["general", "design"]) -> "design_general"

    Parameters
    ----------
    tokens  : list[str]
    min_len : int   minimum length for a token to be considered significant

    Returns
    -------
    str  joined with "_", or "" if fewer than 2 significant tokens
    """
    significant = [t for t in tokens if len(t) >= min_len]
    # Take first 2 significant tokens (preserves order before sorting)
    pair = significant[:2]
    if len(pair) < 2:
        # Fall back to first 2 tokens of any length
        pair = tokens[:2]
    if len(pair) < 2:
        return ""
    return "_".join(sorted(pair))


def generate_keys(
    name_clean: str,
    country_clean: str,
    address_clean: str = "",
) -> list[str]:
    """
    Generate all blocking keys for a single record.

    Keys are prefixed with a short tag so different key types never collide
    in the same inverted-index bucket, preventing false positives.

    Key design
    ----------
    BK1  c5:  country + first 5 chars of name_clean
               More selective than 4 chars; still catches short variations.
    BK2  ct:  country + first significant token (min length 5)
               Skips short common words (e.g. "ram", "new", "sri", "the")
               that previously created mega-buckets of 10k+ candidates.
    BK3  p6:  first 6 chars of name_clean (no country)
               Country-agnostic bridge; helps cross-country matches.
    BK4  st:  sorted two most significant tokens (min length 4)
               Catches word-order transpositions in business names.
    BK5  ca:  country + first 6 chars of address_clean  (FALLBACK ONLY)
               Applied only when name_clean is empty (e.g. Hindi/Devanagari
               names that are stripped to "" after normalization).
               Recovers those records using address similarity instead.

    Parameters
    ----------
    name_clean    : str  cleaned business name (from normalize.py)
    country_clean : str  cleaned country string
    address_clean : str  cleaned address (used only as fallback)

    Returns
    -------
    list[str]  deduplicated list of blocking key strings

    Examples
    --------
    >>> generate_keys("general design innovations limited liability company", "us", "")
    ['c5:us_gener', 'ct:us_general', 'p6:genera', 'st:design_general']
    >>> generate_keys("", "india", "kh no 570 new delhi west delhi delhi")
    ['ca:india_kh no']
    """
    country = country_clean if country_clean else "xx"
    keys: list[str] = []

    if name_clean:
        tokens = name_clean.split()

        # BK1: country + first 5 chars (more selective than 4)
        prefix5 = name_clean[:5]
        if len(prefix5) >= 3:
            keys.append(f"c5:{country}_{prefix5}")

        # BK2: country + first token with >= 5 chars
        # Min length 5 skips common short words that create huge buckets:
        # "ram", "new", "sri", "shri", "the", "co", etc.
        first_tok = _first_significant_token(tokens, min_len=5)
        if first_tok:
            keys.append(f"ct:{country}_{first_tok}")

        # BK3: first 6 chars, no country (cross-country bridge)
        prefix6 = name_clean[:6]
        if len(prefix6) >= 4:
            keys.append(f"p6:{prefix6}")

        # BK4: sorted two significant tokens (min length 4)
        sorted_tok = _sorted_two_tokens(tokens, min_len=4)
        if sorted_tok:
            keys.append(f"st:{sorted_tok}")

    else:
        # BK5: address-based fallback for records with no usable name
        # (e.g. businesses whose name is entirely in Devanagari/Hindi script)
        if address_clean:
            addr_prefix = address_clean[:6].strip()
            if len(addr_prefix) >= 4:
                keys.append(f"ca:{country}_{addr_prefix}")

    # Remove duplicates while preserving order
    seen: set[str] = set()
    unique_keys: list[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            unique_keys.append(k)

    return unique_keys


# ---------------------------------------------------------------------------
# Inverted index construction
# ---------------------------------------------------------------------------

def build_inverted_index(
    source_dfs: list[pd.DataFrame],
    max_per_key: int = MAX_CANDIDATES_PER_KEY,
    desc: str = "Indexing S2+S3",
) -> dict[str, set[str]]:
    """
    Build an inverted index mapping blocking_key -> set of entity_ids.

    Processes all DataFrames in source_dfs in sequence.  Typically called
    with [s2_df, s3_df] so that the index covers all candidate sources.

    Parameters
    ----------
    source_dfs  : list[pd.DataFrame]
        Normalized source DataFrames.  Must contain columns:
        entity_id, name_clean, country_clean.
    max_per_key : int
        Hard cap on how many entity IDs a single key may hold.
        Prevents hot keys from exploding candidate set sizes.
    desc        : str
        Label for the tqdm progress bar.

    Returns
    -------
    dict[str, set[str]]
        Inverted index.  defaultdict is converted to plain dict before
        returning so callers get safe key-miss behaviour (returns None).
    """
    index: dict[str, set[str]] = defaultdict(set)

    # Concatenate all sources into one iterator for a single progress bar.
    total_rows = sum(len(df) for df in source_dfs)

    # Build a unified iterator over all rows across all dataframes.
    def _row_iter():
        for df in source_dfs:
            yield from df.itertuples(index=False)

    for row in tqdm(_row_iter(), total=total_rows, desc=desc, ncols=90):
        entity_id     = row.entity_id
        name_clean    = row.name_clean
        country_clean = row.country_clean
        address_clean = row.address_clean

        for key in generate_keys(name_clean, country_clean, address_clean):
            bucket = index[key]
            if len(bucket) < max_per_key:
                bucket.add(entity_id)

    return dict(index)


# ---------------------------------------------------------------------------
# Candidate lookup
# ---------------------------------------------------------------------------

def generate_candidates(
    s1_df: pd.DataFrame,
    index: dict[str, set[str]],
    desc: str = "Generating candidates",
) -> dict[str, list[str]]:
    """
    For every Source-1 entity, look up all candidates from the inverted index
    and return a deduplicated, sorted list.

    Guarantees
    ----------
    - Every Source-1 entity_id in s1_df has an entry in the returned dict
      (empty list if no candidates found).
    - Candidate IDs are sorted for deterministic output.
    - The Source-1 entity itself is never in its own candidate list.

    Parameters
    ----------
    s1_df : pd.DataFrame
        Normalized Source-1.  Needs: entity_id, name_clean, country_clean.
    index : dict[str, set[str]]
        Inverted index from build_inverted_index().
    desc  : str
        Label for the progress bar.

    Returns
    -------
    dict[str, list[str]]
        source1_entity_id -> sorted list of candidate S2/S3 entity_ids.
    """
    candidates: dict[str, list[str]] = {}

    for row in tqdm(
        s1_df.itertuples(index=False),
        total=len(s1_df),
        desc=desc,
        ncols=90,
    ):
        entity_id     = row.entity_id
        name_clean    = row.name_clean
        country_clean = row.country_clean

        merged: set[str] = set()
        for key in generate_keys(name_clean, country_clean):
            bucket = index.get(key)
            if bucket:
                merged.update(bucket)

        # Safety: remove the entity itself (cross-source shouldn't happen,
        # but guard against edge cases in the data).
        merged.discard(entity_id)

        candidates[entity_id] = sorted(merged)

    return candidates


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_candidates(
    candidates: dict[str, list[str]],
    output_path: str = OUTPUT_FILE,
) -> None:
    """
    Write candidate pairs to a tab-separated file.

    Output format
    -------------
    Header : source1_entity_id <TAB> candidate_entity_ids
    Rows   : one per Source-1 entity; candidate IDs comma-separated.
             Empty string in the second column when no candidates found.

    Parameters
    ----------
    candidates  : dict[str, list[str]]
    output_path : str
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"  Writing {len(candidates):,} rows -> {os.path.abspath(output_path)}")
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for s1_id, cand_list in tqdm(
            candidates.items(), desc="Saving TSV", ncols=90
        ):
            writer.writerow([s1_id, ",".join(cand_list)])

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  File size: {size_mb:.1f} MB")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_blocking(
    candidates: dict[str, list[str]],
    gt_df: pd.DataFrame,
) -> dict[str, float]:
    """
    Measure blocking quality against the training ground truth.

    Metrics computed
    ----------------
    avg_candidates
        Average number of candidate IDs per Source-1 entity.
        Measures the cost side of the trade-off.

    blocking_recall  (pair-level)
        Of all (S1, S2/S3) true-match pairs in the ground truth, what
        fraction have the S2/S3 ID present in the candidate list?
        This is the most important metric: missing a true match here
        means it can never be recovered in later ranking stages.

    entity_recall  (entity-level)
        Fraction of Source-1 entities where EVERY true match is covered.

    Parameters
    ----------
    candidates : dict[str, list[str]]
        Output of generate_candidates().
    gt_df : pd.DataFrame
        Training ground truth; columns: source1_entity_id, matched_entity_ids.

    Returns
    -------
    dict[str, float | int]
    """
    total_candidates = sum(len(v) for v in candidates.values())
    n_s1 = len(candidates)
    avg_candidates = total_candidates / n_s1 if n_s1 else 0.0

    total_true = 0
    covered_true = 0
    entities_fully_covered = 0

    for row in tqdm(
        gt_df.itertuples(index=False),
        total=len(gt_df),
        desc="Evaluating",
        ncols=90,
    ):
        s1_id = row.source1_entity_id
        raw   = row.matched_entity_ids

        # Skip entities with no ground-truth matches (0-match bucket)
        if not isinstance(raw, str) or not raw.strip():
            continue

        true_ids  = set(raw.split(","))
        total_true += len(true_ids)

        cand_set = set(candidates.get(s1_id, []))
        covered  = true_ids & cand_set
        covered_true += len(covered)

        if covered == true_ids:
            entities_fully_covered += 1

    blocking_recall = covered_true / total_true if total_true else 0.0
    # Entity recall denominator: only S1 entities that have >= 1 true match
    entities_with_matches = (gt_df["matched_entity_ids"].str.strip() != "").sum()
    entity_recall = (
        entities_fully_covered / entities_with_matches
        if entities_with_matches else 0.0
    )

    return {
        "avg_candidates":          avg_candidates,
        "total_candidates":        total_candidates,
        "blocking_recall":         blocking_recall,
        "entity_recall":           entity_recall,
        "total_true_matches":      total_true,
        "covered_matches":         covered_true,
        "entities_fully_covered":  entities_fully_covered,
        "entities_with_matches":   int(entities_with_matches),
        "total_s1_entities":       n_s1,
    }


def print_evaluation(metrics: dict) -> None:
    """Pretty-print the evaluation dictionary returned by evaluate_blocking()."""
    sep = "=" * 60
    print(f"\n{sep}")
    print("  BLOCKING EVALUATION RESULTS")
    print(sep)
    print(f"  Avg candidates per S1 entity : {metrics['avg_candidates']:>10.1f}")
    print(f"  Total candidates generated   : {metrics['total_candidates']:>10,}")
    print(f"  Blocking recall (pair-level) : {metrics['blocking_recall']*100:>9.2f}%")
    print(f"  Entity recall  (all covered) : {metrics['entity_recall']*100:>9.2f}%")
    print(f"  ---")
    print(f"  True match pairs in GT       : {metrics['total_true_matches']:>10,}")
    print(f"  Covered by candidates        : {metrics['covered_matches']:>10,}")
    print(f"  Entities with >=1 true match : {metrics['entities_with_matches']:>10,}")
    print(f"  Entities fully covered       : {metrics['entities_fully_covered']:>10,}")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_blocking(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    gt_df: pd.DataFrame | None = None,
    output_path: str = OUTPUT_FILE,
    evaluate: bool = True,
) -> dict[str, list[str]]:
    """
    End-to-end blocking pipeline: normalize -> index -> candidates -> save.

    Parameters
    ----------
    s1_df, s2_df, s3_df : pd.DataFrame
        Raw source DataFrames (pre-load, not yet normalized).
    gt_df : pd.DataFrame or None
        Training ground truth.  Pass None to skip evaluation.
    output_path : str
        Where to write candidate_pairs.tsv.
    evaluate : bool
        Whether to run evaluate_blocking() after generating candidates.

    Returns
    -------
    dict[str, list[str]]
        source1_entity_id -> list of candidate entity_ids
    """
    # -- Normalize --
    print("\n[Normalize] Source 1 ...")
    s1_norm = normalize_dataframe(s1_df)
    print("[Normalize] Source 2 ...")
    s2_norm = normalize_dataframe(s2_df)
    print("[Normalize] Source 3 ...")
    s3_norm = normalize_dataframe(s3_df)

    # -- Build inverted index --
    print("\n[Index] Building inverted index from Source 2 + Source 3 ...")
    index = build_inverted_index([s2_norm, s3_norm])
    n_keys = len(index)
    print(f"  {n_keys:,} unique blocking keys in index.")

    # -- Generate candidates --
    print("\n[Candidates] Generating candidates for Source 1 ...")
    candidates = generate_candidates(s1_norm, index)

    # -- Evaluate --
    if evaluate and gt_df is not None:
        print("\n[Evaluate] Measuring blocking quality ...")
        metrics = evaluate_blocking(candidates, gt_df)
        print_evaluation(metrics)

    # -- Save --
    print("[Save] Writing output file ...")
    save_candidates(candidates, output_path)

    return candidates


def main() -> None:
    """
    Full blocking run using the training split.
    Execute with:  python src/blocking.py
    """
    print("=" * 60)
    print("  Blocking / Candidate Generation")
    print("=" * 60)

    # Load raw data
    print("\n[1] Loading datasets ...")
    s1 = load_source(FILE_PATHS["train_source1"], "train_source1")
    s2 = load_source(FILE_PATHS["train_source2"], "train_source2")
    s3 = load_source(FILE_PATHS["train_source3"], "train_source3")
    gt = load_ground_truth(FILE_PATHS["train_ground_truth"])
    print(f"  S1: {len(s1):,}  S2: {len(s2):,}  S3: {len(s3):,}  GT: {len(gt):,}")

    # Run the full pipeline
    run_blocking(s1, s2, s3, gt_df=gt, evaluate=True)

    print("Done.")


if __name__ == "__main__":
    main()
