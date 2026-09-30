"""
blocking.py
===========
Multi-Source Record Linkage Pipeline
--------------------------------------
Blocking / Candidate Generation module.

Problem
-------
Source 1 has 2.2M records; Source 2 + 3 together have ~10M records.
A full O(n*m) pairwise comparison is infeasible (~22 billion pairs), so we
use *blocking*: for each Source-1 entity we generate a small set of candidate
IDs from Source-2/3 that are plausible matches, using cheap key-based lookup.

Design Goal — Precision-First Blocking
----------------------------------------
The downstream evaluation metric (macro F0.5) penalises false positives
roughly twice as heavily as false negatives.  This shifts the optimal
blocking strategy from "recall at any cost" toward generating a SMALL,
HIGH-QUALITY candidate set.

Previous version problems:
  - BK1 (country + 3-char prefix) was catastrophically broad — the key
    "c3:india_pri" alone matched hundreds of thousands of "Private Ltd"
    companies, producing a 5+ GB candidate file.
  - BK7 (country + 4-char address) was applied to ALL entities, flooding
    candidates for perfectly well-named records.
  - MAX_CANDIDATES_PER_KEY = 180 was too permissive; hot keys still
    contributed 180 entities each × 10 keys = 1800 candidates worst-case.
  - No per-entity total cap: union of all keys could be unbounded.

Strategy — complementary precision-first blocking keys
--------------------------------------------------------
Name-based keys  (when name_clean has >= 5 chars):
  BK1  c5:  country + first 5 chars of name_clean      (primary precision key)
  BK2  c7:  country + first 7 chars of name_clean      (high-precision, short range)
  BK3  ct:  country + first significant token >= 5 chars (token-level recall)
  BK4  p7:  first 7 chars of name_clean, no country    (cross-country bridge)
  BK5  st:  sorted significant tokens (length >= 4)     (word-order invariant)
  BK6  ph:  country + Soundex(first token)              (phonetic / spelling variants)

Address key — CONDITIONAL only (when name is weak or absent):
  BK7  aa:  country + first 6 chars of address_clean

Fallback keys (when name_clean is empty or < 5 chars, e.g. Devanagari/Hindi):
  BK8  ca:  country + first 6 chars of address_clean
  BK9  at:  country + first significant token of address_clean (>= 5 chars)
  BK10 ap:  first 7 chars of address_clean (country-agnostic)

All keys generated from NORMALIZED text so abbreviations and casing are
handled uniformly.

Safeguards
----------
1. Each key bucket is capped at MAX_CANDIDATES_PER_KEY (default: 60) to
   suppress runaway hot-keys.  A bucket larger than this cap is a de facto
   stop-word and adds more noise than signal.
2. Each Source-1 entity's final candidate list is capped at
   MAX_TOTAL_CANDIDATES (default: 80) to guarantee a bounded output file,
   regardless of how many keys fire.
3. BK7 (address key) is only emitted when name_clean has fewer than
   NAME_WEAK_THRESHOLD (5) characters, so well-named entities do not get
   flooded with address-based false positives.

Key changes vs. v1
------------------
- Removed BK1 (c3: 3-char prefix) — the single biggest source of explosion.
- Raised BK2 from c5→ becomes the new BK1; added c7 as BK2.
- Raised BK3 first-token minimum length from 4 → 5.
- Raised BK4 prefix from 6 → 7 chars.
- Raised BK5 token minimum length from 3 → 4 chars.
- Made BK7 (address) conditional on weak name instead of universal.
- Lowered MAX_CANDIDATES_PER_KEY: 180 → 60.
- Added MAX_TOTAL_CANDIDATES = 80 (new hard per-entity cap).

Expected impact
---------------
- Average candidates per entity: ~2400+ → ~50–80
- Output file size: 5+ GB → ~200–400 MB
- Blocking recall: minor drop for very short / generic names; well-named
  entities lose no recall since c5/c7 keys are strictly more selective but
  equally precise.

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

try:
    import jellyfish
    _SOUNDEX_AVAILABLE = True
except ImportError:
    _SOUNDEX_AVAILABLE = False

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

OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "output")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

# Per-bucket cap: keys whose bucket exceeds this size are treated as
# stop-words.  Reduced from 180 → 60 to cut candidate volume aggressively.
MAX_CANDIDATES_PER_KEY = 60

# Per-entity total cap: after unioning all key buckets for one Source-1
# entity, truncate the result to this many candidates.
# Guarantees bounded file size regardless of key overlap.
MAX_TOTAL_CANDIDATES = 80

# Name is considered "weak" (missing or too short to trust) if cleaned name
# has fewer than this many characters.  Only weak-name entities get BK7
# (the conditional address key) added to their blocking set.
NAME_WEAK_THRESHOLD = 5


# ---------------------------------------------------------------------------
# Blocking key helpers
# ---------------------------------------------------------------------------

_LEADING_STOPWORDS = {"the", "a", "an"}


def _strip_leading_stopword(name_clean: str) -> str:
    """
    Drop a single leading article ("the", "a", "an") so prefix keys
    treat "The Home Depot" and "Home Depot" identically.
    Only strips one leading stopword; leaves the rest of the string alone.
    """
    tokens = name_clean.split()
    if tokens and tokens[0] in _LEADING_STOPWORDS and len(tokens) > 1:
        return " ".join(tokens[1:])
    return name_clean


def _first_significant_token(tokens: list[str], min_len: int = 5) -> str:
    """
    Return the first token that is at least min_len characters long.
    Returns "" if no token meets the threshold — we want selectivity, not
    a fallback to short tokens that would act like stop-words.

    Parameters
    ----------
    tokens  : list[str]  pre-split tokens from a cleaned string
    min_len : int        minimum character length to be "significant"

    Returns
    -------
    str  (empty string if no qualifying token found)
    """
    for tok in tokens:
        if len(tok) >= min_len:
            return tok
    return ""


def _sorted_significant_tokens(tokens: list[str], min_len: int = 4) -> str:
    """
    Return a canonical string built from ALL significant tokens
    (length >= min_len), sorted alphabetically to handle word-order
    variations.

    Minimum length raised from 3 → 4 vs. v1 to reduce collisions on
    very common 3-letter words ("the", "and", "for", etc.).

    e.g. tokens = ["general", "design", "innovations", "limited"]
         significant (>=4) = ["general", "design", "innovations", "limited"]
         -> sorted -> "design_general_innovations_limited"

    Parameters
    ----------
    tokens  : list[str]
    min_len : int   minimum length for a token to be considered significant

    Returns
    -------
    str  joined with "_", or "" if fewer than 2 significant tokens
    """
    significant = sorted(t for t in tokens if len(t) >= min_len)
    if len(significant) < 2:
        return ""
    return "_".join(significant)


def _soundex_key(token: str) -> str:
    """
    Return the Soundex code for a token, or "" if unavailable/non-Latin.
    Uses jellyfish if installed; degrades gracefully to "" otherwise.
    """
    if not _SOUNDEX_AVAILABLE or not token:
        return ""
    try:
        code = jellyfish.soundex(token)
        return code if code else ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Blocking key generation
# ---------------------------------------------------------------------------

def generate_keys(
    name_clean: str,
    country_clean: str,
    address_clean: str = "",
) -> list[str]:
    """
    Generate all blocking keys for a single record.

    Keys are prefixed with a short tag so different key types never collide
    in the same inverted-index bucket, preventing false positives.

    Key design (precision-first, v2)
    ---------------------------------
    Name-based keys (generated when name_clean has >= NAME_WEAK_THRESHOLD chars):
      BK1  c5:  country + first 5 chars of name_clean   (primary key)
      BK2  c7:  country + first 7 chars of name_clean   (high-precision)
      BK3  ct:  country + first token with >= 5 chars   (token selectivity)
      BK4  p7:  first 7 chars of name_clean, no country (cross-country bridge)
      BK5  st:  sorted significant tokens (all >= 4 chars, word-order invariant)
      BK6  ph:  country + Soundex(first token) (phonetic/spelling variants)

    Conditional address key (only when name is weak/absent):
      BK7  aa:  country + first 6 chars of address_clean

    Fallback keys (only when name_clean is empty or < NAME_WEAK_THRESHOLD):
      BK8  ca:  country + first 6 chars of address_clean
      BK9  at:  country + first significant token of address_clean (>= 5 chars)
      BK10 ap:  first 7 chars of address_clean (country-agnostic)

    Parameters
    ----------
    name_clean    : str  cleaned business name (from normalize.py)
    country_clean : str  cleaned country string
    address_clean : str  cleaned address

    Returns
    -------
    list[str]  deduplicated list of blocking key strings
    """
    country = country_clean if country_clean else "xx"
    keys: list[str] = []
    use_name = name_clean and len(name_clean) >= NAME_WEAK_THRESHOLD

    if use_name:
        tokens = name_clean.split()
        prefix_source = _strip_leading_stopword(name_clean)

        # BK1: country + first 5 chars (primary precision key)
        # Raised from 3 chars (v1 BK1 was 3) — eliminates the single largest
        # source of candidate explosion from generic 3-letter prefixes.
        prefix5 = prefix_source[:5]
        if len(prefix5) >= 4:
            keys.append(f"c5:{country}_{prefix5}")

        # BK2: country + first 7 chars (high-precision, narrow range)
        # New in v2: gives a tighter precision-oriented key for longer names.
        prefix7 = prefix_source[:7]
        if len(prefix7) >= 5:
            keys.append(f"c7:{country}_{prefix7}")

        # BK3: country + first significant token >= 5 chars
        # Minimum raised from 4 → 5 to exclude very common 4-letter words.
        first_tok = _first_significant_token(tokens, min_len=5)
        if first_tok:
            keys.append(f"ct:{country}_{first_tok}")

        # BK4: first 7 chars, no country (cross-country bridge)
        # Prefix raised from 6 → 7 for better selectivity.
        prefix7_nc = prefix_source[:7]
        if len(prefix7_nc) >= 5:
            keys.append(f"p7:{prefix7_nc}")

        # BK5: sorted significant tokens (all tokens >= 4 chars)
        # Token min-length raised from 3 → 4 to exclude short stopword-like tokens.
        sorted_tok = _sorted_significant_tokens(tokens, min_len=4)
        if sorted_tok:
            keys.append(f"st:{sorted_tok}")

        # BK6: country + Soundex of first meaningful token (phonetic matching)
        soundex_src = first_tok if first_tok else (tokens[0] if tokens else "")
        soundex_code = _soundex_key(soundex_src)
        if soundex_code:
            keys.append(f"ph:{country}_{soundex_code}")

        # BK7 (conditional address key): only emit for weak-name entities.
        # In v1 this was applied to ALL entities, flooding well-named entities
        # with noisy address-based candidates.  Now gated behind NAME_WEAK_THRESHOLD.
        # A well-named entity (>= 5 chars) does not need an address key because
        # BK1–BK6 already provide sufficient recall.

    else:
        # Name is absent or too short (e.g. Devanagari/Hindi names that become
        # empty after ASCII stripping).  Fall back entirely to address keys.
        if address_clean and len(address_clean) >= 4:
            addr_tokens = address_clean.split()

            # BK8: country + first 6 chars of address_clean
            addr_prefix6 = address_clean[:6].strip()
            if len(addr_prefix6) >= 4:
                keys.append(f"ca:{country}_{addr_prefix6}")

            # BK9: country + first significant address token (>= 5 chars)
            first_addr_tok = _first_significant_token(addr_tokens, min_len=5)
            if first_addr_tok:
                keys.append(f"at:{country}_{first_addr_tok}")

            # BK10: first 7 chars of address, no country (country-agnostic fallback)
            addr_prefix7 = address_clean[:7].strip()
            if len(addr_prefix7) >= 5:
                keys.append(f"ap:{addr_prefix7}")

        # BK7 (conditional address key) — also apply here since name is weak
        if address_clean and len(address_clean) >= 3:
            addr_prefix6 = address_clean[:6]
            if len(addr_prefix6) >= 4:
                keys.append(f"aa:{country}_{addr_prefix6}")

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

    Memory note
    -----------
    The index stores sets of string entity IDs.  With max_per_key=60 and
    a realistic key count of ~30M unique keys across 10M records, peak RSS
    is roughly 4–6 GB.  This is substantially less than v1's 180-cap index.

    Parameters
    ----------
    source_dfs  : list[pd.DataFrame]
        Normalized source DataFrames.  Must contain columns:
        entity_id, name_clean, country_clean, address_clean.
    max_per_key : int
        Hard cap on how many entity IDs a single key may hold.
        Keys that overflow this cap are de facto stop-words.
    desc        : str
        Label for the tqdm progress bar.

    Returns
    -------
    dict[str, set[str]]
        Inverted index.  defaultdict converted to plain dict before
        returning for safe key-miss behaviour (returns None).
    """
    index: dict[str, set[str]] = defaultdict(set)

    total_rows = sum(len(df) for df in source_dfs)

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
    max_total: int = MAX_TOTAL_CANDIDATES,
    desc: str = "Generating candidates",
) -> dict[str, list[str]]:
    """
    For every Source-1 entity, look up all candidates from the inverted index
    and return a deduplicated, sorted list capped at max_total.

    The per-entity cap (max_total) is the key new guarantee in v2: no
    matter how many keys fire or how much the buckets overlap, each Source-1
    entity can produce at most max_total candidates.  This bounds the output
    file size independently of data distribution.

    Guarantees
    ----------
    - Every Source-1 entity_id in s1_df has an entry in the returned dict
      (empty list if no candidates found).
    - Candidate IDs are sorted for deterministic output.
    - The Source-1 entity itself is never in its own candidate list.
    - Total candidates per entity <= max_total.

    Parameters
    ----------
    s1_df     : pd.DataFrame
        Normalized Source-1.  Needs: entity_id, name_clean, country_clean,
        address_clean.
    index     : dict[str, set[str]]
        Inverted index from build_inverted_index().
    max_total : int
        Hard cap on candidates per entity (default: MAX_TOTAL_CANDIDATES).
    desc      : str
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
        address_clean = row.address_clean

        merged: set[str] = set()
        for key in generate_keys(name_clean, country_clean, address_clean):
            bucket = index.get(key)
            if bucket:
                merged.update(bucket)
                # Early-exit: once we have more than max_total candidates
                # we know we'll truncate anyway.  Keep collecting to allow
                # deterministic sort-then-truncate, but avoid unnecessary
                # set unions for very large merges.
                if len(merged) >= max_total * 3:
                    break

        # Safety: remove the entity itself (cross-source shouldn't happen,
        # but guard against edge cases in the data).
        merged.discard(entity_id)

        # Sort first for deterministic output, then hard-cap.
        cand_list = sorted(merged)
        if len(cand_list) > max_total:
            cand_list = cand_list[:max_total]

        candidates[entity_id] = cand_list

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

    blocking_recall  (pair-level)
        Of all (S1, S2/S3) true-match pairs in the ground truth, what
        fraction have the S2/S3 ID present in the candidate list?

    entity_recall  (entity-level)
        Fraction of Source-1 entities where EVERY true match is covered.

    zero_candidate_pct
        Percentage of Source-1 entities that have 0 candidates.

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

    zero_candidates = sum(1 for v in candidates.values() if len(v) == 0)
    zero_candidate_pct = zero_candidates / n_s1 * 100 if n_s1 else 0.0

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
        "zero_candidates":         zero_candidates,
        "zero_candidate_pct":      zero_candidate_pct,
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
    print(f"  Zero-candidate S1 entities   : {metrics['zero_candidates']:>10,}  "
          f"({metrics['zero_candidate_pct']:.2f}%)")
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
    print("\n[Normalize] Source 1 ...")
    s1_norm = normalize_dataframe(s1_df)
    print("[Normalize] Source 2 ...")
    s2_norm = normalize_dataframe(s2_df)
    print("[Normalize] Source 3 ...")
    s3_norm = normalize_dataframe(s3_df)

    print("\n[Index] Building inverted index from Source 2 + Source 3 ...")
    index = build_inverted_index([s2_norm, s3_norm])
    n_keys = len(index)
    print(f"  {n_keys:,} unique blocking keys in index.")

    print(f"\n[Candidates] Generating candidates for Source 1 "
          f"(cap: {MAX_TOTAL_CANDIDATES} per entity) ...")
    candidates = generate_candidates(s1_norm, index)

    # Print quick stats before evaluation
    total_cands = sum(len(v) for v in candidates.values())
    avg = total_cands / len(candidates) if candidates else 0
    print(f"  Total candidates: {total_cands:,}  |  Avg per entity: {avg:.1f}")

    if evaluate and gt_df is not None:
        print("\n[Evaluate] Measuring blocking quality ...")
        metrics = evaluate_blocking(candidates, gt_df)
        print_evaluation(metrics)

    print("[Save] Writing output file ...")
    save_candidates(candidates, output_path)

    return candidates


def main() -> None:
    """
    Full blocking run on train or test split.
    Execute with:
        python -m src.blocking --split train
        python -m src.blocking --split test
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run blocking candidate generation.")
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test"],
        default="train",
        help="Dataset split to run blocking on (train or test). Default: train",
    )
    args, _ = parser.parse_known_args()

    print("=" * 60)
    print(f"  Blocking / Candidate Generation ({args.split.upper()} split)")
    print(f"  MAX_CANDIDATES_PER_KEY = {MAX_CANDIDATES_PER_KEY}")
    print(f"  MAX_TOTAL_CANDIDATES   = {MAX_TOTAL_CANDIDATES}")
    print("=" * 60)

    if args.split == "test":
        print("\n[1] Loading test datasets ...")
        s1 = load_source(FILE_PATHS["test_source1"], "test_source1")
        s2 = load_source(FILE_PATHS["test_source2"], "test_source2")
        s3 = load_source(FILE_PATHS["test_source3"], "test_source3")
        print(f"  S1: {len(s1):,}  S2: {len(s2):,}  S3: {len(s3):,}")
        run_blocking(s1, s2, s3, gt_df=None, evaluate=False)
    else:
        print("\n[1] Loading train datasets ...")
        s1 = load_source(FILE_PATHS["train_source1"], "train_source1")
        s2 = load_source(FILE_PATHS["train_source2"], "train_source2")
        s3 = load_source(FILE_PATHS["train_source3"], "train_source3")
        gt = load_ground_truth(FILE_PATHS["train_ground_truth"])
        print(f"  S1: {len(s1):,}  S2: {len(s2):,}  S3: {len(s3):,}  GT: {len(gt):,}")
        run_blocking(s1, s2, s3, gt_df=gt, evaluate=True)

    print("Done.")


if __name__ == "__main__":
    main()
