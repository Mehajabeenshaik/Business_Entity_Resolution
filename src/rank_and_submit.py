"""
rank_and_submit.py
==================
Fast candidate re-ranking and submission file generation.

Approach:
1. Load entity names from source files into memory (entity_id -> name string)
2. Stream candidate_pairs.tsv line-by-line (never loads full 5 GB into RAM)
3. For each S1 entity, score all candidates with rapidfuzz ratio
4. Keep top-K candidates (those above a similarity threshold)
5. Write matching_results.tsv (submission) and reduced candidate_pairs.tsv

Usage:
    python src/rank_and_submit.py --mode train   # rank training candidates
    python src/rank_and_submit.py --mode test    # generate test submission
    python src/rank_and_submit.py --mode both    # do both
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict

from rapidfuzz import fuzz
from tqdm import tqdm

# Force UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.load_data import FILE_PATHS
from src.normalize import clean_business_name, clean_address

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TOP_K = 5                   # Keep top K candidates per S1 entity
MATCH_THRESHOLD = 50.0      # Minimum fuzz ratio to include in final matches
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")

# ---------------------------------------------------------------------------
# Name loading (memory-efficient: only entity_id -> cleaned name + address)
# ---------------------------------------------------------------------------

def load_names(path: str, desc: str = "") -> dict[str, tuple[str, str, str]]:
    """
    Load entity_id -> (name_clean, address_clean, country_clean) from a source TSV.
    Streams file, never loads full DF.
    """
    names: dict[str, tuple[str, str, str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in tqdm(reader, desc=f"Loading {desc}", ncols=90):
            eid = row["entity_id"]
            name = clean_business_name(row.get("business_name", ""))
            addr = clean_address(row.get("business_address", ""))
            country = row.get("country", "").lower().strip()
            names[eid] = (name, addr, country)
    print(f"  {desc}: {len(names):,} entities loaded.")
    return names


def score_candidate(
    s1_name: str, s1_addr: str,
    c_name: str, c_addr: str,
) -> float:
    """
    Fast composite similarity score between two entities.
    Uses rapidfuzz ratio (0-100) on names, with address as tiebreaker.
    """
    # Primary: name similarity
    if s1_name and c_name:
        name_score = fuzz.ratio(s1_name, c_name)
    elif not s1_name and not c_name:
        name_score = 0.0
    else:
        name_score = 0.0

    # Secondary: address similarity (weighted lower)
    if s1_addr and c_addr:
        addr_score = fuzz.ratio(s1_addr, c_addr)
    else:
        addr_score = 0.0

    # Composite: 70% name + 30% address
    return 0.7 * name_score + 0.3 * addr_score


# ---------------------------------------------------------------------------
# Streaming re-ranker
# ---------------------------------------------------------------------------

def rank_candidates(
    candidate_path: str,
    s1_names: dict[str, tuple[str, str, str]],
    s23_names: dict[str, tuple[str, str, str]],
    matching_out: str,
    reduced_cand_out: str,
    top_k: int = TOP_K,
    threshold: float = MATCH_THRESHOLD,
) -> None:
    """
    Stream candidate_pairs.tsv, re-rank each entity's candidates by
    rapidfuzz similarity, keep top-K, and write outputs.
    """
    os.makedirs(os.path.dirname(os.path.abspath(matching_out)), exist_ok=True)

    # Count lines for progress bar
    print(f"  Counting rows in {candidate_path} ...")
    with open(candidate_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f) - 1  # minus header
    print(f"  {total_lines:,} rows to process.")

    match_fh = open(matching_out, "w", newline="", encoding="utf-8")
    cand_fh = open(reduced_cand_out, "w", newline="", encoding="utf-8")

    match_writer = csv.writer(match_fh, delimiter="\t")
    cand_writer = csv.writer(cand_fh, delimiter="\t")

    match_writer.writerow(["source1_entity_id", "matched_entity_ids"])
    cand_writer.writerow(["source1_entity_id", "candidate_entity_ids"])

    total_matches = 0
    zero_matches = 0

    with open(candidate_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header

        for row in tqdm(reader, total=total_lines, desc="Ranking", ncols=90):
            s1_id = row[0]
            raw_cands = row[1] if len(row) > 1 else ""

            if not raw_cands.strip():
                match_writer.writerow([s1_id, ""])
                cand_writer.writerow([s1_id, ""])
                zero_matches += 1
                continue

            cand_ids = raw_cands.split(",")

            # Get S1 info
            s1_info = s1_names.get(s1_id, ("", "", ""))
            s1_name, s1_addr, s1_country = s1_info

            # Score each candidate
            scored = []
            for cid in cand_ids:
                c_info = s23_names.get(cid, ("", "", ""))
                c_name, c_addr, c_country = c_info
                score = score_candidate(s1_name, s1_addr, c_name, c_addr)
                scored.append((score, cid))

            # Sort by score descending, keep top-K
            scored.sort(reverse=True)
            top_candidates = scored[:top_k]

            # Filter by threshold for final matches
            matched_ids = [cid for sc, cid in top_candidates if sc >= threshold]
            reduced_ids = [cid for _, cid in top_candidates]

            if matched_ids:
                total_matches += len(matched_ids)
            else:
                zero_matches += 1

            match_writer.writerow([s1_id, ",".join(matched_ids)])
            cand_writer.writerow([s1_id, ",".join(reduced_ids)])

    match_fh.close()
    cand_fh.close()

    print(f"\n  Results:")
    print(f"    Total S1 entities     : {total_lines:,}")
    print(f"    Total matches written : {total_matches:,}")
    print(f"    Zero-match entities   : {zero_matches:,}")
    print(f"    matching_results.tsv  : {matching_out}")
    print(f"    candidate_pairs.tsv   : {reduced_cand_out}")


# ---------------------------------------------------------------------------
# Full blocking + ranking for test set (no pre-existing candidate_pairs)
# ---------------------------------------------------------------------------

def run_test_blocking_and_ranking(
    s1_names: dict[str, tuple[str, str, str]],
    s23_names: dict[str, tuple[str, str, str]],
    matching_out: str,
    cand_out: str,
    top_k: int = TOP_K,
    threshold: float = MATCH_THRESHOLD,
) -> None:
    """
    For the test set: run blocking + ranking in one pass.
    Generates candidate_pairs.tsv and matching_results.tsv.
    """
    from src.blocking import generate_keys, MAX_CANDIDATES_PER_KEY

    print("\n[1] Building inverted index from test S2+S3 ...")
    index: dict[str, set[str]] = defaultdict(set)
    for eid, (name, addr, country) in tqdm(s23_names.items(), desc="Indexing S2+S3", ncols=90):
        for key in generate_keys(name, country, addr):
            bucket = index[key]
            if len(bucket) < MAX_CANDIDATES_PER_KEY:
                bucket.add(eid)
    index = dict(index)
    print(f"  {len(index):,} unique blocking keys.")

    print("\n[2] Generating candidates + ranking for test S1 ...")
    os.makedirs(os.path.dirname(os.path.abspath(matching_out)), exist_ok=True)

    match_fh = open(matching_out, "w", newline="", encoding="utf-8")
    cand_fh = open(cand_out, "w", newline="", encoding="utf-8")

    match_writer = csv.writer(match_fh, delimiter="\t")
    cand_writer = csv.writer(cand_fh, delimiter="\t")

    match_writer.writerow(["source1_entity_id", "matched_entity_ids"])
    cand_writer.writerow(["source1_entity_id", "candidate_entity_ids"])

    total_matches = 0
    zero_matches = 0

    for s1_id, (s1_name, s1_addr, s1_country) in tqdm(
        s1_names.items(), desc="Blocking+Ranking", ncols=90
    ):
        # Generate blocking keys for this S1 entity
        merged: set[str] = set()
        for key in generate_keys(s1_name, s1_country, s1_addr):
            bucket = index.get(key)
            if bucket:
                merged.update(bucket)
        merged.discard(s1_id)

        if not merged:
            match_writer.writerow([s1_id, ""])
            cand_writer.writerow([s1_id, ""])
            zero_matches += 1
            continue

        # Score and rank
        scored = []
        for cid in merged:
            c_info = s23_names.get(cid, ("", "", ""))
            c_name, c_addr, c_country = c_info
            score = score_candidate(s1_name, s1_addr, c_name, c_addr)
            scored.append((score, cid))

        scored.sort(reverse=True)
        top_candidates = scored[:top_k]

        matched_ids = [cid for sc, cid in top_candidates if sc >= threshold]
        reduced_ids = [cid for _, cid in top_candidates]

        if matched_ids:
            total_matches += len(matched_ids)
        else:
            zero_matches += 1

        match_writer.writerow([s1_id, ",".join(matched_ids)])
        cand_writer.writerow([s1_id, ",".join(reduced_ids)])

    match_fh.close()
    cand_fh.close()

    print(f"\n  Results:")
    print(f"    Total S1 entities     : {len(s1_names):,}")
    print(f"    Total matches written : {total_matches:,}")
    print(f"    Zero-match entities   : {zero_matches:,}")
    print(f"    matching_results.tsv  : {matching_out}")
    print(f"    candidate_pairs.tsv   : {cand_out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Rank candidates and generate submission files")
    parser.add_argument("--mode", choices=["train", "test", "both"], default="test",
                        help="train = re-rank existing train candidate_pairs.tsv; "
                             "test = full blocking+ranking on test set; "
                             "both = do both")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Keep top K candidates")
    parser.add_argument("--threshold", type=float, default=MATCH_THRESHOLD,
                        help="Min fuzz ratio for final match")
    args = parser.parse_args()

    start = time.time()

    if args.mode in ("train", "both"):
        print("=" * 60)
        print("  TRAIN MODE: Re-ranking existing candidate_pairs.tsv")
        print("=" * 60)

        # Load train source names
        print("\n[1] Loading entity names ...")
        s1_names = load_names(FILE_PATHS["train_source1"], "train_S1")
        s2_names = load_names(FILE_PATHS["train_source2"], "train_S2")
        s3_names = load_names(FILE_PATHS["train_source3"], "train_S3")
        s23_names = {**s2_names, **s3_names}
        del s2_names, s3_names

        train_cand = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
        train_match = os.path.join(OUTPUT_DIR, "train_matching_results.tsv")
        train_reduced = os.path.join(OUTPUT_DIR, "train_candidate_pairs_reduced.tsv")

        print("\n[2] Ranking candidates ...")
        rank_candidates(
            train_cand, s1_names, s23_names,
            matching_out=train_match,
            reduced_cand_out=train_reduced,
            top_k=args.top_k,
            threshold=args.threshold,
        )
        del s1_names, s23_names

    if args.mode in ("test", "both"):
        print("\n" + "=" * 60)
        print("  TEST MODE: Blocking + Ranking on test data")
        print("=" * 60)

        # Load test source names
        print("\n[1] Loading test entity names ...")
        s1_names = load_names(FILE_PATHS["test_source1"], "test_S1")
        s2_names = load_names(FILE_PATHS["test_source2"], "test_S2")
        s3_names = load_names(FILE_PATHS["test_source3"], "test_S3")
        s23_names = {**s2_names, **s3_names}
        del s2_names, s3_names

        test_match = os.path.join(OUTPUT_DIR, "matching_results.tsv")
        test_cand = os.path.join(OUTPUT_DIR, "test_candidate_pairs.tsv")

        run_test_blocking_and_ranking(
            s1_names, s23_names,
            matching_out=test_match,
            cand_out=test_cand,
            top_k=args.top_k,
            threshold=args.threshold,
        )

    elapsed = time.time() - start
    print(f"\n  Total time: {elapsed/60:.1f} minutes")
    print("Done.")


if __name__ == "__main__":
    main()
