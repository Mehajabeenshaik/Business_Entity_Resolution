"""
Strict validator for the two required submission files.
Prints VALIDATION PASSED only when every check succeeds.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.load_data import FILE_PATHS, load_source

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
MATCH_PATH = OUTPUT_DIR / "matching_results.tsv"
CAND_PATH  = OUTPUT_DIR / "candidate_pairs.tsv"


def fail(msg: str):
    print(f"VALIDATION FAILED: {msg}")
    sys.exit(1)


def main():
    # 1. files exist
    if not MATCH_PATH.exists():
        fail(f"{MATCH_PATH} does not exist")
    if not CAND_PATH.exists():
        fail(f"{CAND_PATH} does not exist")

    # 2. load test sources
    s1 = load_source(FILE_PATHS["test_source1"], "test_s1")
    s2 = load_source(FILE_PATHS["test_source2"], "test_s2")
    s3 = load_source(FILE_PATHS["test_source3"], "test_s3")
    valid_s1 = set(s1["entity_id"])
    valid_cand = set(s2["entity_id"]) | set(s3["entity_id"])

    # 3. load outputs
    match = pd.read_csv(MATCH_PATH, sep="\t", dtype=str, keep_default_na=False)
    cand  = pd.read_csv(CAND_PATH,  sep="\t", dtype=str, keep_default_na=False)

    # 4. column checks
    if list(match.columns) != ["source1_entity_id", "matched_entity_ids"]:
        fail(f"matching_results columns wrong: {list(match.columns)}")
    if list(cand.columns) != ["source1_entity_id", "candidate_entity_ids"]:
        fail(f"candidate_pairs columns wrong: {list(cand.columns)}")

    # 5. row counts & uniqueness
    if len(match) != len(valid_s1):
        fail(f"matching_results has {len(match)} rows, expected {len(valid_s1)}")
    if len(cand) != len(valid_s1):
        fail(f"candidate_pairs has {len(cand)} rows, expected {len(valid_s1)}")

    if match["source1_entity_id"].duplicated().any():
        fail("duplicate source1_entity_id in matching_results")
    if cand["source1_entity_id"].duplicated().any():
        fail("duplicate source1_entity_id in candidate_pairs")

    match_ids = set(match["source1_entity_id"])
    cand_ids  = set(cand["source1_entity_id"])
    if match_ids != valid_s1:
        fail("matching_results Source-1 IDs do not match test_source1")
    if cand_ids != valid_s1:
        fail("candidate_pairs Source-1 IDs do not match test_source1")

    # 6. build candidate lookup
    cand_map = {}
    for _, row in cand.iterrows():
        ids = [x.strip() for x in row["candidate_entity_ids"].split(",") if x.strip()]
        if len(ids) != len(set(ids)):
            fail(f"duplicate candidate IDs for {row['source1_entity_id']}")
        for cid in ids:
            if cid not in valid_cand:
                fail(f"candidate {cid} is not a valid test S2/S3 ID")
            if cid.startswith("S1-"):
                fail(f"candidate {cid} starts with S1-")
        cand_map[row["source1_entity_id"]] = set(ids)

    # 7. check predictions
    n_links = 0
    n_zero = n_one = n_multi = 0
    for _, row in match.iterrows():
        s1_id = row["source1_entity_id"]
        raw = row["matched_entity_ids"]
        # blank must be truly blank
        if raw in ("None", "null", "NaN", "[]", '""', "nan"):
            fail(f"illegal blank value for {s1_id}: {raw!r}")
        ids = [x.strip() for x in raw.split(",") if x.strip()]
        if len(ids) != len(set(ids)):
            fail(f"duplicate matched IDs for {s1_id}")
        for mid in ids:
            if mid not in valid_cand:
                fail(f"predicted {mid} is not a valid test S2/S3 ID")
            if mid.startswith("S1-"):
                fail(f"predicted {mid} starts with S1-")
            if mid not in cand_map.get(s1_id, set()):
                fail(f"predicted {mid} for {s1_id} was not in its candidate list")
        n_links += len(ids)
        if len(ids) == 0:
            n_zero += 1
        elif len(ids) == 1:
            n_one += 1
        else:
            n_multi += 1

    print("VALIDATION PASSED")
    print(f"  Source-1 test entities     : {len(valid_s1):,}")
    print(f"  Predicted links            : {n_links:,}")
    print(f"  Entities with 0 matches    : {n_zero:,}")
    print(f"  Entities with 1 match      : {n_one:,}")
    print(f"  Entities with ≥2 matches   : {n_multi:,}")
    print(f"  Avg matches per S1 entity  : {n_links / len(valid_s1):.3f}")


if __name__ == "__main__":
    main()
