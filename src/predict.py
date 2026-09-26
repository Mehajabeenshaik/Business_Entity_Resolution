"""
Score test candidates and write output/matching_results.tsv
"""

from __future__ import annotations
import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.load_data import FILE_PATHS, load_source
from src.normalize import normalize_dataframe
from src.features import (
    build_feature_matrix, load_candidate_lookup, load_candidates_tsv,
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
MODEL_DIR = ROOT / "artifacts" / "model"
CAND_PATH = OUTPUT_DIR / "candidate_pairs.tsv"   # must be the TEST candidates
MATCH_PATH = OUTPUT_DIR / "matching_results.tsv"


def main():
    meta = json.loads((MODEL_DIR / "model_config.json").read_text())
    thr = meta["threshold"]

    model = CatBoostClassifier()
    model.load_model(str(MODEL_DIR / "matching_model.cbm"))

    print("Loading test sources …")
    s1 = normalize_dataframe(load_source(FILE_PATHS["test_source1"], "test_s1"))
    s2 = normalize_dataframe(load_source(FILE_PATHS["test_source2"], "test_s2"))
    s3 = normalize_dataframe(load_source(FILE_PATHS["test_source3"], "test_s3"))

    print("Loading test candidates …")
    candidates = load_candidates_tsv(str(CAND_PATH))
    lookup = load_candidate_lookup(s2, s3)

    # ensure every S1 appears even if it has zero candidates
    for eid in s1["entity_id"]:
        candidates.setdefault(eid, [])

    pairs_df, X = build_feature_matrix(s1, lookup, candidates, desc="Test features")
    proba = model.predict_proba(X)[:, 1] if len(X) else np.array([])

    matches = {eid: [] for eid in s1["entity_id"]}
    for i, row in enumerate(pairs_df.itertuples(index=False)):
        if proba[i] >= thr:
            matches[row.source1_entity_id].append(row.candidate_entity_id)

    # write exactly in the order of test_source1
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(MATCH_PATH, "w", encoding="utf-8", newline="") as fh:
        fh.write("source1_entity_id\tmatched_entity_ids\n")
        for eid in s1["entity_id"]:
            ids = sorted(set(matches.get(eid, [])))
            fh.write(f"{eid}\t{','.join(ids)}\n")

    n_with = sum(1 for v in matches.values() if v)
    n_links = sum(len(v) for v in matches.values())
    print(f"Wrote {MATCH_PATH}")
    print(f"  Entities with ≥1 match : {n_with:,}")
    print(f"  Total predicted links  : {n_links:,}")


if __name__ == "__main__":
    main()
