"""
Full evaluation on the training ground-truth using the saved CatBoost model
and selected threshold. Also reports blocking-recall ceiling.
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
from src.load_data import FILE_PATHS, load_source, load_ground_truth
from src.normalize import normalize_dataframe
from src.features import (
    FEATURE_NAMES, build_feature_matrix, load_candidate_lookup, load_candidates_tsv,
)
from src.train import macro_f05   # reuse

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
MODEL_DIR = ROOT / "artifacts" / "model"
VAL_DIR = ROOT / "artifacts" / "validation"
CAND_PATH = OUTPUT_DIR / "candidate_pairs.tsv"


def main():
    meta = json.loads((MODEL_DIR / "model_config.json").read_text())
    thr = meta["threshold"]

    model = CatBoostClassifier()
    model.load_model(str(MODEL_DIR / "matching_model.cbm"))

    s1 = normalize_dataframe(load_source(FILE_PATHS["train_source1"], "s1"))
    s2 = normalize_dataframe(load_source(FILE_PATHS["train_source2"], "s2"))
    s3 = normalize_dataframe(load_source(FILE_PATHS["train_source3"], "s3"))
    gt_df = load_ground_truth(FILE_PATHS["train_ground_truth"])

    candidates = load_candidates_tsv(str(CAND_PATH))
    lookup = load_candidate_lookup(s2, s3)

    pairs_df, X = build_feature_matrix(s1, lookup, candidates, desc="Eval features")
    proba = model.predict_proba(X)[:, 1]

    pred_dict = {}
    for i, row in enumerate(pairs_df.itertuples(index=False)):
        if proba[i] >= thr:
            pred_dict.setdefault(row.source1_entity_id, set()).add(row.candidate_entity_id)

    gt_dict = {}
    for _, row in gt_df.iterrows():
        matched = {m.strip() for m in row["matched_entity_ids"].split(",") if m.strip()}
        gt_dict[row["source1_entity_id"]] = matched

    # blocking recall ceiling
    hits = total = 0
    for s1_id, true in gt_dict.items():
        cands = set(candidates.get(s1_id, []))
        hits += len(true & cands)
        total += len(true)
    blocking_recall = hits / total if total else 0.0

    metrics = macro_f05(pred_dict, gt_dict)
    metrics["blocking_recall_ceiling"] = blocking_recall
    metrics["threshold"] = thr

    print(json.dumps(metrics, indent=2))
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    (VAL_DIR / "full_eval_metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
