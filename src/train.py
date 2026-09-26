"""
Train CatBoost (primary) + optional HistGradientBoosting baseline.
Validation is grouped by source1_entity_id (no leakage).
Threshold is selected on validation macro-F0.5.
"""

from __future__ import annotations
import os
import sys
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier, Pool
from tqdm import tqdm
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.load_data import FILE_PATHS, load_source, load_ground_truth
from src.normalize import normalize_dataframe
from src.features import (
    FEATURE_NAMES, build_feature_matrix, load_candidate_lookup,
    load_candidates_tsv, save_feature_columns,
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
ARTIFACTS = ROOT / "artifacts"
MODEL_DIR = ARTIFACTS / "model"
VAL_DIR = ARTIFACTS / "validation"

CAND_PATH = OUTPUT_DIR / "candidate_pairs.tsv"   # produced by blocking on TRAIN
SEED = 42
VAL_SIZE = 0.20
NEG_RATIO = 5          # negatives kept per positive (candidate-based)


def build_labels(candidates, gt_df):
    gt = {}
    for _, row in gt_df.iterrows():
        s1 = row["source1_entity_id"]
        matched = {m.strip() for m in row["matched_entity_ids"].split(",") if m.strip()}
        gt[s1] = matched

    labels = {}
    for s1, cands in candidates.items():
        true = gt.get(s1, set())
        for c in cands:
            labels[(s1, c)] = 1 if c in true else 0
    return labels, gt


def sample_pairs(labels, neg_ratio=NEG_RATIO, seed=SEED):
    rng = np.random.default_rng(seed)
    by_s1 = {}
    for (s1, c), y in labels.items():
        by_s1.setdefault(s1, []).append((c, y))

    selected = []
    for s1, pairs in by_s1.items():
        pos = [(c, 1) for c, y in pairs if y == 1]
        neg = [(c, 0) for c, y in pairs if y == 0]
        if not pos:
            continue
        n_neg = min(len(neg), len(pos) * neg_ratio)
        if n_neg:
            idx = rng.choice(len(neg), size=n_neg, replace=False)
            neg = [neg[i] for i in idx]
        else:
            neg = []
        for c, y in pos + neg:
            selected.append((s1, c, y))
    return selected


def macro_f05(pred_dict, gt_dict):
    fs, ps, rs = [], [], []
    for s1, true in gt_dict.items():
        pred = pred_dict.get(s1, set())
        tp = len(true & pred)
        fp = len(pred - true)
        fn = len(true - pred)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = (1.25 * p * r) / (0.25 * p + r + 1e-12) if (p + r) else 0.0
        fs.append(f); ps.append(p); rs.append(r)
    return {
        "macro_f05": float(np.mean(fs)),
        "macro_precision": float(np.mean(ps)),
        "macro_recall": float(np.mean(rs)),
        "n_entities": len(gt_dict),
    }


def main():
    print("=== Loading train data ===")
    s1 = normalize_dataframe(load_source(FILE_PATHS["train_source1"], "s1"))
    s2 = normalize_dataframe(load_source(FILE_PATHS["train_source2"], "s2"))
    s3 = normalize_dataframe(load_source(FILE_PATHS["train_source3"], "s3"))
    gt_df = load_ground_truth(FILE_PATHS["train_ground_truth"])

    print("=== Loading candidates ===")
    candidates = load_candidates_tsv(str(CAND_PATH))
    lookup = load_candidate_lookup(s2, s3)

    print("=== Building labels ===")
    labels, gt_dict = build_labels(candidates, gt_df)
    train_pairs = sample_pairs(labels)
    print(f"  Sampled {sum(y for *_, y in train_pairs):,} pos / "
          f"{sum(1-y for *_, y in train_pairs):,} neg")

    # restrict to sampled pairs
    sampled = {}
    for s1_id, c_id, _ in train_pairs:
        sampled.setdefault(s1_id, []).append(c_id)

    pairs_df, X = build_feature_matrix(s1, lookup, sampled, desc="Train features")
    y = np.array([labels[(r.source1_entity_id, r.candidate_entity_id)]
                  for r in pairs_df.itertuples(index=False)], dtype=np.int8)
    groups = pairs_df["source1_entity_id"].values

    # Grouped split
    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=SEED)
    train_idx, val_idx = next(gss.split(X, y, groups))
    X_tr, X_va = X[train_idx], X[val_idx]
    y_tr, y_va = y[train_idx], y[val_idx]
    groups_va = groups[val_idx]
    pairs_va = pairs_df.iloc[val_idx].reset_index(drop=True)

    print(f"  Train pairs: {len(y_tr):,}  Val pairs: {len(y_va):,}")
    print(f"  Unique S1 train: {len(set(groups[train_idx])):,}  "
          f"Val: {len(set(groups_va)):,}")

    # ---------- CatBoost (primary) ----------
    print("=== Training CatBoost ===")
    train_pool = Pool(X_tr, y_tr, feature_names=FEATURE_NAMES)
    val_pool   = Pool(X_va, y_va, feature_names=FEATURE_NAMES)

    cat_model = CatBoostClassifier(
        iterations=3000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=SEED,
        early_stopping_rounds=80,
        verbose=100,
        thread_count=-1,
    )
    cat_model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    # ---------- optional baseline ----------
    print("=== Training HistGBM baseline ===")
    hgb = HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.05, max_depth=8,
        early_stopping=True, random_state=SEED,
    )
    hgb.fit(X_tr, y_tr)

    # ---------- threshold search on validation ----------
    print("=== Threshold search (CatBoost) ===")
    va_proba = cat_model.predict_proba(X_va)[:, 1]

    thresholds = np.round(np.arange(0.50, 1.00, 0.01), 2)
    results = []
    best_f, best_thr = -1.0, 0.5

    for thr in thresholds:
        pred_dict = {}
        for i, row in enumerate(pairs_va.itertuples(index=False)):
            if va_proba[i] >= thr:
                pred_dict.setdefault(row.source1_entity_id, set()).add(row.candidate_entity_id)

        # restrict gt to validation entities only
        va_entities = set(pairs_va["source1_entity_id"])
        gt_va = {k: v for k, v in gt_dict.items() if k in va_entities}
        metrics = macro_f05(pred_dict, gt_va)
        metrics["threshold"] = float(thr)
        results.append(metrics)
        if metrics["macro_f05"] > best_f or (
            abs(metrics["macro_f05"] - best_f) < 1e-6 and thr > best_thr
        ):
            best_f, best_thr = metrics["macro_f05"], thr

    print(f"Best validation macro-F0.5 = {best_f:.4f} @ threshold {best_thr:.2f}")

    # ---------- save everything ----------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    save_feature_columns()

    cat_model.save_model(str(MODEL_DIR / "matching_model.cbm"))
    joblib.dump(hgb, MODEL_DIR / "baseline_hgb.joblib")

    meta = {
        "model": "CatBoostClassifier",
        "threshold": float(best_thr),
        "val_macro_f05": float(best_f),
        "feature_names": FEATURE_NAMES,
        "n_train_pairs": int(len(y_tr)),
        "n_val_pairs": int(len(y_va)),
        "pos_rate_train": float(y_tr.mean()),
        "best_iteration": int(cat_model.get_best_iteration() or 0),
        "seed": SEED,
    }
    (MODEL_DIR / "model_config.json").write_text(json.dumps(meta, indent=2))
    (MODEL_DIR / "selected_threshold.json").write_text(
        json.dumps({"threshold": float(best_thr)}, indent=2)
    )

    pd.DataFrame(results).to_csv(VAL_DIR / "threshold_results.csv", index=False)
    (VAL_DIR / "validation_summary.json").write_text(json.dumps(meta, indent=2))

    # feature importance
    imp = cat_model.get_feature_importance()
    imp_df = pd.DataFrame({"feature": FEATURE_NAMES, "importance": imp})
    imp_df = imp_df.sort_values("importance", ascending=False)
    imp_df.to_csv(VAL_DIR / "feature_importance.csv", index=False)
    print("Top-10 features:\n", imp_df.head(10).to_string(index=False))

    print(f"\nModel saved → {MODEL_DIR}")
    print(f"Threshold   → {best_thr:.2f}")
    print(f"Val F0.5    → {best_f:.4f}")


if __name__ == "__main__":
    main()
