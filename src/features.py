"""
Pairwise feature engineering for (Source-1, candidate) pairs.
All similarities use rapidfuzz. Empty strings are handled safely.
Feature order is fixed and saved so train / val / test stay identical.
"""

from __future__ import annotations
import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.load_data import FILE_PATHS, load_source
from src.normalize import normalize_dataframe

# ----------------------------------------------------------------------
# Fixed feature column order – never change after training
# ----------------------------------------------------------------------
FEATURE_NAMES: List[str] = [
    # ---- name ----
    "name_exact",
    "name_jw",
    "name_lev",
    "name_ratio",
    "name_partial",
    "name_token_sort",
    "name_token_set",
    "name_jaccard",
    "name_containment",
    "name_shared_tokens",
    "name_first_token_match",
    "name_last_token_match",
    "name_prefix4_match",
    "name_len_diff",
    "name_len_ratio",
    "name_digit_jaccard",
    # ---- address ----
    "addr_exact",
    "addr_jw",
    "addr_lev",
    "addr_ratio",
    "addr_token_sort",
    "addr_token_set",
    "addr_jaccard",
    "addr_containment",
    "addr_shared_tokens",
    "addr_len_diff",
    "addr_len_ratio",
    "addr_digit_jaccard",
    "addr_housenum_match",
    "addr_pin_match",
    # ---- country ----
    "country_exact",
    "country_conflict",
    "country_both_present",
    "country_s1_missing",
    "country_cand_missing",
    # ---- source / blocking context ----
    "cand_is_s2",
    "cand_is_s3",
    "n_candidates_for_s1",
    # ---- combined ----
    "name_x_addr",
    "name_plus_addr",
    "name_min_addr",
    "name_max_addr",
    "same_country_high_name",
    "same_country_high_addr",
    "exact_name_exact_country",
    "exact_addr_exact_country",
    "high_name_and_high_addr",
]

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
FEATURE_COLS_PATH = ARTIFACTS_DIR / "features" / "feature_columns.json"


def _safe(s: str) -> str:
    return s if isinstance(s, str) else ""


def _tokens(s: str) -> List[str]:
    return [t for t in _safe(s).split() if t]


def _jaccard(a: str, b: str) -> float:
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _containment(a: str, b: str) -> float:
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not ta:
        return 1.0 if not tb else 0.0
    return len(ta & tb) / len(ta)


def _digit_tokens(s: str) -> set:
    return set(re.findall(r"\d+", _safe(s)))


def _digit_jaccard(a: str, b: str) -> float:
    da, db = _digit_tokens(a), _digit_tokens(b)
    if not da and not db:
        return 1.0
    if not da or not db:
        return 0.0
    return len(da & db) / len(da | db)


def _first_token(s: str) -> str:
    t = _tokens(s)
    return t[0] if t else ""


def _last_token(s: str) -> str:
    t = _tokens(s)
    return t[-1] if t else ""


def _extract_housenum(addr: str) -> str:
    m = re.search(r"\b(\d+[a-zA-Z]?)\b", _safe(addr))
    return m.group(1).lower() if m else ""


def _extract_pin(addr: str) -> str:
    # crude but useful: 5–6 digit sequences common in US/India/France
    m = re.search(r"\b(\d{5,6})\b", _safe(addr))
    return m.group(1) if m else ""


def pair_features(
    s1_name: str, s1_addr: str, s1_country: str,
    c_name: str, c_addr: str, c_country: str,
    n_cands: int = 0,
    cand_id: str = "",
) -> List[float]:
    """Return one row of features in FEATURE_NAMES order."""
    s1_name, s1_addr, s1_country = _safe(s1_name), _safe(s1_addr), _safe(s1_country)
    c_name, c_addr, c_country = _safe(c_name), _safe(c_addr), _safe(c_country)

    # name
    name_exact = 1.0 if s1_name and s1_name == c_name else 0.0
    name_jw = fuzz.WRatio(s1_name, c_name) / 100.0
    name_lev = 1.0 - distance.Levenshtein.normalized_distance(s1_name, c_name)
    name_ratio = fuzz.ratio(s1_name, c_name) / 100.0
    name_partial = fuzz.partial_ratio(s1_name, c_name) / 100.0
    name_tsort = fuzz.token_sort_ratio(s1_name, c_name) / 100.0
    name_tset = fuzz.token_set_ratio(s1_name, c_name) / 100.0
    name_jac = _jaccard(s1_name, c_name)
    name_cont = _containment(s1_name, c_name)
    name_share = float(len(set(_tokens(s1_name)) & set(_tokens(c_name))))
    name_first = 1.0 if _first_token(s1_name) and _first_token(s1_name) == _first_token(c_name) else 0.0
    name_last = 1.0 if _last_token(s1_name) and _last_token(s1_name) == _last_token(c_name) else 0.0
    name_pref = 1.0 if s1_name[:4] and s1_name[:4] == c_name[:4] else 0.0
    name_ldiff = float(abs(len(s1_name) - len(c_name)))
    name_lratio = min(len(s1_name), len(c_name)) / max(len(s1_name), len(c_name), 1)
    name_dj = _digit_jaccard(s1_name, c_name)

    # address
    addr_exact = 1.0 if s1_addr and s1_addr == c_addr else 0.0
    addr_jw = fuzz.WRatio(s1_addr, c_addr) / 100.0
    addr_lev = 1.0 - distance.Levenshtein.normalized_distance(s1_addr, c_addr)
    addr_ratio = fuzz.ratio(s1_addr, c_addr) / 100.0
    addr_tsort = fuzz.token_sort_ratio(s1_addr, c_addr) / 100.0
    addr_tset = fuzz.token_set_ratio(s1_addr, c_addr) / 100.0
    addr_jac = _jaccard(s1_addr, c_addr)
    addr_cont = _containment(s1_addr, c_addr)
    addr_share = float(len(set(_tokens(s1_addr)) & set(_tokens(c_addr))))
    addr_ldiff = float(abs(len(s1_addr) - len(c_addr)))
    addr_lratio = min(len(s1_addr), len(c_addr)) / max(len(s1_addr), len(c_addr), 1)
    addr_dj = _digit_jaccard(s1_addr, c_addr)
    addr_house = 1.0 if _extract_housenum(s1_addr) and _extract_housenum(s1_addr) == _extract_housenum(c_addr) else 0.0
    addr_pin = 1.0 if _extract_pin(s1_addr) and _extract_pin(s1_addr) == _extract_pin(c_addr) else 0.0

    # country
    country_exact = 1.0 if s1_country and s1_country == c_country else 0.0
    country_conflict = 1.0 if (s1_country and c_country and s1_country != c_country) else 0.0
    country_both = 1.0 if (s1_country and c_country) else 0.0
    country_s1_miss = 1.0 if not s1_country else 0.0
    country_c_miss = 1.0 if not c_country else 0.0

    # source
    cand_is_s2 = 1.0 if cand_id.startswith("S2-") else 0.0
    cand_is_s3 = 1.0 if cand_id.startswith("S3-") else 0.0
    n_cands_f = float(n_cands)

    # combined
    name_x_addr = name_jw * addr_jw
    name_plus_addr = name_jw + addr_jw
    name_min_addr = min(name_jw, addr_jw)
    name_max_addr = max(name_jw, addr_jw)
    same_ctry_high_name = 1.0 if country_exact and name_jw >= 0.85 else 0.0
    same_ctry_high_addr = 1.0 if country_exact and addr_jw >= 0.85 else 0.0
    exact_name_ctry = 1.0 if name_exact and country_exact else 0.0
    exact_addr_ctry = 1.0 if addr_exact and country_exact else 0.0
    high_both = 1.0 if name_jw >= 0.85 and addr_jw >= 0.85 else 0.0

    return [
        name_exact, name_jw, name_lev, name_ratio, name_partial,
        name_tsort, name_tset, name_jac, name_cont, name_share,
        name_first, name_last, name_pref, name_ldiff, name_lratio, name_dj,
        addr_exact, addr_jw, addr_lev, addr_ratio, addr_tsort, addr_tset,
        addr_jac, addr_cont, addr_share, addr_ldiff, addr_lratio, addr_dj,
        addr_house, addr_pin,
        country_exact, country_conflict, country_both, country_s1_miss, country_c_miss,
        cand_is_s2, cand_is_s3, n_cands_f,
        name_x_addr, name_plus_addr, name_min_addr, name_max_addr,
        same_ctry_high_name, same_ctry_high_addr,
        exact_name_ctry, exact_addr_ctry, high_both,
    ]


def load_candidates_tsv(path: str) -> Dict[str, List[str]]:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    out = {}
    for _, row in df.iterrows():
        cands = [c.strip() for c in row["candidate_entity_ids"].split(",") if c.strip()]
        # deduplicate while preserving order
        seen = set()
        uniq = []
        for c in cands:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        out[row["source1_entity_id"]] = uniq
    return out


def load_candidate_lookup(*dfs: pd.DataFrame) -> Dict[str, dict]:
    lookup = {}
    for df in dfs:
        for r in df.itertuples(index=False):
            lookup[r.entity_id] = {
                "name_clean": getattr(r, "name_clean", ""),
                "address_clean": getattr(r, "address_clean", ""),
                "country_clean": getattr(r, "country_clean", ""),
            }
    return lookup


def build_feature_matrix(
    s1_df: pd.DataFrame,
    cand_lookup: Dict[str, dict],
    candidates: Dict[str, List[str]],
    desc: str = "Building features",
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Returns
    -------
    pairs_df : source1_entity_id, candidate_entity_id
    X        : float32 array (n_pairs, n_features)
    """
    s1_map = {
        r.entity_id: (
            getattr(r, "name_clean", ""),
            getattr(r, "address_clean", ""),
            getattr(r, "country_clean", ""),
        )
        for r in s1_df.itertuples(index=False)
    }

    rows = []
    feats = []
    for s1_id, cand_ids in tqdm(candidates.items(), desc=desc, ncols=90):
        if s1_id not in s1_map:
            continue
        s1_name, s1_addr, s1_ctry = s1_map[s1_id]
        n_cands = len(cand_ids)
        for cid in cand_ids:
            if cid not in cand_lookup:
                continue
            c = cand_lookup[cid]
            rows.append((s1_id, cid))
            feats.append(pair_features(
                s1_name, s1_addr, s1_ctry,
                c["name_clean"], c["address_clean"], c["country_clean"],
                n_cands=n_cands, cand_id=cid,
            ))

    pairs_df = pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id"])
    X = np.asarray(feats, dtype=np.float32)
    # safety: replace any accidental NaN / Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return pairs_df, X


def save_feature_columns():
    FEATURE_COLS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_COLS_PATH.write_text(json.dumps(FEATURE_NAMES, indent=2))
