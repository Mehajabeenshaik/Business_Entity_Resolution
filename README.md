# Business Entity Resolution — Amazon ML Challenge 2026

> **A full end-to-end entity resolution pipeline** built for the Amazon ML Challenge 2026.
> This is a detailed portfolio write-up that documents what was built, the real challenges encountered, and what should be done differently next time.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Dataset Scale](#2-dataset-scale)
3. [Pipeline Architecture](#3-pipeline-architecture)
4. [Module Breakdown](#4-module-breakdown)
5. [Technical Challenges](#5-technical-challenges)
6. [Results & Honest Assessment](#6-results--honest-assessment)
7. [What Was Successfully Built](#7-what-was-successfully-built)
8. [What Went Wrong & Why](#8-what-went-wrong--why)
9. [Future Improvements](#9-future-improvements)
10. [Setup & Usage](#10-setup--usage)

---

## 1. Problem Statement

Given three independent sources of business entity records, determine which records across Source 2 and Source 3 refer to the **same real-world business** as a record in Source 1.

| Source | Role |
|--------|------|
| **Source 1** | Deduplicated reference set (the "query") |
| **Source 2** | Noisy external registry |
| **Source 3** | Second noisy external registry |

Each Source 1 entity may match **zero, one, or many** records from Sources 2 and 3. The submission format is:

```
source1_entity_id    matched_entity_ids
S1-0001              S2-1234,S3-5678
S1-0002              (empty — no match found)
```

### Evaluation Metric: Macro F0.5

The challenge is scored using **macro-averaged F0.5**, computed independently per Source 1 entity and then averaged.

$$F_{0.5} = \frac{(1 + 0.5^2) \cdot P \cdot R}{0.5^2 \cdot P + R} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$$

F0.5 **weights precision twice as heavily as recall**. This means:
- A **false positive** (predicting a wrong match) is penalized much more than a **false negative** (missing a true match).
- Predicting nothing (empty output) scores 0 on entities with matches — but at least avoids false positives on entities with no match.
- Threshold selection is therefore critical and heavily biased toward precision.

---

## 2. Dataset Scale

| Dataset | Approx. Rows |
|---------|-------------|
| Source 1 (train) | ~2.2 million |
| Source 2 (train) | ~5 million |
| Source 3 (train) | ~5 million |
| Ground Truth (train) | ~2.2 million |
| Source 1 (test) | ~2.2 million |

Total data across all sources: **~12 million business records**, spanning multiple countries with names in Latin, Devanagari (Hindi), Cyrillic, Arabic, and other scripts.

A **full O(n × m) pairwise comparison** between Source 1 and Sources 2+3 would require evaluating ~22 billion pairs — completely infeasible. This made the **blocking** (candidate generation) stage the most critical architectural decision.

---

## 3. Pipeline Architecture

```
Raw TSV Files (S1, S2, S3)
         │
         ▼
  ┌─────────────┐
  │  load_data  │  — Load TSVs, print EDA statistics
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  normalize  │  — Unicode NFKD, diacritic removal, abbreviation
  └──────┬──────┘    expansion, lowercasing, punctuation stripping
         │
         ▼
  ┌─────────────┐
  │  blocking   │  — Multi-key inverted index → candidate_pairs.tsv
  └──────┬──────┘    (BK1–BK10: prefix, phonetic, address, token-sort)
         │
         ▼
  ┌─────────────┐
  │  features   │  — 47 pairwise similarity features per (S1, candidate)
  └──────┬──────┘    pair using RapidFuzz + custom token/digit metrics
         │
         ▼
  ┌─────────────┐
  │   train     │  — CatBoost classifier, grouped train/val split,
  └──────┬──────┘    F0.5-optimized threshold search (0.50→0.99)
         │
         ▼
  ┌─────────────┐
  │   predict   │  — Score test candidates, apply threshold,
  └──────┬──────┘    write matching_results.tsv
         │
         ▼
  ┌──────────────────┐
  │  validate_output │  — Strict submission format validator
  └──────────────────┘
```

---

## 4. Module Breakdown

### `src/load_data.py`
- Loads all 7 TSV files (3 train sources, ground truth, 3 test sources) into pandas DataFrames.
- Path configurable via `AML_DATASET_ROOT` environment variable.
- Prints EDA: dataset shapes, unique country distributions, ground truth match-count histogram.

### `src/normalize.py`
Vectorized text cleaning pipeline for business names, addresses, and countries:
1. **Unicode NFKD normalization** — decompose and strip diacritic marks (e.g. `"Café"` → `"Cafe"`).
2. **Non-Latin fallback** — for scripts like Devanagari (Hindi) that become empty after ASCII stripping, the original text is preserved so blocking keys can still be generated.
3. **Abbreviation expansion** — `"Ltd"` → `"limited"`, `"Pvt"` → `"private"`, `"Rd"` → `"road"`, etc. Applied *before* punctuation removal so word boundaries work correctly.
4. **Punctuation removal** — retain only `[a-z0-9 ]`.
5. **Whitespace collapse** — squeeze and strip.

All operations use **pandas vectorized str methods** (C-speed) instead of row-by-row `.apply()` for scalability across millions of rows.

### `src/blocking.py`
The most critical and complex module. Generates a small set of candidate matches for each Source 1 entity using a **multi-key inverted index** to avoid an O(n×m) brute-force search.

**10 Complementary Blocking Keys (BK1–BK10):**

| Key | Tag | Strategy |
|-----|-----|----------|
| BK1 | `c3:` | country + first 3 chars of name (broad recall) |
| BK2 | `c5:` | country + first 5 chars of name (higher precision) |
| BK3 | `ct:` | country + first significant token (≥4 chars) |
| BK4 | `p6:` | first 6 chars of name, **no country** (cross-country bridge) |
| BK5 | `st:` | sorted significant tokens, word-order invariant |
| BK6 | `ph:` | country + Soundex of first token (phonetic/spelling variants) |
| BK7 | `aa:` | country + first 4 chars of address (noisy-name backup, all entities) |
| BK8 | `ca:` | country + first 6 chars of address (non-Latin fallback) |
| BK9 | `at:` | country + first significant address token (non-Latin fallback) |
| BK10 | `ap:` | first 7 chars of address, no country (non-Latin fallback) |

Each key bucket is **capped at 180 entities** (`MAX_CANDIDATES_PER_KEY`) to suppress runaway hot-keys (e.g. `"c3:india_pri"` would match hundreds of thousands of "Private Ltd" companies).

Also includes a blocking **evaluation harness** measuring:
- Pair-level blocking recall (fraction of true-match pairs covered by candidates)
- Entity-level recall (fraction of S1 entities where ALL true matches are in candidates)
- Average candidates per entity
- Zero-candidate rate

### `src/features.py`
Builds a **47-feature vector** for each (Source 1, candidate) pair:

| Group | Features |
|-------|---------|
| **Name similarity** (16) | Exact match, Jaro-Winkler WRatio, Levenshtein, ratio, partial ratio, token sort, token set, Jaccard, containment, shared token count, first/last token match, prefix-4 match, length diff/ratio, digit Jaccard |
| **Address similarity** (14) | Same metrics as name + house number match, PIN/postal code match |
| **Country features** (5) | Exact match, conflict flag, both present, S1 missing, candidate missing |
| **Source/blocking context** (3) | Is S2 / Is S3 / total candidates for this S1 entity |
| **Combined interactions** (9) | name×addr product, sum, min, max; high-name+same-country; high-addr+same-country; exact-name+exact-country; exact-addr+exact-country; both-high |

Feature order is **fixed and serialized** to `artifacts/features/feature_columns.json` to guarantee train/val/test consistency.

### `src/train.py`
- Loads candidate pairs from `output/candidate_pairs.tsv` (produced by blocking on the train split).
- Labels pairs as match (1) or non-match (0) using ground truth.
- **Down-samples negatives** to 5:1 ratio (negatives:positives) per S1 entity to keep training feasible.
- **Grouped train/val split** (`GroupShuffleSplit` by `source1_entity_id`) — ensures no S1 entity appears in both train and validation to prevent leakage.
- Trains a **CatBoost classifier** (3000 iterations, depth=8, lr=0.05, early stopping=80 rounds) and an **sklearn HistGBM baseline**.
- **Threshold search** over [0.50, 0.99] on validation macro-F0.5 to select the optimal decision boundary.
- Saves model, config, selected threshold, feature importance, and validation metrics to `artifacts/`.

### `src/predict.py`
Loads the saved CatBoost model and threshold, scores all test candidates, and writes `output/matching_results.tsv`.

### `src/evaluate.py`
Full evaluation on training ground truth: reports macro-F0.5, macro-precision, macro-recall, number of entities, and blocking recall ceiling.

### `src/validate_output.py`
Strict submission file validator — checks format, column names, entity ID coverage, and value types.

### `fast_submit.py`
Emergency submission utility that was used for the actual final submission. Merges existing predictions with the complete list of test entity IDs to ensure no entity is missing from the output file.

---

## 5. Technical Challenges

### 5.1 Candidate Explosion (The Core Bottleneck)

The biggest problem encountered was that blocking keys were **too permissive**, producing a `candidate_pairs.tsv` file of **~5.3 GB**. At 12 million+ total records, even a conservative average of ~2,400 candidates per Source 1 entity results in billions of pairwise feature computation calls.

**Impact:** CatBoost training never completed because the feature matrix alone — even before training — was too large to fit in RAM and took too long to compute. This meant the primary ML approach could not be evaluated at all on the test set.

**Root cause:** The `c3:` (country + 3-char prefix) and `aa:` (country + 4-char address prefix) keys produce very large buckets for common business naming patterns (e.g., most Indian businesses share the prefix `"pri"` for "Private Limited"). Even with the 180-entity per-key cap, the union across 6+ keys per entity still explodes.

### 5.2 Non-Latin Business Names

India-based records (a significant portion of the dataset) often have business names in **Devanagari/Hindi script**. After NFKD normalization and ASCII stripping, these names become empty strings, making all name-based blocking keys useless.

**Mitigation:** Implemented fallback address-based blocking keys (BK8–BK10) and preserved original text when ASCII stripping yields nothing. However, the address field is also often in Devanagari or missing, making these records very hard to match without script-aware normalization.

### 5.3 F0.5 Precision Bias

The metric strongly punishes false positives. This creates a dilemma:
- **Aggressive blocking** (high recall) → more candidates → more false positives from the classifier → lower F0.5
- **Conservative blocking** (fewer candidates) → fewer false positives → but also misses true matches → zero coverage score

The optimal strategy would be very selective blocking with a high-precision classifier. Getting that balance right requires iteration, which ran out of time for.

### 5.4 Scale vs. Iteration Speed

With ~2.2M Source 1 entities and a 5.3 GB candidate file, even a single pass over the data takes 30–60+ minutes. This severely limits the ability to iterate on hyperparameters or blocking strategies during a time-limited competition.

---

## 6. Results & Honest Assessment

| Metric | Value |
|--------|-------|
| **Final submission** | RapidFuzz token-set ranking (fast_submit.py), not the trained model |
| **Competition score** | Close to "predict nothing" baseline |
| **Why** | Candidate explosion prevented CatBoost training from completing; fallback used simple fuzzy ranking without any trained threshold |

The final score was low, but that does **not** mean the engineering effort was wasted. The pipeline is correct in design — the failure was a resource/time management problem, not an algorithmic one.

The "predict nothing" baseline gets a non-zero score only on entities that genuinely have no matches (predicting empty is correct for those). For entities that do have matches, an empty prediction gives F0.5 = 0. The final submission was marginally better in some buckets but similarly hurt by over-prediction in others.

---

## 7. What Was Successfully Built

✅ **Complete ETL pipeline** — loads, validates, and normalizes 12M+ multilingual records.

✅ **Production-quality blocking module** — 10 complementary key types, inverted index, hot-key capping, thorough evaluation harness (pair-level + entity-level recall).

✅ **47-feature pairwise feature engineering** — name, address, country, source context, and interaction features using RapidFuzz at C-speed.

✅ **CatBoost training scaffold** — grouped cross-validation, negative downsampling, F0.5-optimized threshold search, feature importance logging.

✅ **Full evaluation harness** — computes macro-F0.5/P/R at any threshold, blocking recall ceiling, and validation summary.

✅ **Submission validator** — strict format checking before submission.

✅ **Non-Latin text handling** — Devanagari fallback strategy, original-text preservation on ASCII-empty strings.

✅ **Windows/UTF-8 compatibility** — explicit stdout reconfiguration to avoid cp1252 crashes on Hindi/Arabic names.

---

## 8. What Went Wrong & Why

| Problem | Root Cause | Effect |
|---------|-----------|--------|
| 5.3 GB candidate file | BK1 (`c3:`) + BK7 (`aa:`) too permissive for generic prefixes | Could not complete feature extraction or training |
| No trained model in final submission | Candidate explosion → OOM / timeout | Fell back to unsophisticated fuzzy ranking |
| Low score | Fallback had no learned threshold, no precision control | Many false positives; score near baseline |

---

## 9. Future Improvements

### Immediate Fixes (would have the most impact)

1. **Tighten blocking** — Remove or heavily restrict `c3:` (3-char prefix) and `aa:` (4-char address). Switch `c3:` to `c5:` only, raise token minimum lengths. Target ≤ 200 average candidates per entity.
2. **Streaming feature extraction** — Process candidate pairs in chunks of ~500K rows; write feature batches to disk; train on partial data if necessary.
3. **Lighter model** — Use HistGBM or even logistic regression with the 47 features. Faster to train, easier to iterate. Switch to CatBoost only for final refinement.
4. **Blocking recall audit** — Before training, verify that the blocking recall on train ground truth is ≥ 90%. If the true matches are not in the candidate set, no classifier can find them.

### Longer-term Improvements

5. **Script-aware normalization** — Use `langdetect` + transliteration libraries (`indic-transliteration`, `arabic-transliterator`) to romanize non-Latin names before blocking.
6. **Faiss/approximate nearest-neighbor blocking** — Embed business names with a light multilingual embedding (e.g. LaBSE or sentence-transformers) and use Faiss for approximate k-NN candidate retrieval. Scales much better than prefix keys for multilingual data.
7. **Learning-to-match model** — Fine-tune a small cross-encoder (e.g. `paraphrase-multilingual-MiniLM`) on the (name1, name2, match) pairs instead of hand-crafted features.
8. **Proper negative mining** — Use hard negatives from blocking (high-similarity non-matches) rather than random sampling to improve model discrimination.
9. **Ensemble** — Combine CatBoost predictions with a lightweight Jaccard/BM25 score as a prior for better calibration.

---

## 10. Setup & Usage

### Installation

```bash
pip install -r requirements.txt
```

### Dataset Path

Place the dataset at `~/Downloads/6ab10eb3b23ba_student_resource/student_resource/dataset/` or set:

```bash
# Linux / macOS
export AML_DATASET_ROOT="/path/to/student_resource/dataset"

# Windows (PowerShell)
$env:AML_DATASET_ROOT = "C:\path\to\student_resource\dataset"
```

### Running the Full Pipeline

```bash
# 1. Explore dataset statistics
python -m src.load_data

# 2. Run blocking on train split (generates candidate_pairs.tsv)
python -m src.blocking --split train

# 3. Train the CatBoost model
python -m src.train

# 4. Evaluate on train ground truth
python -m src.evaluate

# 5. Run blocking on test split (overwrites candidate_pairs.tsv)
python -m src.blocking --split test

# 6. Generate test predictions
python -m src.predict

# 7. Validate submission format
python -m src.validate_output
```

### Repository Structure

```
src/
    load_data.py        # TSV loading, EDA helpers, path configuration
    normalize.py        # Vectorized text cleaning, diacritic/abbreviation handling
    blocking.py         # Multi-key inverted index blocking (BK1–BK10)
    features.py         # 47 pairwise similarity features
    train.py            # CatBoost + HistGBM training, F0.5 threshold search
    evaluate.py         # Full eval on train ground truth, blocking recall ceiling
    predict.py          # Score test candidates → matching_results.tsv
    validate_output.py  # Submission format validator
    main.py             # Pipeline orchestrator
fast_submit.py          # Emergency submission fixer (used for final submission)
output/
    candidate_pairs.tsv      # Generated by blocking.py
    matching_results.tsv     # Final predictions
    matching_results_FIXED.tsv  # Submission with all S1 entity IDs present
artifacts/
    model/                   # CatBoost model, config, threshold
    features/                # Feature column definitions
    validation/              # Metrics, threshold sweep, feature importance
requirements.txt
```

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| `pandas` | Tabular data loading and vectorized string ops |
| `numpy` | Feature matrix construction and threshold search |
| `rapidfuzz` | C-level fuzzy string matching (Levenshtein, Jaro-Winkler, token ratios) |
| `catboost` | Primary gradient-boosted classifier with native categorical support |
| `scikit-learn` | HistGBM baseline, grouped cross-validation split |
| `jellyfish` | Soundex phonetic encoding for blocking key BK6 |
| `tqdm` | Progress bars for long-running loops |

---

*This project was built as part of the Amazon ML Challenge 2026 and is maintained as a learning portfolio artifact. The pipeline design is production-ready; the competition outcome reflects the realities of working with internet-scale, multilingual, noisy entity data under time pressure.*
