# Multi-Source Record Linkage Pipeline

> **An end-to-end scalable entity resolution system** that links business records across three independently-maintained data sources using a multi-key inverted-index blocking strategy and a CatBoost classifier with 47 hand-crafted similarity features.

Built to handle **~12 million multilingual business records** across Latin, Devanagari, Cyrillic, and Arabic scripts.

---

## Table of Contents

1. [What Is Entity Resolution?](#1-what-is-entity-resolution)
2. [Dataset Overview](#2-dataset-overview)
3. [Pipeline Architecture](#3-pipeline-architecture)
4. [Module Breakdown](#4-module-breakdown)
5. [Technical Challenges](#5-technical-challenges)
6. [Blocking: v1 vs. v2 — The Core Fix](#6-blocking-v1-vs-v2--the-core-fix)
7. [Results & Honest Learnings](#7-results--honest-learnings)
8. [Future Improvements](#8-future-improvements)
9. [Setup & Usage](#9-setup--usage)

---

## 1. What Is Entity Resolution?

**Entity Resolution** (also called *Record Linkage* or *Deduplication*) is the problem of identifying which records across different datasets refer to the same real-world entity — despite spelling variations, abbreviations, missing fields, and data noise.

This project tackles a **multi-source, cross-lingual** variant:

| Source | Role |
|--------|------|
| **Source 1** | Deduplicated reference set ("the query set") |
| **Source 2** | Independent external business registry (noisy) |
| **Source 3** | Second independent external registry (noisy) |

**Goal:** For each record in Source 1, find all records in Sources 2 and 3 that describe the same real-world business.

```
Source 1 entity:    "Acme Technologies Pvt Ltd"   | Mumbai, India
                            ↓  matches?
Source 2 entity:    "ACME Tech Private Limited"   | Mumbai, MH, India   ✓ MATCH
Source 3 entity:    "Acme Technologies"           | Pune, India          ? MAYBE
Source 2 entity:    "Acme Chemicals Ltd"          | Delhi, India         ✗ NO MATCH
```

This is a hard problem because:
- The same business can be spelled many different ways across sources
- Abbreviations differ: "Pvt Ltd" vs "Private Limited" vs "P. Ltd."
- Addresses are noisy and inconsistently formatted
- ~12 million records — brute-force pairwise comparison ≈ 22 billion pairs

---

## 2. Dataset Overview

| Split | Source 1 | Source 2 | Source 3 |
|-------|----------|----------|----------|
| Train | ~2.2M    | ~5M      | ~5M      |
| Test  | ~2.2M    | ~5M      | ~5M      |

**Schema** (all source files):

```
entity_id | business_name | business_address | country
```

**Ground truth** (train only):

```
source1_entity_id | matched_entity_ids   (comma-separated list)
```

### Data Challenges

- **Scale**: 12M+ total records — row-by-row Python operations are a non-starter
- **Multilingual names**: significant portions in Devanagari (Hindi), Cyrillic, Arabic, Chinese
- **Noisy addresses**: inconsistent formatting, missing fields, different transliterations
- **Many-to-many matches**: a single Source 1 entity may match 0, 1, or many records
- **Class imbalance**: true matches are a tiny fraction of all possible pairs (< 0.001%)

---

## 3. Pipeline Architecture

```
Raw TSV Files (S1, S2, S3)
         │
         ▼
  ┌─────────────┐
  │  load_data  │  ── Load TSVs, EDA stats, path config
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  normalize  │  ── Unicode NFKD, diacritic strip, abbreviation
  └──────┬──────┘     expansion, lowercase, punctuation removal
         │
         ▼
  ┌─────────────┐    10 complementary key types (BK1–BK10)
  │  blocking   │  ── Multi-key inverted index → candidate_pairs.tsv
  └──────┬──────┘    Per-key cap: 60  |  Per-entity cap: 80
         │
         ▼
  ┌─────────────┐
  │  features   │  ── 47 pairwise similarity features per (S1, candidate)
  └──────┬──────┘     RapidFuzz + custom token/digit/address metrics
         │
         ▼
  ┌─────────────┐
  │   train     │  ── CatBoost classifier, grouped split (no leakage)
  └──────┬──────┘     Threshold search optimising macro-F0.5
         │
         ▼
  ┌─────────────┐
  │   predict   │  ── Score test candidates → matching_results.tsv
  └──────┬──────┘
         │
         ▼
  ┌──────────────────┐
  │  validate_output │  ── Strict submission format validator
  └──────────────────┘
```

---

## 4. Module Breakdown

### `src/load_data.py`
Loads all 7 TSV files into pandas DataFrames. Path configurable via the
`RECORD_LINKAGE_DATASET_ROOT` environment variable. Prints EDA statistics:
shapes, unique country distributions, match-count histogram from ground truth.

### `src/normalize.py`
Fully **vectorized** text cleaning pipeline (no row-by-row `.apply()` bottlenecks):

| Step | Operation |
|------|-----------|
| 1 | Unicode NFKD normalization — decompose + strip diacritic marks |
| 2 | Non-Latin fallback — if ASCII stripping empties the string (e.g. Hindi), keep original |
| 3 | Lowercase |
| 4 | Abbreviation expansion — `"Ltd"→"limited"`, `"Rd"→"road"`, etc. (before punctuation removal so `\b` works) |
| 5 | Punctuation removal — keep only `[a-z0-9 ]` |
| 6 | Whitespace collapse — strip and squeeze |

### `src/blocking.py`

The most critical module. Generates a compact candidate set for each Source 1 entity using a **multi-key inverted index** — the only viable approach at this scale.

**10 Blocking Keys (v2, precision-first design):**

| Key | Tag | Strategy | Added/Changed |
|-----|-----|----------|---------------|
| BK1 | `c5:` | country + first **5** chars | Was 3 chars in v1 — biggest explosion source |
| BK2 | `c7:` | country + first **7** chars | New in v2 |
| BK3 | `ct:` | country + first token ≥ **5** chars | Raised from 4 chars |
| BK4 | `p7:` | first **7** chars, no country | Raised from 6 chars |
| BK5 | `st:` | sorted tokens ≥ **4** chars | Raised from 3 chars |
| BK6 | `ph:` | country + Soundex of first token | Unchanged |
| BK7 | `aa:` | country + first 6 chars of address | **Conditional** (weak name only) |
| BK8 | `ca:` | country + first 6 chars of address | Non-Latin fallback |
| BK9 | `at:` | country + first significant address token | Non-Latin fallback |
| BK10 | `ap:` | first 7 chars of address | Non-Latin fallback |

**Two-level caps:**
- `MAX_CANDIDATES_PER_KEY = 60` — stops hot-key buckets from dominating
- `MAX_TOTAL_CANDIDATES = 80` — hard cap per entity after unioning all keys

Also includes a **blocking evaluation harness** that measures pair-level and entity-level recall against ground truth, so you know your recall ceiling before training.

### `src/features.py`
Builds a **47-feature vector** for each `(Source 1, candidate)` pair using RapidFuzz (C-speed):

| Feature Group | Count | Examples |
|---------------|-------|---------|
| Name similarity | 16 | Jaro-Winkler WRatio, Levenshtein, token sort/set ratio, Jaccard, prefix-4 match, digit Jaccard |
| Address similarity | 14 | Same + house number match, postal code match |
| Country features | 5 | Exact match, conflict flag, missingness indicators |
| Source/context | 3 | Is S2 / Is S3 / total candidates for this entity |
| Interaction features | 9 | name×addr, same_country_high_name, exact_name_exact_country, etc. |

Feature order is fixed and serialized to `artifacts/features/feature_columns.json` to prevent train/test skew.

### `src/train.py`
- Loads `candidate_pairs.tsv` (train blocking output)
- Labels pairs as match/non-match from ground truth
- **5:1 negative downsampling** per entity to keep training feasible
- **Grouped train/val split** (`GroupShuffleSplit` by `source1_entity_id`) — no leakage
- Trains **CatBoost** (3000 iter, depth=8, lr=0.05, early stopping)
- **Threshold sweep** [0.50, 0.99] on validation macro-F0.5 to pick optimal decision boundary
- Saves model, threshold, feature importance, and metrics to `artifacts/`

### `src/predict.py`
Loads saved model + threshold, scores test candidates, writes `matching_results.tsv`.

### `src/evaluate.py`
Full eval on training ground truth: macro-F0.5 / precision / recall + blocking recall ceiling.

### `src/validate_output.py`
Strict submission file validator — checks format, column names, entity coverage, value types.

### `fast_submit.py`
Utility that merges existing predictions with the full list of Source 1 IDs to ensure no entity is missing from the output (a submission requirement).

---

## 5. Technical Challenges

### 5.1 Candidate Explosion — The Core Scalability Problem

The first version of the blocking module produced a **5.3 GB** `candidate_pairs.tsv`.
At ~2.2M Source 1 entities this means ~2,400 candidates per entity on average, which makes feature extraction and training completely infeasible.

**Root causes:**
1. `c3:` key (country + 3-char prefix) was catastrophically broad. The key `"c3:india_pri"` alone matched hundreds of thousands of "Private Ltd" companies — the most common business type in India.
2. `aa:` key (address, 4-char prefix) was applied to **all** entities, not just those with weak names — flooding every well-named entity with noisy address-based candidates.
3. Per-key bucket cap of 180 was too large: 10 keys × 180 = 1,800 worst-case candidates per entity.
4. No total-per-entity cap: the union of all keys had no bound.

**Fix:** See [Blocking v2](#6-blocking-v1-vs-v2--the-core-fix) below.

### 5.2 Non-Latin Business Names

India accounts for a major portion of the dataset; many business names are in **Devanagari/Hindi script**. After NFKD normalization and ASCII stripping, these names become empty strings — making all name-based blocking keys useless.

**Mitigation:**
- Detect when ASCII stripping empties the string; preserve original text so the record at least has something
- BK8–BK10 address fallback keys activate automatically for any entity where `name_clean` is too short
- True fix would require romanization/transliteration of Devanagari (see Future Improvements)

### 5.3 Precision-Weighted Metric

The evaluation metric is **macro-F0.5** — precision weighted twice as heavily as recall:

$$F_{0.5} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$$

This means adding a wrong match hurts roughly twice as much as missing a correct one.
The optimal strategy is therefore **precision-first blocking**: a smaller, cleaner candidate set is better than a larger, noisier one.

### 5.4 Iteration Speed at Scale

With 2.2M entities and a 5 GB candidate file, a single pipeline run (blocking + feature extraction + training) takes hours. This makes hyperparameter tuning and blocking strategy experimentation very slow — a key lesson about building **fast evaluation loops** before scaling up.

---

## 6. Blocking: v1 vs. v2 — The Core Fix

This section documents the most important engineering change in the project.

### What changed

| Parameter | v1 (broken) | v2 (fixed) | Impact |
|-----------|-------------|------------|--------|
| 3-char prefix key (`c3:`) | **Present** | **Removed** | Biggest single source of explosion |
| 5-char prefix key (`c5:`) | Present | Present (now BK1) | Kept — good precision |
| 7-char prefix key (`c7:`) | Absent | **Added** | New high-precision key |
| First token min length | 4 chars | **5 chars** | Excludes common 4-letter words |
| Prefix key width (`p6/p7`) | 6 chars | **7 chars** | More selective |
| Sorted token min length | 3 chars | **4 chars** | Excludes "the", "and", "for" |
| Address key (BK7) scope | ALL entities | **Weak-name only** | Eliminates noise for well-named entities |
| Max per-key bucket | **180** | **60** | 3× fewer candidates from hot buckets |
| Max total per entity | None | **80** | Hard guarantee on output size |

### Why these specific numbers

- **5-char prefix** is the sweet spot for business names: it's selective enough to exclude generic word-starts ("manuf", "trade") but broad enough to tolerate small OCR errors.
- **60 per-key cap**: any blocking key with >60 matching entities is a stop-word by another name. Keeping them adds noise, not signal.
- **80 per-entity cap**: at 2.2M entities × 80 candidates × ~50 bytes per ID, the output file is ~8.8 GB theoretical maximum — practically much less since most entities produce far fewer candidates. This guarantees a tractable file.

### Expected improvement

| Metric | v1 | v2 (estimated) |
|--------|----|----------------|
| Avg candidates per entity | ~2,400 | ~50–80 |
| Output file size | 5.3 GB | ~200–400 MB |
| Feature matrix rows (train) | Billions | ~100–200M |
| CatBoost trainable? | ❌ No | ✅ Yes |

---

## 7. Results & Honest Learnings

### What was built and ran successfully
- Complete ETL, normalization, and blocking pipeline
- 47-feature pairwise feature engineering (RapidFuzz-based)
- CatBoost training scaffold with grouped validation and F0.5 threshold search
- Blocking evaluation harness (recall ceiling measurement)
- Submission format validator

### What didn't work
The v1 blocking produced a 5.3 GB candidate file. This made feature extraction and CatBoost training infeasible to complete within available memory and time. The final output was generated using a lightweight fuzzy-ranking fallback (`fast_submit.py`) rather than the trained classifier — resulting in a score close to the "predict nothing" baseline.

### Key Learnings

1. **Measure your blocking recall ceiling first.** Before spending time on features and training, verify that the true matches are actually in your candidate set. A classifier cannot find matches that blocking missed.

2. **Precision-first blocking is correct for precision-weighted metrics.** It feels counterintuitive, but a smaller, cleaner candidate set produces better downstream precision than a large, noisy one — even if recall drops slightly.

3. **Build a fast end-to-end loop before scaling.** Running the full pipeline on a 1% sample would have revealed the candidate explosion in minutes rather than hours.

4. **Candidate count statistics are your first health check.** Average candidates per entity, max candidates, and 99th-percentile candidates are cheap to compute and immediately reveal blocking quality problems.

5. **UTF-8 is not optional for multilingual data.** On Windows, stdout defaults to cp1252 — without explicit reconfiguration, any Hindi or Arabic character in a print statement crashes the process silently.

---

## 8. Future Improvements

### Immediate (high impact)

1. **Run blocking v2** — validate that average candidates are in the 50–80 range and blocking recall stays above 85% on the train ground truth.
2. **Streaming feature extraction** — process candidate pairs in chunks of ~500K and write feature matrices to disk (HDF5 or numpy memmap). Avoid loading the full feature matrix into RAM.
3. **Faster baseline first** — train HistGBM or Logistic Regression on a 10% data sample to get a quick validation score before waiting for full CatBoost training.

### Algorithmic improvements

4. **Script-aware normalization** — romanize non-Latin scripts using `indic-transliteration` (Hindi), `arabic-transliterator`, `unidecode` as a fallback. This would make BK1–BK6 effective for the currently-untreatable Devanagari/Arabic records.
5. **ANN-based blocking with multilingual embeddings** — embed business names using [LaBSE](https://huggingface.co/sentence-transformers/LaBSE) or `paraphrase-multilingual-MiniLM-L12-v2`, then use Faiss for approximate nearest-neighbour candidate retrieval. Language-agnostic and scales sub-linearly.
6. **Hard negative mining** — the current negative sampling is random. Mine hard negatives (high-similarity non-matches like "Acme Chemicals" vs "Acme Technologies") to force the model to learn finer discrimination.
7. **Cross-encoder reranking** — for the top-K candidates per entity, use a small fine-tuned cross-encoder to rerank and filter, leveraging the full text pair rather than fixed features.

### System improvements

8. **Multiprocessing for feature extraction** — the pairwise feature loop is embarrassingly parallel. Use `multiprocessing.Pool` or Dask to saturate all CPU cores.
9. **Profile hot-key distribution** — log the 100 most common blocking keys and their bucket sizes after building the index. This reveals data-distribution pathologies before they cause downstream problems.

---

## 9. Setup & Usage

### Requirements

```bash
pip install -r requirements.txt
```

### Dataset Path

Set the root directory of the dataset:

```bash
# Linux / macOS
export RECORD_LINKAGE_DATASET_ROOT="/path/to/dataset"

# Windows (PowerShell)
$env:RECORD_LINKAGE_DATASET_ROOT = "C:\path\to\dataset"
```

Expected layout:

```
dataset/
    train/
        train_source1.tsv
        train_source2.tsv
        train_source3.tsv
        train_ground_truth.tsv
    test/
        test_source1.tsv
        test_source2.tsv
        test_source3.tsv
```

### Full Pipeline

```bash
# 1. Explore dataset statistics
python -m src.load_data

# 2. Run blocking on train split
#    → output/candidate_pairs.tsv
python -m src.blocking --split train

# 3. Train CatBoost model, run threshold search
#    → artifacts/model/  and  artifacts/validation/
python -m src.train

# 4. Evaluate on train ground truth
python -m src.evaluate

# 5. Run blocking on test split
#    → output/candidate_pairs.tsv  (overwrites train candidates)
python -m src.blocking --split test

# 6. Generate test predictions
#    → output/matching_results.tsv
python -m src.predict

# 7. Validate output format
python -m src.validate_output
```

### Repository Structure

```
src/
    load_data.py        # TSV loading, EDA, path config
    normalize.py        # Vectorized multilingual text cleaning
    blocking.py         # Multi-key inverted index, v2 precision-first design
    features.py         # 47 pairwise similarity features
    train.py            # CatBoost + HistGBM, grouped CV, F0.5 threshold search
    evaluate.py         # Full evaluation + blocking recall ceiling
    predict.py          # Score test candidates → matching_results.tsv
    validate_output.py  # Output format validator
    main.py             # Pipeline orchestrator
fast_submit.py          # Emergency merge utility (ensures all S1 IDs are in output)
output/
    candidate_pairs.tsv        # From blocking.py
    matching_results.tsv       # From predict.py
    matching_results_FIXED.tsv # From fast_submit.py
artifacts/
    model/          # CatBoost model (.cbm), config, threshold
    features/       # Feature column definitions (JSON)
    validation/     # Threshold sweep CSV, feature importance, eval metrics
requirements.txt
```

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| `pandas` ≥ 2.0 | Vectorized string operations on millions of rows |
| `numpy` | Feature matrix construction, threshold grid search |
| `rapidfuzz` ≥ 3.0 | C-level Levenshtein, Jaro-Winkler, token ratios |
| `catboost` ≥ 1.2 | Primary gradient-boosted classifier |
| `scikit-learn` ≥ 1.3 | HistGBM baseline, GroupShuffleSplit |
| `jellyfish` | Soundex phonetic encoding for blocking key BK6 |
| `tqdm` | Progress bars for long-running data loops |

---

*This project explores the engineering tradeoffs in large-scale entity resolution: the tension between blocking recall and precision, the scalability limits of inverted-index approaches, and the challenge of matching multilingual business names at internet scale.*
