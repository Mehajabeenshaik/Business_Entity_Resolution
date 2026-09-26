"""
normalize.py
============
Amazon ML Challenge 2026 - Business Entity Resolution
------------------------------------------------------
Text cleaning and normalization utilities for business names, addresses,
and countries loaded from the three source datasets.

Pipeline for clean_text()
--------------------------
1. Unicode normalization  - decompose and strip diacritic marks (accents)
2. Lowercase             - uniform casing for comparisons
3. Abbreviation expansion - expand known domain abbreviations BEFORE removing
                            punctuation so word boundaries are intact
4. Punctuation removal   - strip everything that is not alphanumeric or space
5. Whitespace collapse   - squeeze internal whitespace and strip ends

Usage
-----
    from src.normalize import clean_text, normalize_dataframe
"""

import re
import sys
import unicodedata

import pandas as pd

# Force UTF-8 console output on Windows (default codec is cp1252).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Abbreviation expansion tables
# ---------------------------------------------------------------------------

# Business-entity abbreviations (applied to business names).
# Key   : the abbreviation pattern (case-insensitive whole-word match).
# Value : the expanded form (lowercase, as the string is already lowercased
#         before expansion).
BUSINESS_ABBREVS: dict[str, str] = {
    r"\bltd\b":   "limited",
    r"\bpvt\b":   "private",
    r"\bcorp\b":  "corporation",
    r"\binc\b":   "incorporated",
    r"\bllp\b":   "limited liability partnership",
    r"\bllc\b":   "limited liability company",
    r"\bco\b":    "company",
    r"\bplc\b":   "public limited company",
}

# Address abbreviations (applied to business addresses).
ADDRESS_ABBREVS: dict[str, str] = {
    r"\brd\b":    "road",
    r"\bst\b":    "street",
    r"\bave\b":   "avenue",
    r"\blvd\b":   "boulevard",     # already the short form of blvd
    r"\bblvd\b":  "boulevard",
    r"\bdr\b":    "drive",
    r"\bnr\b":    "near",
    r"\bapt\b":   "apartment",
    r"\bste\b":   "suite",
    r"\bhwy\b":   "highway",
    r"\bpkwy\b":  "parkway",
    r"\bln\b":    "lane",
    r"\bct\b":    "court",
    r"\bpl\b":    "place",
    r"\bsq\b":    "square",
    r"\bfwy\b":   "freeway",
}

# Pre-compile all patterns for performance (datasets have millions of rows).
_BUSINESS_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(pat, re.IGNORECASE), repl)
    for pat, repl in BUSINESS_ABBREVS.items()
]

_ADDRESS_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(pat, re.IGNORECASE), repl)
    for pat, repl in ADDRESS_ABBREVS.items()
]

# Single compiled pattern to remove non-alphanumeric, non-space characters.
_PUNCTUATION_RE = re.compile(r"[^a-z0-9\s]")

# Single compiled pattern to collapse multiple spaces.
_WHITESPACE_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Core cleaning function
# ---------------------------------------------------------------------------

def clean_text(
    text: str,
    abbrev_patterns: list[tuple[re.Pattern, str]] | None = None,
) -> str:
    """
    Clean and normalize a raw text string.

    Steps
    -----
    1. Guard against non-string / NaN inputs - return "" for those.
    2. Unicode NFKD normalization - decomposes characters with diacritics
       (e.g. "Cafe\u0301" -> "Cafe") and drops the combining marks, making
       downstream regex simpler and matches more robust.
    3. Lowercase.
    4. Abbreviation expansion - done *before* punctuation removal so word
       boundaries (\\b) work correctly.  Pass ``abbrev_patterns`` to apply
       a domain-specific table (business vs. address).
    5. Remove punctuation and special characters (keep a-z, 0-9, space).
    6. Collapse multiple whitespace characters and strip.

    Parameters
    ----------
    text : str
        Raw input string (business name, address, or country).
    abbrev_patterns : list of (compiled Pattern, str), optional
        Pre-compiled (pattern, replacement) pairs to expand abbreviations.
        Use ``_BUSINESS_PATTERNS`` for names, ``_ADDRESS_PATTERNS`` for
        addresses, or ``None`` to skip expansion (e.g. for country).

    Returns
    -------
    str
        Cleaned, normalized string.

    Examples
    --------
    >>> clean_text("B+ Retail Inc.", _BUSINESS_PATTERNS)
    'b retail incorporated'
    >>> clean_text("1795 Westchester Dr., High Point, NC", _ADDRESS_PATTERNS)
    '1795 westchester drive high point nc'
    """
    # Step 1 - guard: treat None / NaN / non-strings as empty string
    if not isinstance(text, str) or not text.strip():
        return ""

    # Step 2 - Unicode normalization: strip diacritics / accent marks
    nfkd = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in nfkd if not unicodedata.combining(ch))

    # Step 3 - lowercase
    text = text.lower()

    # Step 4 - abbreviation expansion (whole-word, case-insensitive)
    if abbrev_patterns:
        for pattern, replacement in abbrev_patterns:
            text = pattern.sub(replacement, text)

    # Step 5 - remove punctuation and special characters
    text = _PUNCTUATION_RE.sub(" ", text)

    # Step 6 - collapse whitespace
    text = _WHITESPACE_RE.sub(" ", text).strip()

    return text


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def clean_business_name(text: str) -> str:
    """Clean a business name, expanding business-entity abbreviations."""
    return clean_text(text, abbrev_patterns=_BUSINESS_PATTERNS)


def clean_address(text: str) -> str:
    """Clean an address string, expanding street/address abbreviations."""
    return clean_text(text, abbrev_patterns=_ADDRESS_PATTERNS)


def clean_country(text: str) -> str:
    """
    Normalize a country string.

    Only lowercases and strips - deliberately avoids mapping/filtering so
    the function works for any country that appears in the data, including
    unseen ones in the test set (e.g. France).
    """
    if not isinstance(text, str):
        return ""
    return text.lower().strip()


# ---------------------------------------------------------------------------
# Vectorized helpers (used by normalize_dataframe for speed)
# ---------------------------------------------------------------------------

# Build flat dicts for vectorized str.replace calls.
# pandas Series.str.replace() with regex=True runs at C speed over the whole
# column — 10-50x faster than row-by-row .apply() on large datasets.
_BUSINESS_ABBREV_DICT: dict[str, str] = {
    pat: repl for pat, repl in BUSINESS_ABBREVS.items()
}
_ADDRESS_ABBREV_DICT: dict[str, str] = {
    pat: repl for pat, repl in ADDRESS_ABBREVS.items()
}


def _vectorized_clean(
    series: pd.Series,
    abbrev_dict: dict[str, str],
) -> pd.Series:
    """
    Apply the full cleaning pipeline to an entire pandas Series at once.

    Steps (vectorized via pandas str accessor except Step 2)
    ---------------------------------------------------------
    1. Fill NaN / non-string with empty string
    2. Unicode NFKD normalization + diacritic removal  (.apply — unicodedata
       has no vectorized API, but this runs once per series, not per column)
    3. Lowercase
    4. Abbreviation expansion  (one C-speed regex pass per abbreviation)
    5. Remove punctuation / special characters
    6. Collapse whitespace

    Parameters
    ----------
    series      : pd.Series of raw strings
    abbrev_dict : dict mapping regex pattern -> replacement string

    Returns
    -------
    pd.Series of cleaned strings
    """
    # Step 1: coerce NaN and non-strings to ""
    s = series.fillna("").astype(str)

    # Step 2: Unicode NFKD — strip diacritic/accent combining marks.
    # Fastest approach: normalize to NFKD then encode to ASCII ignoring
    # non-representable bytes.  This is ~6x faster than iterating characters
    # with unicodedata.combining().
    #
    # Fallback: for non-Latin scripts (e.g. Hindi/Devanagari) the full string
    # becomes empty after ASCII encoding.  In that case we keep the original
    # lowercased text so those records still get blocking keys.
    def _strip_diacritics(text: str) -> str:
        ascii_text = (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
        # If the entire string was non-ASCII, preserve the original
        return ascii_text if ascii_text.strip() else text

    s = s.apply(_strip_diacritics)

    # Step 3: Lowercase — vectorized
    s = s.str.lower()

    # Step 4: Abbreviation expansion — one C-level regex pass per pattern
    for pattern, replacement in abbrev_dict.items():
        s = s.str.replace(pattern, replacement, regex=True)

    # Step 5: Remove punctuation / special chars (keep a-z, 0-9, space)
    s = s.str.replace(r"[^a-z0-9\s]", " ", regex=True)

    # Step 6: Collapse whitespace — vectorized
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()

    return s


# ---------------------------------------------------------------------------
# DataFrame-level normalizer
# ---------------------------------------------------------------------------

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add normalized columns to a source DataFrame.

    New columns added
    -----------------
    name_clean    : cleaned business_name  (abbreviations expanded)
    address_clean : cleaned business_address  (street abbreviations expanded)
    country_clean : lowercased + stripped country  (no mapping applied)

    Uses fully vectorized pandas str operations for speed — suitable for
    DataFrames with millions of rows.  Operates on a copy; never mutates
    the caller's DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: business_name, business_address, country.

    Returns
    -------
    pd.DataFrame with three additional cleaned columns appended.

    Raises
    ------
    KeyError
        If any expected source columns are missing.
    """
    required = {"business_name", "business_address", "country"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"normalize_dataframe() missing expected columns: {missing}"
        )

    df = df.copy()  # never mutate the caller's frame

    print("  Normalizing business names ...")
    df["name_clean"] = _vectorized_clean(df["business_name"], _BUSINESS_ABBREV_DICT)

    print("  Normalizing addresses ...")
    df["address_clean"] = _vectorized_clean(df["business_address"], _ADDRESS_ABBREV_DICT)

    print("  Normalizing countries ...")
    # Country: just lowercase + strip — no diacritics needed, fully vectorized
    df["country_clean"] = (
        df["country"].fillna("").astype(str).str.lower().str.strip()
    )

    return df


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

def _run_demo() -> None:
    """
    Load train_source1, normalize it, and print 5 before/after examples.
    Run with:  python src/normalize.py
    """
    # Import here to keep normalize.py usable standalone without load_data.py
    import os
    import sys

    # Allow running as `python src/normalize.py` from the project root
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.load_data import load_source, FILE_PATHS

    print("=" * 70)
    print("  normalize.py - Smoke Test")
    print("=" * 70)

    # --- Load ---
    print("\n[1] Loading train_source1 ...")
    df = load_source(FILE_PATHS["train_source1"], name="train_source1")
    print(f"    Loaded {len(df):,} rows.\n")

    # --- Normalize ---
    print("[2] Normalizing ...")
    df_norm = normalize_dataframe(df)
    print(f"    Done. Columns now: {list(df_norm.columns)}\n")

    # --- Show 5 examples ---
    print("[3] Before vs After (5 random examples)")
    print("-" * 70)

    sample = df_norm.sample(5, random_state=42)
    for i, (_, row) in enumerate(sample.iterrows(), start=1):
        print(f"\n  Example {i}")
        print(f"  {'Name    (raw)':<18}: {row['business_name']}")
        print(f"  {'Name    (clean)':<18}: {row['name_clean']}")
        print(f"  {'Address (raw)':<18}: {row['business_address']}")
        print(f"  {'Address (clean)':<18}: {row['address_clean']}")
        print(f"  {'Country (raw)':<18}: {row['country']}")
        print(f"  {'Country (clean)':<18}: {row['country_clean']}")

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    _run_demo()
