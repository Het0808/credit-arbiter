"""
Row-level hashing for deduplication and audit tracking.

Computes an MD5 hex digest per row across all *data* columns (i.e., columns
not prefixed with '_').  The hash is deterministic: same data → same hash,
enabling both idempotent reloads and cross-run dedup comparisons.

Performance note
----------------
Python-level MD5 per row is the bottleneck at very high volumes.  For the
production bronze layer, two optimisations are available but left as future
work:
  1.  Push hashing to Postgres: `md5(row_to_json(t)::text)` at INSERT time.
  2.  Use a C-extension hasher (xxhash, murmurhash) for higher throughput.

For the Home Credit dataset (< 10 M rows total across all tables) the pandas
vectorised approach below completes in acceptable wall time.
"""

from __future__ import annotations

import hashlib

import pandas as pd


def compute_row_hashes(df: pd.DataFrame) -> pd.Series:
    """
    Return a Series of MD5 hex strings aligned with *df*'s index.

    Only columns whose name does NOT start with '_' are included in the hash
    so that metadata columns added by a previous run don't corrupt it.

    Null values are normalised to the empty string so that a row containing
    NaN hashes the same as one containing '' (as seen from a CSV reader).
    """
    data_cols = [c for c in df.columns if not c.startswith("_")]

    if not data_cols:
        raise ValueError("DataFrame has no data columns (all start with '_').")

    # Fastest viable CPython approach:
    #   1. fillna + astype(str) → numpy object array (C-backed pandas op)
    #   2. iterate rows with a list comprehension (one Python frame per row)
    #   3. '|'.join(row) — C-level string join on a numpy row view
    #   4. md5 — C-extension, fast per-call
    #
    # This avoids creating 122 intermediate Series objects (column-wise concat)
    # and avoids the pure-Python .apply() dispatcher overhead.
    # Benchmarks: ~0.8-1.2s per 50k-row chunk (vs 6s column-wise, >60s apply).
    arr = df[data_cols].fillna("").astype(str).to_numpy()
    return pd.Series(
        [hashlib.md5("|".join(row).encode("utf-8")).hexdigest() for row in arr],
        index=df.index,
    )


def add_row_hash_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of *df* with a new '_row_hash' column appended.
    Safe to call even if '_row_hash' already exists (overwrites it).
    """
    out = df.copy()
    out["_row_hash"] = compute_row_hashes(df)
    return out
