"""
Bulk-loading engine for the Credit Arbiter ingestion pipeline.

Loading strategy
----------------
Each CSV file is streamed in chunks (size = CHUNK_SIZE env var, default 50 000
rows).  For each chunk:

  1.  Metadata columns are added: _row_hash, _ingested_at, _source_file.
  2.  The chunk is serialised to an in-memory CSV buffer.
  3.  psycopg2's `copy_expert` transfers the buffer to Postgres via the
      COPY protocol — the fastest server-side bulk-load path available
      without superuser privileges.

full_refresh mode
-----------------
  TRUNCATE the table then COPY all chunks.

incremental mode
-----------------
  Uses a PostgreSQL temporary staging table:
    1.  COPY the chunk into a temp table (no constraints, fastest path).
    2.  INSERT INTO <target> SELECT … FROM _staging WHERE _row_hash NOT IN
        (SELECT _row_hash FROM <target>) — fully server-side, no Python heap
        pressure from loading millions of hashes.
    3.  DROP the temp table.
  This approach scales to hundreds-of-millions of rows without OOMing.

Rejects
-------
  If a COPY fails (type mismatch, oversized value, encoding error), the
  offending chunk is written as a timestamped CSV in REJECTS_DIR rather
  than being silently dropped.  The pipeline continues to the next chunk.
"""

from __future__ import annotations

import csv
import io
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import connection as PgConnection

from .config import CHUNK_SIZE, REJECTS_DIR
from .hasher import add_row_hash_column

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _add_metadata(df: pd.DataFrame, source_file_name: str) -> pd.DataFrame:
    """
    Append the three standard metadata columns to a chunk DataFrame.
    Called *before* the COPY so values are present in the in-memory buffer.
    """
    df = add_row_hash_column(df)                        # _row_hash
    df["_ingested_at"] = datetime.now(tz=timezone.utc)  # _ingested_at
    df["_source_file"] = source_file_name               # _source_file
    return df


def _df_to_copy_buffer(df: pd.DataFrame) -> io.StringIO:
    """
    Serialise a DataFrame to an in-memory CSV buffer for COPY FROM STDIN.

    NULL handling
    -------------
    The COPY statement declares ``NULL ''`` (unquoted empty field = NULL).
    We must match that exactly:
      - ``na_rep=""``           → NaN serialised as an empty field
      - ``quoting=QUOTE_MINIMAL`` → empty fields are written WITHOUT quotes

    Using ``QUOTE_NONNUMERIC`` was wrong: it quoted every non-numeric value,
    so NaN → ``""`` (quoted empty string).  Postgres receives a literal
    two-character empty string, not NULL, and rejects it for numeric columns.
    """
    buf = io.StringIO()
    df.to_csv(
        buf,
        index=False,
        header=False,
        quoting=csv.QUOTE_MINIMAL,   # empty fields stay unquoted → Postgres NULL
        na_rep="",                   # NaN → empty field (matched by NULL '' in COPY)
    )
    buf.seek(0)
    return buf


def _copy_chunk_to_table(
    conn: PgConnection,
    schema: str,
    table: str,
    df: pd.DataFrame,
) -> int:
    """
    COPY *df* into *schema.table* using the binary COPY protocol.
    Returns the number of rows transferred.
    Raises psycopg2.Error on failure (caller decides how to handle).
    """
    if df.empty:
        return 0

    buf = _df_to_copy_buffer(df)
    columns_sql = sql.SQL(", ").join(sql.Identifier(c) for c in df.columns)
    copy_stmt = sql.SQL(
        "COPY {schema}.{table} ({cols}) "
        "FROM STDIN WITH (FORMAT CSV, QUOTE '\"', NULL '')"
    ).format(
        schema=sql.Identifier(schema),
        table=sql.Identifier(table),
        cols=columns_sql,
    )

    with conn.cursor() as cur:
        cur.copy_expert(copy_stmt.as_string(conn), buf)

    conn.commit()
    return len(df)


def _copy_chunk_incremental(
    conn: PgConnection,
    schema: str,
    table: str,
    df: pd.DataFrame,
) -> int:
    """
    Load *df* incrementally using a server-side staging table.

    Workflow:
      1.  CREATE TEMP TABLE _staging AS SELECT … WHERE FALSE  (schema clone)
      2.  COPY df → _staging
      3.  INSERT INTO target SELECT * FROM _staging
              WHERE _row_hash NOT IN (SELECT _row_hash FROM target)
      4.  DROP TABLE _staging
      5.  COMMIT

    Returns the number of *new* rows actually inserted.
    """
    if df.empty:
        return 0

    staging = "_ingestion_staging"
    target_sql = sql.SQL("{}.{}").format(
        sql.Identifier(schema), sql.Identifier(table)
    )
    staging_sql = sql.Identifier(staging)

    with conn.cursor() as cur:
        # 1. Create temp staging table (session-scoped, auto-dropped on disconnect)
        cur.execute(
            sql.SQL(
                "CREATE TEMP TABLE IF NOT EXISTS {staging} "
                "(LIKE {target} INCLUDING DEFAULTS) ON COMMIT DROP"
            ).format(staging=staging_sql, target=target_sql)
        )
        # Clear staging in case it survived a previous iteration
        cur.execute(sql.SQL("TRUNCATE {staging}").format(staging=staging_sql))

        # 2. COPY chunk into staging
        buf = _df_to_copy_buffer(df)
        columns_sql = sql.SQL(", ").join(sql.Identifier(c) for c in df.columns)
        copy_stmt = sql.SQL(
            "COPY {staging} ({cols}) "
            "FROM STDIN WITH (FORMAT CSV, QUOTE '\"', NULL '')"
        ).format(staging=staging_sql, cols=columns_sql)
        cur.copy_expert(copy_stmt.as_string(conn), buf)

        # 3. Insert only rows whose hash is not already in the target
        cur.execute(
            sql.SQL(
                "INSERT INTO {target} "
                "SELECT s.* FROM {staging} s "
                "WHERE NOT EXISTS ("
                "    SELECT 1 FROM {target} t WHERE t._row_hash = s._row_hash"
                ")"
            ).format(target=target_sql, staging=staging_sql)
        )
        inserted = cur.rowcount

    conn.commit()
    return inserted


def truncate_table(conn: PgConnection, schema: str, table: str) -> None:
    """TRUNCATE *schema.table* — used at the start of a full_refresh run."""
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("TRUNCATE TABLE {}.{}").format(
                sql.Identifier(schema), sql.Identifier(table)
            )
        )
    conn.commit()
    logger.info("Truncated '%s'.'%s'.", schema, table)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_file(
    conn: PgConnection,
    csv_path: Path,
    schema: str,
    table: str,
    *,
    mode: str = "full_refresh",
    chunk_size: int = CHUNK_SIZE,
    rejects_dir: Optional[Path] = None,
) -> dict:
    """
    Load a single CSV file into *schema.table*.

    Parameters
    ----------
    conn : PgConnection
        An open psycopg2 connection (managed externally).
    csv_path : Path
        Absolute path to the source CSV file.
    schema : str
        Target Postgres schema.
    table : str
        Target Postgres table name.
    mode : str
        ``'full_refresh'``   — TRUNCATE then reload all rows.
        ``'incremental'``    — Skip rows whose _row_hash already exists.
    chunk_size : int
        Rows per COPY batch.
    rejects_dir : Path | None
        Directory for reject CSV files.  If None, rejects are only logged.

    Returns
    -------
    dict
        ``{rows_attempted, rows_loaded, rows_rejected, duration_seconds}``
    """
    t0 = time.monotonic()
    stats = {
        "rows_attempted": 0,
        "rows_loaded": 0,
        "rows_rejected": 0,
        "duration_seconds": 0.0,
    }
    reject_chunks: list[pd.DataFrame] = []
    _rejects_dir = rejects_dir or REJECTS_DIR

    # ---- Full-refresh: wipe the target before streaming ----
    if mode == "full_refresh":
        truncate_table(conn, schema, table)

    # ---- Stream CSV in chunks ----
    # dtype=str  → preserve every raw value exactly; Postgres coerces as needed.
    # keep_default_na=False + na_values=[""] → only treat blank cells as NULL.
    reader = pd.read_csv(
        csv_path,
        chunksize=chunk_size,
        dtype=str,
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
        encoding="utf-8",
        encoding_errors="replace",   # Degrade gracefully instead of crashing
    )

    for chunk_idx, chunk in enumerate(reader):
        chunk_rows = len(chunk)
        stats["rows_attempted"] += chunk_rows
        logger.debug(
            "[%s] Chunk %d: %d rows ingested so far …",
            table,
            chunk_idx + 1,
            stats["rows_attempted"],
        )

        # Attach metadata before any load attempt
        chunk = _add_metadata(chunk, csv_path.name)

        try:
            if mode == "full_refresh":
                inserted = _copy_chunk_to_table(conn, schema, table, chunk)
            else:  # incremental
                inserted = _copy_chunk_incremental(conn, schema, table, chunk)

            stats["rows_loaded"] += inserted

            if mode == "incremental":
                skipped = chunk_rows - inserted
                if skipped:
                    logger.debug(
                        "[%s] Chunk %d: skipped %d duplicate rows.",
                        table,
                        chunk_idx + 1,
                        skipped,
                    )

        except Exception as exc:
            conn.rollback()
            logger.error(
                "[%s] Chunk %d FAILED (%s). Writing %d rows to rejects.",
                table,
                chunk_idx + 1,
                exc,
                chunk_rows,
            )
            stats["rows_rejected"] += chunk_rows
            reject_chunks.append(chunk)

    # ---- Persist rejects ----
    if reject_chunks:
        _rejects_dir = Path(_rejects_dir)
        _rejects_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        reject_path = _rejects_dir / f"{table}_{ts}_rejects.csv"
        pd.concat(reject_chunks, ignore_index=True).to_csv(
            reject_path, index=False
        )
        logger.warning(
            "[%s] %d rejected rows written to: %s",
            table,
            stats["rows_rejected"],
            reject_path,
        )

    stats["duration_seconds"] = round(time.monotonic() - t0, 2)
    return stats
