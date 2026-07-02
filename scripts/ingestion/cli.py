"""
CLI entrypoint for the Credit Arbiter data ingestion pipeline.

Quick reference
---------------
    # Show all options
    python scripts/ingest.py --help

    # Generate DDL files for review, do NOT touch the database
    python scripts/ingest.py --dry-run

    # Apply DDL (create tables) only — no data load
    python scripts/ingest.py --ddl-only

    # Full reload of all tables (truncate + re-insert)
    python scripts/ingest.py --mode full_refresh

    # Incremental load — skip rows already present (by _row_hash)
    python scripts/ingest.py --mode incremental

    # Reload specific tables only
    python scripts/ingest.py --mode full_refresh --tables raw_bureau,raw_application_train

    # Verbose debug logging
    python scripts/ingest.py --log-level DEBUG

Environment variables (see .env.example)
-----------------------------------------
    RAW_DATA_DIR   — where the source CSV files live
    DB_HOST / DB_PORT / DB_NAME / DB_USER / DB_PASSWORD / DB_SSLMODE
    DB_SCHEMA      — target Postgres schema  (default: raw)
    CHUNK_SIZE     — rows per COPY batch     (default: 50 000)
    DDL_OUTPUT_DIR — where .sql DDL files go (default: ./ddl)
    REJECTS_DIR    — where reject CSVs go    (default: ./rejects)
    SOURCES_CONFIG — path to sources.yaml    (default: config/sources.yaml)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

from .config import (
    DB_NAME,
    DB_SCHEMA,
    DDL_OUTPUT_DIR,
    LOG_LEVEL,
    RAW_DATA_DIR,
    REJECTS_DIR,
    SOURCES_CONFIG,
)
from .db import ensure_database, ensure_schema, managed_connection
from .loader import load_file
from .schema_manager import (
    apply_ddl,
    check_schema_drift,
    generate_ddl,
    get_existing_columns,
    write_ddl_file,
)

logger = logging.getLogger("ingestion.cli")


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s  [%(levelname)-8s]  %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )


# ---------------------------------------------------------------------------
# Sources config loader
# ---------------------------------------------------------------------------

def _load_sources(config_path: str) -> list[dict]:
    """Parse sources.yaml and return a list of enabled source entries."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Sources config not found: {path}\n"
            "Set the SOURCES_CONFIG env var or pass --config."
        )
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return [
        s for s in data.get("sources", [])
        if s.get("enabled", True)      # missing 'enabled' → True
    ]


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ingest",
        description=(
            "Credit Arbiter — CSV → PostgreSQL bronze-layer ingestion.\n\n"
            "All connection and path parameters are driven by environment\n"
            "variables (see .env.example).  No code changes are needed to\n"
            "point this script at a different database or source directory."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=SOURCES_CONFIG,
        metavar="PATH",
        help="Path to sources YAML config  (default: %(default)s)",
    )
    parser.add_argument(
        "--mode",
        choices=["full_refresh", "incremental"],
        default="full_refresh",
        help=(
            "full_refresh: truncate each table then reload all rows.\n"
            "incremental : skip rows already present (via _row_hash).\n"
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--tables",
        default=None,
        metavar="T1,T2,…",
        help=(
            "Comma-separated list of target table names to process.\n"
            "Omit to process all enabled sources in the config."
        ),
    )
    parser.add_argument(
        "--schema",
        default=DB_SCHEMA,
        help="Target Postgres schema  (default: %(default)s; overrides DB_SCHEMA)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Generate DDL files and print the execution plan, but do NOT\n"
            "connect to the database or load any data."
        ),
    )
    parser.add_argument(
        "--ddl-only",
        action="store_true",
        help=(
            "Apply DDL (CREATE TABLE / INDEX) for all configured tables,\n"
            "then exit without loading any rows."
        ),
    )
    parser.add_argument(
        "--log-level",
        default=LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity  (default: %(default)s; overrides LOG_LEVEL)",
    )
    return parser


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:  # noqa: C901
    args = _build_parser().parse_args(argv)
    _setup_logging(args.log_level)

    schema = args.schema

    logger.info("=" * 60)
    logger.info("  Credit Arbiter  —  Ingestion Pipeline")
    logger.info("=" * 60)
    logger.info("Mode          : %s", args.mode)
    logger.info("Schema        : %s", schema)
    logger.info("Source dir    : %s", RAW_DATA_DIR)
    logger.info("Config        : %s", args.config)
    logger.info("DDL output    : %s", DDL_OUTPUT_DIR)
    logger.info("Rejects dir   : %s", REJECTS_DIR)
    logger.info("Chunk size    : %s rows", args.__dict__.get("chunk_size", "env default"))
    logger.info("Dry run       : %s", args.dry_run)
    logger.info("DDL only      : %s", args.ddl_only)
    logger.info("=" * 60)

    # ---- Load source map ----
    try:
        sources = _load_sources(args.config)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    if not sources:
        logger.error("No enabled sources found in config: %s", args.config)
        return 1

    # ---- Apply --tables filter ----
    if args.tables:
        allowed = {t.strip() for t in args.tables.split(",")}
        sources = [s for s in sources if s["table"] in allowed]
        if not sources:
            logger.error(
                "None of the requested tables match enabled entries in config.\n"
                "Requested: %s",
                args.tables,
            )
            return 1

    # ---- Phase 1: DDL generation (always runs, even in dry-run) ----
    logger.info("--- Phase 1: DDL generation ---")
    ddl_map: dict[str, tuple[Path, str]] = {}  # table → (csv_path, ddl_text)

    for source in sources:
        csv_name: str = source["file"]
        table: str = source["table"]
        csv_path = RAW_DATA_DIR / csv_name

        if not csv_path.exists():
            logger.warning("Source file not found — skipping: %s", csv_path)
            continue

        logger.info("Sniffing schema: %s", csv_path.name)
        # Read a small sample for type inference. Use latin-1 (a superset of
        # ASCII that never raises UnicodeDecodeError) so that files with
        # Windows-1252 or other 8-bit encodings (e.g. HomeCredit_columns_description.csv)
        # don't abort DDL generation.  The actual load uses encoding_errors='replace'.
        try:
            sample = pd.read_csv(
                csv_path,
                nrows=5_000,      # larger sample → more reliable int-vs-float inference
                low_memory=False,
                keep_default_na=False,
                na_values=[""],
                encoding="latin-1",
            )
        except Exception as exc:
            logger.error("Could not sniff schema for %s: %s — skipping.", csv_path.name, exc)
            continue

        ddl = generate_ddl(schema, table, sample)
        ddl_path = write_ddl_file(ddl, table, DDL_OUTPUT_DIR)
        ddl_map[table] = (csv_path, ddl)
        logger.info("  [OK] DDL ready: %s -> %s", table, ddl_path.name)

    if not ddl_map:
        logger.error("No source files found.  Check RAW_DATA_DIR=%s", RAW_DATA_DIR)
        return 1

    if args.dry_run:
        logger.info(
            "Dry run complete.  %d DDL file(s) written to: %s",
            len(ddl_map),
            DDL_OUTPUT_DIR,
        )
        logger.info("Review them, then re-run without --dry-run to execute.")
        return 0

    # ---- Phase 2: Connect and bootstrap database + schema ----
    logger.info("--- Phase 2: Database + schema bootstrap ---")

    # ensure_database() connects to DB_MAINTENANCE_DB (default: 'postgres'),
    # checks pg_database, and issues CREATE DATABASE if needed — all with
    # autocommit=True (required by Postgres for database-level DDL).
    try:
        ensure_database()
    except Exception as exc:
        logger.error(
            "Could not ensure database '%s' exists: %s\n"
            "If you are on a managed service (RDS, Azure, Cloud SQL), "
            "create the database manually and re-run.",
            DB_NAME, exc,
        )
        return 1
    with managed_connection() as conn:
        ensure_schema(conn, schema)

        # ---- Phase 3: Apply DDL for each table ----
        logger.info("--- Phase 3: DDL application ---")
        for source in sources:
            table = source["table"]
            if table not in ddl_map:
                continue   # file was missing — already warned above

            csv_path, ddl = ddl_map[table]

            # Schema drift check (only meaningful if table already exists)
            existing_cols = get_existing_columns(conn, schema, table)
            if existing_cols:
                sample_hdr = pd.read_csv(csv_path, nrows=1,
                                         keep_default_na=False, na_values=[""],
                                         encoding="latin-1")
                for warning in check_schema_drift(
                    existing_cols, list(sample_hdr.columns), table
                ):
                    logger.warning(warning)

            apply_ddl(
                conn, ddl, table,
                schema=schema,
                drop_first=(args.mode == "full_refresh"),
            )

        if args.ddl_only:
            logger.info("DDL-only mode complete.  No rows loaded.")
            return 0

        # ---- Phase 4: Data load ----
        logger.info("--- Phase 4: Data load [mode=%s] ---", args.mode)

        total_loaded = 0
        total_rejected = 0
        failures = []

        for source in sources:
            table = source["table"]
            description = source.get("description", "")
            if table not in ddl_map:
                continue

            csv_path, _ = ddl_map[table]
            logger.info(
                "Loading  %s -> %s.%s  %s",
                csv_path.name,
                schema,
                table,
                f"({description})" if description else "",
            )

            try:
                stats = load_file(
                    conn,
                    csv_path,
                    schema,
                    table,
                    mode=args.mode,
                    rejects_dir=REJECTS_DIR,
                )
            except Exception as exc:
                logger.error("[%s] Load aborted: %s", table, exc)
                failures.append(table)
                continue

            total_loaded += stats["rows_loaded"]
            total_rejected += stats["rows_rejected"]

            status = "[OK]  " if stats["rows_rejected"] == 0 else "[WARN]"
            logger.info(
                "  %s  %-40s  loaded=%d  rejected=%d  time=%.2fs",
                status,
                table,
                stats["rows_loaded"],
                stats["rows_rejected"],
                stats["duration_seconds"],
            )

    # ---- Summary ----
    logger.info("=" * 60)
    logger.info("  Ingestion complete")
    logger.info("  Total rows loaded   : %d", total_loaded)
    logger.info("  Total rows rejected : %d", total_rejected)
    if total_rejected:
        logger.info("  Reject files        : %s", REJECTS_DIR)
    if failures:
        logger.error("  Tables with errors  : %s", failures)
    logger.info("=" * 60)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
