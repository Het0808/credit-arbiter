"""
Configuration for the Credit Arbiter ingestion pipeline.

All runtime knobs are driven by environment variables so the same script
runs unchanged against:
  - local Postgres (docker-compose)
  - AWS RDS / Aurora PostgreSQL
  - Azure Database for PostgreSQL
  - Google Cloud SQL
  - Any other Postgres-compatible host

Load order:
  1.  A .env file in the project root (python-dotenv, local dev only).
  2.  Real environment variables (CI/CD, ECS task, K8s Secret, etc.).
  Real env vars always win over .env values.

Nothing in this file is hardcoded beyond the documented local defaults.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Locate project root (two levels above this file: scripts/ingestion/ → scripts/ → root)
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Load .env from project root (silently ignored in prod where it won't exist)
load_dotenv(_PROJECT_ROOT / ".env", override=False)

# ===========================================================================
# Source data
# ===========================================================================
# The production source location (S3, Azure Blob, NFS mount, upstream API,
# etc.) is deliberately OUT OF SCOPE for this iteration.  This variable is
# expected to be overridden in every non-local environment.
RAW_DATA_DIR: Path = Path(
    os.environ.get(
        "RAW_DATA_DIR",
        str(_PROJECT_ROOT / "data" / "home_credit_data"),
    )
)

# ===========================================================================
# PostgreSQL connection
# ===========================================================================
DB_HOST: str = os.environ.get("DB_HOST", "localhost")
DB_PORT: int = int(os.environ.get("DB_PORT", "5432"))
DB_NAME: str = os.environ.get("DB_NAME", "credit_arbiter")
DB_USER: str = os.environ.get("DB_USER", "postgres")
DB_PASSWORD: str = os.environ.get("DB_PASSWORD", "")

# Default to "require" — safe for any managed Postgres service that enforces TLS.
# Set to "disable" only for local docker-compose where no TLS is configured.
# Never set to "disable" in production.
DB_SSLMODE: str = os.environ.get("DB_SSLMODE", "require")

# Target schema for all raw tables (created automatically if absent).
DB_SCHEMA: str = os.environ.get("DB_SCHEMA", "raw")

# Maintenance database used ONLY when DB_NAME does not yet exist and the
# script needs to run CREATE DATABASE.  This must be a database that already
# exists on the server — 'postgres' is the standard Postgres maintenance DB
# and is almost always available.  Override if your server restricts it.
DB_MAINTENANCE_DB: str = os.environ.get("DB_MAINTENANCE_DB", "postgres")

# ===========================================================================
# Operational paths
# ===========================================================================
# DDL files are written here for human review BEFORE being applied.
DDL_OUTPUT_DIR: Path = Path(
    os.environ.get("DDL_OUTPUT_DIR", str(_PROJECT_ROOT / "ddl"))
)

# Rows that fail to load are written here instead of being silently dropped.
REJECTS_DIR: Path = Path(
    os.environ.get("REJECTS_DIR", str(_PROJECT_ROOT / "rejects"))
)

# Path to the YAML file that maps CSV filenames → table names.
SOURCES_CONFIG: str = os.environ.get(
    "SOURCES_CONFIG",
    str(_PROJECT_ROOT / "config" / "sources.yaml"),
)

# ===========================================================================
# Performance tuning
# ===========================================================================
# Number of rows per COPY batch.  Tune per available RAM and network latency:
#   - Local / fast LAN  : 50 000–100 000
#   - High-latency cloud: 10 000–25 000
CHUNK_SIZE: int = int(os.environ.get("CHUNK_SIZE", "50000"))

# ===========================================================================
# Logging
# ===========================================================================
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO").upper()
