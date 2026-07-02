# Credit Arbiter — Raw Data Ingestion Pipeline

> **Bronze / landing layer** — raw CSV → PostgreSQL `raw` schema, zero transformation.

This document covers everything you need to run, configure, and extend the ingestion
pipeline.  No code changes are required to point the script at a different database
or a different source folder — every tunable is an environment variable.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Quick Start (local)](#quick-start-local)
3. [Environment Variables Reference](#environment-variables-reference)
4. [CLI Reference](#cli-reference)
5. [Switching to a Different Postgres Instance](#switching-to-a-different-postgres-instance)
6. [Switching to a Different Source Folder](#switching-to-a-different-source-folder)
7. [Table Layout](#table-layout)
8. [Load Modes](#load-modes)
9. [DDL Review Workflow](#ddl-review-workflow)
10. [Schema Drift Handling](#schema-drift-handling)
11. [Rejects / Error Handling](#rejects--error-handling)
12. [Adding New Source Files](#adding-new-source-files)
13. [Out of Scope](#out-of-scope)
14. [Roadmap — Future Extensions](#roadmap--future-extensions)

---

## Architecture Overview

```
RAW_DATA_DIR/
 ├── application_train.csv
 ├── bureau.csv
 └── …
        │
        │  pandas (chunked, dtype=str)
        ▼
 [ingestion pipeline]
   1. Sniff schema from sample rows
   2. Generate DDL  →  ddl/<table>.sql   ← human review point
   3. Apply DDL     →  CREATE TABLE IF NOT EXISTS
   4. Stream chunks →  COPY FROM STDIN   (psycopg2, bulk protocol)
        │
        │  TCP / TLS (DB_SSLMODE)
        ▼
 PostgreSQL
  └── raw schema
       ├── raw_application_train
       ├── raw_bureau
       └── …   (+ _ingested_at, _source_file, _row_hash on every table)
```

**Why TEXT for most columns?**
A bronze layer is a raw mirror of the source.  Casting `"365243"` to `BIGINT` at
ingest time hides the fact that it's a sentinel value used by Home Credit to denote
"employed since forever".  Downstream silver/gold layers apply business-logic casts
with full context.

---

## Quick Start (local)

### 1. Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.10+ |
| PostgreSQL | 13+ (local docker or managed) |

### 2. Install dependencies

```bash
pip install -r requirements-ingestion.txt
```

### 3. Create your local `.env`

```bash
cp .env.example .env
# Edit .env — set DB_PASSWORD and confirm DB_HOST/DB_NAME
```

### 4. Create the target database (first time only)

```sql
-- Run as a Postgres superuser
CREATE DATABASE credit_arbiter;
```

### 5. Dry run — generate DDL files without touching the DB

```bash
python scripts/ingest.py --dry-run
```

Review the generated SQL files in `ddl/`.

### 6. Apply DDL (create tables) — no data loaded

```bash
python scripts/ingest.py --ddl-only
```

### 7. Load all data

```bash
python scripts/ingest.py --mode full_refresh
```

---

## Environment Variables Reference

Set these in `.env` for local dev, or as real environment variables in production.
Real env vars always override `.env` values.

| Variable | Default | Description |
|----------|---------|-------------|
| `RAW_DATA_DIR` | `./data/home_credit_data` | Directory containing source CSV files |
| `DB_HOST` | `localhost` | Postgres hostname or IP |
| `DB_PORT` | `5432` | Postgres port |
| `DB_NAME` | `credit_arbiter` | Target database name |
| `DB_USER` | `postgres` | Postgres username |
| `DB_PASSWORD` | *(empty)* | Postgres password — **never hardcode** |
| `DB_SSLMODE` | `require` | `disable` / `require` / `verify-ca` / `verify-full` |
| `DB_SCHEMA` | `raw` | Target schema; created automatically if absent |
| `SOURCES_CONFIG` | `config/sources.yaml` | Path to the CSV→table mapping file |
| `DDL_OUTPUT_DIR` | `ddl/` | Where generated `.sql` DDL files are written |
| `REJECTS_DIR` | `rejects/` | Where failed-row CSV files are written |
| `CHUNK_SIZE` | `50000` | Rows per COPY batch |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

> **SSL default is `require`** — safe for any managed Postgres service.
> Change to `disable` only for local docker-compose where TLS is not configured.

---

## CLI Reference

```
python scripts/ingest.py [OPTIONS]
```

| Flag | Default | Effect |
|------|---------|--------|
| `--config PATH` | `SOURCES_CONFIG` env var | Override sources YAML location |
| `--mode {full_refresh,incremental}` | `full_refresh` | Load strategy (see below) |
| `--tables T1,T2,…` | all enabled | Process only the named tables |
| `--schema NAME` | `DB_SCHEMA` env var | Override target schema |
| `--dry-run` | off | Generate DDL files only — no DB connection |
| `--ddl-only` | off | Apply DDL, then exit — no rows loaded |
| `--log-level LEVEL` | `LOG_LEVEL` env var | Verbosity |

### Examples

```bash
# See what would happen — generates ddl/*.sql, no DB writes
python scripts/ingest.py --dry-run

# Create all tables (idempotent), then exit
python scripts/ingest.py --ddl-only

# Full reload of every table
python scripts/ingest.py --mode full_refresh

# Incremental load of two tables only
python scripts/ingest.py --mode incremental --tables raw_bureau,raw_bureau_balance

# Load into a non-default schema
python scripts/ingest.py --schema staging --mode full_refresh

# Debug a single table with verbose output
python scripts/ingest.py --mode full_refresh --tables raw_application_train --log-level DEBUG
```

---

## Switching to a Different Postgres Instance

**No code changes needed.**  Update only the DB_ variables:

### AWS RDS / Aurora PostgreSQL

```bash
# In .env or as real env vars in your ECS task / Lambda
DB_HOST=my-cluster.cluster-xxxxxxxxxxxx.us-east-1.rds.amazonaws.com
DB_PORT=5432
DB_NAME=credit_arbiter
DB_USER=arbiter_ingest
DB_PASSWORD=<from Secrets Manager>
DB_SSLMODE=require
```

### Azure Database for PostgreSQL (Flexible Server)

```bash
DB_HOST=my-server.postgres.database.azure.com
DB_PORT=5432
DB_NAME=credit_arbiter
DB_USER=arbiter_ingest@my-server
DB_PASSWORD=<from Key Vault>
DB_SSLMODE=require
```

### Google Cloud SQL (PostgreSQL)

```bash
DB_HOST=<Cloud SQL public IP or private IP>
DB_PORT=5432
DB_NAME=credit_arbiter
DB_USER=arbiter_ingest
DB_PASSWORD=<from Secret Manager>
DB_SSLMODE=require
```

### Local docker-compose

```bash
DB_HOST=localhost
DB_PORT=5432
DB_NAME=credit_arbiter
DB_USER=postgres
DB_PASSWORD=postgres
DB_SSLMODE=disable   # no TLS in docker-compose by default
```

---

## Switching to a Different Source Folder

Set `RAW_DATA_DIR` to any local path.  The same override works for mounted volumes
in containers:

```bash
# Local alternative path
RAW_DATA_DIR=/mnt/data/home_credit

# Docker run — mount a host directory
docker run -e RAW_DATA_DIR=/data -v /host/csvs:/data my-image python scripts/ingest.py

# Kubernetes — mount a PVC
env:
  - name: RAW_DATA_DIR
    value: /data/csvs
volumeMounts:
  - mountPath: /data/csvs
    name: csv-pvc
```

---

## Table Layout

Every raw table is created under the configured schema (default: `raw`) and has
the following structure:

```sql
CREATE TABLE IF NOT EXISTS "raw"."raw_application_train" (
    -- All original CSV columns (TEXT or numeric, conservatively typed)
    "SK_ID_CURR"          DOUBLE PRECISION,
    "NAME_CONTRACT_TYPE"  TEXT,
    -- … (all source columns)

    -- Standard metadata columns (appended by the pipeline, never in the CSV)
    _ingested_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    _source_file  TEXT         NOT NULL,
    _row_hash     TEXT         NOT NULL
);

CREATE INDEX IF NOT EXISTS "raw_application_train__row_hash_idx"
    ON "raw"."raw_application_train" (_row_hash);
```

### Metadata columns

| Column | Type | Purpose |
|--------|------|---------|
| `_ingested_at` | `TIMESTAMPTZ` | UTC timestamp of the load run |
| `_source_file` | `TEXT` | Originating CSV filename (not full path) |
| `_row_hash` | `TEXT` | MD5 of all data column values — used for dedup and audit |

---

## Load Modes

### `full_refresh` (default)

```
TRUNCATE raw.raw_application_train;
COPY raw.raw_application_train (...) FROM STDIN …  -- chunk 1
COPY raw.raw_application_train (...) FROM STDIN …  -- chunk 2
…
```

Use when: the source CSV is a complete snapshot and you want a clean reload.

### `incremental`

For each chunk:
1. COPY chunk into a session-scoped temp staging table.
2. INSERT INTO target only rows whose `_row_hash` does NOT already exist.
3. DROP the staging table.

```sql
INSERT INTO "raw"."raw_bureau"
SELECT s.* FROM _ingestion_staging s
WHERE NOT EXISTS (
    SELECT 1 FROM "raw"."raw_bureau" t WHERE t._row_hash = s._row_hash
);
```

Use when: you want to append new rows from a re-export without duplicating rows
that were already loaded.  The dedup is fully server-side — no Python-side hash
set — so it scales to hundreds of millions of rows.

---

## DDL Review Workflow

1.  Run `--dry-run` (no DB connection):
    ```bash
    python scripts/ingest.py --dry-run
    ```
2.  Review files in `ddl/` — one `.sql` file per table.
3.  Optionally commit the DDL files to version control for an audit trail.
4.  Run `--ddl-only` to apply DDL without loading data:
    ```bash
    python scripts/ingest.py --ddl-only
    ```
5.  Run the full load:
    ```bash
    python scripts/ingest.py --mode full_refresh
    ```

The pipeline will never auto-apply DDL silently — it always writes the file first.

---

## Schema Drift Handling

When a table already exists and the incoming CSV has a different set of columns,
the pipeline logs warnings:

```
[SCHEMA DRIFT] raw_bureau: CSV has new columns not in DB table — ['new_col'].
               Run --ddl-only after updating sources.yaml to add them.

[SCHEMA DRIFT] raw_bureau: DB table has columns absent from CSV — ['old_col'].
               Those columns will remain NULL for this load.
```

The pipeline does NOT auto-ALTER the table.  To handle drift:
- **New column in CSV**: regenerate DDL (`--dry-run`), review the `.sql` file,
  then manually `ALTER TABLE … ADD COLUMN` or drop-and-recreate.
- **Column removed from CSV**: the DB column is simply not written — it receives
  NULL for that load.

---

## Rejects / Error Handling

If a COPY batch fails (encoding error, value too long, constraint violation), the
offending chunk is:
1.  Logged with the error message.
2.  Written to `rejects/<table_name>_<YYYYMMDDTHHMMSSZ>_rejects.csv`.

The pipeline continues to the next chunk.  At the end, a summary shows total
`rows_loaded` vs `rows_rejected`.

To investigate rejects:
```bash
ls rejects/
# raw_bureau_20260101T120000Z_rejects.csv

# Open in any CSV viewer or load into a scratch table for investigation
```

---

## Adding New Source Files

1.  Place the new CSV in `RAW_DATA_DIR`.
2.  Add an entry to `config/sources.yaml`:
    ```yaml
    - file: my_new_table.csv
      table: raw_my_new_table
      description: "Description of what this table contains"
      enabled: true
    ```
3.  Run `--dry-run` to generate and review the DDL.
4.  Run `--ddl-only` to create the table.
5.  Run the full load.

**No Python changes required.**

---

## Out of Scope

> This section documents limitations explicitly so they are not discovered in production.

### Production source-data location

**This script operates on locally-accessible CSV files only.**

In production, the raw data will originate from one or more upstream sources —
an Loan Origination System (LOS) export, a managed S3 bucket, an Azure Blob
Storage container, a SFTP drop, an upstream API, etc.  **The mechanism for
getting data from that upstream source into a directory accessible to this
script is entirely out of scope for this iteration.**

Specifically, the following are NOT handled here:
- Downloading files from S3 / Azure Blob / GCS
- Authenticating to an upstream API
- Streaming data from a Kafka / Kinesis topic
- Decrypting PGP-encrypted drops
- Decompressing `.gz` / `.zip` archives automatically

The expected pattern is that a separate "data acquisition" step (a managed ETL
service, a cron job, a cloud function, etc.) places the CSV files into a
directory (local disk, EFS mount, FUSE-mounted object storage, etc.) and then
invokes this script via `RAW_DATA_DIR=<path> python scripts/ingest.py`.

### Cloud-native authentication

Standard Postgres username/password authentication is used exclusively.  The
following are intentionally NOT implemented in this iteration:

| Mechanism | Status |
|-----------|--------|
| AWS IAM database authentication | ❌ Out of scope |
| Azure Active Directory auth | ❌ Out of scope |
| Google Cloud SQL IAM auth | ❌ Out of scope |
| AWS Secrets Manager auto-rotation | ❌ Out of scope |
| HashiCorp Vault dynamic credentials | ❌ Out of scope |

See [Roadmap](#roadmap--future-extensions) for the planned extension path.

### Transformation / business logic

This script is a **bronze layer only**.  It performs zero transformation:
- No column renaming
- No type coercion beyond what Postgres does implicitly during COPY
- No NULL filling
- No feature engineering
- No joins

All of that lives in the silver and gold layers downstream.

---

## Roadmap — Future Extensions

| Extension | Notes |
|-----------|-------|
| AWS IAM auth | Replace password with `generate_db_auth_token()` from `boto3` |
| Azure AD auth | Use `azure-identity` to mint an access token as the password |
| S3 source adapter | `boto3.download_file` → temp dir → existing pipeline |
| Azure Blob adapter | `azure-storage-blob` → temp dir → existing pipeline |
| Compressed file support | Auto-detect `.gz` and pass `compression='gzip'` to `pd.read_csv` |
| Parallel table loading | `concurrent.futures.ThreadPoolExecutor` over the sources list |
| Great Expectations integration | Add data quality checkpoint after DDL, before load |
| dbt snapshot compatibility | Ensure `_row_hash` column name aligns with dbt snapshot strategy |
