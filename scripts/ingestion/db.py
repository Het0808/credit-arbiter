"""
Database connection management for the ingestion pipeline.

Provides:
  - ensure_database()        → CREATE DATABASE if it doesn't exist (autocommit)
  - get_connection()         → raw psycopg2 connection (caller manages lifecycle)
  - managed_connection()     → context manager with auto-rollback on error
  - ensure_schema()          → CREATE SCHEMA IF NOT EXISTS (idempotent)
  - table_exists()           → quick existence check
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator

import psycopg2
import psycopg2.extras
from psycopg2 import sql
from psycopg2.extensions import connection as PgConnection

from .config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, DB_SSLMODE, DB_MAINTENANCE_DB

logger = logging.getLogger(__name__)


def _make_dsn(dbname: str | None = None) -> dict:
    """Return connection kwargs.  Password excluded from logs — never call repr() on this."""
    return dict(
        host=DB_HOST,
        port=DB_PORT,
        dbname=dbname or DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        sslmode=DB_SSLMODE,
        connect_timeout=30,
        # Keep connections alive on high-latency managed services
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5,
    )


def ensure_database() -> None:
    """
    Create the target database (DB_NAME) if it does not already exist.

    Why this is non-trivial in PostgreSQL
    --------------------------------------
    You cannot run ``CREATE DATABASE`` inside a transaction block, and you
    must already be connected to *some* existing database before you can
    issue any SQL.  The solution:

    1.  Connect to ``DB_MAINTENANCE_DB`` (default: ``postgres``) — a database
        that is guaranteed to exist on every standard Postgres installation.
    2.  Query ``pg_database`` to check whether ``DB_NAME`` already exists.
    3.  If not, run ``CREATE DATABASE`` with ``autocommit = True`` (the only
        way Postgres allows DDL that creates a database).
    4.  Close the maintenance connection — all subsequent work uses DB_NAME.

    On managed services (RDS, Azure, Cloud SQL) the user must have the
    ``CREATEDB`` privilege granted to run this.  If they don't, have a DBA
    create the database manually and skip this step.
    """
    maintenance_dsn = _make_dsn(dbname=DB_MAINTENANCE_DB)
    logger.info(
        "Checking whether database '%s' exists on %s:%s …",
        DB_NAME, DB_HOST, DB_PORT,
    )

    conn = psycopg2.connect(**maintenance_dsn)
    try:
        # autocommit required — CREATE DATABASE cannot run in a transaction
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (DB_NAME,),
            )
            exists = cur.fetchone() is not None

        if exists:
            logger.info("Database '%s' already exists — nothing to do.", DB_NAME)
        else:
            logger.info(
                "Database '%s' not found — creating it now "
                "(connected via maintenance DB '%s') …",
                DB_NAME, DB_MAINTENANCE_DB,
            )
            with conn.cursor() as cur:
                # sql.Identifier safely quotes the name, preventing injection
                cur.execute(
                    sql.SQL("CREATE DATABASE {}").format(sql.Identifier(DB_NAME))
                )
            logger.info("Database '%s' created successfully.", DB_NAME)
    finally:
        conn.close()


def get_connection() -> PgConnection:
    """
    Open and return a psycopg2 connection.
    Caller is responsible for calling conn.close() when done.
    """
    dsn = _make_dsn()
    logger.info(
        "Connecting to host=%s  port=%s  dbname=%s  user=%s  sslmode=%s",
        dsn["host"],
        dsn["port"],
        dsn["dbname"],
        dsn["user"],
        dsn["sslmode"],
    )
    conn = psycopg2.connect(**dsn)
    conn.autocommit = False
    return conn


@contextmanager
def managed_connection() -> Generator[PgConnection, None, None]:
    """
    Context manager that opens a connection, yields it, and guarantees
    rollback on any exception before closing.

    Usage::

        with managed_connection() as conn:
            do_work(conn)
    """
    conn = get_connection()
    try:
        yield conn
    except Exception:
        logger.exception("Rolling back transaction due to error.")
        conn.rollback()
        raise
    finally:
        conn.close()
        logger.debug("Database connection closed.")


def ensure_schema(conn: PgConnection, schema: str) -> None:
    """
    Create *schema* if it doesn't already exist.
    Uses psycopg2.sql to safely quote the identifier — no SQL injection risk.
    """
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema))
        )
    conn.commit()
    logger.info("Schema '%s' is ready.", schema)


def table_exists(conn: PgConnection, schema: str, table: str) -> bool:
    """Return True if *schema.table* exists in the current database."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM   information_schema.tables
            WHERE  table_schema = %s
              AND  table_name   = %s
            """,
            (schema, table),
        )
        return cur.fetchone() is not None
