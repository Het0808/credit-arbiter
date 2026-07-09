import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db")

if DATABASE_URL.startswith("sqlite"):
    # ponytail: default pool_size=5+max_overflow=10=15 connections can't
    # serve 50 concurrent /assess calls (see scripts/load_test.py), so bump
    # the pool. `timeout` is sqlite3's busy-wait: concurrent writers queue
    # instead of raising "database is locked" immediately. (StaticPool - one
    # shared raw connection across threads - looks tempting but isn't
    # thread-safe even with check_same_thread=False; don't.) SQLite is still
    # single-writer at the file level; real concurrency needs Postgres
    # (DATABASE_URL already supports it, see claude.md).
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False, "timeout": 15},
        pool_size=50,
        max_overflow=0,
    )
else:
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
