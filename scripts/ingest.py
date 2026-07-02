#!/usr/bin/env python3
"""
Credit Arbiter — raw data ingestion pipeline entrypoint.

Run from the project root:

    python scripts/ingest.py --help
    python scripts/ingest.py --dry-run
    python scripts/ingest.py --mode full_refresh
    python scripts/ingest.py --mode incremental --tables raw_bureau

All options and environment variables are documented in docs/INGESTION.md.
The ingestion package lives alongside this file under scripts/ingestion/.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add the scripts/ directory to sys.path so that `import ingestion` resolves
# to scripts/ingestion/ without requiring an editable install.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from ingestion.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
