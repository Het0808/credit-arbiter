"""
Credit Arbiter — raw data ingestion package.

Bronze / landing layer: CSV → PostgreSQL raw schema, zero transformation.
Every CSV column lands as TEXT (or its natural numeric type when unambiguous).
No business logic, no joins, no feature engineering lives here.
"""

__version__ = "0.1.0"
__author__ = "Abe Kuriachan"
