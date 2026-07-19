"""Load test & P95 latency certification (AC-9 / US-403).

Fires N concurrent /assess requests against an in-process TestClient (isolated
in-memory DB), records end-to-end latency per request, and certifies P95 <= 20s.
Also runs one instrumented assessment to attribute latency to each pipeline
stage, so a breached budget names the failing stage. Writes reports/ops/load_test.json.

Run:  python -m scripts.load_test [concurrency] [total_requests]
"""

import json
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.database import Base, get_db
from src.api.main import app
from src.api.models import Application

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "reports" / "ops" / "load_test.json"
P95_BUDGET_S = 20.0


def _make_client():
    # File-based sqlite so each worker thread gets its own connection; a busy
    # timeout makes concurrent writes wait (serialize) instead of erroring.
    db_path = Path(tempfile.mkdtemp()) / "loadtest.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    client.SessionLocal = TestingSessionLocal
    return client


def _seed(client, n=5):
    session = client.SessionLocal()
    ids = []
    for i in range(n):
        a = Application(external_id=f"load-{i}", loan_scheme="Personal Loan", amt_income_total=250000,
                        amt_credit=300000, amt_annuity=15000, days_employed=-3650, status="COMPLETE")
        session.add(a)
        session.commit()
        session.refresh(a)
        ids.append(a.id)
    session.close()
    return ids


def _percentile(values, pct):
    if not values:
        return None
    s = sorted(values)
    k = min(len(s) - 1, int(round((pct / 100) * (len(s) - 1))))
    return s[k]


def _stage_timings(app_id):
    """Time each pipeline stage for one application (attribution)."""
    from src.api.models import Application as App
    from src.api.services import assessment as A

    timings = {}
    # Fresh single-threaded in-memory db with one seeded app.
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    application = App(external_id="stage-1", loan_scheme="Personal Loan", amt_income_total=250000,
                      amt_credit=300000, amt_annuity=15000, days_employed=-3650, status="COMPLETE")
    session.add(application); session.commit(); session.refresh(application)
    profile = A._profile_from_application(application)

    def _t(label, fn):
        t0 = time.perf_counter(); fn(); timings[label] = round((time.perf_counter() - t0) * 1000, 2)

    _t("scoring_ms", lambda: A.score_application(profile))
    _t("retrieval_ms", lambda: A.retrieve_for_profile(profile, scheme="Personal Loan"))
    _t("regulatory_ms", lambda: A.verify_regulatory(application.external_id))
    _t("document_ms", lambda: A.verify_documents(session, application))
    session.close()
    return timings


def main():
    concurrency = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    total = int(sys.argv[2]) if len(sys.argv) > 2 else concurrency
    client = _make_client()
    ids = _seed(client)

    # Authenticate once (registration returns an access token).
    reg = client.post("/api/auth/register", json={"email": "load@test.com", "password": "loadpass123"})
    headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}

    def one(i):
        t0 = time.perf_counter()
        r = client.post("/api/assess", json={"application_id": ids[i % len(ids)]}, headers=headers)
        return time.perf_counter() - t0, r.status_code

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        results = list(ex.map(one, range(total)))
    wall = time.perf_counter() - t_start

    latencies = [r[0] for r in results]
    errors = sum(1 for r in results if r[1] != 200)
    p95 = _percentile(latencies, 95)

    report = {
        "concurrency": concurrency,
        "total_requests": total,
        "errors": errors,
        "wall_time_s": round(wall, 3),
        "throughput_rps": round(total / wall, 2) if wall else None,
        "p50_latency_s": round(_percentile(latencies, 50), 4),
        "p95_latency_s": round(p95, 4),
        "max_latency_s": round(max(latencies), 4),
        "p95_budget_s": P95_BUDGET_S,
        "p95_within_budget": p95 <= P95_BUDGET_S,
        "stage_attribution_ms": _stage_timings(ids[0]),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    app.dependency_overrides.clear()

    print(json.dumps(report, indent=2))
    if not report["p95_within_budget"]:
        slowest = max(report["stage_attribution_ms"].items(), key=lambda kv: kv[1])
        print(f"\nP95 BUDGET BREACHED. Slowest stage: {slowest[0]} ({slowest[1]}ms)")
        return 1
    print(f"\nP95 {p95:.3f}s within {P95_BUDGET_S}s budget - CERTIFIED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
