"""Load test for AC-9 (P95 <= 20s @ 50 concurrent /api/assess calls).

Not a certification - run this yourself against a running instance and read
the printed P95 against the NFR budget. Stdlib only.

    python -m scripts.load_test [--concurrency 50] [--requests 200]
"""

import argparse
import time
import urllib.request
import json
from concurrent.futures import ThreadPoolExecutor

BASE = "http://localhost:8000/api"
DEMO_LOGIN = {"username": "underwriter@halcyon.com", "password": "halcyon-demo-1"}


def _post(path, data, token=None, form=False):
    headers = {"Content-Type": "application/x-www-form-urlencoded" if form else "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = "&".join(f"{k}={v}" for k, v in data.items()).encode() if form else json.dumps(data).encode()
    req = urllib.request.Request(f"{BASE}{path}", data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _one_assess(token, application_id):
    start = time.perf_counter()
    try:
        _post("/assess", {"application_id": application_id}, token=token)
        return time.perf_counter() - start
    except Exception as e:
        print(f"request failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--requests", type=int, default=200)
    args = parser.parse_args()

    token = _post("/auth/login", DEMO_LOGIN, form=True)["access_token"]
    req = urllib.request.Request(f"{BASE}/applications", headers={"Authorization": f"Bearer {token}"})
    app_ids = [a["id"] for a in json.loads(urllib.request.urlopen(req).read())]

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        durations = list(pool.map(lambda i: _one_assess(token, app_ids[i % len(app_ids)]), range(args.requests)))

    durations = sorted(d for d in durations if d is not None)
    if not durations:
        print("all requests failed")
        return
    p95 = durations[int(len(durations) * 0.95)]
    print(f"{len(durations)}/{args.requests} succeeded")
    print(f"avg={sum(durations)/len(durations):.2f}s  p95={p95:.2f}s  max={max(durations):.2f}s")
    print(f"AC-9 (P95 <= 20s @ {args.concurrency} concurrent): {'PASS' if p95 <= 20 else 'FAIL'}")


if __name__ == "__main__":
    main()
