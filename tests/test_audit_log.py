import json

from src.api import audit_log


def test_log_external_call_writes_one_json_line(tmp_path, monkeypatch):
    log_path = tmp_path / "external_calls.log"
    monkeypatch.setattr(audit_log, "_LOG_PATH", str(log_path))

    audit_log.log_external_call("groq", model="llama-3.1-8b-instant", success=True, cost_usd=0.00003)

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["service"] == "groq"
    assert entry["success"] is True
    assert entry["cost_usd"] == 0.00003
    assert "timestamp" in entry
