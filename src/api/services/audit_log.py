"""Immutable, hash-chained audit log (FR-10 / US-401).

Every decision and every external (regulatory) call is appended as an
:class:`AuditEvent`. Each entry's ``entry_hash`` is SHA-256 over the previous
entry's hash plus this entry's canonical content, so the log is tamper-evident:
changing any historical payload invalidates the hash of that row and every row
after it. :func:`verify_chain` re-derives the chain and reports the first break.

The stored payload is the full evidence artefact, so an auditor can reconstruct
the complete reasoning for any decision from the log alone (AC-10).
"""

from __future__ import annotations

import hashlib
import json
import threading

from sqlalchemy.orm import Session

from ..models import AuditEvent

GENESIS_HASH = "0" * 64

# Appending is read-last-hash → insert → commit, which must be atomic or two
# concurrent appends could read the same prev_hash and fork the chain. This
# process-wide lock serialises appends so the chain stays linear under the
# threaded server / load test. (Multi-process deployments need a DB-level lock.)
_APPEND_LOCK = threading.Lock()


def _canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _compute_hash(prev_hash: str, event_type: str, payload_str: str) -> str:
    return hashlib.sha256(f"{prev_hash}|{event_type}|{payload_str}".encode("utf-8")).hexdigest()


def _last_hash(db: Session) -> str:
    last = db.query(AuditEvent).order_by(AuditEvent.id.desc()).first()
    return last.entry_hash if last else GENESIS_HASH


def append_event(
    db: Session,
    event_type: str,
    payload: dict,
    *,
    application_id: int | None = None,
    decision_record_id: int | None = None,
) -> AuditEvent:
    """Append one event to the chain and return it."""
    payload_str = _canonical(payload)
    with _APPEND_LOCK:  # atomic read-last-hash → insert → commit
        prev_hash = _last_hash(db)
        entry_hash = _compute_hash(prev_hash, event_type, payload_str)
        event = AuditEvent(
            event_type=event_type,
            application_id=application_id,
            decision_record_id=decision_record_id,
            payload_json=payload_str,
            prev_hash=prev_hash,
            entry_hash=entry_hash,
        )
        db.add(event)
        db.commit()
        db.refresh(event)
    return event


def verify_chain(db: Session) -> dict:
    """Recompute the whole chain; report integrity and the first broken row (if any)."""
    events = db.query(AuditEvent).order_by(AuditEvent.id.asc()).all()
    prev_hash = GENESIS_HASH
    for event in events:
        expected = _compute_hash(prev_hash, event.event_type, event.payload_json)
        if event.prev_hash != prev_hash or event.entry_hash != expected:
            return {"intact": False, "broken_at_id": event.id, "event_count": len(events)}
        prev_hash = event.entry_hash
    return {"intact": True, "broken_at_id": None, "event_count": len(events)}


def reconstruct(db: Session, decision_record_id: int) -> list[dict]:
    """Return the stored artefacts for a decision, for auditor reconstruction (AC-10)."""
    events = (
        db.query(AuditEvent)
        .filter(AuditEvent.decision_record_id == decision_record_id)
        .order_by(AuditEvent.id.asc())
        .all()
    )
    return [
        {"id": e.id, "event_type": e.event_type, "entry_hash": e.entry_hash,
         "payload": json.loads(e.payload_json), "created_at": e.created_at}
        for e in events
    ]
