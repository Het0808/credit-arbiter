"""Global kill-switch & degraded-mode state (PRD §11 / US-405).

A single operator-controlled flag stored in system_flag. When active, the
assessment service must route every new application to human review rather than
auto-deciding (enforced in assessment.py). Reads are cheap (one indexed row),
so the switch takes effect on the very next assessment - well within the 60s
requirement.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from ..models import SystemFlag

KILL_SWITCH_KEY = "global_kill_switch"


def _flag(db: Session) -> SystemFlag | None:
    return db.query(SystemFlag).filter(SystemFlag.key == KILL_SWITCH_KEY).first()


def is_active(db: Session) -> bool:
    flag = _flag(db)
    return bool(flag and flag.value == "on")


def set_kill_switch(db: Session, active: bool, actor: str | None = None) -> dict:
    flag = _flag(db)
    if flag is None:
        flag = SystemFlag(key=KILL_SWITCH_KEY)
        db.add(flag)
    flag.value = "on" if active else "off"
    db.commit()
    return {"kill_switch": flag.value, "actor": actor}


def status(db: Session) -> dict:
    return {"active": is_active(db)}
