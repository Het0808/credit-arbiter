"""External-call audit trail (US-406, NFR-Security).

Every real external network call the system makes (currently: the Groq LLM
call in services/explanation.py) is appended here as one JSON line,
independent of app request logs, so a security review can grep a single
file for what left the building and when.
"""

import json
import os
from datetime import datetime

_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs", "external_calls.log"
)


def log_external_call(service: str, **fields) -> None:
    os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
    entry = {"timestamp": datetime.utcnow().isoformat(), "service": service, **fields}
    with open(_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
