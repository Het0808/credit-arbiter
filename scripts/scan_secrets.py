"""US-406 AC: 'Given a code scan, When run, Then no secret literals are
present.' Stdlib regex scan for hardcoded secret-shaped literals - this
codebase's actual convention is os.environ.get(...) lookups (see
src/api/database.py, src/api/services/explanation.py), so any literal
key/token/password assignment is a real finding, not a false positive.

Run directly: `python -m scripts.scan_secrets` (exit 1 if anything found).
Also exercised as the US-406 test in tests/test_secrets_scan.py.
"""

import re
import subprocess
import sys

_PLACEHOLDER_MARKERS = ("replace", "your-", "changeme", "example", "xxx", "<", "{", "todo")

_PREFIXED_SECRET = re.compile(r"""["'](gsk_|sk-|AKIA[0-9A-Z]*|ghp_|xox[baprs]-)[A-Za-z0-9_\-]{10,}["']""")
_GENERIC_ASSIGNMENT = re.compile(
    r"""(?i)\b(api[_-]?key|secret[_-]?key|access[_-]?key|password|token)\b\s*[:=]\s*["']([^"']+)["']"""
)


def _is_placeholder(value: str) -> bool:
    if not value:
        return True
    lowered = value.lower()
    return any(marker in lowered for marker in _PLACEHOLDER_MARKERS)


_DOC_EXTENSIONS = (".md", ".txt", ".rst")
# ponytail: this scanner's own test embeds fake secret-shaped literals to
# verify detection - exclude it or it flags itself every run.
_SELF_TEST_FILE = "tests/test_secrets_scan.py"


def find_secret_literals(paths):
    findings = []
    for path in paths:
        if str(path).lower().endswith(_DOC_EXTENSIONS) or str(path) == _SELF_TEST_FILE:
            continue
        try:
            with open(path, encoding="utf-8", errors="ignore") as fh:
                for line_no, line in enumerate(fh, start=1):
                    if _PREFIXED_SECRET.search(line):
                        findings.append((path, line_no, line.strip()))
                        continue
                    m = _GENERIC_ASSIGNMENT.search(line)
                    if m and not _is_placeholder(m.group(2)):
                        findings.append((path, line_no, line.strip()))
        except (FileNotFoundError, IsADirectoryError):
            continue
    return findings


def main():
    tracked = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout.splitlines()
    findings = find_secret_literals(tracked)
    for path, line_no, line in findings:
        print(f"{path}:{line_no}: {line}")
    if findings:
        print(f"\n{len(findings)} secret-shaped literal(s) found.")
        sys.exit(1)
    print("No secret literals found.")


if __name__ == "__main__":
    main()
