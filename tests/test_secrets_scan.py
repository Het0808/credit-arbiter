"""US-406 AC: 'Given a code scan, When run, Then no secret literals are
present.' Scans every git-tracked file for hardcoded secret-shaped literals
(vs. the codebase's actual convention of os.environ.get(...) lookups)."""

from scripts.scan_secrets import find_secret_literals


def test_flags_hardcoded_secret_assignment(tmp_path):
    f = tmp_path / "bad.py"
    f.write_text('GROQ_API_KEY = "gsk_abcdefghijklmnopqrstuvwxyz1234"\n', encoding="utf-8")

    findings = find_secret_literals([f])

    assert len(findings) == 1
    assert findings[0][0] == f


def test_ignores_env_lookup(tmp_path):
    f = tmp_path / "good.py"
    f.write_text('GROQ_API_KEY = os.environ.get("GROQ_API_KEY")\n', encoding="utf-8")

    assert find_secret_literals([f]) == []


def test_ignores_empty_placeholder(tmp_path):
    f = tmp_path / "example.env"
    f.write_text('JWT_SECRET_KEY="replace-this-with-a-very-secret-key-in-production"\nGROQ_API_KEY=""\n', encoding="utf-8")

    assert find_secret_literals([f]) == []


def test_ignores_prose_in_markdown_docs(tmp_path):
    f = tmp_path / "runbook.md"
    f.write_text('Example: `password = "literal"` is a hardcoded assignment.\n', encoding="utf-8")

    assert find_secret_literals([f]) == []


def test_repo_scan_is_clean():
    """The actual US-406 acceptance check: run it over every tracked file."""
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files"], cwd=".", capture_output=True, text=True, check=True
    ).stdout.splitlines()
    findings = find_secret_literals(tracked)
    assert findings == [], f"secret-shaped literals found: {findings}"
