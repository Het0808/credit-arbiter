"""Secrets management & least-privilege tool access (NFR-Security / US-406).

Secrets are read exclusively from the environment (a stand-in for a real
secrets manager such as AWS Secrets Manager / Vault) - never hard-coded. Each
external tool is granted an explicit, minimal scope set; the assessment layer
checks the scope before invoking a tool, so a compromised or mis-wired tool
cannot exceed its declared privileges.
"""

from __future__ import annotations

import os


class MissingSecretError(RuntimeError):
    """Raised when a required secret is not present in the environment."""


def get_secret(name: str, default: str | None = None, *, required: bool = False) -> str | None:
    """Fetch a secret from the environment. Never returns a hard-coded literal."""
    value = os.environ.get(name, default)
    if required and not value:
        raise MissingSecretError(
            f"Required secret {name!r} is not set. Configure it in the secrets manager / env."
        )
    return value


# Per-tool least-privilege scopes. A tool may only perform the actions listed.
TOOL_SCOPES: dict[str, set[str]] = {
    "risk_model": {"model:infer"},
    "policy_retrieval": {"policy:read"},
    "regulatory": {"regulatory:verify"},
    "document_store": {"documents:read", "documents:write"},
    "llm_explanation": {"llm:generate"},  # reserved for FR-9 when an LLM is wired
}


class ScopeError(PermissionError):
    """Raised when a tool is asked to perform an action outside its granted scope."""


def check_scope(tool: str, action: str) -> None:
    """Assert that ``tool`` is permitted to perform ``action`` (least privilege)."""
    allowed = TOOL_SCOPES.get(tool, set())
    if action not in allowed:
        raise ScopeError(f"Tool {tool!r} is not permitted to perform {action!r}. Allowed: {sorted(allowed)}")


# JWT config, read from env with dev-safe fallbacks that are NOT secrets
# (the fallback secret key is only used for local dev and is clearly labelled).
JWT_ALGORITHM = get_secret("JWT_ALGORITHM", "HS256")
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(get_secret("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
