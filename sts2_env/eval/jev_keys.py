"""TypeSafe / Jev API key management and environment auth wiring.

Loads, parses, and provides rotation helpers for single keys and key pools
from environment variables and box-secrets configuration.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

TYPESAFE_API_KEY_ENV = "TYPESAFE_API_KEY"
TYPESAFE_API_KEYS_ENV = "TYPESAFE_API_KEYS"
# Box-secrets card fields Surplus stores: TYPESAFE_API_KEY + TYPESAFE_API_KEY_1..4.
TYPESAFE_BOX_SECRET_NUMBERED_MAX = 4
TYPESAFE_NUMBERED_KEY_MAX = 16
BOX_SECRETS_PATH = Path("/home/box/agent-data/box-secrets.json")
RECOMMENDED_N_ENVS_MAX = 4


def numbered_typesafe_key_env_names(*, upto: int | None = None) -> tuple[str, ...]:
    last = TYPESAFE_NUMBERED_KEY_MAX if upto is None else int(upto)
    names = [TYPESAFE_API_KEY_ENV]
    names.extend(f"{TYPESAFE_API_KEY_ENV}_{i}" for i in range(1, last + 1))
    return tuple(names)


def box_secret_key_names() -> tuple[str, ...]:
    """card.* names in /home/box/agent-data/box-secrets.json."""
    return numbered_typesafe_key_env_names(upto=TYPESAFE_BOX_SECRET_NUMBERED_MAX)


def _dedupe_keys(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in values:
        key = str(raw or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def parse_typesafe_keys_blob(raw: str) -> list[str]:
    """Parse TYPESAFE_API_KEYS: JSON list or comma-separated. Never logs values."""
    text = str(raw or "").strip()
    if not text:
        return []
    if text[:1] in "[{":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, list):
            return _dedupe_keys([str(x) for x in data])
    return _dedupe_keys(text.split(","))


def apply_box_secrets_to_environ(
    environ: Any | None = None,
    secrets_path: str | Path | None = None,
) -> bool:
    """Copy card.TYPESAFE_API_KEY plus card.TYPESAFE_API_KEY_1..4 into env.

    Never overwrites a name that is already exported. Never logs values.
    Process env is often empty on Surplus; callers must run this.
    """
    env = os.environ if environ is None else environ
    path = Path(secrets_path) if secrets_path is not None else BOX_SECRETS_PATH
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    card = data.get("card") if isinstance(data, dict) else None
    if not isinstance(card, dict):
        return False
    wrote = False
    for name in box_secret_key_names():
        val = card.get(name)
        if isinstance(val, str) and val.strip() and not str(env.get(name) or "").strip():
            env[name] = val.strip()
            wrote = True
    blob = card.get(TYPESAFE_API_KEYS_ENV)
    if isinstance(blob, list):
        blob = ",".join(str(x) for x in blob if str(x).strip())
    if (
        isinstance(blob, str)
        and blob.strip()
        and not str(env.get(TYPESAFE_API_KEYS_ENV) or "").strip()
    ):
        env[TYPESAFE_API_KEYS_ENV] = blob.strip()
        wrote = True
    return wrote


def _sync_primary_key_env(keys: list[str], env: Mapping[str, str]) -> None:
    if not keys:
        return
    try:
        if not str(env.get(TYPESAFE_API_KEY_ENV) or "").strip():
            env[TYPESAFE_API_KEY_ENV] = keys[0]  # type: ignore[index]
    except (TypeError, KeyError):
        pass


def load_typesafe_api_keys(
    environ: Mapping[str, str] | None = None,
    *,
    secrets_path: str | Path | None = None,
    hydrate_box_secrets: bool = True,
) -> list[str]:
    """Load TypeSafe key pool for hang-protocol Jev workers.

    Always reads ``/home/box/agent-data/box-secrets.json`` ``card`` fields
    ``TYPESAFE_API_KEY`` and ``TYPESAFE_API_KEY_1``..``_4`` unless
    ``hydrate_box_secrets`` is False. Already-exported
    ``TYPESAFE_API_KEY`` / ``TYPESAFE_API_KEY_N`` / ``TYPESAFE_API_KEYS``
    win over the file (not overwritten). Injects filled names into env so
    spawn workers inherit them. Never logs key material.
    """
    env: Mapping[str, str]
    if environ is None:
        env = os.environ
    else:
        env = environ
    if hydrate_box_secrets:
        apply_box_secrets_to_environ(env, secrets_path)
    keys: list[str] = []
    keys.extend(parse_typesafe_keys_blob(str(env.get(TYPESAFE_API_KEYS_ENV) or "")))
    for name in numbered_typesafe_key_env_names():
        val = env.get(name)
        if val:
            keys.append(str(val))
    keys = _dedupe_keys(keys)
    if keys:
        _sync_primary_key_env(keys, env)
    return keys


def key_for_worker(keys: list[str], worker_id: int) -> str:
    if not keys:
        return ""
    return keys[int(worker_id) % len(keys)]


def typesafe_key_pool_summary(keys: list[str] | None = None) -> dict[str, Any]:
    pool = list(keys) if keys is not None else load_typesafe_api_keys()
    return {
        "typesafe_key_count": len(pool),
        "typesafe_key_pool": len(pool) > 1,
        "recommended_n_envs_max": RECOMMENDED_N_ENVS_MAX,
    }


def warn_n_envs(n_envs: int) -> str | None:
    n = int(n_envs)
    if n > RECOMMENDED_N_ENVS_MAX:
        return (
            f"n_envs={n} exceeds first-recipe cap {RECOMMENDED_N_ENVS_MAX}; "
            "use 2-4 (not 16)"
        )
    return None


__all__ = [
    "BOX_SECRETS_PATH",
    "RECOMMENDED_N_ENVS_MAX",
    "TYPESAFE_API_KEYS_ENV",
    "TYPESAFE_API_KEY_ENV",
    "TYPESAFE_BOX_SECRET_NUMBERED_MAX",
    "TYPESAFE_NUMBERED_KEY_MAX",
    "apply_box_secrets_to_environ",
    "box_secret_key_names",
    "key_for_worker",
    "load_typesafe_api_keys",
    "numbered_typesafe_key_env_names",
    "parse_typesafe_keys_blob",
    "typesafe_key_pool_summary",
    "warn_n_envs",
]
