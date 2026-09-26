"""Transport and client adapters for TypeSafe / Jev System One calls.

Provides HTTP transport, header construction, Cloudflare error handling, key rotation,
SDK fallback, and Protocol/Stub/Live client implementations.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any, Protocol

from sts2_env.eval.jev_keys import (
    TYPESAFE_API_KEY_ENV,
    _dedupe_keys,
    load_typesafe_api_keys,
)
from sts2_env.eval.jev_types import (
    JevAnswer,
    JevError,
    _parse_answer,
)

logger = logging.getLogger(__name__)

TYPESAFE_API_URL = "https://api.typesafe.ai/v1/systemone"
TYPESAFE_MODEL = "jev-1.13.0"
TYPESAFE_HTTP_USER_AGENT = "sts2-rl-agent-jev/1.0"
CLOUDFLARE_1010_MIN_INTERVAL_S = 1.0
HTTP_TIMEOUT_S = 20.0
TYPESAFE_NETWORK_RETRY_MAX = 4
TYPESAFE_NETWORK_BACKOFF_S = (0.25, 0.5, 1.0)


def _is_transient_network_url_error(err: urllib.error.URLError) -> bool:
    """True for DNS / other errors that often clear on a quick retry."""
    msg = str(err).lower()
    if "temporary failure in name resolution" in msg:
        return True
    reason = err.reason
    if isinstance(reason, OSError) and getattr(reason, "errno", None) == -3:
        return True
    return False


class JevClient(Protocol):
    """Surplus/box can swap this without rewriting the eval loop."""

    def system_one(
        self,
        state: Any,
        questions: dict[str, dict[str, Any]],
    ) -> dict[str, JevAnswer]:
        ...


class StubJevClient:
    """Used when --jev off. Never called for the chosen action."""

    def system_one(self, state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, JevAnswer]:
        return {
            name: JevAnswer(status="stub", fallback_reason="jev_off")
            for name in questions
        }


def typesafe_http_headers(api_key: str) -> dict[str, str]:
    """Headers for the TypeSafe HTTP fallback (not the SDK path)."""
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": TYPESAFE_HTTP_USER_AGENT,
    }


def is_cloudflare_1010(http_code: int, body: str) -> bool:
    return int(http_code) == 403 and "1010" in (body or "")


def is_typesafe_forbidden(http_code: int) -> bool:
    return int(http_code) == 403


class LiveJevClient:
    """Live TypeSafe System One call.

    Prefers ``typesafe-sdk`` if installed; otherwise POSTs the public HTTP API.
    Missing key or transport/parse errors raise :class:`JevError`.

    Optional key pool: on HTTP 403 / Cloudflare 1010 rotate to the next key
    and back off. MAP/CARD stay on Jev (do not switch to random to buy fps).
    Default is a single ``TYPESAFE_API_KEY``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = TYPESAFE_MODEL,
        *,
        api_keys: list[str] | None = None,
        key_index: int = 0,
    ):
        if api_keys is not None:
            pool = _dedupe_keys([str(k) for k in api_keys])
        elif api_key is not None:
            pool = _dedupe_keys([api_key] if api_key else [])
        else:
            pool = load_typesafe_api_keys()
        self._keys = pool
        self._idx = int(key_index) % len(pool) if pool else 0
        self.api_key = pool[self._idx] if pool else ""
        self.model = model
        if self.api_key:
            os.environ[TYPESAFE_API_KEY_ENV] = self.api_key
        else:
            logger.warning(
                "%s unset; --jev on will log error and fall back to legal random "
                "at each non-combat decision (combat zip path unchanged)",
                TYPESAFE_API_KEY_ENV,
            )

    def _rotate_key(self) -> bool:
        if len(self._keys) <= 1:
            return False
        prev = self._idx
        self._idx = (self._idx + 1) % len(self._keys)
        self.api_key = self._keys[self._idx]
        if self.api_key:
            os.environ[TYPESAFE_API_KEY_ENV] = self.api_key
        logger.warning(
            "TypeSafe HTTP 403; rotated key pool index %s -> %s (count=%s)",
            prev,
            self._idx,
            len(self._keys),
        )
        return True

    def _urlopen_json_with_network_retry(
        self, req: urllib.request.Request
    ) -> dict[str, Any]:
        last_url_err: urllib.error.URLError | None = None
        for net_attempt in range(TYPESAFE_NETWORK_RETRY_MAX):
            try:
                with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.URLError as e:
                if isinstance(e, urllib.error.HTTPError):
                    raise
                last_url_err = e
                if (
                    not _is_transient_network_url_error(e)
                    or net_attempt + 1 >= TYPESAFE_NETWORK_RETRY_MAX
                ):
                    raise JevError(f"typesafe network error: {e}") from e
                delay = TYPESAFE_NETWORK_BACKOFF_S[
                    min(net_attempt, len(TYPESAFE_NETWORK_BACKOFF_S) - 1)
                ]
                logger.warning(
                    "TypeSafe transient network/DNS error; backoff %.2fs then retry "
                    "(%s/%s): %s",
                    delay,
                    net_attempt + 1,
                    TYPESAFE_NETWORK_RETRY_MAX - 1,
                    e,
                )
                time.sleep(delay)
        raise JevError(f"typesafe network error: {last_url_err}") from last_url_err

    def system_one(
        self,
        state: Any,
        questions: dict[str, dict[str, Any]],
    ) -> dict[str, JevAnswer]:
        if not self.api_key:
            raise JevError(f"{TYPESAFE_API_KEY_ENV} unset")
        try:
            raw = self._call(state, questions)
        except JevError:
            raise
        except Exception as e:
            raise JevError(f"typesafe call failed: {e}") from e
        answers_raw = raw.get("answers") or raw.get("Answers") or {}
        if not isinstance(answers_raw, dict):
            raise JevError("typesafe response missing answers object")
        out: dict[str, JevAnswer] = {}
        for name in questions:
            payload = answers_raw.get(name) or {}
            if not isinstance(payload, dict):
                payload = {}
            out[name] = _parse_answer(payload)
        return out

    def _call(self, state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        sdk_result = self._try_sdk(state, questions)
        if sdk_result is not None:
            return sdk_result
        body = json.dumps(
            {"model": self.model, "state": state, "questions": questions}
        ).encode("utf-8")
        last_http: JevError | None = None
        payload: Any = None
        attempts = max(2, len(self._keys) + 1)
        for attempt in range(attempts):
            req = urllib.request.Request(
                TYPESAFE_API_URL,
                data=body,
                method="POST",
                headers=typesafe_http_headers(self.api_key),
            )
            try:
                payload = self._urlopen_json_with_network_retry(req)
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", errors="replace")[:400]
                last_http = JevError(f"typesafe HTTP {e.code}: {detail}")
                forbidden = is_typesafe_forbidden(e.code)
                hit_1010 = is_cloudflare_1010(e.code, detail)
                if forbidden:
                    rotated = self._rotate_key()
                    if attempt + 1 < attempts and (rotated or hit_1010):
                        logger.warning(
                            "TypeSafe HTTP 403%s; backoff %.1ss then retry (no random MAP/CARD)",
                            " Cloudflare 1010" if hit_1010 else "",
                            CLOUDFLARE_1010_MIN_INTERVAL_S,
                        )
                        time.sleep(CLOUDFLARE_1010_MIN_INTERVAL_S)
                        continue
                raise last_http from e
        else:
            raise last_http or JevError("typesafe HTTP failed")
        if not isinstance(payload, dict):
            raise JevError("typesafe response is not a JSON object")
        return payload

    def _try_sdk(self, state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
        # TODO(Surplus/Jev): prefer the box-pinned typesafe-sdk once Surplus lands it.
        try:
            from typesafe_sdk import Choice, Score, TypeSafeClient
        except ImportError:
            return None
        q_objs: dict[str, Any] = {}
        for name, spec in questions.items():
            qtype = spec.get("type")
            if qtype == "choice":
                q_objs[name] = Choice(
                    instructions=spec.get("instructions", ""),
                    criteria=spec.get("criteria") or {},
                )
            elif qtype == "score":
                q_objs[name] = Score(
                    instructions=spec.get("instructions", ""),
                    criteria=list(spec.get("criteria") or []),
                )
            else:
                raise JevError(f"unsupported question type {qtype!r}")
        with TypeSafeClient() as client:
            response = client.system_one(state=state, questions=q_objs, model=self.model)
        answers = getattr(response, "answers", {}) or {}
        dumped: dict[str, Any] = {"answers": {}}
        for name, ans in answers.items():
            dumped["answers"][name] = {
                "choice": getattr(ans, "choice", None),
                "confidence": getattr(ans, "confidence", None),
                "score": getattr(ans, "score", None),
                "probabilities": dict(getattr(ans, "probabilities", None) or {}),
            }
        return dumped


def build_jev_adapter(
    *,
    enabled: bool,
    client: JevClient | None = None,
    key_index: int = 0,
    api_keys: list[str] | None = None,
) -> JevClient:
    if client is not None:
        return client
    if not enabled:
        return StubJevClient()
    return LiveJevClient(api_keys=api_keys, key_index=key_index)


__all__ = [
    "CLOUDFLARE_1010_MIN_INTERVAL_S",
    "HTTP_TIMEOUT_S",
    "TYPESAFE_NETWORK_BACKOFF_S",
    "TYPESAFE_NETWORK_RETRY_MAX",
    "JevClient",
    "LiveJevClient",
    "StubJevClient",
    "TYPESAFE_API_URL",
    "TYPESAFE_HTTP_USER_AGENT",
    "TYPESAFE_MODEL",
    "build_jev_adapter",
    "is_cloudflare_1010",
    "is_typesafe_forbidden",
    "typesafe_http_headers",
]
