"""Jev / TypeSafe adapter for Act1 RunEnv non-combat decisions.

Lab-hung thresholds (do not retune in this eval):

* Choice confidence ≥ 0.65, else uncertain → legal random
* rest_or_continue also uses hp_pressure Score: ≥ 2.0 prefer rest,
  ≤ 1.0 prefer continue, middle trust Choice
* card_reward also uses card_fit Score (4-level): Choice 0.65 is
  unchanged; if confidence < 0.65 but card_fit ≥ 2.0 and choice is not
  skip, land with reason ``jev_card_fit_assist``
* Potion / relic screens that share ``PHASE_CARD_REWARD`` must not call
  card_reward Jev (legal random, reason ``potion_or_relic_reward_random``)
* Strip invisible / illegal candidates before Choice
* Act1 reward ``+`` cards are not natural drops (Smith / Neow only)
* Card-reward Choice is Neow+early natural Act1, not mid-act fixtures

Criteria text references ``docs/act1_content_map.md``.

TODO(Surplus/Jev): plug the box's live TypeSafe client here without rewriting
``scripts/eval_act1_runenv.py``. ``TYPESAFE_API_KEY`` is the env var both the
HTTP fallback and ``typesafe-sdk`` read. Pin ``jev-1.13.0`` while these
thresholds are frozen (``jev-latest`` can drift).
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)

CHOICE_CONFIDENCE_MIN = 0.65
HP_PRESSURE_REST = 2.0
HP_PRESSURE_CONTINUE = 1.0
CARD_FIT_ASSIST_MIN = 2.0
POTION_OR_RELIC_REASON = "potion_or_relic_reward_random"
CARD_FIT_ASSIST_REASON = "jev_card_fit_assist"

TYPESAFE_API_URL = "https://api.typesafe.ai/v1/systemone"
TYPESAFE_MODEL = "jev-1.13.0"
TYPESAFE_API_KEY_ENV = "TYPESAFE_API_KEY"
CONTENT_MAP_REF = "docs/act1_content_map.md"
HTTP_TIMEOUT_S = 20.0

HP_PRESSURE_SCORE_CRITERIA = [
    "0 — comfortable HP; keep pushing, a rest site is not needed",
    "1 — some missing HP; continuing the path is still reasonable",
    "2 — HP pressure high enough that a rest site is the better fork",
    "3 — critically low HP; rest if a rest node is legal",
]

CARD_FIT_SCORE_CRITERIA = [
    "0 — poor fit for this Neow+early Act1 deck; skip or ignore",
    "1 — weakly useful; take only if nothing better",
    "2 — solid Act1 pickup for this Neow+early deck",
    "3 — high-priority Act1 card for this Neow+early deck",
]

PLUS_CARD_CRITERION = (
    "Act1 combat-reward '+' / upgraded cards are NOT natural drops "
    "(Smith rest-site upgrade or Neow only). Do not prefer them as if "
    "they were reward-upgraded. See " + CONTENT_MAP_REF + "."
)

NEOW_EARLY_CARD_INSTRUCTIONS = (
    "Neow+early natural Act1 (not mid-act fixtures). "
    "Choose a card reward or skip. "
    + PLUS_CARD_CRITERION
)


class JevError(RuntimeError):
    """Raised when a TypeSafe/Jev call fails. Callers must not swallow the decision point."""


@dataclass
class JevAnswer:
    status: str  # ok | uncertain | error | stub | skipped
    choice: str | None = None
    confidence: float | None = None
    score: float | None = None
    card_fit: float | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    fallback_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def as_log(self) -> dict[str, Any]:
        log = {
            "shadow_suggestion": self.choice,
            "shadow_status": self.status,
            "shadow_confidence": self.confidence,
            "shadow_hp_pressure": self.score,
            "shadow_fallback_reason": self.fallback_reason,
        }
        if self.card_fit is not None:
            log["jev_card_fit"] = self.card_fit
        return log


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


class LiveJevClient:
    """Live TypeSafe System One call.

    Prefers ``typesafe-sdk`` if installed; otherwise POSTs the public HTTP API.
    Missing key or transport/parse errors raise :class:`JevError`.
    """

    def __init__(self, api_key: str | None = None, model: str = TYPESAFE_MODEL):
        self.api_key = api_key if api_key is not None else os.environ.get(TYPESAFE_API_KEY_ENV, "")
        self.model = model
        if not self.api_key:
            logger.warning(
                "%s unset; --jev on will log error and fall back to legal random "
                "at each non-combat decision (combat zip path unchanged)",
                TYPESAFE_API_KEY_ENV,
            )

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
        req = urllib.request.Request(
            TYPESAFE_API_URL,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:400]
            raise JevError(f"typesafe HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise JevError(f"typesafe network error: {e}") from e
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


def build_jev_adapter(*, enabled: bool, client: JevClient | None = None) -> JevClient:
    if client is not None:
        return client
    if not enabled:
        return StubJevClient()
    return LiveJevClient()


def _parse_answer(payload: dict[str, Any]) -> JevAnswer:
    choice = payload.get("choice")
    if choice is not None:
        choice = str(choice)
    conf = payload.get("confidence")
    score = payload.get("score")
    probs = payload.get("probabilities") or {}
    if not isinstance(probs, dict):
        probs = {}
    return JevAnswer(
        status="ok",
        choice=choice,
        confidence=float(conf) if conf is not None else None,
        score=float(score) if score is not None else None,
        probabilities={str(k): float(v) for k, v in probs.items()},
        raw=payload,
    )


def local_hp_pressure(hp: int, max_hp: int) -> float:
    """Fallback Score-shaped pressure: 1.0 at full HP, 2.0 at half HP."""
    return float(max_hp) / float(max(int(hp), 1))


def apply_choice_confidence(answer: JevAnswer) -> JevAnswer:
    if answer.status != "ok":
        return answer
    conf = answer.confidence
    if conf is None or conf < CHOICE_CONFIDENCE_MIN:
        answer.status = "uncertain"
        answer.fallback_reason = (
            f"choice confidence {conf} < {CHOICE_CONFIDENCE_MIN}"
        )
    return answer


def rest_or_continue_override(hp_pressure: float) -> str | None:
    """Return 'rest', 'continue', or None (trust Choice)."""
    if hp_pressure >= HP_PRESSURE_REST:
        return "rest"
    if hp_pressure <= HP_PRESSURE_CONTINUE:
        return "continue"
    return None
