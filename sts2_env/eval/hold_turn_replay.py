"""Bounded turn-plan replay logs for HOLD (fail + Boss episodes).

Append-only JSONL under ``evals/`` (or ``--turn-replay-dir``). Logging only;
does not change combat policy, prune, prompts, or assist ranker.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

HOLD_TURN_REPLAY_ENV_ATTR = "_hold_turn_plan_replay"
HOLD_TURN_REPLAY_PROTOCOL = "hold_turn_plan_replay_v1"
DEFAULT_HOLD_TURN_REPLAY_DIR = "evals/hold_turn_replay"
# Additive turn keys (v1 readers ignore unknown fields). n=1 replay smoke is diagnostic only —
# not a promotion / eval gate (see lock_eval_then_v3 formal n=20).
REPLAY_TURN_TRAJECTORY_KEYS = (
    "player_hp_start",
    "player_hp_end",
    "enemies_hp",
    "replan_count",
    "replan_cap_hit",
    "fail_open",
    "fail_open_reason",
)


def enemies_hp_from_board(board: Mapping[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, row in enumerate(board.get("enemies") or []):
        if not isinstance(row, Mapping):
            continue
        slot = row.get("slot", i)
        name = row.get("name")
        ident = str(name) if name else f"slot_{int(slot)}"
        out.append({"id": ident, "hp": int(row.get("hp") or 0)})
    return out


def player_hp_from_board(board: Mapping[str, Any]) -> int:
    self_row = board.get("self") if isinstance(board, Mapping) else {}
    if not isinstance(self_row, Mapping):
        return 0
    return int(self_row.get("hp") or 0)


def should_retain_hold_turn_replay(*, win: bool, bucket: str) -> bool:
    """Keep replay for losses and all Boss-bucket fights (wins included)."""
    if not win:
        return True
    return str(bucket) == "boss"


def resolve_hold_turn_replay_dir(
    explicit: str | None,
    *,
    combat_policy: str,
) -> Path | None:
    if combat_policy != "jev-turn":
        return None
    raw = explicit
    if raw is None:
        return Path(DEFAULT_HOLD_TURN_REPLAY_DIR)
    text = str(raw).strip()
    if not text or text.lower() in ("none", "off", "0"):
        return None
    return Path(text)


def replay_recorder_from_env(env: Any) -> HoldTurnPlanEpisodeReplay | None:
    rec = getattr(env, HOLD_TURN_REPLAY_ENV_ATTR, None)
    if rec is None:
        return None
    return rec if isinstance(rec, HoldTurnPlanEpisodeReplay) else None


def terminal_snapshot_from_combat(combat: Any, mask: np.ndarray) -> dict[str, Any]:
    from sts2_env.eval.combat_turn_plan import serialize_combat_board_full

    board = serialize_combat_board_full(combat, mask)
    self_row = board.get("self") or {}
    piles = board.get("piles") or {}
    enemies = board.get("enemies") or []
    return {
        "player_hp": int(self_row.get("hp") or 0),
        "player_max_hp": int(self_row.get("max_hp") or 0),
        "player_block": int(self_row.get("block") or 0),
        "player_energy": int(self_row.get("energy") or 0),
        "deck_summary": {
            "hand": list(self_row.get("hand") or []),
            "draw_n": int(piles.get("draw_n") or 0),
            "discard_n": int(piles.get("discard_n") or 0),
            "exhaust_n": int(piles.get("exhaust_n") or 0),
            "draw_top": list(piles.get("draw") or [])[:12],
            "discard_top": list(piles.get("discard") or [])[:12],
        },
        "enemy_intents": {
            f"slot_{int(e.get('slot', i))}": str(e.get("intent") or "?")
            for i, e in enumerate(enemies)
        },
    }


@dataclass
class HoldTurnPlanEpisodeReplay:
    """In-memory turn-plan Choice trace for one HOLD fight."""

    seed: int
    fixture_index: int
    enc_id: int
    bucket: str
    ep: int
    run_meta: dict[str, Any] = field(default_factory=dict)
    turns: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_hold_job(
        cls, job: Mapping[str, Any], *, run_meta: Mapping[str, Any] | None = None
    ) -> HoldTurnPlanEpisodeReplay:
        return cls(
            seed=int(job["seed"]),
            fixture_index=int(job["fixture_index"]),
            enc_id=int(job["enc_id"]),
            bucket=str(job.get("bucket") or ""),
            ep=int(job.get("ep", 0)),
            run_meta=dict(run_meta or {}),
        )

    def record_plan_choice(
        self,
        *,
        player_turn: int,
        plans: Sequence[Any],
        board: Mapping[str, Any],
        pruned_plan_count: int,
        picked_plan_id: str | None,
        pick_error: str | None,
        bh_assist: Any | None,
        shadow: Mapping[str, Any] | None = None,
        replan_count: int = 0,
        replan_cap_hit: bool = False,
        fail_open: bool = False,
        fail_open_reason: str | None = None,
    ) -> None:
        from sts2_env.eval.combat_turn_plan import _plan_criteria_summary

        hp_start = player_hp_from_board(board)
        enemies = enemies_hp_from_board(board)
        shortlist = [
            {
                "plan_id": str(p.plan_id),
                "criteria": _plan_criteria_summary(p, board=dict(board)),
            }
            for p in plans
        ]
        choice: dict[str, Any] = {
            "plan_id": picked_plan_id,
            "error": pick_error,
        }
        if shadow:
            if shadow.get("turn_plan_failopen"):
                choice["failopen"] = True
                choice["failopen_reason"] = shadow.get("turn_plan_failopen_reason")
            if shadow.get("turn_plan_id"):
                choice["plan_id"] = shadow.get("turn_plan_id")
        entry: dict[str, Any] = {
            "player_turn": int(player_turn),
            "plan_shortlist": shortlist,
            "pruned_plan_count": int(pruned_plan_count),
            "choice_pick": choice,
            "player_hp_start": hp_start,
            "player_hp_end": hp_start,
            "enemies_hp": enemies,
            "replan_count": int(replan_count),
            "replan_cap_hit": bool(replan_cap_hit),
            "fail_open": bool(fail_open),
            "fail_open_reason": fail_open_reason if fail_open else None,
        }
        if bh_assist is not None:
            entry["bh_assist"] = bh_assist.as_dict()
        if shadow and shadow.get("turn_plan_step"):
            entry["executed_step"] = shadow.get("turn_plan_step")
        self.turns.append(entry)

    def patch_last_turn_trajectory(
        self,
        board: Mapping[str, Any],
        *,
        replan_count: int,
    ) -> None:
        """End-of-player-turn HP/enemy snapshot (additive update to last turn row)."""
        if not self.turns:
            return
        last = self.turns[-1]
        last["player_hp_end"] = player_hp_from_board(board)
        last["enemies_hp"] = enemies_hp_from_board(board)
        last["replan_count"] = int(replan_count)

    def to_document(
        self,
        *,
        win: bool,
        steps: int,
        terminal: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "protocol": HOLD_TURN_REPLAY_PROTOCOL,
            "seed": self.seed,
            "fixture_index": self.fixture_index,
            "enc_id": self.enc_id,
            "bucket": self.bucket,
            "ep": self.ep,
            "win": bool(win),
            "steps": int(steps),
            **self.run_meta,
            "turns": list(self.turns),
            "terminal": dict(terminal),
        }


class HoldTurnReplayWriter:
    """Append retained episodes to a JSONL file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.written = 0

    def maybe_write_episode(
        self,
        recorder: HoldTurnPlanEpisodeReplay,
        *,
        win: bool,
        steps: int,
        combat: Any | None,
        mask: np.ndarray | None,
    ) -> bool:
        if not should_retain_hold_turn_replay(win=win, bucket=recorder.bucket):
            return False
        terminal: dict[str, Any] = {}
        if combat is not None and mask is not None:
            terminal = terminal_snapshot_from_combat(combat, np.asarray(mask))
        doc = recorder.to_document(win=win, steps=steps, terminal=terminal)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        self.written += 1
        return True


__all__ = [
    "DEFAULT_HOLD_TURN_REPLAY_DIR",
    "HOLD_TURN_REPLAY_ENV_ATTR",
    "HOLD_TURN_REPLAY_PROTOCOL",
    "REPLAY_TURN_TRAJECTORY_KEYS",
    "enemies_hp_from_board",
    "player_hp_from_board",
    "HoldTurnPlanEpisodeReplay",
    "HoldTurnReplayWriter",
    "replay_recorder_from_env",
    "resolve_hold_turn_replay_dir",
    "should_retain_hold_turn_replay",
    "terminal_snapshot_from_combat",
]
