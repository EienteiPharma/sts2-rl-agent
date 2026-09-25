"""Assist ranker apply: ranked_semantic must not collapse to end_turn (action-0 tie bug)."""
from __future__ import annotations

import numpy as np

from sts2_env.eval.bh_assist import rank_semantic_keys_with_ranker
from sts2_env.eval.combat_turn_plan import SEMANTIC_END_TURN
from sts2_env.gym_env.observation import OBS_SIZE


def _lethal_board() -> dict:
    return {
        "self": {
            "hp": 10,
            "max_hp": 70,
            "block": 0,
            "energy": 3,
            "hand": [
                {"name": "Strike", "cost": "1", "hand_index": 0},
                {"name": "Defend", "cost": "1", "hand_index": 1},
            ],
        },
        "enemies": [
            {
                "name": "Slime",
                "slot": 0,
                "hp": 20,
                "max_hp": 20,
                "block": 0,
                "intent": "attack 20",
            }
        ],
        "turn": {"end_turn_legal": True, "player_turn_index": 1},
        "piles": {"draw_n": 5, "discard_n": 0, "exhaust_n": 0, "draw": [], "discard": [], "exhaust": []},
    }


def test_ranker_apply_prefers_defend_over_end_turn_on_lethal():
    ranker = {
        "w": np.zeros(OBS_SIZE, dtype=np.float32),
        "obs_size": np.array([OBS_SIZE], dtype=np.int64),
    }
    keys = ("end_turn", "play:Defend:h1@self", "play:Strike:h0@e0")
    ranked = rank_semantic_keys_with_ranker(
        ranker,
        np.zeros(OBS_SIZE, dtype=np.float32),
        keys,
        combat=None,
        mask=np.zeros(1),
        board=_lethal_board(),
    )
    assert ranked
    assert ranked[0] != SEMANTIC_END_TURN
    assert ranked[0].startswith("play:Defend")


def test_ranker_apply_varies_top_across_key_sets():
    ranker = {
        "w": np.zeros(OBS_SIZE, dtype=np.float32),
        "obs_size": np.array([OBS_SIZE], dtype=np.int64),
    }
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    board = _lethal_board()
    r1 = rank_semantic_keys_with_ranker(
        ranker, obs, ("end_turn", "play:Defend:h1@self"), combat=None, mask=np.zeros(1), board=board
    )
    board_safe = dict(board)
    board_safe["enemies"] = [
        {**board["enemies"][0], "intent": "unknown", "hp": 20}
    ]
    r2 = rank_semantic_keys_with_ranker(
        ranker,
        obs,
        ("end_turn", "play:Strike:h0@e0"),
        combat=None,
        mask=np.zeros(1),
        board=board_safe,
    )
    assert r1[0] != r2[0] or r1[0] == "play:Strike:h0@e0"
