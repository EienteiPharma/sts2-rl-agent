"""bh_assist train entry: outdir guards and buffer label smoke."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from sts2_env.eval.bh_assist import bh_assist
from sts2_env.eval.bh_assist_train import (
    DEFAULT_BH_ASSIST_OUTDIR,
    build_assist_rows_from_buffer,
    dry_run_manifest,
    refuse_bh_assist_output_path,
    refuse_hang_ppo_continue,
    save_assist_checkpoint,
    train_linear_assist_ranker,
)
from sts2_env.gym_env.combat_buffer import HUNG_OUTDIR_NAME, synthetic_combat_buffer


def test_refuses_hang_outdir():
    with pytest.raises(SystemExit, match="frozen"):
        refuse_bh_assist_output_path(f"output/{HUNG_OUTDIR_NAME}")


def test_refuses_hang_final_model_zip_path():
    with pytest.raises(SystemExit, match="frozen|hang"):
        refuse_bh_assist_output_path(
            f"/workspace/sts2-sim/output/{HUNG_OUTDIR_NAME}/final_model.zip"
        )


def test_refuses_continue_from_hang_zip():
    with pytest.raises(SystemExit, match="continue-from"):
        refuse_hang_ppo_continue(
            f"/workspace/sts2-sim/output/{HUNG_OUTDIR_NAME}/final_model.zip"
        )


def test_build_rows_and_train_checkpoint(tmp_path: Path):
    arrays = synthetic_combat_buffer(n=20, n_episodes=4, seed=1)
    rows = build_assist_rows_from_buffer(arrays)
    assert rows
    ckpt = train_linear_assist_ranker(rows, steps=8, seed=0)
    out = tmp_path / "combat_bh_assist_v1"
    path = save_assist_checkpoint(
        out,
        ckpt,
        {"protocol": "test", "n_assist_rows": len(rows)},
    )
    assert path.is_file()
    assert (out / "bh_assist_train_manifest.json").is_file()


def test_dry_run_manifest_default_outdir():
    payload = dry_run_manifest(buffer_path=None, output_dir=DEFAULT_BH_ASSIST_OUTDIR, n_train_steps=16)
    assert payload["dry_run"] is True
    assert payload["hang_policy_swap"] is False
    assert payload["n_assist_rows"] > 0


def test_bh_assist_heuristic_subset_of_legal():
    board = {
        "self": {"hp": 12, "max_hp": 70, "block": 0, "energy": 3, "hand": []},
        "enemies": [{"intent": "attack 20", "hp": 30, "max_hp": 30, "block": 0, "slot": 0}],
        "turn": {"end_turn_legal": True},
    }
    legal = ("end_turn", "play:Defend:h0@self")
    res = bh_assist(board, legal_semantic_keys=legal)
    assert set(res.ranked_semantic) <= set(legal)


def test_train_cli_dry_run():
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/train_bh_assist_from_buffer.py",
            "--dry-run",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["dry_run"] is True
    assert DEFAULT_BH_ASSIST_OUTDIR in payload["output_dir"]
