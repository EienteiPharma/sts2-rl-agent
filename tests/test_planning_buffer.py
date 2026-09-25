"""Planning buffer schema + colab_v1 collect wiring (no TypeSafe)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sts2_env.gym_env.planning_buffer import (
    EMPIRICAL_PLANNING_YIELD_V0,
    PLANNING_DEFAULT_OUT,
    PLANNING_TARGET_ROWS,
    PlanningStepRecorder,
    combat_steps_for_planning_rows,
    phase_to_code,
    planning_shard_path,
    refuse_planning_path,
    save_planning_buffer,
    validate_planning_buffer,
)
from sts2_env.gym_env.run_env import RUN_OBS_SIZE, TOTAL_ACTIONS
from sts2_env.run.run_manager import RunManager


def test_planning_recorder_stack_and_validate(tmp_path: Path):
    rec = PlanningStepRecorder()
    obs = np.zeros(RUN_OBS_SIZE, dtype=np.float32)
    mask = np.zeros(TOTAL_ACTIONS, dtype=np.int8)
    mask[115] = 1
    rec.record_step(
        obs=obs,
        next_obs=obs + 0.01,
        action=115,
        reward=0.0,
        done=False,
        action_mask=mask,
        phase=RunManager.PHASE_MAP_CHOICE,
        policy_tag="noncombat_ppo_bh_v1_map_proxy",
    )
    arrays = rec.to_arrays()
    assert arrays is not None
    assert arrays["obs"].shape == (1, RUN_OBS_SIZE)
    assert arrays["action_mask"].shape == (1, TOTAL_ACTIONS)
    assert int(arrays["phase_code"][0]) == phase_to_code(RunManager.PHASE_MAP_CHOICE)
    out = tmp_path / "planning.npz"
    save_planning_buffer(out, arrays)
    loaded = dict(np.load(out, allow_pickle=True))
    validate_planning_buffer(loaded)
    assert loaded["policy_tag"][0] == "noncombat_ppo_bh_v1_map_proxy"


def test_planning_skips_combat_phase():
    rec = PlanningStepRecorder()
    obs = np.zeros(RUN_OBS_SIZE, dtype=np.float32)
    mask = np.zeros(TOTAL_ACTIONS, dtype=np.int8)
    rec.record_step(
        obs=obs,
        next_obs=obs,
        action=0,
        reward=0.0,
        done=False,
        action_mask=mask,
        phase=RunManager.PHASE_COMBAT,
        policy_tag="skip",
    )
    assert rec.to_arrays() is None


def test_run_obs_size_is_201():
    assert RUN_OBS_SIZE == 201


def test_planning_yield_formula():
    steps = combat_steps_for_planning_rows(PLANNING_TARGET_ROWS)
    assert steps == int(__import__("math").ceil(PLANNING_TARGET_ROWS / EMPIRICAL_PLANNING_YIELD_V0))


def test_planning_recorder_stats_skip_phase():
    rec = PlanningStepRecorder()
    obs = np.zeros(RUN_OBS_SIZE, dtype=np.float32)
    rec.note_auto_noncombat_step()
    rec.record_step(
        obs=obs,
        next_obs=obs,
        action=0,
        reward=0.0,
        done=False,
        action_mask=np.zeros(TOTAL_ACTIONS, dtype=np.int8),
        phase="COMBAT",
        policy_tag="skip",
    )
    assert rec.stats.recorded == 0
    assert rec.stats.skipped_phase == 1


def test_planning_shard_path():
    p = planning_shard_path("shard01")
    assert p.name == "planning_shard01.npz"
    assert "shards" in p.parts


def test_refuse_planning_v0_backup():
    with pytest.raises(SystemExit, match="runenv_planning_buffer_colab_v0"):
        refuse_planning_path(
            "/workspace/sts2-sim/output/runenv_planning_buffer_colab_v0/planning_transitions.npz"
        )


def test_refuse_planning_overwrite_colab_v1_combat():
    with pytest.raises(SystemExit, match="runenv_combat_buffer_colab_v1"):
        refuse_planning_path(
            "/workspace/sts2-sim/output/runenv_combat_buffer_colab_v1/planning_transitions.npz"
        )


def test_planning_only_dry_run_skips_combat_out():
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[1] / "scripts" / "collect_runenv_combat.py"
    spec = importlib.util.spec_from_file_location("collect_planning_only", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    report = mod.dry_run(
        mod.parse_args(
            [
                "--dry-run",
                "--planning-only",
                "--out",
                "/workspace/sts2-sim/output/runenv_combat_buffer_colab_v1/transitions.npz",
            ]
        )
    )
    assert report["planning_only"] is True
    assert report["out"] is None
    assert report["planning_out"] is not None
    assert report["jev_enabled"] is False


def test_colab_v1_dry_run_includes_planning_out():
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[1] / "scripts" / "collect_runenv_combat.py"
    spec = importlib.util.spec_from_file_location("collect_runenv_combat_plan", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    report = mod.dry_run(
        mod.parse_args(["--dry-run", "--jev", "off", "--no-planning"])
    )
    assert report["planning_out"] is None
    report2 = mod.dry_run(mod.parse_args(["--dry-run", "--jev", "off"]))
    assert report2["planning_out"] == PLANNING_DEFAULT_OUT
    assert "201" in report2["planning_schema"]
