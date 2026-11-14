"""Assist v3 outdir guards + boss inject buffer collect (synthetic)."""
from __future__ import annotations

import json

import numpy as np
import pytest

import sts2_env.cards  # noqa: F401
from sts2_env.eval.bh_assist_train import (
    DEFAULT_BH_ASSIST_V3_OUTDIR,
    build_assist_rows_from_buffer,
    refuse_bh_assist_output_path,
    save_assist_checkpoint,
    train_linear_assist_ranker,
)
from sts2_env.eval.boss_fail_inject_collect import collect_transitions_for_hold_jobs
from sts2_env.eval.boss_forward_inject import write_inject_jobs
from sts2_env.eval.combat_hold import hold_jobs
from sts2_env.gym_env.combat_buffer import synthetic_combat_buffer


def test_refuses_protected_v1_outdir():
    with pytest.raises(SystemExit, match="protected assist"):
        refuse_bh_assist_output_path("output/combat_bh_assist_v1")


def test_v3_outdir_allowed(tmp_path):
    out = tmp_path / "combat_bh_assist_v3"
    path = refuse_bh_assist_output_path(out)
    assert path == out


def test_collect_inject_jobs_tiny_buffer():
    boss = next(j for j in hold_jobs(1) if j["bucket"] == "boss")

    class _Ppo:
        def predict(self, obs, action_masks=None, deterministic=True):
            valid = np.flatnonzero(np.asarray(action_masks) == 1)
            return int(valid[0]), None

    ppo = _Ppo()

    def predict(obs, mask):
        action, _ = ppo.predict(obs, action_masks=mask)
        return int(action)

    arrays = collect_transitions_for_hold_jobs([boss], predict, max_steps=8)
    assert arrays["obs"].shape[0] >= 1


def test_eval_default_ckpt_is_v3_not_write_protected_v1():
    from sts2_env.eval.bh_assist import (
        DEFAULT_BH_ASSIST_CKPT,
        resolve_bh_assist_ckpt_path,
        resolve_turn_plan_bh_assist_from_flags,
    )

    assert "combat_bh_assist_v3" in DEFAULT_BH_ASSIST_CKPT
    assert "combat_bh_assist_v1" not in DEFAULT_BH_ASSIST_CKPT
    assert DEFAULT_BH_ASSIST_CKPT.endswith("bh_assist_ranker.npz")
    assert resolve_bh_assist_ckpt_path(None) == DEFAULT_BH_ASSIST_CKPT
    cfg = resolve_turn_plan_bh_assist_from_flags(bh_assist="on", bh_assist_ckpt=None)
    assert cfg is not None and cfg.enabled
    assert cfg.ckpt_path == DEFAULT_BH_ASSIST_CKPT


def test_load_ranker_from_v1_named_path_does_not_sysexit(tmp_path):
    from sts2_env.eval.bh_assist import _RANKER_CACHE, load_bh_assist_ranker
    from sts2_env.gym_env.observation import OBS_SIZE

    v1_dir = tmp_path / "combat_bh_assist_v1"
    v1_dir.mkdir()
    ckpt = v1_dir / "bh_assist_ranker.npz"
    np.savez(
        ckpt,
        w=np.zeros(OBS_SIZE, dtype=np.float32),
        obs_size=np.array([OBS_SIZE], dtype=np.int64),
    )
    _RANKER_CACHE.clear()
    loaded = load_bh_assist_ranker(ckpt)
    assert loaded is not None
    assert int(loaded["w"].size) == OBS_SIZE


def test_train_v3_checkpoint(tmp_path):
    arrays = synthetic_combat_buffer(n=12, n_episodes=2, seed=0)
    rows = build_assist_rows_from_buffer(arrays)
    ckpt = train_linear_assist_ranker(rows, steps=2, seed=0)
    out = tmp_path / "combat_bh_assist_v3"
    refuse_bh_assist_output_path(out)
    path = save_assist_checkpoint(out, ckpt, {"protocol": "test_v3"})
    assert path.is_file()
    assert DEFAULT_BH_ASSIST_V3_OUTDIR == "output/combat_bh_assist_v3"


def test_collect_dry_run_manifest(tmp_path):
    jobs = hold_jobs(1)
    boss_jobs = [j for j in jobs if j["bucket"] == "boss"][:1]
    meta = {"protocol": "test"}
    inj = tmp_path / "inject_jobs.json"
    write_inject_jobs(inj, boss_jobs, meta)
    from sts2_env.eval.boss_fail_inject_collect import dry_run_collect_manifest

    manifest = dry_run_collect_manifest(
        inject_jobs_path=inj,
        n_jobs=1,
        max_steps=400,
        model_path="/tmp/model.zip",
        pack_dir=str(tmp_path),
    )
    assert manifest["dry_run"] is True
    assert manifest["n_jobs"] == 1
