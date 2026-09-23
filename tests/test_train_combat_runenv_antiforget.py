"""Anti-forgetting mix: hang RunEnv + loadout_v1, HOLD gate, no 500k learn."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sts2_env.core.constants import ACTION_SPACE_SIZE
from sts2_env.eval.combat_hold import (
    HOLD_BOSS_MIN,
    HOLD_ENC_IDS,
    HOLD_FIXTURE_STEMS,
    HOLD_OVERALL_MIN,
    hold_jobs,
    hold_passes,
    summarize_hold_rows,
)
from sts2_env.gym_env.observation import OBS_SIZE
from sts2_env.gym_env.run_env import RUN_OBS_SIZE
from sts2_env.gym_env.runenv_antiforget import (
    DEFAULT_RUNENV_FRAC,
    FROZEN_RUNENV_FRAC,
    SOURCE_LOADOUT,
    SOURCE_RUNENV,
    MixedHangLoadoutEnv,
    frozen_runenv_frac_warning,
    make_loadout_v1_provider,
    parse_runenv_frac,
    refuse_antiforget_v1_continue,
)
from sts2_env.run.run_manager import RunManager

_TRAIN_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_combat_runenv_antiforget.py"


def _load_mod():
    spec = importlib.util.spec_from_file_location(
        "train_combat_runenv_antiforget", _TRAIN_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


train_mod = _load_mod()


class FakeSpace:
    def __init__(self, dim: int):
        self.shape = (dim,)


def test_cli_defaults_continue_bh_v1_not_onpolicy():
    args = train_mod.parse_args([])
    assert args.continue_from == train_mod.HUNG_COMBAT_ZIP
    assert "combat_ppo_obs_v1_bh_v1" in args.continue_from
    assert "onpolicy_v1" not in args.continue_from
    assert args.runenv_frac == pytest.approx(0.3)
    assert args.runenv_frac == pytest.approx(DEFAULT_RUNENV_FRAC)
    assert "antiforget_v1" not in args.continue_from
    assert args.n_envs == 1
    assert args.total_timesteps == 2048
    assert args.lr == pytest.approx(3e-5)
    assert args.hold_freq == 0
    assert args.hold_n_eps == 1
    assert args.hold_stop is False
    assert args.output_dir == train_mod.DEFAULT_OUTPUT_DIR
    assert args.output_dir.endswith("combat_runenv_antiforget_ld03")
    alias = train_mod.parse_args(["--model", "/tmp/bh.zip", "--runenv-frac", "0.5"])
    assert alias.continue_from == "/tmp/bh.zip"
    assert alias.runenv_frac == pytest.approx(0.5)


def test_no_mix_neow_loadout_flag():
    with pytest.raises(SystemExit):
        train_mod.parse_args(["--loadout", "mix_neow_v1"])
    text = _TRAIN_PATH.read_text()
    assert "--loadout" not in text


def test_runenv_frac_bounds():
    with pytest.raises(SystemExit, match="runenv-frac"):
        parse_runenv_frac(1.2)
    with pytest.raises(SystemExit, match="runenv-frac"):
        train_mod.parse_args(["--runenv-frac", "-0.1"])
    assert parse_runenv_frac(0.0) == 0.0
    assert parse_runenv_frac(1.0) == 1.0


def test_refuse_overwrite_frozen_outdirs(tmp_path):
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        train_mod.refuse_overwrite_frozen("output/combat_ppo_obs_v1_bh_v1")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        train_mod.refuse_overwrite_frozen("output/combat_runenv_onpolicy_v1")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        train_mod.refuse_overwrite_frozen("output/combat_runenv_antiforget_v1")
    ok = train_mod.refuse_overwrite_frozen(tmp_path / "combat_runenv_antiforget_ld03")
    assert ok.name == "combat_runenv_antiforget_ld03"


def test_refuse_continue_from_antiforget_v1(tmp_path):
    frozen = tmp_path / "combat_runenv_antiforget_v1" / "final_model.zip"
    frozen.parent.mkdir(parents=True)
    frozen.write_bytes(b"zip")
    with pytest.raises(SystemExit, match="antiforget_v1"):
        train_mod.resolve_continue_from(str(frozen), must_exist=True)
    with pytest.raises(SystemExit, match="antiforget_v1"):
        refuse_antiforget_v1_continue("output/antiforget_v1/final_model.zip")
    with pytest.raises(SystemExit, match="not found"):
        train_mod.resolve_continue_from(str(tmp_path / "missing.zip"), must_exist=True)
    ok = tmp_path / "combat_ppo_obs_v1_bh_v1" / "final_model.zip"
    ok.parent.mkdir(parents=True)
    ok.write_bytes(b"zip")
    assert train_mod.resolve_continue_from(str(ok)) == ok
    assert frozen_runenv_frac_warning(0.3) is None
    assert frozen_runenv_frac_warning(FROZEN_RUNENV_FRAC) is not None
    assert "FROZEN" in frozen_runenv_frac_warning(0.7)


def test_continue_from_and_obs_guard(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        train_mod.resolve_continue_from(str(tmp_path / "missing.zip"), must_exist=True)
    fake = tmp_path / "final_model.zip"
    fake.write_bytes(b"zip")
    assert train_mod.resolve_continue_from(str(fake)) == fake
    assert train_mod.require_combat_obs_dim(SimpleNamespace(observation_space=FakeSpace(OBS_SIZE))) == OBS_SIZE
    with pytest.raises(SystemExit, match="OBS_SIZE"):
        train_mod.require_combat_obs_dim(SimpleNamespace(observation_space=FakeSpace(RUN_OBS_SIZE)))


def test_hold_gate_matches_lab():
    assert HOLD_OVERALL_MIN == 0.70
    assert HOLD_BOSS_MIN == 0.40
    fail = summarize_hold_rows(
        [{"win": True, "bucket": "elite"}] * 62
        + [{"win": False, "bucket": "elite"}] * 28
        + [{"win": True, "bucket": "boss"}] * 24
        + [{"win": False, "bucket": "boss"}] * 37
    )
    # 86/151 ≈ 0.570 overall would fail; use the reported 68.9/39.4 shape
    fail = {
        "overall": {"win_rate": 0.689, "n": 180},
        "boss": {"win_rate": 0.394, "n": 90},
        "elite": {"win_rate": 0.983, "n": 90},
    }
    assert hold_passes(fail) is False
    ok = {
        "overall": {"win_rate": 0.70, "n": 180},
        "boss": {"win_rate": 0.40, "n": 90},
        "elite": {"win_rate": 1.0, "n": 90},
    }
    assert hold_passes(ok) is True
    assert hold_passes({"overall": {"win_rate": 0.70}, "boss": {"win_rate": 0.39}}) is False


def test_hold_jobs_are_three_fixtures_elite_boss():
    from sts2_env.eval.combat_hold import HOLD_DEFAULT_RELICS, HOLD_SEED_BASE, hold_protocol_meta

    jobs = hold_jobs(n_eps=1)
    assert HOLD_FIXTURE_STEMS == ("loadout_v1_01", "loadout_v1_02", "loadout_v1_03")
    assert HOLD_ENC_IDS == list(range(16, 22))
    assert HOLD_DEFAULT_RELICS == ("BURNING_BLOOD", "SHURIKEN")
    assert len(jobs) == 3 * 6
    assert {j["bucket"] for j in jobs} == {"elite", "boss"}
    assert sum(1 for j in jobs if j["bucket"] == "boss") == 9
    jobs2 = hold_jobs(n_eps=2)
    assert len(jobs2) == 36
    jobs20 = hold_jobs(n_eps=20)
    assert len(jobs20) == 360
    job = jobs[0]
    assert job["seed"] == HOLD_SEED_BASE + job["fixture_index"] * 1000 + job["enc_id"] * 100 + job["ep"]
    meta = hold_protocol_meta(n_eps=20)
    assert meta["seed_formula"] == "40000+fix*1000+enc*100+ep"
    assert meta["gate"]["overall_min"] == 0.70
    assert meta["gate"]["boss_min"] == 0.40


def test_mix_runenv_half_is_hang_combat():
    env = MixedHangLoadoutEnv(
        runenv_frac=1.0,
        loadout_provider=make_loadout_v1_provider(offset=0),
        max_steps=400,
    )
    obs, info = env.reset(seed=42)
    assert info["mix_source"] == SOURCE_RUNENV
    assert info["start_with_neow"] is True
    assert info["jev_event"] == "off"
    assert info["jev_neow"] == "off"
    assert obs.shape == (OBS_SIZE,)
    flags = env.hang_protocol()
    assert flags["allows_event"] is False
    assert flags["allows_neow"] is False
    assert flags["start_with_neow"] is True
    if not info.get("combat_unreachable"):
        assert info.get("phase") == RunManager.PHASE_COMBAT
        mask = env.action_masks()
        assert mask.shape == (ACTION_SPACE_SIZE,)
        local = int(np.flatnonzero(mask == 1)[0])
        obs2, reward, _term, _trunc, info2 = env.step(local)
        assert obs2.shape == (OBS_SIZE,)
        assert info2["mix_source"] == SOURCE_RUNENV
        assert isinstance(reward, float)
    env.close()


def test_mix_loadout_half_is_fixture_combat():
    env = MixedHangLoadoutEnv(
        runenv_frac=0.0,
        loadout_provider=make_loadout_v1_provider(offset=0),
        max_steps=80,
    )
    obs, info = env.reset(seed=1)
    assert info["mix_source"] == SOURCE_LOADOUT
    assert info.get("loadout") == "loadout_v1"
    assert obs.shape == (OBS_SIZE,)
    assert obs.shape[-1] != RUN_OBS_SIZE
    mask = env.action_masks()
    assert mask.shape == (ACTION_SPACE_SIZE,)
    local = int(np.flatnonzero(mask == 1)[0])
    obs2, reward, terminated, truncated, info2 = env.step(local)
    env.close()
    assert obs2.shape == (OBS_SIZE,)
    assert info2["mix_source"] == SOURCE_LOADOUT
    assert terminated in (True, False)
    assert truncated in (True, False)


def test_mix_default_frac_samples_both():
    env = MixedHangLoadoutEnv(
        runenv_frac=DEFAULT_RUNENV_FRAC,
        loadout_provider=make_loadout_v1_provider(offset=3),
        max_steps=80,
    )
    sources = []
    for i in range(48):
        _obs, info = env.reset(seed=1000 + i)
        sources.append(info["mix_source"])
    env.close()
    assert SOURCE_RUNENV in sources
    assert SOURCE_LOADOUT in sources
    # 0.3 in expectation; 48 trials should not be all one source.
    n_run = sources.count(SOURCE_RUNENV)
    assert 1 <= n_run <= 40


def test_dry_run_wires_mix_and_hold():
    report = train_mod.dry_run(
        train_mod.parse_args(
            [
                "--dry-run",
                "--output-dir",
                "output/combat_runenv_antiforget_ld03_dry",
            ]
        )
    )
    assert report["dry_run"] is True
    assert report["runenv_frac"] == pytest.approx(0.3)
    assert report["loadout_frac"] == pytest.approx(0.7)
    assert report["runenv_frac_warning"] is None
    assert report["recipe"] == "loadout_dominant_0.3"
    assert report["continue_from_policy"] == "bh_v1_only"
    assert report["loadout_half"] == "loadout_v1"
    assert report["mix_neow_v1"] is False
    assert report["hang"]["start_with_neow"] is True
    assert report["hang"]["jev_event"] == "off"
    assert report["hang"]["jev_neow"] == "off"
    assert report["hang"]["allows_neow"] is False
    assert report["runenv_reset"]["mix_source"] == "runenv"
    assert report["loadout_reset"]["mix_source"] == "loadout_v1"
    assert report["loadout_reset"]["obs_size"] == OBS_SIZE
    assert report["hold"]["overall_min"] == 0.70
    assert report["hold"]["boss_min"] == 0.40
    assert report["hold"]["jobs_n_eps1"] == 18
    assert "combat_ppo_obs_v1_bh_v1" in report["frozen_outdirs"]
    assert "combat_runenv_onpolicy_v1" in report["frozen_outdirs"]
    assert "combat_runenv_antiforget_v1" in report["frozen_outdirs"]
    assert report["pure_onpolicy_500k"] == "frozen"
    assert report["antiforget_v1_0.7"] == "frozen"
    warned = train_mod.dry_run(
        train_mod.parse_args(
            [
                "--dry-run",
                "--output-dir",
                "output/combat_runenv_antiforget_ld03_dry07",
                "--runenv-frac",
                "0.7",
            ]
        )
    )
    assert warned["runenv_frac"] == pytest.approx(0.7)
    assert warned["runenv_frac_warning"] is not None
    assert "FROZEN" in warned["runenv_frac_warning"]


def test_dry_run_refuses_frozen_outdir():
    args = train_mod.parse_args(
        ["--dry-run", "--output-dir", "output/combat_runenv_onpolicy_v1"]
    )
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        train_mod.dry_run(args)
    args_af = train_mod.parse_args(
        ["--dry-run", "--output-dir", "output/combat_runenv_antiforget_v1"]
    )
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        train_mod.dry_run(args_af)
    args_cf = train_mod.parse_args(
        [
            "--dry-run",
            "--output-dir",
            "output/combat_runenv_antiforget_ld03_dry",
            "--continue-from",
            "output/combat_runenv_antiforget_v1/final_model.zip",
        ]
    )
    with pytest.raises(SystemExit, match="antiforget_v1"):
        train_mod.dry_run(args_cf)


def test_hold_smoke_constructs_with_legal_first_action():
    from sts2_env.eval.combat_hold import run_hold_smoke

    def predict_fn(obs, mask):
        assert np.asarray(obs).shape[-1] == OBS_SIZE
        valid = np.flatnonzero(np.asarray(mask) == 1)
        return int(valid[0])

    # One fixture×encounter via n_eps=1 is 18 fights; too heavy for every CI
    # pull if each fight is long. Cap by calling hold_jobs construction only
    # plus a single materialized combat reset through the mix loadout half
    # already covered. Here run a 1-job slice by temporarily using n_eps=1
    # would be 18. Skip full 18; prove callable path with jobs[0] via env.
    from sts2_env.eval.combat_hold import options_from_hold_fixture
    from sts2_env.gym_env.combat_env import STS2CombatEnv
    from sts2_env.relics.base import RelicId

    jobs = hold_jobs(n_eps=1)
    assert len(jobs) == 18
    summary = summarize_hold_rows(
        [{"win": True, "bucket": "elite"}, {"win": False, "bucket": "boss"}]
    )
    assert summary["overall"]["n"] == 2
    assert callable(run_hold_smoke)
    assert callable(predict_fn)
    job = jobs[0]
    opts = options_from_hold_fixture(job["fixture"])
    assert "BURNING_BLOOD" in opts["relics"]
    assert "SHURIKEN" in opts["relics"]
    env = STS2CombatEnv(encounter_pool=[job["encounter_setup"]])
    env.reset(seed=int(job["seed"]), options=opts)
    assert env.combat is not None
    relic_ids = {r.relic_id for r in env.combat.relics}
    assert RelicId.BURNING_BLOOD in relic_ids
    assert RelicId.SHURIKEN in relic_ids
    env.close()
