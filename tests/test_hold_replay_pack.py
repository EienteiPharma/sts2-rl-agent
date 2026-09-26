"""Pack slicer + forward inject (synthetic JSONL)."""
from __future__ import annotations

import json
from pathlib import Path

from sts2_env.eval.boss_forward_inject import (
    build_inject_jobs_from_pack,
    hold_job_from_replay_episode,
    infer_ep_from_seed,
)
from sts2_env.eval.combat_hold import HOLD_SEED_BASE, load_hold_fixtures
from sts2_env.eval.hold_replay_pack import (
    filter_replay_docs,
    pack_hold_turn_replay_files,
)
from sts2_env.eval.hold_turn_replay import HOLD_TURN_REPLAY_PROTOCOL


def _ep_doc(*, seed: int, fix: int, enc: int, ep: int, bucket: str, win: bool) -> dict:
    return {
        "protocol": HOLD_TURN_REPLAY_PROTOCOL,
        "seed": seed,
        "fixture_index": fix,
        "enc_id": enc,
        "ep": ep,
        "bucket": bucket,
        "win": win,
        "steps": 10,
        "turns": [],
        "terminal": {},
    }


def test_pack_boss_fail_filter(tmp_path):
    inp = tmp_path / "hold_turn_replay_w0.jsonl"
    docs = [
        _ep_doc(seed=41900, fix=0, enc=19, ep=0, bucket="boss", win=False),
        _ep_doc(seed=41901, fix=0, enc=19, ep=1, bucket="boss", win=True),
        _ep_doc(seed=41600, fix=0, enc=16, ep=0, bucket="elite", win=False),
    ]
    inp.write_text("\n".join(json.dumps(d) for d in docs) + "\n")
    out_dir = tmp_path / "pack"
    manifest = pack_hold_turn_replay_files(
        [inp], out_dir=out_dir, filter_mode="boss_fail"
    )
    assert manifest["n_episodes"] == 1
    assert manifest["enc_counts"] == {"19": 1}
    kept = filter_replay_docs(docs, "fail_all")
    assert len(kept) == 2


def test_forward_inject_jobs_match_seeds(tmp_path):
    inp = tmp_path / "replay.jsonl"
    doc = _ep_doc(seed=41900, fix=0, enc=19, ep=0, bucket="boss", win=False)
    inp.write_text(json.dumps(doc) + "\n")
    pack_dir = tmp_path / "pack"
    pack_hold_turn_replay_files([inp], out_dir=pack_dir, filter_mode="retain_writer")
    jobs, meta = build_inject_jobs_from_pack(pack_dir)
    assert len(jobs) == 1
    assert jobs[0]["seed"] == 41900
    assert jobs[0]["enc_id"] == 19
    assert meta["seed_base"] == HOLD_SEED_BASE
    fixtures = load_hold_fixtures()
    job = hold_job_from_replay_episode(doc, fixtures)
    assert job["seed"] == HOLD_SEED_BASE + 0 * 1000 + 19 * 100 + 0
    assert infer_ep_from_seed(41900, 0, 19) == 0
