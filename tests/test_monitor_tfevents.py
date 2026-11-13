from __future__ import annotations

import struct
from pathlib import Path

from sts2_env.web._monitor.tfevents import (
    _masked_crc32,
    build_event_payload,
    parse_tfevents,
    wrap_tfrecord,
)


def test_parse_single_event(tmp_path: Path):
    payload = build_event_payload(42, "rollout/ep_rew_mean", -0.5)
    record = wrap_tfrecord(payload)
    p = tmp_path / "events.out.tfevents.0"
    p.write_bytes(record)
    cur, facts, delivered = parse_tfevents(p, None)
    assert delivered
    assert cur.max_step == 42
    assert cur.byte_offset == len(record)
    assert any(f.tag == "rollout/ep_rew_mean" for f in facts)


def test_truncated_record_does_not_advance(tmp_path: Path):
    payload = build_event_payload(1, "time/fps", 100.0)
    record = wrap_tfrecord(payload)
    broken = record[: len(record) // 2]
    p = tmp_path / "events.out.tfevents.0"
    p.write_bytes(broken)
    cur, _, delivered = parse_tfevents(p, None)
    assert not delivered
    assert cur.byte_offset == 0


def test_shrink_resets_cursor_and_reads_new_step(tmp_path: Path):
    p = tmp_path / "events.out.tfevents.0"
    first = wrap_tfrecord(build_event_payload(10, "rollout/ep_rew_mean_very_long_tag", 1.0))
    p.write_bytes(first)
    cur, _facts, delivered = parse_tfevents(p, None)
    assert delivered
    assert cur.byte_offset == len(first)
    second = wrap_tfrecord(build_event_payload(3, "time/fps", 2.0))
    assert len(second) < cur.byte_offset
    p.write_bytes(second)
    cur2, facts, delivered2 = parse_tfevents(p, cur)
    assert delivered2
    assert cur2.byte_offset == len(second)
    assert cur2.max_step == 3
    assert any(f.step == 3 for f in facts)


def test_multiple_files_union_max_step(tmp_path: Path):
    d = tmp_path / "tb_logs" / "MaskablePPO_0"
    d.mkdir(parents=True)
    a = d / "events.out.tfevents.0"
    b = d / "events.out.tfevents.1"
    a.write_bytes(wrap_tfrecord(build_event_payload(10, "rollout/ep_rew_mean", 1.0)))
    b.write_bytes(wrap_tfrecord(build_event_payload(25, "rollout/ep_rew_mean", 2.0)))
    cur_a, _, _ = parse_tfevents(a, None)
    cur_b, _, _ = parse_tfevents(b, None)
    assert max(cur_a.max_step, cur_b.max_step) == 25


def test_tf_masked_crc32c_matches_real_length_header():
    length = struct.pack("<Q", 72)
    assert _masked_crc32(length) == 0x9D4892FE


def test_parse_real_e3_tfevents_has_a_step():
    root = Path(r"D:\workspace\sts2-sim\output\run_ppo_act2_obs1309_gate_e3\tb_logs")
    files = sorted(root.rglob("events.out.tfevents.*")) if root.is_dir() else []
    if not files:
        return
    cur, _facts, delivered = parse_tfevents(files[-1], None)
    assert cur.byte_offset > 0
    assert delivered
    assert cur.max_step > 0
