"""Offline pack slicer for HOLD turn-replay JSONL (Boss / fail episodes)."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Literal

from sts2_env.eval.hold_turn_replay import (
    HOLD_TURN_REPLAY_PROTOCOL,
    should_retain_hold_turn_replay,
)

BOSS_FAIL_PACK_PROTOCOL = "boss_fail_pack_v1"
DEFAULT_BOSS_FAIL_PACK_DIR = "evals/boss_fail_pack_v1"

PackFilterMode = Literal["boss_fail", "boss_all", "fail_all", "retain_writer"]


def iter_replay_jsonl(paths: Iterable[str | Path]) -> tuple[list[Path], list[dict[str, Any]]]:
    resolved: list[Path] = []
    docs: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"replay JSONL not found: {path}")
        resolved.append(path.resolve())
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            text = line.strip()
            if not text:
                continue
            doc = json.loads(text)
            if not isinstance(doc, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object per line")
            docs.append(doc)
    return resolved, docs


def matches_pack_filter(doc: dict[str, Any], mode: PackFilterMode) -> bool:
    bucket = str(doc.get("bucket") or "")
    win = bool(doc.get("win"))
    if mode == "retain_writer":
        return should_retain_hold_turn_replay(win=win, bucket=bucket)
    if mode == "boss_fail":
        return bucket == "boss" and not win
    if mode == "boss_all":
        return bucket == "boss"
    if mode == "fail_all":
        return not win
    raise ValueError(f"unknown pack filter mode {mode!r}")


def filter_replay_docs(
    docs: list[dict[str, Any]], mode: PackFilterMode
) -> list[dict[str, Any]]:
    return [d for d in docs if matches_pack_filter(d, mode)]


def _file_sha256(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in sorted(paths):
        h.update(str(path).encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def write_boss_fail_pack(
    *,
    source_paths: list[Path],
    episodes: list[dict[str, Any]],
    out_dir: str | Path,
    filter_mode: PackFilterMode,
    tip_note: str = "T3 hold replay pack; forward inject from seed — no engine rewind",
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    episodes_path = out / "episodes.jsonl"
    with episodes_path.open("w", encoding="utf-8") as f:
        for doc in episodes:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    enc_counts = Counter(str(int(d.get("enc_id") or 0)) for d in episodes)
    manifest: dict[str, Any] = {
        "protocol": BOSS_FAIL_PACK_PROTOCOL,
        "replay_protocol": HOLD_TURN_REPLAY_PROTOCOL,
        "filter_mode": filter_mode,
        "source_paths": [str(p) for p in source_paths],
        "source_bundle_sha256_16": _file_sha256(source_paths) if source_paths else "",
        "n_episodes": len(episodes),
        "enc_counts": dict(sorted(enc_counts.items(), key=lambda kv: int(kv[0]))),
        "episodes_file": episodes_path.name,
        "tip_note": tip_note,
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest["pack_dir"] = str(out.resolve())
    return manifest


def pack_hold_turn_replay_files(
    inputs: list[str | Path],
    *,
    out_dir: str | Path = DEFAULT_BOSS_FAIL_PACK_DIR,
    filter_mode: PackFilterMode = "boss_fail",
) -> dict[str, Any]:
    source_paths, docs = iter_replay_jsonl(inputs)
    filtered = filter_replay_docs(docs, filter_mode)
    return write_boss_fail_pack(
        source_paths=source_paths,
        episodes=filtered,
        out_dir=out_dir,
        filter_mode=filter_mode,
    )


def load_pack_manifest(pack_dir: str | Path) -> dict[str, Any]:
    path = Path(pack_dir) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"pack manifest missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"invalid manifest: {path}")
    return data


def load_pack_episodes(pack_dir: str | Path) -> list[dict[str, Any]]:
    manifest = load_pack_manifest(pack_dir)
    ep_file = Path(pack_dir) / str(manifest.get("episodes_file") or "episodes.jsonl")
    if not ep_file.is_file():
        raise FileNotFoundError(f"pack episodes missing: {ep_file}")
    docs: list[dict[str, Any]] = []
    for line in ep_file.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            doc = json.loads(text)
            if isinstance(doc, dict):
                docs.append(doc)
    return docs


__all__ = [
    "BOSS_FAIL_PACK_PROTOCOL",
    "DEFAULT_BOSS_FAIL_PACK_DIR",
    "PackFilterMode",
    "filter_replay_docs",
    "iter_replay_jsonl",
    "load_pack_episodes",
    "load_pack_manifest",
    "matches_pack_filter",
    "pack_hold_turn_replay_files",
    "write_boss_fail_pack",
]
