#!/usr/bin/env python3
"""Build forward-inject HOLD jobs from a Boss/fail replay pack (no rewind)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sts2_env.eval.boss_forward_inject import (
    build_inject_jobs_from_pack,
    smoke_forward_inject,
    write_inject_jobs,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Forward-inject HOLD jobs from boss_fail pack (seed reset only)"
    )
    p.add_argument(
        "--pack-dir",
        default="evals/boss_fail_pack_v1",
        help="Pack directory with manifest.json + episodes.jsonl",
    )
    p.add_argument(
        "--out",
        default="evals/boss_fail_pack_v1/inject_jobs.json",
        help="Write inject job list JSON",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only write inject_jobs.json (default behavior; flag for clarity)",
    )
    p.add_argument(
        "--smoke-n",
        type=int,
        default=0,
        help="If >0, run N jobs with random-legal predict (no TypeSafe)",
    )
    args = p.parse_args(argv)
    jobs, meta = build_inject_jobs_from_pack(args.pack_dir)
    out_path = write_inject_jobs(args.out, jobs, meta)
    print(f"inject_jobs: {out_path} ({len(jobs)} jobs)")
    if int(args.smoke_n) > 0:
        rows = smoke_forward_inject(jobs, n=int(args.smoke_n))
        print(json.dumps({"smoke_n": len(rows), "rows": rows}, indent=2))
    if args.dry_run:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
