# Boss forward inject (T3 — pack + seed replay)

**Status:** offline pack from HOLD turn-replay JSONL; forward inject only. **No engine rewind** — mid-board restore from `turns` / `terminal` is out of scope (would be rewind/materialize). Next use after Lab go: assist **v3** training data source from Boss residual failures.

## Turn replay source

- Protocol: `hold_turn_plan_replay_v1` (`sts2_env/eval/hold_turn_replay.py`)
- Files: `evals/hold_turn_replay/hold_turn_replay_w*.jsonl` from `jev-turn` HOLD runs

## Pack (`boss_fail_pack_v1`)

```bash
PYTHONPATH=. python scripts/pack_hold_turn_replay.py \
  evals/hold_turn_replay/hold_turn_replay_w0.jsonl \
  --out-dir evals/boss_fail_pack_v1 \
  --filter boss_fail
```

| `--filter` | Keeps |
|---|---|
| `boss_fail` (default) | `bucket==boss` **and** `win==false` |
| `boss_all` | all Boss-bucket episodes |
| `fail_all` | all losses |
| `retain_writer` | same as replay writer: loss **or** Boss |

Outputs:

- `evals/boss_fail_pack_v1/episodes.jsonl` — filtered episode docs (unchanged)
- `evals/boss_fail_pack_v1/manifest.json` — `protocol`, sources, `n_episodes`, `enc_counts`

## Forward inject (not rewind)

```bash
PYTHONPATH=. python scripts/boss_forward_inject.py \
  --pack-dir evals/boss_fail_pack_v1 \
  --out evals/boss_fail_pack_v1/inject_jobs.json
```

Builds HOLD-style jobs from each episode’s `seed`, `fixture_index`, `enc_id`, `ep` (validates `HOLD_SEED_BASE=40000` formula in `combat_hold.py`). Fights **start from `env.reset(seed, fixture loadout)`** — same as formal HOLD — not from logged board state.

Optional smoke (no TypeSafe):

```bash
PYTHONPATH=. python scripts/boss_forward_inject.py --pack-dir evals/boss_fail_pack_v1 --smoke-n 2
```

Helpers: `sts2_env/eval/hold_replay_pack.py`, `sts2_env/eval/boss_forward_inject.py`.

## Assist v3 train path (EP)

```bash
# 1) Collect buffer from inject jobs (hung PPO; no TypeSafe)
PYTHONPATH=. python scripts/collect_boss_fail_inject_buffer.py \
  --pack-dir evals/boss_fail_pack_v1 \
  --out output/boss_fail_inject_buffer_v1/transitions.npz

# 2) Train assist ranker to v3 outdir (does not overwrite v1/v2)
PYTHONPATH=. python scripts/train_bh_assist_from_buffer.py \
  --buffer output/boss_fail_inject_buffer_v1/transitions.npz \
  --output-dir output/combat_bh_assist_v3 \
  --pack-dir evals/boss_fail_pack_v1 \
  --train-steps 128
```

`refuse_bh_assist_output_path` blocks writes into `combat_bh_assist_v1` / `combat_bh_assist_v2` and frozen hang trees. In-service assist remains **v1** until Lab promotes v3.

Related: `docs/BH_ASSIST_CONTRACT.md` (assist v1 in-service; v3 data path).
