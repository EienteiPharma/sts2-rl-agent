# bh_assist contract (track 2)

**Status:** train entry + turn-plan eval wiring (`--bh-assist`, `73fc1ed`). Default train/eval checkpoint tree is **`output/combat_bh_assist_v3`**. Lab in-service nail: **`d9d9fff` + `combat_bh_assist_v3`** (ranker apply no longer ties `end_turn` at gym action 0). v1/v2 outdirs are **write-protected** (do not overwrite). Later HEAD tips after `d9d9fff` (remap default off `7bbe40d`, short context `4ec6b26`, colab_v1 collect `3794cb8`/`064ff81`) are on the branch but **not** re-nailed as 现役 until Lab re-HOLDs. **No hang runtime swap** — execution stays `bh_v1` / `--combat-policy ppo`. Assist +3/+2 vs jev-turn **not yet passed**. **Pre-train eval gates** and **eval stability (`lock_eval_then_v3`)** below; hang dual gate vs `bh_v1` unchanged (`docs/HOLD_PROTOCOL.md`). Do not treat assist as primary actor.  
**Successor to:** abandoned stepwise `combat_step_choice` / `--combat-policy jev` (see `docs/COMBAT_JEV_HOLD_FAIL.md`).  
**Related:** turn-plan path A in `sts2_env/eval/combat_turn_plan.py`; hang execution stays **`bh_v1`** until a future **turn-plan HOLD** clears.

## Scope

| In | Out |
|---|---|
| Jev-callable **advisory** helper on a full combat **board** | Primary combat actor |
| Ranked **semantic** step hints + **risk notes** | `MaskablePPO.predict` as hang policy |
| State tidy for prompts (deterministic, code-side) | `combat_step_choice` or opaque `aN` ids |
| Contract + offline train entry (assist ranker, not hang PPO) | Hang zip swap, eval flag changes, wired turn-plan Choice |

Hang combat execution remains **`--combat-policy ppo`** on hung zip `combat_ppo_obs_v1_bh_v1` (`bh_v1`). Noncombat Jev (MAP / REST / CARD) unchanged. Dual gate unchanged: `docs/HOLD_PROTOCOL.md`, `docs/HANG_PROTOCOL_2026-09-22.md`.

## API

```text
bh_assist(board, *, legal_semantic_keys, context?) → BhAssistResult
```

**Inputs**

- `board`: full combat board dict (same shape as `serialize_combat_board_full` in `combat_turn_plan.py`: self, piles, enemies, turn).
- `legal_semantic_keys`: sorted unique keys for **currently legal** gym actions (same family as `legal_semantic_keys()` / `semantic_key_for_gym_action()`).
- `context` (optional): small fixed dict for callers only — e.g. `fixture_id`, `encounter_id`, shadow run id. **Not** a free-form model scratchpad.

**Output (`BhAssistResult`)**

| field | type | meaning |
|---|---|---|
| `ranked_semantic` | `tuple[str, ...]` | Subset/reordering of **legal** semantic keys, best-first. Must ⊆ `legal_semantic_keys`. Empty allowed. |
| `risk_notes` | `tuple[str, ...]` | Short, human-readable bullets (intent leak, lethal line, energy cliff, etc.). **Advisory**; must not invent illegal steps. |

Equivalent dict wire shape for logging: `{ "ranked_semantic": [...], "risk_notes": [...] }`.

**Invariants**

1. Every entry in `ranked_semantic` must be in `legal_semantic_keys`.
2. Helper **must not** emit gym action indices, `plan_id`, or keys outside the legal set.
3. Helper **must not** call hung PPO or override the step actually taken.
4. **`combat_step_choice` is banned** on this track (do not add Choice names, criteria, or fail-open paths that mirror `sts2_env/eval/combat_jev.py`).

## Jev role (not actor)

`bh_assist` is a **Jev-callable helper**, same *class* as noncombat assists (`jev_card_fit_assist`, `jev_hp_pressure_assist`): it may **tidy** board text, **propose** ranked semantics, and **surface** risks for a downstream Choice or shadow log.

It is **not** the combat decision engine:

- Does **not** replace `MaskablePPO.predict` on the hang path.
- Does **not** own turn execution; at most informs `combat_turn_plan_choice` (plan pick among code-enumerated plans) in a **later** tip after HOLD.
- Live HTTP belongs in a future adapter; runtime hints today are heuristic-only (`sts2_env/eval/bh_assist.py`) until eval gates pass and wiring lands.

Suggested call flow (future, not wired on tip④):

```text
board ← serialize_combat_board_full(combat, mask)
keys  ← legal_semantic_keys(combat, mask)
assist ← bh_assist(board, legal_semantic_keys=keys)
# Jev system prompt may include assist.risk_notes + assist.ranked_semantic as hints only
plans ← enumerate_candidate_plans(keys)
# Jev Choice: combat_turn_plan_choice among plan_id (unchanged from tip①)
# Step execution: still bh_v1 per plan step until turn-plan HOLD clears
```

## Train entry (assist-only)

- Script: `scripts/train_bh_assist_from_buffer.py` (buffer → linear assist ranker + manifest).
- Outdir: **`output/combat_bh_assist_v3`** (CLI default). v1/v2 paths refused by `refuse_bh_assist_output_path`.
- **Hard ban:** never write into `combat_ppo_obs_v1_bh_v1`, `combat_runenv_onpolicy_v1`, `combat_runenv_antiforget_v1`, or overwrite hang `final_model.zip`.
- **Hard ban:** no `MaskablePPO.learn` / no `--continue-from` hang zip on this path.
- Labels target Jev-facing **`ranked_semantic`** (subset/reorder of legal keys) and **`risk_notes`**; buffer v1 uses partial semantic proxy (`end_turn` when expert ends turn) — full semantic labels when collect adds board keys is a later tip.
- Runtime `bh_assist()` uses heuristic rank, then the loaded v3 ranker when `--bh-assist on` finds a checkpoint (missing ckpt fail-opens to assist-off / heuristic). Eval default ckpt: `/workspace/sts2-sim/output/combat_bh_assist_v3/bh_assist_ranker.npz`.

## Pre-train eval gates (Lab / Jev lock)

Run **before** promoting a trained assist checkpoint into turn-plan Choice wiring. Hang step execution remains **`bh_v1`** (`--combat-policy ppo` on the hang zip); assist is hints only and must **not** replace hang policy or swap the frozen zip.

### Contrast arms (same HOLD protocol)

| Arm | Combat policy | Assist |
|---|---|---|
| **A** | `jev-turn` | off |
| **B** | `jev-turn` | on (`bh_assist` hints in plan Choice state/prompt) |

Both arms use the same hung combat zip for **execution** (plan steps / fail-open catastrophe path still fail-open to `bh_v1`, not assist-as-actor). Do not use stepwise `combat_step_choice` or `--combat-policy jev`.

### Eval stability (`lock_eval_then_v3`)

Eval-only lock (no new train on this tip). Hang stays `bh_v1`; assist checkpoint promotion uses **formal** HOLD contrast only.

| Rule | Locked value |
|---|---|
| Episodes for **promotion / gate decisions** | **`--n-eps 20`** only (360 fights = hang table shape) |
| **`n_eps=5` / small-n smoke** | **Diagnostic only** — telemetry sanity, not assist +3/+2, not HOLD promotion, not train promotion |
| Formal parallel shape | **`--workers 8`** default for Lab formal runs (A/B when possible; same as hang sentry in `docs/HOLD_PROTOCOL.md`) |
| Contrast seeds | **`HOLD_SEED_BASE=40000`**, formula **`40000 + fixture_index*1000 + enc_id*100 + ep`** (`env.reset(seed=…)`). Implemented as `HOLD_SEED_BASE` / `HOLD_SEED_FORMULA` in `sts2_env/eval/combat_hold.py`. **Do not change** the formula; arms A and B must share the same job list and seeds. |

Suite JSON includes `seed_base` / `seed_formula` via `hold_protocol_meta()`.

### Protocol

- **Smoke (diagnostic):** `scripts/eval_combat_suite.py --suite loadout_v1 --workers 1` (`w1`) on A and B; optional `--n-eps 5` or `--n-eps 1` for quick checks only — **never** for promotion decisions.
- **Formal (gates):** `--suite loadout_v1 --n-eps 20 --workers 8` on A and B; compare summary JSON overall / Boss for the +3 / +2 bar.

Example formal arm:

```bash
PYTHONPATH=. python scripts/eval_combat_suite.py \
  --suite loadout_v1 --n-eps 20 --workers 8 \
  --combat-policy jev-turn --bh-assist off \
  --model /workspace/sts2-sim/output/combat_ppo_obs_v1_bh_v1/final_model.zip
```

(B arm: same flags with `--bh-assist on` and the same ckpt path.)

### Episode buckets — catastrophe vs plan quality (reporting)

Overall WR (`wr_any`, same as `overall.win_rate`) mixes Jev-fulfilled turns with turn-plan **catastrophe** fail-open (timeout / error / illegal_plan / replan_cap / …). Reporting-only split (does **not** change `passed` dual-gate or assist +3/+2 math):

| Summary key | Meaning |
|---|---|
| `wr_any` | Alias of existing overall WR (unchanged gate input) |
| `episodes_clean` / `episodes_with_catastrophe` | Episode had **no** / **any** turn-plan catastrophe fail-open |
| `win_rate_clean` / `win_rate_had_catastrophe` | WR within those episode sets |
| `boss_win_rate_clean` / `boss_win_rate_had_catastrophe` | Boss-bucket subset |
| `combat_jev` block | Turn-level `jev_fulfilled_rate`, `catastrophe_failopen_rate`, reason histogram (unchanged) |
| `combat_jev.turn_plan_replan_cap_bucket` | **Reporting-only** split when `replan_cap` fires: `true_replan_exhaustion` (invalidation-driven replans ate the budget) vs `shortlist_or_short_plan_idle` (degenerate ≤1 legal key / single-step shortlist Jev spin). Does **not** change caps or gates. Formal HOLD `d9d9fff` A166/B140 replan_cap: use with `turn_plan_replan_trigger` to prioritize fixes. |

Per-fight rows (jev-turn) include `had_turn_plan_catastrophe`, `turn_plan_fulfilled_turns`, `turn_plan_catastrophe_turns`, `turn_plan_catastrophe_reasons`. Turn replay JSONL rows with `replan_cap_hit` may add `replan_cap_bucket` + `replan_cap_diag` (旁证).

### Lab acceptance — `replan_cap` tips (reporting only)

Any tip whose goal is to **reduce** turn-plan `replan_cap` must ship evidence as **bucket split**, not `jev_turn_catastrophe_reason.replan_cap` total alone:

| Required in tip / summary JSON | Field |
|---|---|
| Per-arm bucket counts | `replan_cap_lab.buckets.true_replan_exhaustion` / `shortlist_or_short_plan_idle` |
| Trigger mix | `replan_cap_lab.replan_triggers` (`illegal_step`, …) |
| Coverage check | `replan_cap_lab.lab_acceptance.bucket_coverage_ok` (sum(buckets) == events when events > 0) |

`eval_combat_suite.py` (`--combat-policy jev-turn`) writes top-level **`replan_cap_lab`** with `arm` = `A_assist_off` or `B_assist_on` and prints one stdout line. **Formal A/B both high** (e.g. `d9d9fff` replan_cap A166 / B140): contrast **two** summaries’ `replan_cap_lab` blocks. Execute-path fixes should move **both** arms similarly (classifier is assist-invariant); if only B moves, explain via Choice/plan-length skew, not execute-only wiring. Does **not** change +3/+2, dual gate, or `MAX_REPLANS`.

### Assist effectiveness bar (A vs B on `jev-turn`)

Assist counts as **effective** only if arm B beats arm A on loadout_v1 HOLD win rates by:

- **overall ≥ +3pp** (percentage points), **and**
- **Boss ≥ +2pp**

Report Δ as `B − A` on overall / elite / Boss columns from the suite summary JSON. This bar is **assist-on vs assist-off** on the turn-plan arm; it does **not** replace the hang-table dual gate (overall ≥70 / Boss ≥40 vs frozen `bh_v1` ppo baseline).

### Collapse / STOP (still applies)

Any HOLD run that includes hang comparison still uses existing **collapse STOP** thresholds vs hang arm A (`--combat-policy ppo` / `bh_v1`) where Lab protocol requires it (large negative Δ vs 74.2 / 49.4 table — see `docs/HOLD_PROTOCOL.md`, `docs/COMBAT_JEV_HOLD_FAIL.md`). The **+3 / +2** rule above is additional: even without hang collapse, assist-off vs assist-on must clear the effectiveness bar before train promotion.

### Three-arm contrast (hang \| A \| B) + hang-gap STOP

Helper: `sts2_env/eval/hold_assist_contrast.py` and `scripts/contrast_hold_assist.py --hang … --a … --b …` on three HOLD summary JSONs.

| Column | Arm |
|---|---|
| hang | `--combat-policy ppo` / `bh_v1` |
| A | `jev-turn` assist off |
| B | `jev-turn` assist on |

Emits overall / elite / Boss WR table, Δ_A vs hang, Δ_B vs hang, Δ_assist (B−A) in **percentage points**. **Hang-gap STOP (locked):** trigger if **overall** Δ_A ≤ **−3.0pp** or overall Δ_B ≤ **−3.0pp** (hang baseline worsened by ≥3pp). Assist promotion bar remains **B−A** overall ≥ +3pp and Boss ≥ +2pp (unchanged). STOP does not replace dual-gate `passed` on a single summary file.

## Why not stepwise Jev

HOLD smoke collapsed twice with Jev **selecting combat steps** (`combat_step_choice`), including after confidence gate 0.45 — fail-open dropped but win rates worsened. Poison = **selected steps**, not gate alone. Archive: `docs/COMBAT_JEV_HOLD_FAIL.md`. Track 2 separates **hinting** (`bh_assist`) from **execution** (`bh_v1`) and from **plan Choice** (`combat_turn_plan_choice`).

## Stub

Reference API: `sts2_env/eval/bh_assist.py` (heuristic hints); train helpers: `sts2_env/eval/bh_assist_train.py`.

### Ranker apply (fixed 2026-09-24)

Runtime ``rank_semantic_keys_with_ranker`` must **not** tie-break on gym action index: ``ACTION_END_TURN==0`` made ``ranked_semantic[0]`` always ``end_turn``. Apply now scores each legal semantic with turn-plan heuristic ``score_turn_plan_candidate`` (+ tiny ``dot(w,obs)`` nudge). Single-episode rank checks are **diagnostic only** (not promotion gates).

## Boss residual pack + forward inject (T3)

Offline only — no policy/prompt/ranker changes. Slice `hold_turn_plan_replay_v1` JSONL → `boss_fail_pack_v1` (`scripts/pack_hold_turn_replay.py`); rebuild HOLD reset jobs for the same seeds (`scripts/boss_forward_inject.py`); collect buffer with hung PPO (`scripts/collect_boss_fail_inject_buffer.py`); train to **`output/combat_bh_assist_v3`** (`scripts/train_bh_assist_from_buffer.py` — refuses v1/v2 outdirs). **No engine rewind** from turn logs. See `docs/BOSS_FORWARD_INJECT.md`.
