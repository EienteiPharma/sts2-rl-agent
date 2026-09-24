# bh_assist contract (track 2)

**Status:** train entry present (`scripts/train_bh_assist_from_buffer.py`, `sts2_env/eval/bh_assist_train.py`). Default outdir `output/combat_bh_assist_v1`. **No hang runtime swap** — execution stays `bh_v1` / `--combat-policy ppo`. Eval±assist HOLD dual gate is **later**; do not treat assist as primary actor.  
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
- Live HTTP belongs in a future adapter; this tip keeps **offline-safe** stub only (`sts2_env/eval/bh_assist.py`).

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
- Outdir: `output/combat_bh_assist_v1` (or `--output-dir` with `refuse_bh_assist_output_path` guard).
- **Hard ban:** never write into `combat_ppo_obs_v1_bh_v1`, `combat_runenv_onpolicy_v1`, `combat_runenv_antiforget_v1`, or overwrite hang `final_model.zip`.
- **Hard ban:** no `MaskablePPO.learn` / no `--continue-from` hang zip on this path.
- Labels target Jev-facing **`ranked_semantic`** (subset/reorder of legal keys) and **`risk_notes`**; buffer v1 uses partial semantic proxy (`end_turn` when expert ends turn) — full semantic labels when collect adds board keys is a later tip.
- Runtime `bh_assist()` uses heuristic rank + risk notes until checkpoint wiring lands in a later tip.

## Why not stepwise Jev

HOLD smoke collapsed twice with Jev **selecting combat steps** (`combat_step_choice`), including after confidence gate 0.45 — fail-open dropped but win rates worsened. Poison = **selected steps**, not gate alone. Archive: `docs/COMBAT_JEV_HOLD_FAIL.md`. Track 2 separates **hinting** (`bh_assist`) from **execution** (`bh_v1`) and from **plan Choice** (`combat_turn_plan_choice`).

## Stub

Reference API: `sts2_env/eval/bh_assist.py` (heuristic hints); train helpers: `sts2_env/eval/bh_assist_train.py`.
