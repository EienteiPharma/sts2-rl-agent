# Jev Choice 契约：EVENT / Unknown /（可选）Neow

**状态：** LOCKED for Surplus box smoke → `EienteiPharma/sts2-rl-agent` PR#1  
**前提：** 战斗硬冻；Choice ≥ **0.65** 不变；MAP/REST/CARD 边界不改。  
**权威：** 本文件为 Jev router 契约。先前「EVENT pending 一律合法随机」的 stub **作废**。

## 范围

| 目标 | RunEnv 相位 | 说明 |
|------|-------------|------|
| **事件选项** | `EVENT` | `event_choice` 列表；`event_id=Neow` 走 `neow_boon` |
| **Unknown 地图点** | `MAP_CHOICE` | 已有 `map_fork`；`point_type=UNKNOWN` 加强 criteria，**不新开相位** |
| **事件内二次选择** | `EVENT` + pending `choose`/`confirm_choice` | **接线 Jev**（mask 走 combat 槽：`combat_start` / `combat_start+1+i`） |

**不做：** SHOP 全量（仍合法随机）、宝箱、Boss relic、combat 训。

## Choice 名（TypeSafe）

| id | 相位 | 何时建 options |
|----|------|----------------|
| `event_choice` | `EVENT` | `action==event_choice` 且 `enabled!=False`；pending 时 `choose`/`confirm_choice` |
| `neow_boon` | `EVENT` 且检出 Neow/boon 屏 | 同上；criteria 强调开局祝福。未检出则 **静默跳过** `neow_boon`（走 `event_choice`） |
| `map_fork` | `MAP_CHOICE` | **已有**；UNKNOWN 节点 criteria 补风险句 |

## A) MAP Unknown defer

合法选项含 `point_type=UNKNOWN` 时：

- **仍打** `hp_pressure` Score（与 rest 同档：≥2.0 高压）。
- 若 `hp_pressure >= 2` **且** 存在非 Unknown 合法点 **且** Choice 选了 Unknown 且 `confidence < 0.80` → **defer**：合法随机（优先非 Unknown 池），reason `unknown_deferred`，写入 shadow 日志。
- Choice 门槛仍是 **0.65**；0.80 只用于 Unknown defer，不改主阈值。

UNKNOWN 只走 `map_fork`，不要在 EVENT 相位伪造 Unknown。

**MAP low-HP (`map_lowhp` v2，hang 默认 on):** `hp_pressure>=2` 且合法含 shop/rest 时 **hard-select** rest-then-shop（`map_lowhp_hard`），不论 Jev 自信与否。仅剩 fight 则全池随机。`--map-lowhp off` 关。EVENT 仍默认 off。

## B) EVENT（`--jev-event on`，默认 off）

`EVENT ∈ JEV_PHASES` 仅当 `--jev-event on`（或 `--jev-phases` 含 `event`）。默认 `map,rest,card`，现表不变。

`build_options`：

```
event_choice list index i  →  _EVENT_START + i   (i < event_size=4, mask==1, enabled only)
pending confirm_choice     →  _COMBAT_START
pending choose index i     →  _COMBAT_START + 1 + i
```

- `option_id`：优先 `a["option_id"]`；否则 `event_{i}_{label}`
- `label`：`"{label}: {description}"`
- meta：`{event_id, option_id, list_index, is_neow}`
- 不可见 / `enabled=False`：不建 option
- Choice 名：`event_choice`（非 Neow）
- state 带 `content_map`、血/金/牌库；criteria 按选项字面效果。标「待核」的事件 **只写字面风险，勿写死规则**。
- `--jev-event off`：普通 EVENT 合法随机，reason `jev_event_off_random`；检出的 Neow 仍走 `neow_boon`（≥0.65）
- 药水/遗物奖励屏仍走 CARD 既有规则，**不**进 `event_choice`

## C) Neow → `neow_boon`

检出 Neow/boon 屏（`event_id==Neow` 或 meta/id 含 boon）时 Choice 名为 `neow_boon`。未检出则静默跳过该 Choice 名。`--jev-neow` 默认 **off**（随机祝福，`neow_jev_off_random`）。`--jev-neow on` 为可选 A/B（`neow_boon` ≥0.65，即使 `--jev-event off`）。Hang：`--start-with-neow` + `--jev-neow off`。REST 0.50 / heal/smith assist **不**套到 Neow。shadow 可选 `phase=NEOW`。

**测 `neow_boon` 必须 `--jev-neow on` 且 `--start-with-neow`。Hang 带 `--start-with-neow` 但 `--jev-neow off`（随机祝福）。** Gym：`reset(..., options={"start_with_neow": True})` → `RunManager(..., start_with_neow=True)`。`RunEnv` 与 `_enter_neow` 在 `get_event("Neow")` **之前** `import sts2_env.events`。`docs/CARDS_REFERENCE.md` 从 package root 解析（不靠 cwd）。

若 Neow 屏合法非 Leave 选项 **< 2**（Leave-only / 空祝福）：**不调 Jev**，reason `neow_options_empty`，合法随机。正常开局为 3 个 `event_choice` 祝福。

EVENT 同样：合法非 Leave **< 2** 时不调 Choice，reason `event_options_empty`（不算落地率）。`_actions_event` **不得**在 `_event_model` 仍在、`_event_options` 空（reward/pending 间隙）时伪造 Leave；Leave 仅当 `event_model is None`。

禁止把 Neow 当 mid fixture 选。开局容错（血/金/牌/遗物）。

## D) Shop

SHOP **仍合法随机**，本刀不扩。

## 开关（eval CLI）

```
--jev-phases map,rest,card          # 现默认等价
--jev-phases map,rest,card,event    # 打开普通 EVENT（Neow 仍默认随机）
--jev-event on|off   (default off)
--jev-neow on|off    (default off; on = optional A/B neow_boon)
--start-with-neow    (hang 带上，与 --jev-neow off 配对)
```

`--jev off` 时 EVENT 仍 masked random。阈值：Choice ≥0.65；事件不用 rest Score 覆写选项；错误 → `error` + 合法随机。

## Shadow 日志

既有字段外：

- `phase=EVENT`（Neow 步可 `NEOW`）
- `reason` 例：`jev_suggest_live` | `low_confidence_random` | `no_jev_options_random` | `unknown_deferred` | `jev_event_off_random` | `neow_jev_off_random` | `neow_options_empty` | `event_options_empty`
- `legal_ids` / `executed_id` / `jev_choice_id` / `jev_confidence`
- meta：`event_id`, `is_neow`

复表：EVENT 落地率与 MAP/REST/CARD 分列；Neow 可再拆 `is_neow=true`。

## Surplus 冒烟

n=2 `suggest_live` + `--jev-event on`（不在本 PR 跑 live TypeSafe）。测 `neow_boon` 另加 `--jev-neow on --start-with-neow`。**Hang：`--start-with-neow --jev-neow off`（随机祝福）。** n=100 Leave-only 是 events import / package-rooted `CARDS_REFERENCE` **之前**的表。

## REST calibration v1 (2026-09-22, Jev)

- Global Choice ≥0.65 **unchanged** for MAP/CARD/EVENT/Neow.
- REST only: `REST_CHOICE_MIN_CONFIDENCE = 0.50`.
- After `_apply_hp_pressure_bias`:
  - `hp_pressure≥2` + HEAL + `conf≥0.30` → `jev_hp_pressure_assist`
  - `hp_pressure≤1` + SMITH + `conf≥0.40` → `jev_smith_assist`
- Smoke gate: REST used ≥30% (split suggest_live / hp_assist / smith_assist / low_conf).
- Code may stay; **no win claim**. Hang bar is 3%/6.5 (n50 4%/8 is history).

## Done when

1. 本文档在仓 `docs/JEV_EVENT_NEOW_CONTRACT.md`
2. `scripts/jev_noncombat.py`：EVENT ∈ JEV_PHASES（开关）；`build_event_options` 接 `event_choice` + pending choose/confirm
3. `eval_act1_runenv.py`：`--jev-event` 默关
4. `docs/JEV_NONCOMBAT_WIRE.md` 含 EVENT 小节

**不**改战斗 zip / **不**训 combat / **不**扩 SHOP。
