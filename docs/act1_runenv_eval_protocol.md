# Act1 RunEnv 评测口径（冻结）

**状态：LOCKED 2026-09-22** · 评分员：评测哨兵  
**范围：** Act1 整局通关评测（`STS2RunEnv`），**≠** combat 套件（bare / loadout_v0 / loadout_v1）。

## 种子集

| 项 | 值 |
|----|----|
| 种子 | `200000 .. 200049`（含端点，共 **50** 局） |
| 注入 | `env.reset(seed=S)`（Gymnasium 种子；同种确定性已冒烟） |
| 角色 | Ironclad |
| 进阶 | 0 |
| `max_steps` | 2000（env 默认） |

## 主指标 / 辅指标

1. **主：`act1_clear_rate`** — 当局过程中 `max(info['act']) >= 1` 的比例（打过 Act1 Boss 并进入 Act2 地图）。
2. **辅：**
   - `median_floor` / `mean_floor`（`info['floor']` 终局）
   - `mean_hp_on_clear`（仅通关局终局 `hp`；未通关不计入）
   - `full_run_win_rate`（`terminated and reward > 0`，三幕全通；旁证）
   - `trunc_rate`（`truncated`）
   - `error_rate`（step 异常被 env 吞成死亡的局，若 info 可区分则记；否则并入失败）

## 禁止混谈

- **不得**把 bare / loadout_* combat 胜率写成 Act1 通关率。
- loadout_v1 战斗套件报赢（74.2/98.9/49.4）**不**自动等于本表通关。HOLD 口径见 `docs/HOLD_PROTOCOL.md`（fixture relics/potions 必须落地）。
- Jev 战略 Choice 由 `--jev on` 接入非战斗决策；阈值见下，**不**等于 Act1 通关完成。
- **`hierarchical` ≠ Act1 通关完成。** 战斗步只接 hung combat zip；非战斗默认合法随机。通关门禁仍由实验室另挂。

## 策略接口

脚本：`scripts/eval_act1_runenv.py`

```bash
# random 基线
python scripts/eval_act1_runenv.py --policy random --out /workspace/sts2-sim/evals/act1_runenv_random_s200000.json

# RunEnv 兼容 MaskablePPO（obs=RUN_OBS_SIZE=201；combat zip 在此路径拒评）
python scripts/eval_act1_runenv.py --policy model --model path/to/run_model.zip --out ...

# hierarchical（评测哨兵首表）：战斗步用 hung combat zip（obs_v1=181）；非战斗合法随机（Jev 仅影子）
python scripts/eval_act1_runenv.py --policy hierarchical --model /workspace/sts2-sim/output/combat_ppo_obs_v1_bh_v1/final_model.zip --jev off --out /workspace/sts2-sim/evals/act1_runenv_hierarchical_s200000.json

# 同上；`--combat-model` 是 `--model` 的别名
python scripts/eval_act1_runenv.py --policy hierarchical --combat-model /workspace/sts2-sim/output/combat_ppo_obs_v1_bh_v1/final_model.zip --jev off

# hierarchical + Jev hang (MAP/REST/CARD; EVENT off; random Neow boon; map_lowhp on)
python scripts/eval_act1_runenv.py --policy hierarchical --model /workspace/sts2-sim/output/combat_ppo_obs_v1_bh_v1/final_model.zip --jev on --jev-event off --jev-neow off --start-with-neow --n 100 --out /workspace/sts2-sim/evals/act1_runenv_map_lowhp_n100.json
```

- `--policy random|model|hierarchical`
- `--model`：
  - `--policy model`：必须 `obs_dim == RUN_OBS_SIZE`（当前 201）。combat-only zip **拒评**并明文报错，不静默。
  - `--policy hierarchical`：hung combat zip，必须 `obs_dim == OBS_SIZE`（obs_v1 = **181**）。Surplus 箱路径：`/workspace/sts2-sim/output/combat_ppo_obs_v1_bh_v1/final_model.zip`。**禁止**把 RunEnv obs 喂给该 zip。
- `--combat-model`：hierarchical `--model` 的别名（同一 combat zip）。
- `--jev off|on`（默认 `off`；`--strategic jev` 等同 `--jev on`）。仅 hierarchical。首表用 `--jev off`（非战斗合法随机，Jev 仅影子、不改动作）。
- `--jev-event off|on`（默认 **`off`**）。仅 hierarchical + `--jev on`。默认普通 EVENT 合法随机（`jev_event_off_random`），**不改**既有 MAP/REST/CARD 表。`--jev-phases map,rest,card,event` 等价于 `--jev-event on`。`--jev-neow` 默认 **off**（随机祝福，reason `neow_jev_off_random`）。`--jev-neow on` 为可选 A/B：`neow_boon` @ Choice ≥0.65，即使 `--jev-event off`。契约：`docs/JEV_EVENT_NEOW_CONTRACT.md`。
- `--map-lowhp on|off`（默认 **`on`**，hang 开，**v1 uncertain filter**）。`hp_pressure>=2.0`（Score 缺失时用本地 `max_hp/hp`）且合法 MAP 含 `SHOP`/`REST_SITE` 时，低置信度/错误/越界重抽 safe 点（`map_lowhp_random`）。自信决策（>=0.65）**不**覆写。
- `--map-lowhp-hard on|off`（默认 **`off`**；仅供实验 opt-in）。开启时在 `hp_pressure>=2.0` 且含 safe 点时无论自信与否硬选 rest-then-shop（`map_lowhp_hard`）。实验室在 n100 clear 0% 后冻结该路线，hang 协议默认关闭。评测按步累计 `map_lowhp_hard_n`。
- `--n`（默认 **50** = 冻结种子 `200000..200049`）。Hang 对照 bar 用 `--n 100`（`200000..200099`）。不改 hang 旗标，只加局数。
- `--start-with-neow`（CLI 默认 **关**；**hang 协议带上**）。开局 Neow 屏 + `--jev-neow off` = 随机祝福。Gym：`reset(..., options={"start_with_neow": True})`。`get_event("Neow")` 前注册 events；`CARDS_REFERENCE.md` 走 package root。EVENT / Neow 合法非 Leave **< 2** 时不调 Jev（`event_options_empty` / `neow_options_empty`，不算落地率）。`_actions_event` 仅在 `event_model is None` 时给 Leave；model 仍在、options 空（reward/pending 间隙）**不得**伪造 Leave。n=100 Leave-only 是 import 修前的表。
- 战斗步：从 `RunManager.get_combat_state()` 取 `CombatState`，`encode_observation(combat)` + combat `get_action_mask`，`MaskablePPO.predict`，再把 combat action index 映射到 RunEnv combat slice（layout offset 0）。**Jev 不进战斗。**
- 非战斗步：
  - `--jev off`：`action_masks()==1` 合法随机；日志 `shadow_status=stub`。
  - `--jev on`：TypeSafe/Jev Choice（`map_fork` / `card_reward` / 同类决策）与 rest_or_continue 的 hp_pressure Score。Choice confidence **≥ 0.65** 否则 uncertain → 合法随机。hp_pressure **≥ 2.0** 优先 rest，**≤ 1.0** 优先 continue，中间信 Choice。MAP 含 `UNKNOWN` 时仍打 hp_pressure；pressure≥2 且存在非 Unknown 且 Choice 选 Unknown 且 conf **< 0.80** → `unknown_deferred`（0.80 只用于 defer，主阈值仍 0.65）。**MAP low-HP `map_lowhp` 默认 on（v1 uncertain filter）**：pressure≥2 且 shop/rest 合法时，仅在 low-conf/error/越界时重抽 safe 点（`map_lowhp_random`），自信决策不覆写；硬选 rest-then-shop（`map_lowhp_hard`）需显式 `--map-lowhp-hard on`（默认 off）；仅剩 fight 则全池随机。`--map-lowhp off` 关闭软过滤。先剥 invisible/illegal。Act1 reward `+` 卡不是自然掉落（仅 Smith/Neow）——criteria 见 `docs/act1_content_map.md`。真 `pick_card` 另打 4 档 Score `card_fit`：**不改** Choice 0.65；confidence < 0.65 但 `card_fit >= 2.0` 且 choice ≠ skip 时落地，原因 `jev_card_fit_assist`。`PHASE_CARD_REWARD` 上的 `pick_potion` / `pick_relic_reward` **不**走 card_reward Jev，合法拿取（`potion_or_relic_safe_fallback`，不进 CARD 落地率分母）。`--jev-event on` 时普通 EVENT 走 `event_choice`（pending choose/confirm 接 combat 槽；低置信度/错误走 `event_safe_fallback` 避开负面项）。`--jev-event off` 时普通 EVENT 合法随机（`jev_event_off_random`）。`--jev-neow` 默认 off：检出的 Neow 合法随机（`neow_jev_off_random`）；`--jev-neow on` 才走 `neow_boon`（≥0.65）。SHOP **仍合法随机**。API/模型错误记 `shadow_status=error` 并回退合法安全/随机，**不吞决策点**。
  - `--jev on` 时日志含 `shadow_suggestion`、`shadow_confidence`、`shadow_fallback_reason`（及 `shadow_hp_pressure`）；真卡屏可选 `jev_card_fit`。EVENT 可含 `event_id` / `is_neow` / `phase=EVENT|NEOW`。
- 输出：完整 JSON + 同名 `.summary.json`（无逐局 rows 的精简版）。
- Live 调用读 `TYPESAFE_API_KEY`（`sts2_env/eval/jev.py` LiveJevClient）。池在 box-secrets `card.TYPESAFE_API_KEY` + `card.TYPESAFE_API_KEY_1`..`_4`；启动脚本自动 load，**不必**预注入 process env，也**不要**往聊天重贴。已 export 的 `TYPESAFE_API_KEY` / `_N` 仍认。403/1010 换下一把 key + 短 backoff，**不**把 MAP/CARD 改随机。缺失时按 error 回退，不改 combat 路径。TODO(Surplus/Jev) 可替换该 adapter，不必改 eval 脚本。

## 观测尺寸（obs_v1）

| 向量 | 宽度 | 用途 |
|------|------|------|
| Combat `OBS_SIZE` | **181** | 全量 `IntentType` one-hot；hung combat zip（`combat_ppo_obs_v1_*`） |
| RunEnv `RUN_OBS_SIZE` | **201** | 181 combat + 20 run-state extras |

两套观测不可互换。`--model` 与 `--combat-model` 分轨就是为了挡住误喂。

## 门禁

本表门禁由实验室另挂；冻结时门禁为空。未出通关数字前不谈「落地完成」。
hierarchical 评测通过 **不等于** Act1 通关达标。

## Hang protocol LOCKED 2026-09-22 (post Neow A/B)

详见 `docs/HANG_PROTOCOL_2026-09-22.md`。

- Combat zip: `combat_ppo_obs_v1_bh_v1` until swapped by gate.
- Hierarchical: `--jev on --jev-event off --jev-neow off --start-with-neow`
- Hang metrics (n=100): **clear 3% / median 6.5** (A random-Neow arm).
- Prior n50 4%/8 kept as history only.
- REST calibrate may remain in code; no win claim from it.
- Strategic claim freeze until lab unfreezes (EVENT off, neow Jev off).

