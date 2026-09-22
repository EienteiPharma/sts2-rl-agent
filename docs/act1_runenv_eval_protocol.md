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
- loadout_v1 战斗套件报赢（74.2/98.9/49.4）**不**自动等于本表通关。
- Jev 战略 Choice 可 shadow 对齐本表时间线；阈值另 brief。
- **`hierarchical` ≠ Act1 通关完成。** 本策略只把 hung combat zip 接到 RunEnv 战斗步；通关门禁仍由实验室另挂，未出通关数字前不谈「落地完成」。

## 策略接口

脚本：`scripts/eval_act1_runenv.py`

```bash
# random 基线
python scripts/eval_act1_runenv.py --policy random --out /workspace/sts2-sim/evals/act1_runenv_random_s200000.json

# RunEnv 兼容 MaskablePPO（obs=RUN_OBS_SIZE，非 combat zip）
python scripts/eval_act1_runenv.py --policy model --model path/to/run_model.zip --out ...

# hierarchical：战斗步用 combat zip（obs=OBS_SIZE / obs_v1=181）；非战斗步合法随机
python scripts/eval_act1_runenv.py --policy hierarchical --combat-model path/to/combat_ppo_obs_v1.zip --out ...
```

- `--policy random|model|hierarchical`
- `--model`：仅 `--policy model`。必须 `obs_dim == RUN_OBS_SIZE`（当前 201）。combat-only zip **拒评**并明文报错，不静默。
- `--combat-model`：仅 `--policy hierarchical`。必须 `obs_dim == OBS_SIZE`（obs_v1 = 181）。**禁止**把 RunEnv obs 喂给 combat zip。
- 战斗步：从 `RunManager.get_combat_state()` 取 `CombatState`，`encode_observation(combat)` + combat `get_action_mask`，`MaskablePPO.predict`，再把 combat action index 映射到 RunEnv combat slice（layout offset 0）。
- 非战斗步：在 `action_masks()==1` 中合法随机。Jev **只 shadow**（日志字段 `shadow_suggestion` / `shadow_status=skipped|stub`），**不**改非战斗动作。
- 输出：完整 JSON + 同名 `.summary.json`（无逐局 rows 的精简版）。

## 观测尺寸（obs_v1）

| 向量 | 宽度 | 用途 |
|------|------|------|
| Combat `OBS_SIZE` | **181** | 全量 `IntentType` one-hot；hung combat zip（`combat_ppo_obs_v1_*`） |
| RunEnv `RUN_OBS_SIZE` | **201** | 181 combat + 20 run-state extras |

两套观测不可互换。`--model` 与 `--combat-model` 分轨就是为了挡住误喂。

## 门禁

本表门禁由实验室另挂；冻结时门禁为空。未出通关数字前不谈「落地完成」。
hierarchical 评测通过 **不等于** Act1 通关达标。
