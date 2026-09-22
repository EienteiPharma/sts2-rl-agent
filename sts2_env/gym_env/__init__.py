"""STS2 Gymnasium environments."""

from sts2_env.gym_env.combat_env import STS2CombatEnv
from sts2_env.gym_env.run_env import STS2RunEnv

__all__ = [
    "STS2CombatEnv",
    "STS2RunEnv",
    "RunEnvOnPolicyCombatEnv",
    "MixedHangLoadoutEnv",
]


def __getattr__(name: str):
    # Lazy: on-policy/mix envs import eval.jev_policy, which imports this package.
    if name == "RunEnvOnPolicyCombatEnv":
        from sts2_env.gym_env.runenv_onpolicy_combat import RunEnvOnPolicyCombatEnv

        return RunEnvOnPolicyCombatEnv
    if name == "MixedHangLoadoutEnv":
        from sts2_env.gym_env.runenv_antiforget import MixedHangLoadoutEnv

        return MixedHangLoadoutEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
