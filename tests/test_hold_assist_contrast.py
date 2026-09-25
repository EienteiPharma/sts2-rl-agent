"""hang | A | B contrast + STOP (synthetic summaries)."""
from __future__ import annotations

from sts2_env.eval.hold_assist_contrast import contrast_hang_a_b, pp_delta


def _summary(overall: float, boss: float) -> dict:
    return {
        "overall": {"n": 360, "win_rate": overall},
        "elite": {"n": 180, "win_rate": 0.9},
        "boss": {"n": 180, "win_rate": boss},
    }


def test_pp_delta():
    assert pp_delta(0.50, 0.40) == 10.0


def test_hang_gap_stop_when_arm_falls_3pp():
    hang = _summary(0.74, 0.49)
    a = _summary(0.70, 0.45)  # -4pp overall
    b = _summary(0.73, 0.48)
    out = contrast_hang_a_b(hang, a, b)
    assert out["hang_gap_stop"]["triggered"] is True
    assert any("overall_delta_a" in r for r in out["hang_gap_stop"]["reasons"])


def test_assist_bar_independent_of_stop():
    hang = _summary(0.74, 0.49)
    a = _summary(0.72, 0.47)
    b = _summary(0.76, 0.50)  # +4pp overall, +3pp boss vs A
    out = contrast_hang_a_b(hang, a, b)
    assert out["hang_gap_stop"]["triggered"] is False
    assert out["assist_effectiveness_bar"]["passed"] is True
    assert out["delta_assist_b_minus_a_pp"]["overall"] == 4.0


def test_three_column_table_keys():
    hang = _summary(0.74, 0.49)
    out = contrast_hang_a_b(hang, hang, hang)
    assert set(out["win_rate_table"]["overall"]) == {"hang", "a", "b"}
