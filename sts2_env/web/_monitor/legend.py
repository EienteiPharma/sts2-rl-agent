from __future__ import annotations

from sts2_env.web._monitor.types import (
    LegendItem,
    OverlayBinding,
    OverlayId,
    StaticLegend,
)


def overlay_bindings() -> tuple[OverlayBinding, ...]:
    # Canvas nodes only. Flow-tab step ids (ft*, fg*) are not in #cv-world.
    return (
        OverlayBinding(
            OverlayId("train-go"),
            "node",
            "train",
            True,
            None,
            OverlayId("train-zips"),
            "train learn / TB stream",
        ),
        OverlayBinding(
            OverlayId("train-to-combat"),
            "node",
            "train",
            False,
            None,
            None,
            "combat train not disambiguated on disk",
        ),
        OverlayBinding(
            OverlayId("train-zips"),
            "node",
            "train",
            True,
            OverlayId("train-go"),
            None,
            "terminal final_model.zip",
        ),
        OverlayBinding(
            OverlayId("train-to-eval"),
            "node",
            "train",
            True,
            None,
            None,
            "headline eval 10000+ / evaluations.npz",
        ),
        OverlayBinding(
            OverlayId("train-to-gate"),
            "node",
            "gate_c",
            True,
            None,
            None,
            "lab gates / gate shards",
        ),
        OverlayBinding(
            OverlayId("te-run"),
            "node",
            "train",
            False,
            None,
            None,
            "evaluate_checkpoint stdout only",
        ),
        OverlayBinding(
            OverlayId("tg-rows"),
            "node",
            "legend",
            False,
            None,
            None,
            "B3 table; not a verdict",
        ),
    )


def static_legend() -> StaticLegend:
    return StaticLegend(
        kind="static_constants",
        items=(
            LegendItem(
                id="b3_keep",
                text="KeepBar skill≥2335 reach_ge16≥872 sel_wins≥750",
            ),
            LegendItem(
                id="act2_confirm",
                text="act2_report GATE_CONFIRM 301800+400 need 108 wins",
            ),
            LegendItem(
                id="act2_sel",
                text="act2_report GATE_SEL_NEED=300 sel wins",
            ),
        ),
    )
