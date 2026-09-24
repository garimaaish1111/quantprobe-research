"""Figures.

Palette and mark specs follow the dataviz guidance, with the categorical slots
validated rather than eyeballed:

    node scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a" --mode light
    -> ALL CHECKS PASS (lightness band, chroma floor, CVD separation,
       normal-vision floor); one WARN on aqua contrast 2.74:1.

That contrast WARN is not dismissable - it obliges visible labels or a table
view. Both are provided: every line is directly labelled at its right end, and
`02_probe.py` writes the same numbers to `results/metrics/*.json`.

Two rules that matter more here than they would on a dashboard:

**Identity is never carried by colour alone.** Each precision gets its own
line style and marker as well as its own hue. A research figure ends up
printed in greyscale, photocopied, and projected through a bad beamer, and the
worst CVD-adjacent pair in this palette (orange vs aqua, deutan dE 9.2) sits
just above the floor. Style redundancy makes all of that a non-issue.

**Colours are assigned to precisions in fixed order and never cycled.** FP16
is always blue, INT8 always orange, INT4 always aqua, in every figure in the
project. A reader who learns the mapping on E1 keeps it for E2 and E3.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger("quantprobe.viz.plots")

#: Fixed precision -> visual identity. Never reassigned, never cycled.
PRECISION_STYLE: dict[str, dict[str, Any]] = {
    "fp16": {"color": "#2a78d6", "linestyle": "-", "marker": "o", "label": "FP16"},
    "int8": {"color": "#eb6834", "linestyle": "--", "marker": "s", "label": "INT8"},
    "int4": {"color": "#1baf7a", "linestyle": "-.", "marker": "^", "label": "INT4"},
}

#: Ink. Text never wears a series colour.
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8983"
SURFACE = "#fcfcfb"


def _style_axes(ax) -> None:
    """Recessive grid and axes: the data is the loud thing, not the frame."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color="#e6e5e1", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d5d4cf")
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=3, width=0.8)


def plot_e1_auroc_by_layer(
    rows: Sequence[dict[str, Any]],
    out_path: Path,
    probe_type: str = "logistic",
    model_name: str = "",
    n_folds: int | None = None,
    dpi: int = 200,
) -> Path:
    """E1 - the headline figure. AUROC against layer, one curve per precision.

    Args:
        rows: output of `probes.linear.aggregate`.
        out_path: PNG destination.
        probe_type: which probe's results to plot.
        model_name: shown in the subtitle.
        n_folds: shown in the subtitle, so the bands are interpretable.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [r for r in rows if r["probe_type"] == probe_type]
    if not selected:
        raise ValueError(f"no rows for probe_type={probe_type!r}")

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)

    # Chance. Drawn first so it sits under the data, and labelled in ink so it
    # never competes with a series for identity.
    ax.axhline(0.5, color=INK_MUTED, linewidth=1.0, linestyle=":", zorder=1)
    # No inline label for the chance line. Anchoring it anywhere collides with
    # something - at x=0 the series themselves sit near 0.5, and at x=max the
    # legend does - and where the collision lands changes with the data. A
    # dotted rule at a 0.5 tick is self-evident on an AUROC plot, so it is
    # named in the subtitle instead, where nothing can overlap it.

    max_layer = 0
    end_labels: list[tuple[float, int, dict]] = []
    for precision in ("fp16", "int8", "int4"):  # fixed order
        series = sorted(
            (r for r in selected if r["precision"] == precision),
            key=lambda r: r["layer"],
        )
        if not series:
            continue
        style = PRECISION_STYLE[precision]
        layers = [r["layer"] for r in series]
        mean = [r["auroc_mean"] for r in series]
        std = [r["auroc_std"] for r in series]
        lo = [m - s for m, s in zip(mean, std)]
        hi = [m + s for m, s in zip(mean, std)]
        max_layer = max(max_layer, max(layers))

        ax.fill_between(layers, lo, hi, color=style["color"], alpha=0.13, linewidth=0, zorder=2)
        ax.plot(
            layers, mean,
            color=style["color"], linestyle=style["linestyle"], linewidth=2.0,
            marker=style["marker"], markersize=4.5,
            markeredgecolor=SURFACE, markeredgewidth=0.8,  # 2px-ish surface ring
            label=style["label"], zorder=3,
        )
        end_labels.append((mean[-1], layers[-1], style))

    # Direct labels at the right end - the relief the contrast WARN requires.
    # Nudged apart when curves converge: on the real data all three arms land
    # within 0.007 AUROC of each other at layer 16 and the labels overprint.
    span = max(m for m, _, _ in end_labels) - min(m for m, _, _ in end_labels)
    y_range = max(span, 0.02)
    min_gap = y_range * 0.9 if span < 0.02 else 0.0
    placed: list[float] = []
    for value, x, style in sorted(end_labels, key=lambda e: e[0]):
        y = value
        while any(abs(y - other) < min_gap for other in placed):
            y += min_gap
        placed.append(y)
        ax.annotate(
            style["label"],
            xy=(x, value),
            xytext=(7, (y - value) * 72 * 4),  # offset in points
            textcoords="offset points",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=style["color"],
        )

    ax.set_xlabel("layer  (0 = embeddings)", fontsize=10, color=INK_SECONDARY)
    ax.set_ylabel("AUROC", fontsize=10, color=INK_SECONDARY)
    ax.set_xlim(-0.4, max_layer + 1.6)
    ax.set_xticks(range(0, max_layer + 1, 2))

    subtitle = f"{probe_type} probe"
    if model_name:
        subtitle = f"{model_name} - {subtitle}"
    if n_folds:
        subtitle += f" - mean ± 1 s.d. over {n_folds} leave-one-topic-out folds"
    subtitle += " - dotted rule = chance (0.50)"

    ax.set_title(
        "Truthfulness probe accuracy by layer and precision",
        fontsize=12, fontweight="bold", color=INK_PRIMARY, loc="left", pad=26,
    )
    ax.annotate(
        subtitle,
        xy=(0, 1.015), xycoords="axes fraction",
        fontsize=8.5, color=INK_SECONDARY, va="bottom",
    )

    legend = ax.legend(
        loc="lower right", frameon=True, fontsize=9,
        facecolor=SURFACE, edgecolor="#e6e5e1",
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out_path)
    return out_path


def plot_probe_comparison(
    rows: Sequence[dict[str, Any]],
    out_path: Path,
    model_name: str = "",
    dpi: int = 200,
) -> Path:
    """Logistic vs mass-mean, side by side, sharing one y-axis.

    Two panels rather than two y-scales on one axis. A dual-axis chart is the
    single most misleading chart form there is; small multiples say the same
    thing without inviting a false comparison of slopes.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, probe_type in zip(axes, ("logistic", "mass_mean")):
        _style_axes(ax)
        ax.axhline(0.5, color=INK_MUTED, linewidth=1.0, linestyle=":", zorder=1)
        for precision in ("fp16", "int8", "int4"):
            series = sorted(
                (r for r in rows if r["probe_type"] == probe_type and r["precision"] == precision),
                key=lambda r: r["layer"],
            )
            if not series:
                continue
            style = PRECISION_STYLE[precision]
            layers = [r["layer"] for r in series]
            mean = [r["auroc_mean"] for r in series]
            std = [r["auroc_std"] for r in series]
            ax.fill_between(
                layers, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)],
                color=style["color"], alpha=0.13, linewidth=0, zorder=2,
            )
            ax.plot(
                layers, mean, color=style["color"], linestyle=style["linestyle"],
                linewidth=2.0, marker=style["marker"], markersize=4.0,
                markeredgecolor=SURFACE, markeredgewidth=0.8,
                label=style["label"], zorder=3,
            )
        ax.set_title(
            {"logistic": "Logistic regression", "mass_mean": "Mass-mean (difference of means)"}[probe_type],
            fontsize=10.5, fontweight="bold", color=INK_PRIMARY, loc="left",
        )
        ax.set_xlabel("layer", fontsize=10, color=INK_SECONDARY)

    axes[0].set_ylabel("AUROC", fontsize=10, color=INK_SECONDARY)
    legend = axes[1].legend(loc="lower right", frameon=True, fontsize=9,
                            facecolor=SURFACE, edgecolor="#e6e5e1")
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.suptitle(
        f"Two estimators of the same signal{' - ' + model_name if model_name else ''}",
        fontsize=12, fontweight="bold", color=INK_PRIMARY, x=0.02, ha="left",
    )
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out_path)
    return out_path


# ---------------------------------------------------------------- E2 / E3 / E4

#: Sequential blue ramp, steps 100 -> 700. For continuous magnitude only
#: (heatmap cells), where the lightest step is allowed to recede into the
#: surface because it means "near zero".
SEQUENTIAL_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]

#: Diverging pair: blue <-> red on a neutral gray midpoint. Warm/cool poles
#: that read as opposite, with a midpoint that reads as "nothing" - which is
#: the whole requirement for a zero-centred loss scale. (blue<->aqua was
#: rejected upstream: both cool, so the midpoint does not read as zero.)
DIVERGING_BLUE_RED = [
    "#1c5cab", "#5598e7", "#b7d3f6", "#f0efec", "#f6c9c9", "#e88a89", "#c0332f",
]


def _sequential_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("qp_seq", SEQUENTIAL_BLUE)


def _diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("qp_div", DIVERGING_BLUE_RED)


def _cell_ink(rgba) -> str:
    """Black or white text, whichever the cell can actually carry.

    Relative luminance per WCAG; 0.55 is the crossover that keeps both ends of
    a sequential ramp readable.
    """
    r, g, b = rgba[:3]

    def lin(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    lum = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
    return "#ffffff" if lum < 0.55 else INK_PRIMARY


def plot_e2_transfer_heatmap(
    matrix,
    loss_matrix,
    precisions,
    out_path,
    layer: int,
    model_name: str = "",
    dpi: int = 200,
):
    """E2 - the 3x3 transfer matrix, as AUROC and as loss against native.

    Two panels because they answer different questions and want different
    colour jobs. AUROC is a magnitude, so sequential, one hue. Loss is a
    polarity - better or worse than recalibrating - so it is diverging on a
    neutral midpoint, and scaled symmetrically about zero so the midpoint
    really means "no loss" rather than "middle of the observed range".
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [PRECISION_STYLE[p]["label"] for p in precisions]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6))
    fig.patch.set_facecolor(SURFACE)

    specs = [
        (axes[0], matrix, _sequential_cmap(), None, "AUROC", "{:.3f}"),
        (
            axes[1],
            loss_matrix,
            _diverging_cmap(),
            float(np.abs(loss_matrix).max()) or 0.01,
            "AUROC lost vs native probe",
            "{:+.3f}",
        ),
    ]

    for ax, data, cmap, sym, title, fmt in specs:
        if sym is None:
            vmin, vmax = float(data.min()), float(data.max())
        else:
            vmin, vmax = -sym, sym
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9.5, color=INK_SECONDARY)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9.5, color=INK_SECONDARY)
        ax.set_xlabel("tested on", fontsize=10, color=INK_SECONDARY)
        ax.set_ylabel("trained on", fontsize=10, color=INK_SECONDARY)
        ax.set_title(title, fontsize=10.5, fontweight="bold", color=INK_PRIMARY, loc="left")

        # Every cell labelled. A 3x3 has the room, and the sequential ramp's
        # light end sits below 3:1 against the surface by design.
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(
                    j,
                    i,
                    fmt.format(data[i, j]),
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold" if i == j else "normal",
                    color=_cell_ink(im.cmap(im.norm(data[i, j]))),
                )
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=8, colors=INK_SECONDARY, length=2)

    fig.suptitle(
        "E2 - cross-precision probe transfer at layer {}{}".format(
            layer, " - " + model_name if model_name else ""
        ),
        fontsize=12,
        fontweight="bold",
        color=INK_PRIMARY,
        x=0.02,
        ha="left",
    )
    fig.text(
        0.02,
        0.005,
        "Diagonal = trained and tested on the same precision. Off-diagonal = the "
        "deployment case: a detector calibrated at one precision, run at another.",
        fontsize=8,
        color=INK_SECONDARY,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out_path)
    return out_path


def plot_e3_drift(drift_rows, out_path, model_name: str = "", dpi: int = 200):
    """E3 - three drift metrics against layer, INT8 and INT4 versus FP16.

    Three panels rather than three lines on one axis: relative drift is a
    distance in [0, inf), while cosine and CKA are similarities in [0, 1].
    Sharing a y-axis between them would be the dual-axis mistake in disguise.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        ("relative_drift", "Relative drift", "distance / |h|; lower is closer to FP16"),
        ("mean_cosine", "Mean cosine similarity", "1.0 = no rotation"),
        ("linear_cka", "Linear CKA", "1.0 = identical geometry"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))
    fig.patch.set_facecolor(SURFACE)

    for ax, (key, title, hint) in zip(axes, metrics):
        _style_axes(ax)
        for precision in ("int8", "int4"):
            series = sorted(
                (r for r in drift_rows if r["precision"] == precision),
                key=lambda r: r["layer"],
            )
            if not series:
                continue
            style = PRECISION_STYLE[precision]
            ax.plot(
                [r["layer"] for r in series],
                [r[key] for r in series],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.0,
                marker=style["marker"],
                markersize=4.0,
                markeredgecolor=SURFACE,
                markeredgewidth=0.8,
                label=style["label"] + " vs FP16",
                zorder=3,
            )
        ax.set_title(title, fontsize=10.5, fontweight="bold", color=INK_PRIMARY, loc="left", pad=18)
        ax.annotate(
            hint, xy=(0, 1.015), xycoords="axes fraction",
            fontsize=8, color=INK_MUTED, va="bottom",
        )
        ax.set_xlabel("layer", fontsize=10, color=INK_SECONDARY)

    legend = axes[0].legend(loc="best", frameon=True, fontsize=8.5,
                            facecolor=SURFACE, edgecolor="#e6e5e1")
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.suptitle(
        "E3 - representational drift from FP16" + (" - " + model_name if model_name else ""),
        fontsize=12, fontweight="bold", color=INK_PRIMARY, x=0.02, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out_path)
    return out_path


def plot_e4_rotation(rotation_rows, out_path, model_name: str = "", dpi: int = 200):
    """E4 - angle between the FP16 probe direction and each quantized one."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)

    # 90 degrees = the two probes are orthogonal, i.e. one carries no
    # information at all about the other arm's direction.
    ax.axhline(90.0, color=INK_MUTED, linewidth=1.0, linestyle=":", zorder=1)

    for precision in ("int8", "int4"):
        series = sorted(
            (r for r in rotation_rows if r["precision"] == precision),
            key=lambda r: r["layer"],
        )
        if not series:
            continue
        style = PRECISION_STYLE[precision]
        ax.plot(
            [r["layer"] for r in series],
            [r["angle_deg"] for r in series],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            marker=style["marker"],
            markersize=4.5,
            markeredgecolor=SURFACE,
            markeredgewidth=0.8,
            label=style["label"] + " vs FP16",
            zorder=3,
        )

    ax.set_xlabel("layer  (0 = embeddings)", fontsize=10, color=INK_SECONDARY)
    ax.set_ylabel("angle between probe directions (deg)", fontsize=10, color=INK_SECONDARY)
    ax.set_title(
        "E4 - does the probe direction rotate under quantization?",
        fontsize=12, fontweight="bold", color=INK_PRIMARY, loc="left", pad=26,
    )
    ax.annotate(
        (model_name + " - " if model_name else "")
        + "0 deg = identical direction; dotted rule at 90 deg = orthogonal, no shared signal",
        xy=(0, 1.015), xycoords="axes fraction",
        fontsize=8.5, color=INK_SECONDARY, va="bottom",
    )
    legend = ax.legend(loc="best", frameon=True, fontsize=9,
                       facecolor=SURFACE, edgecolor="#e6e5e1")
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out_path)
    return out_path


def plot_coverage_risk(results, out_path, layer: int, model_name: str = "", dpi: int = 200):
    """Coverage against false-assertion rate, one curve per precision.

    The two axes point in opposite directions on purpose. Risk alone is
    trivially minimised by abstaining from everything, and coverage alone by
    answering everything; only the pair is a claim. The dotted rule is the
    base error rate - the rate you get with no filter at all - so the vertical
    gap between a curve and that line is exactly what the probe bought.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)

    baseline = None
    for precision in ("fp16", "int8", "int4"):
        if precision not in results:
            continue
        rows = sorted(results[precision]["curve"], key=lambda r: r["coverage"])
        style = PRECISION_STYLE[precision]
        baseline = results[precision]["baseline_false_rate"]
        ax.plot(
            [r["coverage"] * 100 for r in rows],
            [r["false_assertion_rate"] * 100 for r in rows],
            color=style["color"], linestyle=style["linestyle"], linewidth=2.0,
            marker=style["marker"], markersize=4.0,
            markeredgecolor=SURFACE, markeredgewidth=0.8,
            label=style["label"], zorder=3,
        )

    if baseline is not None:
        ax.axhline(baseline * 100, color=INK_MUTED, linewidth=1.0, linestyle=":", zorder=1)

    ax.set_xlabel("coverage - % of statements the model still asserts",
                  fontsize=10, color=INK_SECONDARY)
    ax.set_ylabel("false assertions (%)", fontsize=10, color=INK_SECONDARY)
    ax.set_title(
        "Probe-guided abstention: what refusing to answer buys",
        fontsize=12, fontweight="bold", color=INK_PRIMARY, loc="left", pad=26,
    )
    ax.annotate(
        (model_name + " - " if model_name else "")
        + f"layer {layer} - dotted rule = base error rate with no filter - "
          "down and to the right is better",
        xy=(0, 1.015), xycoords="axes fraction",
        fontsize=8.5, color=INK_SECONDARY, va="bottom",
    )
    legend = ax.legend(loc="best", frameon=True, fontsize=9,
                       facecolor=SURFACE, edgecolor="#e6e5e1")
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out_path)
    return out_path
