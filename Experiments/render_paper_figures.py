#!/usr/bin/env python3
"""
Paper-ready figure renderer for the EHR-Dataset-Processing pipeline (mimic-iii, mean aggregation).

Regenerates a fixed set of figures as scalable PDFs styled for inclusion in a paper:
white background, black text, no chart titles, slightly larger axis labels, and filenames
that describe each chart.

Reuses the plotting functions in Managers.visualization_manager_v2 (called exactly as the
notebook / gallery renderer does) and post-processes the returned matplotlib Figures for the
paper style. The McNemar plot is built fresh here because it needs structural changes
(two labeled thresholds, tiered bar colours, no legend).

Reads only the existing caches under Data/mimic-iii/ — no GPU, DB, or Marimo server needed.

Two evaluation designs live side by side in Data/mimic-iii/<agg>/, and which one the
figures describe is chosen with --impact-suffix:

    '_cv'  (default)  <label>_filter_impact_cv.pkl — stratified 5-fold x 4 repeats,
                      every record predicted out-of-fold, correctness-paired McNemar.
    ''     (legacy)   <label>_filter_impact.pkl — one 10% holdout, McNemar on
                      prediction agreement rather than correctness.

The two are not comparable, so the suffix also drives the axis labels and, for
McNemar, whether the bars can be coloured by direction at all. A missing pickle is
an error, never a silent fall back to the other design — same rule as
notebook.py's IMPACT_SUFFIX guard.

Only the filter-impact and McNemar figures depend on that choice; the centroid,
ECDF and observation-count figures describe the data, not a model.

Usage:
    module load scipy-stack/2026a
    MPLBACKEND=Agg python3 Experiments/render_paper_figures.py
    MPLBACKEND=Agg python3 Experiments/render_paper_figures.py --impact-suffix '' \
        --out /tmp/legacy_check
    MPLBACKEND=Agg python3 Experiments/render_paper_figures.py --only mcnemar
    MPLBACKEND=Agg python3 Experiments/render_paper_figures.py --only mcnemar \
        --mcnemar-threshold baseline

The last of those answers a different question from the published McNemar panel.
By default each arm is scored at its own mean Youden threshold, so it differs from
raw both in its features and in its operating point; --mcnemar-threshold baseline
re-scores every arm at raw's threshold, which isolates the filter. It writes
mcnemar_<label>_mean_baseline_threshold.pdf and never overwrites the published pair.

--mcnemar-style significance renders that panel in the pre-CV paper's plainer
encoding — grey/amber/crimson by p alone, no legend, no n01/n10 counts — and
--mcnemar-cap sets the bar-height ceiling (capped bars stay hatched and keep their
true p-value label):

    MPLBACKEND=Agg python3 Experiments/render_paper_figures.py --only mcnemar \
        --mcnemar-threshold baseline --mcnemar-style significance --mcnemar-cap 10
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from types import ModuleType

# ── Paths ─────────────────────────────────────────────────────────────────────
EHR_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = EHR_ROOT / "Data" / "mimic-iii"
DEFAULT_OUT_DIR = EHR_ROOT / "paper_figures"

# Rebound by main() when --out is given. Module-level because every figure
# function saves through finalize(), and threading a directory through all of
# them buys nothing over one assignment at start-up.
OUT_DIR = DEFAULT_OUT_DIR
# Output format. PDF is what the paper embeds; PNG exists so the same figures can be
# placed in a slide deck, which cannot embed PDF. Set by --format/--dpi in main().
FORMAT = "pdf"
DPI = 200
# Multiplies every type size finalize() applies. 1.0 is the paper. The poster places
# these figures at about half their natural width, which would print 24pt ticks at
# 13pt, so it renders them with larger type instead of shrinking the published look.
FONT_SCALE = 1.0

# ── Matplotlib must be Agg before any other import ───────────────────────────
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.transforms import ScaledTranslation
import numpy as np
import pandas as pd
from scipy.stats import binomtest


# ── Stubs for GPU/torch deps so RecordEHR can be unpickled ───────────────────
def _make_stub(name: str) -> ModuleType:
    m = ModuleType(name)
    m.__spec__ = None

    class _Anything:
        def __init__(self, *a, **kw): pass
        def __call__(self, *a, **kw): return self
        def __getattr__(self, n): return _Anything()
        def __iter__(self): return iter([])
        def __len__(self): return 0

    m._Anything = _Anything
    sys.modules[name] = m

    class _StubMod(ModuleType):
        def __getattr__(self, n):
            return _Anything()
    m.__class__ = _StubMod
    return m


# marimo is stubbed for the same reason as the GPU stack, not because it is heavy:
# visualization_manager_v2 imports it at module scope for one function
# (filter_impact_table, :233) that this renderer never calls, and it is not
# installed under the scipy-stack module this script runs on.
for _stub in ("torch", "torch.nn", "torch.optim", "torch.utils", "torch.utils.data",
              "cupy", "cudf", "cuml", "cuml.ensemble", "cuml.model_selection",
              "cuml.metrics", "cugraph", "marimo"):
    if _stub not in sys.modules:
        _make_stub(_stub)

# ── Add EHR project to sys.path and import viz functions ─────────────────────
if str(EHR_ROOT) not in sys.path:
    sys.path.insert(0, str(EHR_ROOT))

from Managers.visualization_manager_v2 import (  # noqa: E402
    centroid_plot, heatmap, centroid_shift_plot, filter_impact_plot,
)

# ── Constants mirrored from notebook.py ───────────────────────────────────────
WHITE = "#ffffff"
AGG = "mean"

VITALS = {
    "heart rate":               [(1, 600), "bpm"],
    "systolic blood pressure":  [(1, 400), "mmHg"],
    "diastolic blood pressure": [(1, 300), "mmHg"],
    "mean blood pressure":      [(1, 300), "mmHg"],
    "respiration rate":         [(1, 70),  "breaths/min"],
    "temperature":              [(21, 50), "C"],
    "oxygen saturation":        [(1, 100),  "%"],
}
VITAL_NAMES = list(VITALS.keys())
VITAL_UNITS = [v[-1] for v in VITALS.values()]

LABELS = ["icu", "mortality"]
FILTER_NAMES = [
    "heart rate", "systolic blood pressure", "diastolic blood pressure",
    "mean blood pressure", "respiration rate", "temperature", "oxygen saturation",
    "fill missing data", "long missing segment", "long gap", "high invalid data",
    "all vitals",
]
LENGTH_METHODS = ["mean", "median", "std", "max", "min"]

# Which evaluation design the model figures describe; rebound by main().
# '_cv' is the default because the single-holdout numbers the legacy pickles carry
# are not what the paper should report — see bug_register.py E-02.
IMPACT_SUFFIX = "_cv"

# Positions 1 and 3 of the results tuple mean different things in the two designs,
# so the axis label has to move with the suffix. Legacy [1] is one 10% holdout at
# the Youden threshold; CV [1] is the mean out-of-fold accuracy across four
# stratified 5-fold partitions covering every record.
METRIC_LABELS = {
    "": ("Testing Accuracy", "Testing F1"),
    "_cv": ("Out-of-fold Accuracy", "Out-of-fold Macro F1"),
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_pkl(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_impact(label: str):
    """The five-position filter-impact tuple for `label` under IMPACT_SUFFIX.

    Raises rather than returning None. The old behaviour — skip the figure and
    carry on — was survivable when there was one design; with two it would let a
    run silently publish eleven CV figures and no McNemar, or worse, invite a
    fall back that writes single-holdout numbers under a filename claiming to be
    cross-validated. Same reasoning as notebook.py:576-585.
    """
    path = DATA_ROOT / AGG / f"{label}_filter_impact{IMPACT_SUFFIX}.pkl"
    impact = load_pkl(path)
    if impact is None:
        raise FileNotFoundError(
            f"{path.name} not found under {path.parent}. "
            f"IMPACT_SUFFIX={IMPACT_SUFFIX!r} results come from "
            "rerun/job_f_filter_impact_cv.sh (_cv) or rerun/job_b_filter_impact.sh "
            "(legacy); this renderer will not substitute one design for the other."
        )
    return impact


def load_mcnemar_directions(label: str):
    """Per-arm (n01, n10) from the CV diagnostics sidecar, keyed by arm name.

    n01 = records the arm got right that the baseline got wrong (the arm fixed
    them); n10 = the reverse (the arm broke them). Only the CV design computes
    these — the legacy McNemar tabulates prediction agreement and never sees the
    labels, so it cannot say which model is right (E-06). Returns None for the
    legacy suffix, and the caller falls back to undirected colouring.

    Keyed by name deliberately. The sidecar is written with sort_keys=True, so
    its iteration order is alphabetical and does NOT match FILTER_NAMES; zipping
    the two would mislabel every arm without erroring. render_balance_comparison.py
    documents the same trap for the stage-H sidecars.
    """
    if IMPACT_SUFFIX != "_cv":
        return None
    path = DATA_ROOT / AGG / f"{label}_filter_impact_cv_diagnostics.json"
    if not path.exists():
        print(f"    [warn] {path.name} not found; McNemar bars will be undirected")
        return None
    diagnostics = json.loads(path.read_text())
    return {name: (detail["mcnemar"]["n01"], detail["mcnemar"]["n10"])
            for name, detail in diagnostics.items() if "mcnemar" in detail}


def fit_axis_labels(fig, *, pad: float = 4.0, min_size: float = 14.0,
                    step: float = 0.97, max_iterations: int = 40):
    """Keep every axis label inside the canvas: slide it first, shrink only if it must.

    Half of the "Deviation from Baseline" fix. This half keeps the label inside
    the figure canvas so it is drawn at all; the other half is the
    `bbox_extra_artists` argument in `finalize`, which keeps the tight crop from
    cutting it back off. Both are needed and they fail differently, so neither
    substitutes for the other.

    Sliding before shrinking, because the overflow is a *placement* problem, not
    a size one. The y-label is centred on the axes, and `finalize` grows the
    rotated x-tick labels from 10pt to 18pt after `filter_impact_plot` has
    already run tight_layout, which lifts the axes and carries the y-label's
    centre well above the middle of the figure. At 22pt that label is 687px in
    an 800px canvas — it fits comfortably; it was just centred 67px too high.
    Nudging it down by the overhang costs nothing, where shrinking it to fit in
    place would have cost four points of type.

    The shrink loop is the fallback for a label genuinely taller than the canvas,
    which no amount of sliding can rescue. `min_size` keeps that from silently
    producing unreadable type; a label still overflowing at 14pt wants a shorter
    name or a taller figure, and this is not the place to decide which.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        axes_box = ax.get_window_extent(renderer=renderer)
        for axis, vertical in ((ax.yaxis, True), (ax.xaxis, False)):
            label = axis.label
            if not label.get_text():
                continue

            limit = fig.bbox.height if vertical else fig.bbox.width
            if (axes_box.height if vertical else axes_box.width) <= 0:
                continue

            for _ in range(max_iterations):
                bb = label.get_window_extent(renderer=renderer)
                low, high = (bb.y0, bb.y1) if vertical else (bb.x0, bb.x1)

                if low >= 0 and high <= limit:
                    break

                # Slide by exactly the overhang, and only while the label is
                # short enough for sliding to resolve it.
                if high - low <= limit - 2 * pad:
                    shift = (limit - pad - high) if high > limit else (pad - low)
                    # A transform offset, not set_label_coords. The axis rewrites
                    # label.set_position() on every draw to keep the label clear
                    # of the tick labels, so a position we set is discarded; and
                    # its stored position is not in one coordinate system anyway
                    # (a y-label's x is display pixels while its y is an axes
                    # fraction), so feeding get_position() back into
                    # set_label_coords — which reads both as axes fractions —
                    # moves the label by multiples of the axes width. Composing an
                    # offset survives the redraw and leaves the automatic
                    # placement in charge of the other coordinate.
                    dx, dy = (0.0, shift) if vertical else (shift, 0.0)
                    label.set_transform(label.get_transform() + ScaledTranslation(
                        dx / fig.dpi, dy / fig.dpi, fig.dpi_scale_trans))
                    fig.canvas.draw()
                    continue

                size = label.get_size()
                if size <= min_size:
                    print(f"    [warn] {'y' if vertical else 'x'}-label still "
                          f"overflows at {size:.0f}pt: {label.get_text()!r}")
                    break
                label.set_size(max(min_size, size * step))
                fig.canvas.draw()


def save(fig, stem, **kwargs):
    """Write one figure in the configured format and report the path."""
    out = OUT_DIR / f"{stem}.{FORMAT}"
    fig.savefig(str(out), facecolor=WHITE,
                **({"dpi": DPI} if FORMAT == "png" else {}), **kwargs)
    plt.close(fig)
    print(f"    saved {out.name}")
    return out


def finalize(fig, stem: str, *, label_size: int = 16, tick_size: int = 12,
             strip_titles: bool = True, text_color: str = "black", title_size: int = None,
             legend_size: int = None, annot_size: int = None,
             xtick_size: int = None, ytick_size: int = None):
    """Restyle a screen figure for print and save it as PDF.

    Post-processing the returned Figure rather than adding a "paper mode" to
    every plotting function — that keeps the paper style in one place and means
    the notebook's plots can't drift away from the published ones.

    Why the changes are what they are: titles come off because the caption
    carries that in a paper; text scales up because these get printed at a
    fraction of screen size; PDF because reviewers zoom.

    `text_color` is separate from the rest because in-axes annotations sometimes
    need to stay white — heatmap cell labels sit on dark Aurora cells even after
    the figure background goes white. Axis furniture is always black.
    """
    if fig is None:
        print(f"    [skip] {stem}: figure is None")
        return

    fig.set_facecolor(WHITE)

    def scaled(size):
        return None if size is None else size * FONT_SCALE

    label_size, tick_size = scaled(label_size), scaled(tick_size)
    title_size, legend_size, annot_size = (scaled(title_size), scaled(legend_size),
                                           scaled(annot_size))
    xtick_size, ytick_size = scaled(xtick_size), scaled(ytick_size)

    for ax in fig.axes:
        ax.set_facecolor(WHITE)

        if strip_titles:
            ax.set_title("")
        elif title_size is not None:
            ax.title.set_size(title_size)

        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")
        ax.xaxis.label.set_size(label_size)
        ax.yaxis.label.set_size(label_size)

        # Per-axis overrides exist because some figures need one axis larger than
        # the other — the heatmap's 24 hourly y-ticks overlap if grown to match
        # its seven x-ticks.
        ax.tick_params(colors="black", labelsize=tick_size)
        for lbl in ax.get_xticklabels():
            lbl.set_color("black")
            if xtick_size is not None:
                lbl.set_size(xtick_size)
        for lbl in ax.get_yticklabels():
            lbl.set_color("black")
            if ytick_size is not None:
                lbl.set_size(ytick_size)

        # Only recolour spines that are already shown — the plotting functions
        # deliberately hide top/right, and forcing a colour would resurrect them.
        for spine in ax.spines.values():
            if spine.get_visible():
                spine.set_color("black")

        for txt in ax.texts:
            txt.set_color(text_color)
            if annot_size is not None:
                txt.set_size(annot_size)

        # Legends come in two flavours (per-axis and figure-level) and matplotlib
        # keeps them in different places, so both get handled — here and below.
        if legend_size is not None and ax.get_legend() is not None:
            lg = ax.get_legend()
            for t in lg.get_texts():
                t.set_size(legend_size)
            if lg.get_title() is not None:
                lg.get_title().set_size(legend_size)

    if legend_size is not None:
        for lg in fig.legends:
            for t in lg.get_texts():
                t.set_size(legend_size)
            if lg.get_title() is not None:
                lg.get_title().set_size(legend_size)

    fit_axis_labels(fig)

    # `bbox_inches="tight"` alone truncates a long axis label, and not by
    # clipping: `Axes.get_tightbbox` deliberately collapses an axis label to a
    # 1px sliver at its centre so a long label cannot drive the layout. The crop
    # is therefore computed as if the label were a point, and anything past the
    # edge of the other artists is cut. Naming the labels as extra artists puts
    # their real extents back into the crop. This is the half that made
    # "Deviation from Baseline" whole; fit_axis_labels is the half that keeps the
    # label inside the canvas so it is drawn in the first place.
    axis_labels = [a.xaxis.label for a in fig.axes] + [a.yaxis.label for a in fig.axes]
    # Passing bbox_extra_artists replaces the default list rather than adding to
    # it, and that default is where legends live — so a legend placed outside the
    # axes is cropped away entirely unless it is named here too.
    legends = [a.get_legend() for a in fig.axes if a.get_legend() is not None]

    save(fig, stem, bbox_inches="tight",
         bbox_extra_artists=axis_labels + legends + list(fig.legends))


# ─────────────────────────────────────────────────────────────────────────────
# Centroid cache loading (mean aggregation only)
# ─────────────────────────────────────────────────────────────────────────────
def load_centroids():
    """Return centroids[label][filter][sign] and points[label][filter][sign] for AGG."""
    centroids, points = {}, {}
    cent_dir = DATA_ROOT / AGG / "centroids"
    for label in LABELS:
        centroids[label], points[label] = {}, {}
        for filter_name in ["raw"] + FILTER_NAMES:
            centroids[label][filter_name], points[label][filter_name] = {}, {}
            for sign in ["pos", "neg"]:
                data = load_pkl(cent_dir / f"{label}_{filter_name}_{sign}.pkl")
                if data is not None:
                    c, p = data
                    centroids[label][filter_name][sign] = c
                    points[label][filter_name][sign] = p
    return centroids, points


def centroid_pairs(centroids, filter_name):
    return [
        (centroids["icu"]["raw"]["pos"],       centroids["icu"][filter_name]["pos"]),
        (centroids["icu"]["raw"]["neg"],       centroids["icu"][filter_name]["neg"]),
        (centroids["mortality"]["raw"]["pos"], centroids["mortality"][filter_name]["pos"]),
        (centroids["mortality"]["raw"]["neg"], centroids["mortality"][filter_name]["neg"]),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Centroid density of the raw dataset (95th percentile, mean agg)
# ─────────────────────────────────────────────────────────────────────────────
def fig_centroid_density_raw(centroids, points, *, use_percentile=True,
                             stem="centroid_density_raw_mimic_iii"):
    print(f"  {stem} ...", flush=True)
    try:
        raw_c = [
            centroids["icu"]["raw"]["pos"], centroids["icu"]["raw"]["neg"],
            centroids["mortality"]["raw"]["pos"], centroids["mortality"]["raw"]["neg"],
        ]
        raw_p = [
            points["icu"]["raw"]["pos"], points["icu"]["raw"]["neg"],
            points["mortality"]["raw"]["pos"], points["mortality"]["raw"]["neg"],
        ]
    except KeyError as e:
        print(f"    [SKIP] missing raw centroid data: {e}")
        return
    cats = ["Raw ICU Pos", "Raw ICU Neg", "Raw Mort Pos", "Raw Mort Neg"]
    fig = centroid_plot(raw_c, raw_p, VITAL_NAMES, VITAL_UNITS, cats,
                        use_percentile=use_percentile, background="white")

    # Move the legend off the top of the figure and into the empty (2,4) grid cell
    # (centroid_plot leaves the 8th panel blank for 7 vitals), and size its labels to
    # match the rest of the graph.
    handles, labels = fig.axes[0].get_legend_handles_labels()
    for lg in list(fig.legends):   # drop centroid_plot's top figure-level legend
        lg.remove()
    legend_ax = fig.axes[-1]       # the blank (2,4) panel, added last by centroid_plot
    legend_ax.axis("off")
    legend_ax.legend(handles, labels, loc="center", facecolor=WHITE, edgecolor="black",
                     labelcolor="black", fontsize=26, framealpha=1.0,
                     title="Subpopulation", title_fontsize=30)

    # Keep the per-vital panel labels (subplot titles); enlarge the vital names, the
    # y-axis numbers, and the axis labels significantly for readability at small sizes.
    finalize(fig, stem, strip_titles=False, title_size=36, tick_size=27, label_size=28)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Heatmap of mean per-hour observation counts per vital (raw)
# ─────────────────────────────────────────────────────────────────────────────
def fig_observation_count_heatmap():
    print("  observation_count_heatmap_raw_mimic_iii ...", flush=True)
    records = load_pkl(DATA_ROOT / "processed_record_ehr.pkl")
    if records is None:
        print("    [SKIP] processed_record_ehr.pkl not found")
        return

    length_dfs = []
    for rec in records:
        try:
            ldf = rec.timeseries.map(
                lambda x: x.size if hasattr(x, "size") else (len(x) if isinstance(x, list) else 0)
            )
            length_dfs.append(ldf)
        except Exception:
            pass
    if not length_dfs:
        print("    [SKIP] no length DataFrames built")
        return

    combined = pd.concat(length_dfs)
    mean_lengths = combined.groupby(level=0).agg("mean")

    fig = heatmap(mean_lengths, y_title="Hour", background_colour=WHITE, value_format=".3f")
    # Cell value annotations white (readable on the dark aurora cells); axes/ticks stay black.
    # The 24 hourly y-ticks cap how large the y-ticks can grow before overlapping
    # vertically, so they stay modest while the "Hour" label and vital-name x-ticks grow.
    finalize(fig, "observation_count_heatmap_raw_mimic_iii", text_color="white",
             label_size=26, xtick_size=11.33, ytick_size=12, annot_size=10)


# ─────────────────────────────────────────────────────────────────────────────
# Figures 3 & 6: Centroid deviation from baseline for a given filter (mean agg)
# ─────────────────────────────────────────────────────────────────────────────
def fig_centroid_deviation(centroids, filter_name, stem):
    print(f"  {stem} ...", flush=True)
    try:
        cp = centroid_pairs(centroids, filter_name)
    except KeyError as e:
        print(f"    [SKIP] missing centroid data for '{filter_name}': {e}")
        return
    cats = ["icu pos", "icu neg", "mortality pos", "mortality neg"]
    fig = centroid_shift_plot(cp, cats, filter_name, VITAL_NAMES, background=WHITE)
    finalize(fig, stem, label_size=28, tick_size=24, legend_size=22)


def fig_centroid_deviation_split(centroids, filter_name, stem):
    """The same deviation plot split in two: heart rate alone, and the other six.

    Under the all-vitals filter heart rate moves ~30 units where no other vital
    moves more than ~4, so on a shared y-axis the other six read as flat. Each
    half gets its own y-scale; axis titles, hatching and legend are unchanged.

    The legend goes above the axes rather than at matplotlib's "best" spot: with
    bars both sides of zero there is no empty corner that is guaranteed to stay
    empty, and above the plot it cannot cover a bar or a value label.
    """
    try:
        cp = centroid_pairs(centroids, filter_name)
    except KeyError as e:
        print(f"    [SKIP] missing centroid data for '{filter_name}': {e}")
        return
    cats = ["icu pos", "icu neg", "mortality pos", "mortality neg"]
    hr = VITAL_NAMES.index("heart rate")
    parts = [
        ("heart_rate", [hr], (7, 7)),
        ("other_vitals", [i for i in range(len(VITAL_NAMES)) if i != hr], (20, 7)),
    ]
    for suffix, idx, size in parts:
        part_stem = f"{stem}_{suffix}"
        print(f"  {part_stem} ...", flush=True)
        pairs = [([b[i] for i in idx], [s[i] for i in idx]) for b, s in cp]
        fig = centroid_shift_plot(pairs, cats, filter_name,
                                  [VITAL_NAMES[i] for i in idx],
                                  fig_size=size, background=WHITE)
        ax = fig.axes[0]
        if len(idx) == 1:
            # One tick needs no 45° slant to clear its neighbours.
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
        else:
            # 24 bars: a "+0.0194" label is wider than its bar, so neighbours
            # near zero print over each other. Stand them upright, then widen
            # the y-limits until every label sits inside the axes.
            for txt in ax.texts:
                txt.set_rotation(90)
            fig.canvas.draw()
            to_data = ax.transData.inverted()
            ys = [y for t in ax.texts if t.get_text()
                  for y in to_data.transform(
                      t.get_window_extent(fig.canvas.get_renderer()))[:, 1]]
            low, high = ax.get_ylim()
            span = max(high, *ys) - min(low, *ys)
            ax.set_ylim(min(low, *ys) - 0.02 * span, max(high, *ys) + 0.02 * span)
        # Colour is direction, hatch is group. Default swatches borrow each
        # group's first bar colour, so "icu pos" would be red in one half and
        # green in the other; neutral swatches keep the two legends identical.
        handles = [Patch(facecolor=WHITE, edgecolor="black", linewidth=0.5,
                         hatch=bars.patches[0].get_hatch(), label=cat)
                   for bars, cat in zip(ax.containers, cats)]
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
                  ncol=2 if len(idx) == 1 else 4, frameon=False,
                  handlelength=2.4, handleheight=1.4)
        finalize(fig, part_stem, label_size=28, tick_size=24, legend_size=22)


# ─────────────────────────────────────────────────────────────────────────────
# Figures 4,5,8,9: Filter impact (accuracy / macro F1) per label (mean agg)
# ─────────────────────────────────────────────────────────────────────────────
def fig_filter_impact(label):
    impact = load_impact(label)
    accuracy, f1 = impact[1], impact[3]
    accuracy_label, f1_label = METRIC_LABELS[IMPACT_SUFFIX]

    # Stems keep the word "testing" under both designs. They are what the paper
    # \includegraphics references; renaming them would break the build silently
    # while the figures themselves still rendered. The axis label carries the
    # distinction instead.
    print(f"  filter_impact_{label}_testing_accuracy_mean ...", flush=True)
    fig = filter_impact_plot(accuracy, list(FILTER_NAMES), label.upper(),
                             accuracy_label, AGG, background=WHITE)
    finalize(fig, f"filter_impact_{label}_testing_accuracy_mean",
             label_size=22, tick_size=18)

    print(f"  filter_impact_{label}_testing_f1_mean ...", flush=True)
    fig = filter_impact_plot(f1, list(FILTER_NAMES), label.upper(),
                             f1_label, AGG, background=WHITE)
    finalize(fig, f"filter_impact_{label}_testing_f1_mean",
             label_size=22, tick_size=18)


# ─────────────────────────────────────────────────────────────────────────────
# Figures 7 & 13: McNemar significance per label (mean agg) — paper variant
# ─────────────────────────────────────────────────────────────────────────────
# Direction x significance. Grey is "no detectable difference"; the saturated
# shades are the p < 0.0001 tier that used to be the lone red.
# Bar heights are capped here rather than left to run free. Under the CV design
# the mortality p-values reach 1e-147, and an axis that tall flattens the 0.05 and
# 0.0001 reference lines into the baseline — the two levels the figure exists to
# show. Capped bars are hatched and the annotation always prints the *true*
# p-value, so the cap is never mistaken for the measurement. (The pre-existing
# `max(p, 1e-20)` clamp did the same thing to the height but then printed the
# clamped value, so a bar with p=1.3e-147 was labelled "1.0e-20".)
MCNEMAR_CAP = 20.0

MCNEMAR_GREY = "#a0a0a0"
MCNEMAR_HELPED = ("#7fbfa2", "#3b8465")     # (p < 0.05, p < 0.0001)
MCNEMAR_HURT = ("#d98c96", "#bd3140")
MCNEMAR_RULE = "#bd3140"

# The pre-CV paper's amber, recovered from 2d26d3b. Used only by
# style="significance", which tiers bars by p alone the way the published paper
# figure does: grey -> amber -> crimson, no direction encoding.
MCNEMAR_AMBER = "#f39c12"


def _neg_log10(p):
    """-log10(p), with exact zero mapped to the smallest representable exponent.

    `mcnemar(exact=True)` underflows to a hard 0.0 well before the true p-value
    would; every such arm plots at the same height, which is the honest thing to
    do since the data cannot distinguish them.
    """
    return 308.0 if p <= 0.0 else float(-np.log10(p))


def _format_p(p):
    """The measured p-value as text. Never the capped or clamped one."""
    if p <= 0.0:
        return r"$p<10^{-308}$"
    if p > 0.001:
        return f"{p:.3f}"
    return f"{p:.1e}"


def _shrink_to_fit(fig, ax, texts, min_size=8.5, step=0.94, max_iterations=30):
    """Shrink a row of bar annotations until neighbours stop overlapping.

    Twelve arms in a 14-inch axes leave roughly 75 points per bar, and a
    two-count annotation like "4,731/2,873" needs more than that at size 13, so
    adjacent labels collide and read as one number — "2,8731,710/519" in the
    mortality panel. Bar order carries meaning here, so the fix is type size
    rather than dropping or rotating labels.

    Deliberately a no-op when nothing overlaps: a figure whose labels already fit
    keeps the size-13 text it has always had, so this cannot quietly restyle the
    panels that were fine.
    """
    if len(texts) < 2:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def overlaps():
        boxes = [t.get_window_extent(renderer=renderer) for t in texts]
        order = sorted(boxes, key=lambda b: b.x0)
        # 2px of clear air between neighbours, matching the mark spacer elsewhere.
        return any(b.x1 + 2.0 > n.x0 for b, n in zip(order, order[1:]))

    size = texts[0].get_fontsize()
    for _ in range(max_iterations):
        if not overlaps():
            return
        size *= step
        if size < min_size:
            break
        for t in texts:
            t.set_fontsize(size)
        fig.canvas.draw()

    # Still colliding at the floor: stagger every other label upward by one line
    # rather than shrink into illegibility.
    if overlaps():
        # Offset in points via the figure's dpi transform — the y positions are in
        # data units, and "one line up" has no data-space meaning.
        for i, t in enumerate(texts):
            if i % 2:
                t.set_transform(t.get_transform()
                                + ScaledTranslation(0, 1.4 * size / 72.0, fig.dpi_scale_trans))
        fig.canvas.draw()


def paper_mcnemar_plot(p_values, filter_names, directions=None, label_size=26,
                       tick_size=18, figsize=(14, 7), style="directional",
                       cap=MCNEMAR_CAP, show_counts=True, show_legend=True):
    """White bg, black text, no title, two labeled thresholds, tiered bar colours.

    `directions` is the {arm: (n01, n10)} map from load_mcnemar_directions, or
    None for the legacy design. When present, bars are coloured by whether the arm
    fixed more records than it broke and annotated with the two counts.

    The direction encoding is not decoration. Bar height is -log10(p), so a filter
    that made the model reliably *worse* draws exactly the same tall bar as one
    that helped, and the legacy figure was read as "taller means better" in both
    notebook.py:515-516 and noahNotes.md:448-460 (bug_register E-06). Colour and
    the n01/n10 annotation are what make the two distinguishable.

    `style` picks between that and the published paper's plainer encoding:

        'directional' (default)  green/red by n01 vs n10, as described above.
        'significance'           grey -> amber -> crimson by p alone, the pre-CV
                                 paper figure's scheme (2d26d3b). Carries no
                                 direction information — see the E-06 note above
                                 for what that costs the reader.

    `cap`, `show_counts` and `show_legend` are independent of `style`; their
    defaults reproduce the current published figures exactly.
    """
    neg_log_p_true = np.array([_neg_log10(p) for p in p_values])
    neg_log_p = np.minimum(neg_log_p_true, cap)
    capped = neg_log_p_true > cap
    indices = np.arange(len(p_values))
    filter_names = filter_names[:len(p_values)]

    t1, t2 = 0.05, 0.0001          # significance thresholds
    y1, y2 = -np.log10(t1), -np.log10(t2)

    counts = [None] * len(p_values)
    if directions is not None:
        counts = [directions.get(name) for name in filter_names]

    colors = []
    if style == "significance":
        for p in p_values:
            colors.append(MCNEMAR_RULE if p < t2 else
                          (MCNEMAR_AMBER if p < t1 else MCNEMAR_GREY))
    else:
        for p, count in zip(p_values, counts):
            if p >= t1 or count is None or count[0] == count[1]:
                colors.append(MCNEMAR_GREY)
                continue
            palette = MCNEMAR_HELPED if count[0] > count[1] else MCNEMAR_HURT
            colors.append(palette[1] if p < t2 else palette[0])

    if not show_counts:
        counts = [None] * len(p_values)

    fig, ax = plt.subplots(figsize=figsize, facecolor=WHITE)
    ax.set_facecolor(WHITE)
    bars = ax.bar(indices, neg_log_p, color=colors, alpha=0.9, zorder=3)

    for bar, is_capped in zip(bars, capped):
        if is_capped:
            bar.set_hatch("///")
            bar.set_edgecolor(WHITE)
            bar.set_linewidth(0.0)

    # Two dashed reference lines, labeled just outside the right spine so they
    # never land on top of a bar or its annotation.
    for y, txt in ((y1, r"$p=0.05$"), (y2, r"$p=0.0001$")):
        ax.axhline(y=y, color=MCNEMAR_RULE, linestyle="--", linewidth=1.5, zorder=4)
        ax.text(1.012, y, txt, transform=ax.get_yaxis_transform(), clip_on=False,
                ha="left", va="center", color="black", fontsize=tick_size)

    # p-value annotation above each bar, with the discordant counts beneath it
    # where they exist. Always the measured p, never the capped one.
    annotations = []
    for bar, p, count in zip(bars, p_values, counts):
        h = bar.get_height()
        disp = _format_p(p)
        if count is not None:
            disp = f"{disp}\n{count[0]:,}/{count[1]:,}"
        annotations.append(ax.text(
            bar.get_x() + bar.get_width() / 2., h + 0.1, disp,
            ha="center", va="bottom", color="black", fontsize=13, linespacing=1.3))

    _shrink_to_fit(fig, ax, annotations)

    ax.set_xticks(indices)
    ax.set_xticklabels(filter_names, rotation=45, ha="right", color="black", fontsize=tick_size)
    ax.tick_params(colors="black", labelsize=tick_size)
    ax.set_ylabel(r"$-\log_{10}(p\text{-value})$", color="black", fontsize=label_size)
    ax.set_xlabel("Filter Name", color="black", fontsize=label_size)

    ax.grid(axis="y", color="gray", linestyle="--", linewidth=0.5, alpha=0.2, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_color("black")

    handles = []
    if show_legend and directions is not None and style != "significance":
        handles += [
            Patch(facecolor=MCNEMAR_HELPED[1], label="filter fixed more than it broke"),
            Patch(facecolor=MCNEMAR_HURT[1], label="filter broke more than it fixed"),
            Patch(facecolor=MCNEMAR_GREY, label=r"no significant difference ($p \geq 0.05$)"),
        ]
    if show_legend and capped.any():
        handles.append(Patch(facecolor=MCNEMAR_GREY, hatch="///", edgecolor=WHITE,
                             label=rf"bar height capped at $-\log_{{10}}(p)={cap:.0f}$"))
    if not handles and not show_legend:
        # The legend is what normally forces headroom for the annotation above the
        # tallest bar; without it the top label would clip.
        ax.set_ylim(top=max(ax.get_ylim()[1], float(neg_log_p.max()) * 1.18))
    if handles:
        # Above the axes, not inside them. Inside, "upper left" sat on top of the
        # annotations of any bar that reaches the cap, and on mortality four of
        # them do — there is no free corner once several bars are capped.
        title = "Direction (fixed / broken)" if directions is not None else None
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.01),
                  ncol=2, facecolor=WHITE, edgecolor="black", labelcolor="black",
                  fontsize=tick_size - 2, framealpha=1.0, title=title,
                  title_fontsize=tick_size - 2, alignment="left")
        # Headroom for the two-line annotation above the tallest bar.
        ax.set_ylim(top=max(ax.get_ylim()[1], float(neg_log_p.max()) * 1.18))

    plt.tight_layout()
    return fig


def _exact_mcnemar_p(n01: int, n10: int) -> float:
    """Two-sided exact (binomial) McNemar p-value for a discordant pair count.

    Identical to `statsmodels mcnemar(table, exact=True)` — for a symmetric
    binomial the two-sided test is `2 * cdf(min(n01, n10))` capped at 1 — but
    written against scipy because the renderer runs under scipy-stack, which has
    no statsmodels. `mcnemar_at_baseline_threshold` asserts the equivalence
    against the stored p-values before it returns, so a divergence is caught
    rather than quietly plotted.
    """
    if n01 + n10 == 0:
        return 1.0
    return float(binomtest(min(n01, n10), n01 + n10, 0.5).pvalue)


def mcnemar_at_baseline_threshold(label: str):
    """Re-run McNemar with every arm scored at the *baseline's* mean threshold.

    Why this exists. In the published design each arm thresholds its consensus
    scores at its own 20-fold mean Youden threshold, so an arm differs from raw
    in two ways at once: the filter changed the features, and the operating point
    moved. The second dominates. Across the seven per-vital arms,
    corr(delta threshold, n01 - n10) = 0.985 and corr(delta threshold, delta
    accuracy) = 0.992, while the threshold shifts themselves are not
    distinguishable from zero (paired over the 20 matched folds, diastolic blood
    pressure t = 0.69, mean blood pressure t = 0.82). Holding the threshold fixed
    isolates the filter.

    The arithmetic is `evaluate_filter_impact_cv`'s, replayed from the stored
    out-of-fold scores rather than refit: consensus score = the mean of the four
    repeats' OOF scores, then one threshold. Only the threshold changes, so no
    forest is retrained and the result is deterministic.

    Caveat carried by the caller, not fixable here: the three record-dropping
    arms (`long missing segment`, `long gap`, `high invalid data`) score a
    different cohort than raw, so their own threshold is legitimately different
    and forcing raw's on them is less apples-to-apples than it is for the
    full-cohort arms. The comparison is still paired on the intersection.

    Returns (p_values, directions) shaped exactly like the published path:
    p_values in FILTER_NAMES order, directions as {arm: (n01, n10)}.
    """
    scores_path = DATA_ROOT / AGG / f"{label}_oof_scores_cv.npz"
    diagnostics_path = DATA_ROOT / AGG / f"{label}_filter_impact_cv_diagnostics.json"
    for path in (scores_path, diagnostics_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path.name} not found under {path.parent}. The baseline-threshold "
                "McNemar is derived from stage F's stored out-of-fold scores; run "
                "rerun/job_f_filter_impact_cv.sh first."
            )

    bundle = np.load(scores_path, allow_pickle=True)
    diagnostics = json.loads(diagnostics_path.read_text())
    baseline_threshold = diagnostics["raw"]["mean_threshold"]

    label_by_id = dict(zip(bundle["raw_admission_ids"].tolist(),
                           bundle["raw_labels"].tolist()))

    def consensus(arm):
        """(ids, consensus score) for one arm. -1 marks a repeat that never covered
        the record; nanmean over the rest is what the CV path itself averages."""
        key = arm.replace(" ", "_")
        raw_scores = bundle[f"{key}__oof_scores"].astype(np.float64)
        covered = np.where(raw_scores < 0, np.nan, raw_scores)
        with np.errstate(invalid="ignore"):
            return bundle[f"{key}__admission_ids"], np.nanmean(covered, axis=0)

    def tabulate(arm_ids, arm_predictions, base_ids, base_predictions):
        """n01/n10 over the records both arms hold, scored against truth —
        the same correctness tabulation as calculate_mcnemar_paired, including
        its rule that -1 marks a record the arm never predicted and so cannot
        contribute a pair."""
        arm_predictions = np.asarray(arm_predictions)
        base_predictions = np.asarray(base_predictions)
        shared = np.intersect1d(base_ids[base_predictions >= 0],
                                arm_ids[arm_predictions >= 0])
        base_lookup = dict(zip(base_ids.tolist(), base_predictions.tolist()))
        arm_lookup = dict(zip(arm_ids.tolist(), arm_predictions.tolist()))
        truth = np.array([label_by_id[i] for i in shared.tolist()])
        base_correct = np.array([base_lookup[i] for i in shared.tolist()]) == truth
        arm_correct = np.array([arm_lookup[i] for i in shared.tolist()]) == truth
        return (int((~base_correct & arm_correct).sum()),
                int((base_correct & ~arm_correct).sum()))

    # The baseline side of every comparison is the pipeline's own stored decision,
    # not a replay of it. `oof_scores` is saved as float32 while the CV path
    # thresholded in float64, so a score sitting exactly on a threshold can round
    # either way — at its own threshold `temperature` flips 34 records that way,
    # because a forest's scores are k/n_trees rationals and a mean-of-thresholds
    # lands on one often enough to matter. Reading the stored predictions sidesteps
    # the round trip entirely.
    base_ids = bundle["raw__admission_ids"]
    base_predictions = bundle["raw__consensus_predictions"]

    p_values, directions = [], {}
    for arm in FILTER_NAMES:
        key = arm.replace(" ", "_")
        arm_ids, arm_scores = consensus(arm)

        # Self-check: the stored predictions, tabulated here, must reproduce the
        # sidecar's counts. This validates the pairing and the correctness
        # tabulation against evaluate_filter_impact_cv with no arithmetic of our
        # own in the way — if it fails, the figure would be fiction.
        published = diagnostics[arm]["mcnemar"]
        replayed = tabulate(arm_ids, bundle[f"{key}__consensus_predictions"],
                            base_ids, base_predictions)
        if replayed != (published["n01"], published["n10"]):
            raise AssertionError(
                f"{label}/{arm}: the stored consensus predictions tabulate to n01/n10 "
                f"{replayed}, but the diagnostics sidecar recorded "
                f"({published['n01']}, {published['n10']}). The two artifacts "
                "disagree; regenerate stage F before rendering."
            )

        # Re-thresholding IS a float comparison, so guard the one thing that could
        # corrupt it. Currently zero arms have a score within 1e-6 of the baseline
        # threshold in either label, so nothing is riding on the rounding — but a
        # future rerun could land on one, and it must not do so silently.
        covered = ~np.isnan(arm_scores)
        ties = int((np.abs(arm_scores[covered] - baseline_threshold) < 1e-6).sum())
        if ties:
            raise AssertionError(
                f"{label}/{arm}: {ties} record(s) score within 1e-6 of the baseline "
                f"threshold {baseline_threshold!r}. oof_scores is stored as float32, "
                "so their side of the cut is not recoverable and the counts below "
                "would be arbitrary to that many records."
            )

        arm_predictions = np.full(arm_scores.shape, -1, dtype=np.int8)
        arm_predictions[covered] = (arm_scores[covered] >= baseline_threshold).astype(np.int8)

        n01, n10 = tabulate(arm_ids, arm_predictions, base_ids, base_predictions)
        p_values.append(_exact_mcnemar_p(n01, n10))
        directions[arm] = (n01, n10)

    return p_values, directions


def fig_mcnemar(label, threshold_mode="own", style="directional", cap=MCNEMAR_CAP):
    """The McNemar panel. `threshold_mode` picks which comparison it shows.

    'own' (default) is the published design, read straight from the impact
    pickle. 'baseline' recomputes every arm at raw's threshold — a different
    measurement, so it gets its own filename rather than overwriting a figure the
    paper's \\includegraphics already resolves.

    `style='significance'` renders the plainer published-paper encoding: no
    legend, no n01/n10 counts, bars tiered by p alone. It takes a further filename
    suffix for the same reason — it is a different chart, not a restyle of one the
    paper already cites.
    """
    if threshold_mode == "baseline":
        stem = f"mcnemar_{label}_mean_baseline_threshold"
        print(f"  {stem} ...", flush=True)
        p_values, directions = mcnemar_at_baseline_threshold(label)
    else:
        stem = f"mcnemar_{label}_mean"
        print(f"  {stem} ...", flush=True)
        impact = load_impact(label)
        p_values = [x[1] for x in impact[-1]][1:]  # drop raw baseline, which is p=1 against itself
        directions = load_mcnemar_directions(label)
    if style == "significance":
        stem = f"{stem}_plain"
        print(f"    -> {stem}", flush=True)
    fig = paper_mcnemar_plot(
        p_values, list(FILTER_NAMES), directions=directions, style=style, cap=cap,
        show_counts=(style != "significance"), show_legend=(style != "significance"))
    # finalize would re-strip; this figure is already paper-styled, just save.
    save(fig, stem, bbox_inches="tight")


# ─────────────────────────────────────────────────────────────────────────────
# ECDF grid: raw vs filtered per-vital centroid distributions (fill missing data, mean)
# ─────────────────────────────────────────────────────────────────────────────
# 4 vitals × {raw, filtered}; one panel each, four subpopulation curves per panel.
ECDF_FILTER = "fill missing data"
ECDF_VITALS = [  # (name, unit, index into the centroid vector)
    ("heart rate", "bpm", 0),
    ("mean blood pressure", "mmHg", 3),
    ("temperature", "C", 5),
    ("oxygen saturation", "%", 6),
]
ECDF_SUBPOPS = [("icu", "pos"), ("icu", "neg"), ("mortality", "pos"), ("mortality", "neg")]
ECDF_SUBPOP_LABELS = {("icu", "pos"): "ICU+", ("icu", "neg"): "ICU−",
                      ("mortality", "pos"): "Mort+", ("mortality", "neg"): "Mort−"}
ECDF_COLORS = {("icu", "pos"): "#1b9e77", ("icu", "neg"): "#7570b3",
               ("mortality", "pos"): "#d95f02", ("mortality", "neg"): "#e7298a"}


def _ecdf(values):
    v = np.sort(values[~np.isnan(values)])
    if v.size == 0:
        return v, v
    return v, np.arange(1, len(v) + 1) / len(v)


def _p95_clip(values, center):
    d = np.abs(values - center)
    lim = np.nanpercentile(d, 95)
    return values[d <= lim]


def fig_ecdf_grid(layout="4x2"):
    """Render the raw-vs-filtered ECDF grid in either a 4x2 or 2x4 panel arrangement.

    layout="4x2": one vital per row (raw col 0, filtered col 1).
    layout="2x4": two vitals per row, each as a raw/filtered pair of adjacent columns.
    """
    stem = f"ecdf_fill_missing_data_mean_{layout}"
    print(f"  {stem} ...", flush=True)
    cent_dir = DATA_ROOT / AGG / "centroids"

    def load(condition, label, pol):
        data = load_pkl(cent_dir / f"{label}_{condition}_{pol}.pkl")
        if data is None:
            return None, None
        c, pts = data
        return c, np.asarray(pts)

    if layout == "4x2":
        nrows, ncols, figsize = 4, 2, (12, 18)
        # vital i → raw at (i, 0), filtered at (i, 1)
        placement = [((i, 0), (i, 1)) for i in range(len(ECDF_VITALS))]
    elif layout == "2x4":
        nrows, ncols, figsize = 2, 4, (22, 9)
        # vital i → raw/filtered pair of adjacent columns, two vitals per row
        placement = [((i // 2, (i % 2) * 2), (i // 2, (i % 2) * 2 + 1))
                     for i in range(len(ECDF_VITALS))]
    else:
        raise ValueError(f"unknown layout {layout!r}")

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=WHITE)

    def draw_panel(ax, name, unit, vital_idx, condition, nice):
        ax.set_facecolor(WHITE)
        for (label, pol) in ECDF_SUBPOPS:
            c, pts = load(condition, label, pol)
            if c is None:
                continue
            x, y = _ecdf(_p95_clip(pts[:, vital_idx], c[vital_idx]))
            if x.size == 0:
                continue
            ax.plot(x, y, color=ECDF_COLORS[(label, pol)], linewidth=1.8,
                    label=ECDF_SUBPOP_LABELS[(label, pol)])
        ax.set_title(f"{name} — {nice}", color="black", fontsize=18)
        ax.set_xlabel(unit, color="black", fontsize=17)
        ax.set_ylabel("cumulative fraction", color="black", fontsize=17)
        ax.tick_params(colors="black", labelsize=14)
        for s in ax.spines.values():
            s.set_color("black")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(color="gray", linestyle="--", linewidth=0.4, alpha=0.4)
        ax.set_ylim(0, 1.02)

    first_ax = None
    for (name, unit, vital_idx), (raw_cell, filt_cell) in zip(ECDF_VITALS, placement):
        raw_ax = axes[raw_cell]
        draw_panel(raw_ax, name, unit, vital_idx, "raw", "raw")
        draw_panel(axes[filt_cell], name, unit, vital_idx, ECDF_FILTER, "filtered")
        if first_ax is None:
            first_ax = raw_ax

    # Legend incorporated into the graph: placed in the empty lower-right region of the
    # first panel (an ECDF leaves that corner clear).
    handles, labels = first_ax.get_legend_handles_labels()
    first_ax.legend(handles, labels, loc="lower right", facecolor=WHITE,
                    edgecolor="black", labelcolor="black", fontsize=15,
                    framealpha=1.0, title="Subpopulation", title_fontsize=16)

    fig.tight_layout()
    save(fig, stem, bbox_inches="tight")


# ─────────────────────────────────────────────────────────────────────────────
# --only selects whole groups rather than individual stems: the two figures in a
# group share a load (centroids, or one impact pickle), so splitting them would
# save nothing.
GROUPS = ("centroid_density", "heatmap", "centroid_deviation", "filter_impact",
          "mcnemar", "ecdf")

# Groups that need the centroid cache. Loading it is ~30 MB of pickles and about a
# second, but the heatmap group instead needs the 286 MB processed_record_ehr.pkl,
# which is why --only exists at all.
CENTROID_GROUPS = {"centroid_density", "centroid_deviation", "ecdf"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--impact-suffix", default="_cv", choices=sorted(METRIC_LABELS),
        help="which evaluation design the model figures describe: '_cv' (default, "
             "cross-validated) or '' (legacy single holdout)")
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT_DIR,
        help=f"output directory (default: {DEFAULT_OUT_DIR})")
    parser.add_argument(
        "--mcnemar-threshold", default="own", choices=("own", "baseline"),
        help="which operating point the McNemar panel compares at: 'own' "
             "(default, each arm at its own mean Youden threshold — the "
             "published design) or 'baseline' (every arm re-scored at raw's "
             "threshold, isolating the filter from the operating-point shift). "
             "'baseline' writes mcnemar_<label>_mean_baseline_threshold.pdf and "
             "requires --impact-suffix _cv")
    parser.add_argument(
        "--mcnemar-style", default="directional", choices=("directional", "significance"),
        help="McNemar bar encoding: 'directional' (default) colours by n01 vs n10 "
             "and annotates both counts; 'significance' is the published paper's "
             "plainer scheme — grey/amber/crimson by p alone, no legend, no counts "
             "— and writes a _plain filename")
    parser.add_argument(
        "--mcnemar-cap", type=float, default=MCNEMAR_CAP, metavar="NEGLOG10P",
        help=f"cap McNemar bar height at this -log10(p) (default: {MCNEMAR_CAP:.0f}). "
             "Capped bars are hatched and always annotated with the true p-value")
    parser.add_argument(
        "--format", default="pdf", choices=("pdf", "png"),
        help="output format (default: pdf, what the paper embeds). 'png' is for "
             "slide decks, which cannot embed PDF")
    parser.add_argument(
        "--dpi", type=int, default=200, metavar="DPI",
        help="raster resolution, --format png only (default: 200)")
    parser.add_argument(
        "--font-scale", type=float, default=1.0, metavar="X",
        help="multiply every type size finalize() applies (default: 1.0, the paper). "
             "For figures placed smaller than their natural width, e.g. a poster")
    parser.add_argument(
        "--only", default=None,
        help="comma-separated subset of " + ",".join(GROUPS) + " (default: all)")
    return parser.parse_args(argv)


def main(argv=None):
    global IMPACT_SUFFIX, OUT_DIR, FORMAT, DPI, FONT_SCALE
    args = parse_args(argv)

    IMPACT_SUFFIX = args.impact_suffix
    OUT_DIR = args.out
    FORMAT = args.format
    DPI = args.dpi
    FONT_SCALE = args.font_scale
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.only:
        selected = [g.strip() for g in args.only.split(",") if g.strip()]
        unknown = [g for g in selected if g not in GROUPS]
        if unknown:
            raise SystemExit(f"unknown group(s) {unknown}; choose from {list(GROUPS)}")
    else:
        selected = list(GROUPS)

    if args.mcnemar_threshold == "baseline" and IMPACT_SUFFIX != "_cv":
        raise SystemExit(
            "--mcnemar-threshold baseline needs the out-of-fold scores that only "
            "the _cv design stores; it cannot be combined with --impact-suffix ''.")

    design = "cross-validated (_cv)" if IMPACT_SUFFIX == "_cv" else "legacy single holdout"
    print(f"Output dir: {OUT_DIR}")
    print(f"Format: {FORMAT}" + (f" @ {DPI} dpi" if FORMAT == "png" else "")
          + (f", type x{FONT_SCALE:g}" if FONT_SCALE != 1.0 else ""))
    print(f"Evaluation design: {design}")
    print(f"Groups: {', '.join(selected)}")
    if "mcnemar" in selected:
        print(f"McNemar operating point: {args.mcnemar_threshold}")

    centroids, points = ({}, {})
    if CENTROID_GROUPS.intersection(selected):
        centroids, points = load_centroids()

    if "centroid_density" in selected:
        print("\n=== Centroid density (raw) ===")
        fig_centroid_density_raw(centroids, points)  # 95th-percentile clipped (default)
        fig_centroid_density_raw(centroids, points, use_percentile=False,
                                 stem="centroid_density_raw_mimic_iii_full_range")

    if "heatmap" in selected:
        print("\n=== Observation-count heatmap ===")
        fig_observation_count_heatmap()

    if "centroid_deviation" in selected:
        print("\n=== Centroid deviations ===")
        fig_centroid_deviation(centroids, "all vitals",
                               "centroid_deviation_all_vitals_filter_mean")
        fig_centroid_deviation_split(centroids, "all vitals",
                                     "centroid_deviation_all_vitals_filter_mean")
        fig_centroid_deviation(centroids, "high invalid data",
                               "centroid_deviation_high_invalid_data_filter_mean")

    if "filter_impact" in selected:
        print("\n=== Filter impact ===")
        fig_filter_impact("mortality")
        fig_filter_impact("icu")

    if "mcnemar" in selected:
        print("\n=== McNemar ===")
        fig_mcnemar("icu", args.mcnemar_threshold, args.mcnemar_style, args.mcnemar_cap)
        fig_mcnemar("mortality", args.mcnemar_threshold, args.mcnemar_style, args.mcnemar_cap)

    if "ecdf" in selected:
        print("\n=== ECDF grid (fill missing data) ===")
        fig_ecdf_grid(layout="4x2")
        fig_ecdf_grid(layout="2x4")

    print("\nDone.")


if __name__ == "__main__":
    main()
