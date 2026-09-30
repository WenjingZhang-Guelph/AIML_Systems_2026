"""Render the new figures and the number macros for the A0 "Beyond the Model" poster.

    python Experiments/render_poster_figures.py [--out DIR]

Runs under scipy-stack (matplotlib). Reads every number through
`filtering_results.load()`, the same loader and cross-checks the talk uses, and
writes two things into `paper_figures/poster/`:

  figures/*.pdf   the poster's own diagrams and charts, each drawn at its final
                  printed size, so a 20pt label here prints at 20pt on the A0 sheet
  numbers.tex     one \\newcommand per number the poster text quotes, so the .tex
                  types no data (the published Table III aside)

The published paper figure the poster reuses (centroid deviation) is
rendered by `render_paper_figures.py --font-scale`, not here, so it stays the
published figure rather than a lookalike. See rerun/job_m_poster.sh.

Two conventions, fixed on purpose and shared with the talk:
  - MIMIC-III, mean aggregation, `_cv` design only.
  - McNemar is the baseline-threshold comparison (every arm at raw's threshold).

The "not like-for-like" figure deliberately rests only on quantities that do not
depend on the threshold convention: the paper's plotted deltas, and the unchanged
baseline model re-scored on each arm's survivors (raw at raw's own threshold, which
*is* the baseline threshold). The own-threshold paired accuracies in the CV
diagnostics flip sign under the baseline convention, so they are not used.
"""
import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from filtering_results import (  # noqa: E402
    COHORT_STEPS, LEVELS, RECORD_ARMS, ROOT, SHORT, TABLE_III, THRESHOLDS, VITAL_ARMS,
    arm_stat, delta,
    fmt_p, load, p_value,
)

DEFAULT_OUT = ROOT / "paper_figures" / "poster"
STANDALONE_OUT = DEFAULT_OUT / "standalone"

# ── house style: the deck's palette (build_balance_slides.py), as hex ─────────
INK, BODY, MUTED = "#0E1618", "#2C3C40", "#61767A"
LINE, GROUND, SUNK, SURFACE = "#CDD8D9", "#EEF2F2", "#E4EAEA", "#FFFFFF"
TEAL, RUST, SLATE, GREEN = "#0D6B6B", "#A4552B", "#4A6572", "#3D7A52"
LEVEL_COLOR = {"value": TEAL, "feature": SLATE, "record": RUST}
# Tints of the level colours, for "removed" / "not significant" fills.
TINT = {TEAL: "#B9D6D6", RUST: "#EBCDBD", SLATE: "#C6D1D6"}
# The two tasks need colours of their own: the level colours are already spoken for.
TASK_COLOR = {"icu": "#2F4B7C", "mortality": "#9C6B98"}
TINT.update({"#2F4B7C": "#B6C0D1", "#9C6B98": "#DCCBDB"})
TASK_NAME = {"icu": "Prolonged ICU stay", "mortality": "In-hospital mortality"}

# ── fonts: the poster's own faces, straight out of the TeX tree ───────────────
TEXMF = Path("/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/share/"
             "texmf-dist/fonts/opentype")
FONT_FILES = ["adobe/sourcesanspro/SourceSansPro-Regular.otf",
              "adobe/sourcesanspro/SourceSansPro-RegularIt.otf",
              "adobe/sourcesanspro/SourceSansPro-Semibold.otf",
              "adobe/sourcesanspro/SourceSansPro-Bold.otf",
              "adobe/sourceserifpro/SourceSerifPro-Semibold.otf",
              "public/fira/FiraMono-Regular.otf",
              "public/fira/FiraMono-Medium.otf"]
SANS, SERIF, MONO = "Source Sans Pro", "Source Serif Pro", "Fira Mono"

# ── geometry: must match Experiments/poster/beyond_the_model_poster.tex ───────
# A column's inner text width is 237 mm; the bottom band's wide box is 509 mm.
MM = 1 / 25.4
COL_W = 237 * MM
SIZE = {                                   # (width, height) in inches, as printed
    "pipeline": (COL_W, 118 * MM),
    "levels": (COL_W, 78 * MM),
    "survivors": (COL_W, 92 * MM),
    "task_divergence": (COL_W, 150 * MM),
    "mcnemar": (COL_W, 148 * MM),
    "readings": (COL_W, 70 * MM),
    "prevalence": (COL_W, 92 * MM),
    "population": (COL_W, 96 * MM),
    "mcnemar_sample": (COL_W, 72 * MM),
}

TYPE = 20          # smallest type on any figure, in printed points
MCNEMAR_CAP = 20.0


def setup_style():
    for rel in FONT_FILES:
        path = TEXMF / rel
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing — is CVMFS mounted?")
        font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({
        "font.family": SANS, "font.size": TYPE,
        "axes.labelsize": TYPE, "xtick.labelsize": TYPE, "ytick.labelsize": TYPE,
        "legend.fontsize": TYPE, "axes.titlesize": TYPE + 2,
        "text.color": BODY, "axes.labelcolor": BODY, "axes.edgecolor": MUTED,
        "xtick.color": BODY, "ytick.color": BODY,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 1.2, "xtick.major.width": 1.2, "ytick.major.width": 1.2,
        "grid.color": LINE, "grid.linewidth": 1.0,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "pdf.fonttype": 3,
        "hatch.linewidth": 1.5, "mathtext.fontset": "custom",
        "mathtext.rm": SANS, "mathtext.it": f"{SANS}:italic",
    })


def new_figure(name):
    return plt.figure(figsize=SIZE[name])


def save(fig, out, name):
    """Saved at exactly the declared size: no tight crop, so the .tex can place it
    at 100% and a point here is a point on paper."""
    path = out / f"poster_{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"    saved {path.name}")
    return path


def box(ax, x, y, w, h, text, *, face=SURFACE, edge=LINE, color=BODY, size=TYPE,
        weight="normal", family=SANS, lw=1.6, pad=0.012):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad},rounding_size=0.02",
                                facecolor=face, edgecolor=edge, linewidth=lw,
                                transform=ax.transAxes, mutation_aspect=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=color,
            fontsize=size, fontweight=weight, family=family, linespacing=1.15,
            transform=ax.transAxes)


def arrow(ax, start, end, *, color=MUTED, lw=2.4, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=26,
                                 color=color, linewidth=lw, transform=ax.transAxes,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0, shrinkB=0))


def blank_axes(fig):
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# A. Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def fig_pipeline(d, out):
    fig = new_figure("pipeline")
    ax = blank_axes(fig)
    n = d["survivors"]["raw"]

    w, h = 0.29, 0.20
    xs = [0.0 + 0.012, 0.355, 0.698]
    y1, y2, y3 = 0.73, 0.40, 0.03
    # Row 1 — data
    box(ax, xs[0], y1, w, h, f"MIMIC-III\n{n:,} ICU stays", weight="bold", color=INK)
    box(ax, xs[1], y1, w, h, "first 24 h\n7 vitals × 24 hours")
    box(ax, xs[2], y1, w, h, "VALUE filters\n(pre-aggregation)",
        face=TINT[TEAL], edge=TEAL, color=INK, weight="bold")
    arrow(ax, (xs[0] + w + 0.012, y1 + h / 2), (xs[1] - 0.012, y1 + h / 2))
    arrow(ax, (xs[1] + w + 0.012, y1 + h / 2), (xs[2] - 0.012, y1 + h / 2))
    # Row 2 — right to left
    box(ax, xs[2], y2, w, h, "hourly mean\n→ 168 cells")
    box(ax, xs[1], y2, w, h, "FEATURE / RECORD\nfilters (post-agg.)",
        face="#EAD9CF", edge=RUST, color=INK, weight="bold")
    box(ax, xs[0], y2, w, h, "random forest\n300 trees")
    arrow(ax, (xs[2] + w / 2, y1 - 0.012), (xs[2] + w / 2, y2 + h + 0.012))
    arrow(ax, (xs[2] - 0.012, y2 + h / 2), (xs[1] + w + 0.012, y2 + h / 2))
    arrow(ax, (xs[1] - 0.012, y2 + h / 2), (xs[0] + w + 0.012, y2 + h / 2))
    # Row 3 — evaluation
    box(ax, xs[0], y3, 0.976, h + 0.02,
        "5-fold × 4 repeats · folds fixed once on the raw cohort by admission id\n"
        "threshold by Youden's J on an inner split  →  compare every arm with raw",
        face=SUNK, edge=SUNK, size=TYPE - 1)
    arrow(ax, (xs[0] + w / 2, y2 - 0.012), (xs[0] + w / 2, y3 + h + 0.032))
    return save(fig, out, "pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# B. Three levels, on a schematic record
# ─────────────────────────────────────────────────────────────────────────────
def fig_levels(d, out):
    fig = new_figure("levels")
    rows, cols = 7, 12
    # A fixed, hand-placed pattern: this is a picture of an idea, not of data.
    missing = {(1, 3), (1, 4), (4, 8), (5, 1), (6, 6), (6, 7), (2, 10)}
    removed = {(0, 2), (3, 9), (5, 5)}

    panels = [("VALUE", TEAL, "drops impossible readings\npatient kept"),
              ("FEATURE", SLATE, "fills empty cells\npatient kept"),
              ("RECORD", RUST, "deletes the whole stay")]
    width = 0.305
    for index, (name, color, caption) in enumerate(panels):
        left = 0.012 + index * (width + 0.035)
        ax = fig.add_axes([left, 0.30, width, 0.52])
        ax.set_xlim(0, cols)
        ax.set_ylim(rows, 0)
        ax.set_axis_off()
        for r in range(rows):
            for c in range(cols):
                empty = (r, c) in missing
                face = SURFACE if empty else SUNK
                if name == "FEATURE" and empty:
                    face = TINT[SLATE]
                ax.add_patch(Rectangle((c + 0.06, r + 0.06), 0.88, 0.88, facecolor=face,
                                       edgecolor=LINE if empty else SURFACE,
                                       linewidth=1.0,
                                       hatch="//" if name == "FEATURE" and empty else None))
                if name == "VALUE" and (r, c) in removed:
                    ax.plot([c + 0.2, c + 0.8], [r + 0.2, r + 0.8], color=TEAL, lw=3)
                    ax.plot([c + 0.2, c + 0.8], [r + 0.8, r + 0.2], color=TEAL, lw=3)
        if name == "RECORD":
            ax.add_patch(Rectangle((0, 0), cols, rows, facecolor=SURFACE, alpha=0.55,
                                   edgecolor="none"))
            ax.plot([0.2, cols - 0.2], [0.2, rows - 0.2], color=RUST, lw=6)
            ax.plot([0.2, cols - 0.2], [rows - 0.2, 0.2], color=RUST, lw=6)
        fig.text(left, 0.88, name, color=color, family=MONO, fontweight="medium",
                 fontsize=TYPE + 2, ha="left", va="bottom")
        fig.text(left, 0.25, caption, color=BODY, fontsize=TYPE, ha="left", va="top",
                 linespacing=1.15)
    return save(fig, out, "levels")


# ─────────────────────────────────────────────────────────────────────────────
# D. What the value-level filters touch: readings removed per vital
# ─────────────────────────────────────────────────────────────────────────────
def fig_readings(d, out):
    fig = new_figure("readings")
    ax = fig.add_axes([0.30, 0.05, 0.60, 0.92])
    # Not d["value_level"]: the filter logs count in-cell NaN as removed readings.
    counts = [d["value_level_genuine"][v] for v in VITAL_ARMS]
    ys = range(len(VITAL_ARMS))
    ax.barh(list(ys), counts, height=0.66, color=TEAL)
    for y, n in zip(ys, counts):
        ax.text(n + max(counts) * 0.02, y, f"{n:,}", va="center", ha="left",
                fontsize=TYPE, family=MONO, color=TEAL)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([SHORT[v] for v in VITAL_ARMS])
    ax.invert_yaxis()
    ax.set_xlim(0, max(counts) * 1.22)
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    return save(fig, out, "readings")


# ─────────────────────────────────────────────────────────────────────────────
# C. Survivors — one waffle per distinct cohort size, 1 square = 1% of stays
# ─────────────────────────────────────────────────────────────────────────────
def fig_survivors(d, out):
    fig = new_figure("survivors")
    raw = d["survivors"]["raw"]
    heads = ["10 value &\nfeature arms", "long missing\nsegment", "long gap",
             "high invalid\ndata"]
    width, gap = 0.215, 0.047
    for index, (arm, head) in enumerate(zip(COHORT_STEPS, heads)):
        kept = d["survivors"][arm]
        squares = round(100 * kept / raw)
        color = TEAL if arm == "raw" else RUST
        left = 0.012 + index * (width + gap)
        ax = fig.add_axes([left, 0.25, width, width * SIZE["survivors"][0] /
                           SIZE["survivors"][1]])
        ax.set_xlim(0, 10)
        ax.set_ylim(10, 0)
        ax.set_axis_off()
        for k in range(100):
            r, c = divmod(k, 10)
            alive = k < squares
            ax.add_patch(Rectangle((c + 0.07, r + 0.07), 0.86, 0.86,
                                   facecolor=color if alive else TINT[RUST],
                                   edgecolor="none"))
        fig.text(left, 0.98, head, fontsize=TYPE, color=INK, ha="left", va="top",
                 fontweight="semibold", linespacing=1.05)
        fig.text(left, 0.225, f"{kept:,}", family=MONO, fontsize=TYPE + 6,
                 color=color, ha="left", va="top", fontweight="medium")
        fig.text(left, 0.01, f"{kept / raw:.0%} kept", fontsize=TYPE, color=MUTED,
                 ha="left", va="bottom")
    return save(fig, out, "survivors")


# ─────────────────────────────────────────────────────────────────────────────
# I. Accuracy change per arm, both tasks (the numbers behind paper Figs. 5a & 8)
# ─────────────────────────────────────────────────────────────────────────────
def arm_order():
    return VITAL_ARMS + ["all vitals", "fill missing data"] + RECORD_ARMS


def colour_arm_ticks(ax, arms):
    for label, arm in zip(ax.get_yticklabels(), arms):
        label.set_color(LEVEL_COLOR[LEVELS[arm]])
        if LEVELS[arm] == "record":
            label.set_fontweight("semibold")


def fig_task_divergence(d, out):
    fig = new_figure("task_divergence")
    ax = fig.add_axes([0.34, 0.12, 0.63, 0.78])
    arms = arm_order()
    ys = range(len(arms))
    bar = 0.38
    for offset, label in ((-bar / 2, "mortality"), (bar / 2, "icu")):
        values = [delta(d, label, arm, "accuracy_mean") for arm in arms]
        ax.barh([y + offset for y in ys], values, height=bar,
                color=TASK_COLOR[label], label=TASK_NAME[label])
    ax.axvline(0, color=INK, lw=1.4)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([SHORT[a] for a in arms])
    colour_arm_ticks(ax, arms)
    ax.invert_yaxis()
    ax.set_xlabel("Δ accuracy vs. unfiltered baseline")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="lower left", frameon=False, bbox_to_anchor=(-0.52, 1.0), ncol=2,
              handlelength=1.2, columnspacing=1.2, borderaxespad=0.3)
    return save(fig, out, "task_divergence")


# ─────────────────────────────────────────────────────────────────────────────
# H. McNemar, both tasks, baseline threshold
# ─────────────────────────────────────────────────────────────────────────────
def fig_mcnemar(d, out):
    fig = new_figure("mcnemar")
    arms = arm_order()
    lines = (0.05, 1e-4)
    for index, label in enumerate(("mortality", "icu")):
        ax = fig.add_axes([0.345 + index * 0.345, 0.2, 0.28, 0.71])
        for y, arm in enumerate(arms):
            p = p_value(d, label, arm)
            height = min(-math.log10(max(p, 1e-308)), MCNEMAR_CAP)
            # The task's colour, as in the accuracy panel; the tick labels carry the level.
            color = TASK_COLOR[label]
            significant = p < 0.05
            ax.barh(y, height, height=0.66,
                    color=color if significant else TINT[color],
                    edgecolor=color, linewidth=1.2)
            if height >= MCNEMAR_CAP:
                ax.text(MCNEMAR_CAP - 0.5, y, f"p = {fmt_p(p)}", ha="right", va="center",
                        color=SURFACE, fontsize=TYPE - 2)
            elif label == "icu" and LEVELS[arm] == "record":
                n = d["mcnemar"][label][arm]["n_paired"]
                ax.text(max(height, 4.0) + 0.8, y, f"n = {n:,}", ha="left", va="center",
                        color=MUTED, fontsize=TYPE - 1)
        for p_line in lines:
            ax.axvline(-math.log10(p_line), color=INK, lw=1.4, ls=(0, (4, 3)))
        ax.set_xlim(0, MCNEMAR_CAP)
        ax.set_ylim(len(arms) - 0.5, -0.5)
        ax.set_yticks(range(len(arms)))
        if index == 0:
            ax.set_yticklabels([SHORT[a] for a in arms])
            colour_arm_ticks(ax, arms)
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_xticklabels(["0", "5", "10", "15", "≥20"])
        ax.set_title(TASK_NAME[label], fontsize=TYPE, color=TASK_COLOR[label],
                     fontweight="semibold", loc="left", pad=12)
    fig.text(0.345, 0.01, "−log₁₀ p   ·   dashed lines: p = 0.05 and p = 0.0001",
             ha="left", va="bottom", fontsize=TYPE, color=BODY)
    return save(fig, out, "mcnemar")


# ─────────────────────────────────────────────────────────────────────────────
# E. Prevalence among the survivors
# ─────────────────────────────────────────────────────────────────────────────
def fig_prevalence(d, out):
    fig = new_figure("prevalence")
    ax = fig.add_axes([0.10, 0.30, 0.86, 0.56])
    xs = list(range(len(COHORT_STEPS)))
    ax.axhline(0.5, color=MUTED, lw=1.4, ls=(0, (4, 3)))
    ax.text(len(xs) - 0.62, 0.5, "50%", color=MUTED, va="bottom", ha="right",
            fontsize=TYPE)
    for label, marker in (("icu", "o"), ("mortality", "s")):
        ys = [arm_stat(d, label, arm, "population_prevalence") for arm in COHORT_STEPS]
        ax.plot(xs, ys, color=TASK_COLOR[label], lw=3.2, marker=marker, ms=13,
                label=TASK_NAME[label])
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.045, f"{y:.0%}", ha="center", va="bottom",
                    color=TASK_COLOR[label], fontsize=TYPE, fontweight="semibold",
                    family=MONO)
    ax.set_ylim(0, 0.8)
    ax.set_xlim(-0.35, len(xs) - 0.65)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0", "20%", "40%", "60%", "80%"])
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{'raw' if a == 'raw' else SHORT[a].replace(' ', chr(10), 1)}"
                        f"\n{d['survivors'][a]:,}" for a in COHORT_STEPS],
                       linespacing=1.05)
    for tick, arm in zip(ax.get_xticklabels(), COHORT_STEPS):
        tick.set_color(TEAL if arm == "raw" else RUST)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.1, 1.04), frameon=False,
              handlelength=1.6, ncol=2, borderaxespad=0, columnspacing=1.4)
    return save(fig, out, "prevalence")


# ─────────────────────────────────────────────────────────────────────────────
# F. How much of each plotted delta is a change of patients
# ─────────────────────────────────────────────────────────────────────────────
def population_shift(d, label, arm):
    """Accuracy of the *unchanged* baseline model on this arm's survivors, minus its
    accuracy on the whole cohort. Both are raw's consensus predictions at raw's own
    threshold, so this is the same number under either McNemar convention."""
    m = d["diagnostics"][label][arm]["mcnemar"]
    whole = d["diagnostics"][label]["raw"]["mcnemar"]["baseline_accuracy_on_paired"]
    return m["baseline_accuracy_on_paired"] - whole


def fixed_threshold_delta(d, label, arm):
    """The filter's own effect on its survivors, both models at raw's threshold."""
    m = d["mcnemar"][label][arm]
    return (m["n01"] - m["n10"]) / m["n_paired"]


def fig_population(d, out):
    fig = new_figure("population")
    arms = RECORD_ARMS
    for index, label in enumerate(("icu", "mortality")):
        ax = fig.add_axes([0.285 + index * 0.365, 0.36, 0.325, 0.54])
        bar = 0.36
        plotted = [delta(d, label, a, "accuracy_mean") for a in arms]
        shift = [population_shift(d, label, a) for a in arms]
        ys = list(range(len(arms)))
        ax.barh([y - bar / 2 for y in ys], plotted, height=bar, color=INK,
                label="Δ as published (paper Figs. 5, 8)")
        ax.barh([y + bar / 2 for y in ys], shift, height=bar, color=TINT[RUST],
                edgecolor=RUST, linewidth=1.2, hatch="//",
                label="Δ from the change of patients alone")
        ax.axvline(0, color=INK, lw=1.4)
        ax.set_xlim(-0.2, 0.02)
        ax.set_xticks([-0.2, -0.1, 0])
        ax.set_xticklabels(["−0.2", "−0.1", "0"])
        ax.set_ylim(len(arms) - 0.5, -0.5)
        ax.set_yticks(ys)
        if index == 0:
            ax.set_yticklabels([SHORT[a].replace(" ", "\n", 1) for a in arms],
                               color=RUST, fontweight="semibold", linespacing=1.0)
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
        ax.set_title(TASK_NAME[label], fontsize=TYPE, color=TASK_COLOR[label],
                     fontweight="semibold", loc="left", pad=10)
        if index == 0:
            handles, texts = ax.get_legend_handles_labels()
    fig.legend(handles, texts, loc="lower left", bbox_to_anchor=(0.0, -0.01),
               frameon=False, ncol=1, handlelength=1.4, labelspacing=0.3)
    return save(fig, out, "population")


# ─────────────────────────────────────────────────────────────────────────────
# G. What the McNemar test for high-invalid-data (ICU) is actually decided by
# ─────────────────────────────────────────────────────────────────────────────
def fig_mcnemar_sample(d, out, label="icu", arm="high invalid data"):
    """The whole cohort as one bar, then the survivors blown up underneath: the
    test's verdict rests on the discordant slice of the second bar alone."""
    fig = new_figure("mcnemar_sample")
    raw = d["survivors"]["raw"]
    m = d["mcnemar"][label][arm]
    discordant = m["n01"] + m["n10"]
    agree = m["n_paired"] - discordant
    deleted = raw - m["n_paired"]

    top = fig.add_axes([0.012, 0.66, 0.976, 0.13])
    top.barh(0, deleted, height=1.0, color=TINT[RUST], edgecolor=RUST,
             linewidth=1.2, hatch="//")
    top.barh(0, m["n_paired"], left=deleted, height=1.0, color=INK, edgecolor=INK)
    top.set_xlim(0, raw)
    top.set_ylim(-0.5, 0.5)
    top.set_axis_off()
    fig.text(0.012, 0.815, f"all {raw:,} stays:  {deleted:,} deleted by the filter — "
             "invisible to the test", color=RUST, fontsize=TYPE, fontweight="semibold",
             ha="left", va="bottom")

    zoom = fig.add_axes([0.012, 0.22, 0.976, 0.13])
    zoom.barh(0, agree, height=1.0, color=SUNK, edgecolor=MUTED, linewidth=1.2)
    zoom.barh(0, discordant, left=agree, height=1.0, color=INK, edgecolor=INK)
    zoom.set_xlim(0, m["n_paired"])
    zoom.set_ylim(-0.5, 0.5)
    zoom.set_axis_off()
    zoom.text(agree / 2, 0, f"{agree:,} agree — ignored", ha="center", va="center",
              color=BODY, fontsize=TYPE)

    # Zoom lines from the kept slice down to the full width of the second bar.
    kept_left = 0.012 + 0.976 * deleted / raw
    for x_top, x_bottom in ((kept_left, 0.012), (0.988, 0.988)):
        fig.add_artist(plt.Line2D([x_top, x_bottom], [0.66, 0.35], color=MUTED, lw=1.4,
                                  ls=(0, (3, 3))))
    fig.text(0.988, 0.19, f"the {m['n_paired']:,} kept: {discordant:,} disagree — "
             f"{m['n01']} fixed, {m['n10']} broken"
             f"  →  p = {fmt_p(m['p'])}", color=INK, fontsize=TYPE,
             fontweight="semibold", ha="right", va="top")
    return save(fig, out, "mcnemar_sample")


# ─────────────────────────────────────────────────────────────────────────────
# numbers.tex
# ─────────────────────────────────────────────────────────────────────────────
def pct(x, digits=0):
    return f"{100 * x:.{digits}f}\\%"


def tex_p(p):
    """fmt_p, typeset in the body face: "1.2e-84" becomes 1.2 × 10⁻⁸⁴. Text, not
    math, because math mode would set it in Computer Modern."""
    text = fmt_p(p)
    if "e-" not in text:
        return text
    mantissa, exponent = text.split("e-")
    return f"{mantissa}\\,×\\,10\\textsuperscript{{−{exponent}}}"


def signed(x, digits=3):
    return f"{x:+.{digits}f}".replace("-", "−")


def write_numbers(d, out):
    s = d["survivors"]
    raw = s["raw"]
    hid, lms, lg = "high invalid data", "long missing segment", "long gap"
    readings, edited = d["readings"], d["value_level"]["all vitals"]
    vital_ps = [p_value(d, label, a) for a in VITAL_ARMS for label in ("mortality", "icu")]
    base = d["mcnemar"]["icu"][hid]

    numbers = {
        "NRaw": f"{raw:,}",
        "NLMS": f"{s[lms]:,}", "NLG": f"{s[lg]:,}", "NHID": f"{s[hid]:,}",
        "RemovedLMS": f"{raw - s[lms]:,}", "RemovedLG": f"{raw - s[lg]:,}",
        "RemovedHID": f"{raw - s[hid]:,}",
        "PctRemovedHID": pct(1 - s[hid] / raw),
        "PctKeptHID": pct(s[hid] / raw),
        "Readings": f"{readings / 1e6:.2f}~million",
        "ReadingsEdited": f"{edited:,}",
        "PctReadingsEdited": pct(edited / readings, 2),
        "PrevICURaw": pct(arm_stat(d, "icu", "raw", "population_prevalence")),
        "PrevICUHID": pct(arm_stat(d, "icu", hid, "population_prevalence")),
        "PrevMortRaw": pct(arm_stat(d, "mortality", "raw", "population_prevalence")),
        "PrevMortHID": pct(arm_stat(d, "mortality", hid, "population_prevalence")),
        "AccICURaw": f"{arm_stat(d, 'icu', 'raw', 'accuracy_mean'):.3f}",
        "AccMortRaw": f"{arm_stat(d, 'mortality', 'raw', 'accuracy_mean'):.3f}",
        "DeltaAccHIDicu": signed(delta(d, "icu", hid, "accuracy_mean")),
        "DeltaAccHIDmort": signed(delta(d, "mortality", hid, "accuracy_mean")),
        "BaseAccWholeICU":
            f"{d['diagnostics']['icu']['raw']['mcnemar']['baseline_accuracy_on_paired']:.3f}",
        "BaseAccHIDicu":
            f"{d['diagnostics']['icu'][hid]['mcnemar']['baseline_accuracy_on_paired']:.3f}",
        "ShiftHIDicu": signed(population_shift(d, "icu", hid)),
        "FixedDeltaHIDicu": signed(fixed_threshold_delta(d, "icu", hid)),
        "PHIDicu": tex_p(p_value(d, "icu", hid)),
        "PHIDmort": tex_p(p_value(d, "mortality", hid)),
        "PAllVitalsMort": tex_p(p_value(d, "mortality", "all vitals")),
        "PAllVitalsICU": tex_p(p_value(d, "icu", "all vitals")),
        "PVitalMin": f"{min(vital_ps):.2f}",
        "PFillMort": tex_p(p_value(d, "mortality", "fill missing data")),
        "PairedHIDicu": f"{base['n_paired']:,}",
        "DiscordantHIDicu": f"{base['n01'] + base['n10']:,}",
    }

    # Definitions (paper Table II) and the heart-rate row of Table III, so the .tex
    # quotes them rather than retyping them.
    hr_raw, hr_after = (int(TABLE_III[i][1].replace(",", "")) for i in (0, 1))
    numbers.update({
        "ThrHR": THRESHOLDS["heart rate"], "ThrTemp": THRESHOLDS["temperature"],
        "ThrFill": THRESHOLDS["fill missing data"],
        "ThrLMS": THRESHOLDS["long missing segment"],
        "ThrLG": THRESHOLDS["long gap"], "ThrHID": THRESHOLDS["high invalid data"],
        "HIDTolerated": f"{math.floor(0.10 * 168)}",
        "HRRemoved": f"{hr_raw - hr_after:,}",
        "HRPctRemoved": pct((hr_raw - hr_after) / hr_raw, 2),
        "PrevShiftICUHID": f"{100 * (arm_stat(d, 'icu', hid, 'population_prevalence') - arm_stat(d, 'icu', 'raw', 'population_prevalence')):.0f}",
    })
    for key in ("ThrHR", "ThrTemp", "ThrFill", "ThrLMS", "ThrLG", "ThrHID"):
        numbers[key] = numbers[key].replace("%", "\\%")

    lines = ["% Written by Experiments/render_poster_figures.py — do not edit by hand.",
             "% MIMIC-III, mean aggregation, _cv design; McNemar at the baseline threshold."]
    for key, value in numbers.items():
        lines.append(f"\\newcommand\\{key}{{{value}}}")
    path = out / "numbers.tex"
    path.write_text("\n".join(lines) + "\n")
    print(f"    wrote {path.name} ({len(numbers)} macros)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None,
                        help=f"default {DEFAULT_OUT}, or {STANDALONE_OUT} with --standalone")
    parser.add_argument("--only", nargs="+", choices=list(SIZE), metavar="NAME",
                        help=f"render just these figures: {', '.join(SIZE)}")
    parser.add_argument("--standalone", action="store_true",
                        help="write the PDFs straight into --out and skip numbers.tex")
    args = parser.parse_args()
    out = args.out or (STANDALONE_OUT if args.standalone else DEFAULT_OUT)
    figures = out if args.standalone else out / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    setup_style()
    d = load()
    print("=== poster figures ===")
    for render in (fig_pipeline, fig_levels, fig_readings, fig_survivors,
                   fig_task_divergence,
                   fig_mcnemar, fig_prevalence, fig_population, fig_mcnemar_sample):
        if args.only is None or render.__name__.removeprefix("fig_") in args.only:
            render(d, figures)
    if not args.standalone:
        write_numbers(d, out)


if __name__ == "__main__":
    main()
