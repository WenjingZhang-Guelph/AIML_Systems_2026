"""Standalone version of poster panel 10: what the published record-filter deltas are made of.

    python Experiments/render_population_standalone.py [--out DIR]

Runs under scipy-stack. Writes `population_decomposition.pdf` into
`paper_figures/poster/standalone/` by default.

The poster panel set each published delta beside the change-of-patients term alone,
which left the mortality/high-invalid-data row looking contradictory (a -0.184 shift
against a +0.001 published delta) because the other two terms were not drawn. This
figure draws all three, stacked so they sum to the published delta:

    published delta  =  change of patients   (unchanged baseline, raw's threshold,
                                              survivors minus the whole cohort)
                     +  filter itself        (filtered vs baseline model on the same
                                              survivors, both at raw's threshold)
                     +  re-tuned threshold   (the remainder)

The first two are the same quantities the poster already quotes (`population_shift`,
`fixed_threshold_delta`). The remainder is mostly the filtered arm choosing its own
Youden threshold on a population with a different base rate, but it also absorbs the
gap between the 4-repeat mean accuracy the paper plots and the consensus predictions
the other two terms use (0.004 for ICU and 0.008 for mortality on the raw arm).
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from filtering_results import RECORD_ARMS, ROOT, delta, load
from render_poster_figures import (
    INK, RUST, TASK_COLOR, TASK_NAME, TEAL, fixed_threshold_delta,
    population_shift, setup_style,
)

DEFAULT_OUT = ROOT / "paper_figures" / "poster" / "standalone"
ROW_NAME = {"long missing segment": "Long missing\nsegment",
            "long gap": "Long gap", "high invalid data": "High invalid\ndata"}
# Lighter than SLATE so it cannot be mistaken for TEAL beside it.
THRESHOLD_GREY = "#9DB0B8"
PARTS = [("patients", RUST, "Change of patients (who is left to score)"),
         ("filter", TEAL, "The filter itself (same patients, same threshold)"),
         ("threshold", THRESHOLD_GREY, "Re-tuned decision threshold")]


def decompose(d, label, arm):
    published = delta(d, label, arm, "accuracy_mean")
    patients = population_shift(d, label, arm)
    filtered = fixed_threshold_delta(d, label, arm)
    return {"published": published, "patients": patients, "filter": filtered,
            "threshold": published - patients - filtered}


def draw(d, out):
    setup_style()
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10,
                         "ytick.labelsize": 11, "legend.fontsize": 10,
                         "axes.titlesize": 12, "hatch.linewidth": 1.0})
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9), sharey=True)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.86, bottom=0.34, wspace=0.08)

    for ax, label in zip(axes, ("icu", "mortality")):
        for row, arm in enumerate(RECORD_ARMS):
            parts = decompose(d, label, arm)
            left, right = 0.0, 0.0
            for key, color, _ in PARTS:
                value = parts[key]
                start = left + value if value < 0 else right
                ax.barh(row, abs(value), left=start, height=0.5, color=color,
                        edgecolor="white", linewidth=0.8)
                if value < 0:
                    left += value
                else:
                    right += value
            ax.plot(parts["published"], row, marker="D", markersize=8, color=INK,
                    markeredgecolor="white", markeredgewidth=1.2, zorder=5)
            ax.annotate(f"{parts['published']:+.3f}".replace("-", "−"),
                        (parts["published"], row), xytext=(0, 11),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=10, color=INK, fontweight="semibold", zorder=6,
                        bbox=dict(boxstyle="square,pad=0.15", facecolor="white",
                                  edgecolor="none"))
        ax.axvline(0, color=INK, lw=1.0)
        ax.set_xlim(-0.26, 0.26)
        ticks = [-0.2, -0.1, 0, 0.1, 0.2]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:+.1f}".replace("-", "−") if t else "0" for t in ticks])
        ax.set_xlabel("Change in accuracy vs. unfiltered baseline")
        ax.set_ylim(len(RECORD_ARMS) - 0.4, -0.75)
        ax.set_yticks(range(len(RECORD_ARMS)))
        ax.set_yticklabels([ROW_NAME[a] for a in RECORD_ARMS])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
        ax.set_title(TASK_NAME[label], color=TASK_COLOR[label], fontweight="semibold",
                     loc="left")

    handles = [Patch(facecolor=c, label=text) for _, c, text in PARTS]
    handles.append(Line2D([], [], marker="D", color=INK, markeredgecolor="white",
                          linestyle="none", markersize=8,
                          label="Published Δ (the sum; paper Figs. 5, 8)"))
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.56, 0.0), handlelength=1.4, columnspacing=1.6)

    out.mkdir(parents=True, exist_ok=True)
    path = out / "population_decomposition.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    d = load()
    for label in ("icu", "mortality"):
        for arm in RECORD_ARMS:
            p = decompose(d, label, arm)
            print(f"{label:9s} {arm:22s} " + "  ".join(
                f"{k}={p[k]:+.3f}" for k in ("patients", "filter", "threshold", "published")))
    print(f"saved {draw(d, args.out)}")


if __name__ == "__main__":
    main()
