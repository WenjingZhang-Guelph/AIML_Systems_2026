#!/usr/bin/env python3
"""
Re-draws the EHR-Dataset-Processing Marimo notebook's plots without Marimo,
without a GPU, and without a database. mimic-iii only.

Why this exists: the gallery needs PNGs of everything in Experiments/notebook.py,
but that notebook wants a CUDA box with cuDF/cuML and a live Postgres, and the
gallery build runs on whatever CPU node is free. Rather than keep a GPU node
warm just to make pictures, this reads the pickled caches under Data/mimic-iii/
and calls the same visualization_manager_v2 functions the notebook calls, with
the same arguments. Anything that changes in those functions shows up here for
free — which is the point, the plots stay honest — but if the notebook's *call
sites* change, this file has to be updated by hand to match. The constants
below (VITALS, FILTER_NAMES, AGGREGATION_METHODS) are copied from notebook.py
for the same reason and drift the same way.

Everything is wrapped in attempt()/skip logic because a partial gallery beats no
gallery: caches for some (label, aggregation, filter) combinations were never
generated, and I'd rather render the 400-odd figures that do work than have the
whole run die on the first missing pickle.

    MPLBACKEND=Agg python render_marimo_mimic_iii.py --section all

Figures land in ../figures/EHR-Dataset-Processing/mimic-iii/, and the manifest
fragment for build_manifest.py goes to ../figures/marimo_manifest_fragment.json.
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from types import ModuleType

# ── Paths ─────────────────────────────────────────────────────────────────────
GALLERY = Path(__file__).parent.resolve()
TOOLBOX_ROOT = GALLERY.parent
EHR_ROOT = TOOLBOX_ROOT / "pipelines" / "EHR-Dataset-Processing"
DATA_ROOT = EHR_ROOT / "Data" / "mimic-iii"
OUT_DIR = TOOLBOX_ROOT / "figures" / "EHR-Dataset-Processing" / "mimic-iii"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "thumbs").mkdir(exist_ok=True)

# Agg has to be locked in before anything imports pyplot, otherwise matplotlib
# picks a GUI backend and dies on a headless compute node. Belt and braces here
# (env var *and* use()) because import order through visualization_manager_v2
# isn't something I want to have to keep verifying.
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# The pickles under Data/ were written on a GPU box, so unpickling a RecordEHR
# drags in torch and the whole RAPIDS stack via module references — none of which
# installs on a CPU-only node. Registering do-nothing stubs in sys.modules first
# lets pickle resolve those names and move on. It works because we only ever read
# the plain DataFrame/array attributes; call anything that actually needs a GPU
# and you'll get an _Anything back and a very confusing plot rather than an error.
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

    # Swapping __class__ is the trick that makes `torch.whatever` resolve for
    # *any* attribute — a plain ModuleType raises AttributeError, and pickle
    # needs the lookups to succeed no matter what name it asks for.
    class _StubMod(ModuleType):
        def __getattr__(self, n):
            return _Anything()
    m.__class__ = _StubMod
    return m

for _stub in ("torch", "torch.nn", "torch.optim", "torch.utils", "torch.utils.data",
              "cupy", "cudf", "cuml", "cuml.ensemble", "cuml.model_selection",
              "cuml.metrics", "cugraph"):
    if _stub not in sys.modules:
        _make_stub(_stub)

# Import the real plotting code straight out of the sibling pipeline rather than
# vendoring a copy — a vendored copy would go stale and the gallery would start
# showing plots that no longer match what the pipeline produces.
if str(EHR_ROOT) not in sys.path:
    sys.path.insert(0, str(EHR_ROOT))

from Managers.visualization_manager_v2 import (  # noqa: E402
    surface_plot, heatmap,
    filter_impact_plot, mcnemar_plot,
    centroid_shift_plot, centroid_plot,
)

# Copied verbatim from notebook.py — see the module docstring. Note "respiration
# rate" here, not "respiratory rate": that's the EHR pipeline's spelling and the
# key the cached pickles are written under, so don't "fix" it.
DATASET_NAME = "mimic-iii"
PLOT_THEME = "#191a1c"

# Hard physiological ranges for the seven vitals, keyed by the repo-wide lowercase
# names. `_remove_outliers` tests these as a CLOSED interval (`lo <= v <= hi`), so
# the upper bound is a value the filter keeps, not the first one it rejects. Until
# 2026-08-30 every table here sat one unit low (SpO2 99, so a perfectly normal — and
# modal — reading of 100% was deleted as a charting error); see bug register F-02.
#
# Six copies of this table exist and they must stay literally identical. Drift
# between them is what produced register entry R-30:
#   Experiments/apply_dataset_filter.py     Experiments/notebook.py
#   rerun/_common.py                        gallery/render_marimo_mimic_iii.py
#   Experiments/fill_missing_data_analysis.py
#   Experiments/render_paper_figures.py     (names/units only; ranges inert there)
VITALS = {
    "heart rate":             [(1, 600), "bpm"],
    "systolic blood pressure": [(1, 400), "mmHg"],
    "diastolic blood pressure": [(1, 300), "mmHg"],
    "mean blood pressure":    [(1, 300), "mmHg"],
    "respiration rate":       [(1, 70),  "breaths/min"],
    "temperature":            [(21, 50), "C"],
    "oxygen saturation":      [(1, 100),  "%"],
}
VITAL_NAMES = list(VITALS.keys())
VITAL_UNITS = [v[-1] for v in VITALS.values()]

AGGREGATION_METHODS = ["mean", "median", "standard deviation", "mean deviation", "maximum deviation"]
LABELS = ["icu", "mortality"]

# The 11 filters from ehr_filter_manager's OrderedDict, plus a synthetic "all
# vitals" entry the notebook appends. Order has to match the pipeline's — these
# names are zipped positionally against the results arrays, not looked up by key,
# so a reordering silently mislabels every bar in every filter-impact chart.
FILTER_NAMES = [
    "heart rate", "systolic blood pressure", "diastolic blood pressure",
    "mean blood pressure", "respiration rate", "temperature", "oxygen saturation",
    "fill missing data", "long missing segment", "long gap", "high invalid data",
    "all vitals",
]
LENGTH_METHODS = ["mean", "median", "std", "max", "min"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_pkl(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def save_fig(fig, stem: str, manifest: list, meta: dict):
    """Write a figure out in both formats and record it for the manifest.

    Passing fig.get_facecolor() through to savefig is not optional — the Aurora
    theme sets a dark figure background, and savefig otherwise helpfully renders
    it white, giving you dark text on a white card in the gallery.

    Closing the figure matters more than it looks: this renders several hundred
    figures in one process and matplotlib holds every un-closed one in memory.
    Note the early returns close it too.
    """
    if fig is None:
        print(f"    [skip] {stem}: figure is None")
        return

    png_path = OUT_DIR / f"{stem}.png"
    svg_path = OUT_DIR / f"{stem}.svg"
    thumb_path = OUT_DIR / "thumbs" / f"{stem}_thumb.png"

    try:
        fig.savefig(str(png_path), dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
        fig.savefig(str(svg_path), bbox_inches="tight", facecolor=fig.get_facecolor())
    except Exception as e:
        print(f"    [ERROR] {stem}: {e}")
        plt.close(fig)
        return

    # Rescaling through matplotlib instead of Pillow so this has no dependency
    # beyond what's already imported — see the Pillow note in
    # extract_notebook_graphs.py for the other half of this story.
    try:
        import matplotlib.image as mpimg
        import numpy as np
        img = mpimg.imread(str(png_path))
        h, w = img.shape[:2]
        scale = min(300 / w, 200 / h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        thumb_fig, thumb_ax = plt.subplots(figsize=(new_w / 100, new_h / 100))
        thumb_ax.imshow(img)
        thumb_ax.axis("off")
        thumb_fig.savefig(str(thumb_path), dpi=100, bbox_inches="tight")
        plt.close(thumb_fig)
    except Exception:
        pass  # no thumbnail just means a slower grid, not a broken gallery

    plt.close(fig)

    manifest.append({
        "id": f"marimo_{stem}",
        "pipeline": "EHR-Dataset-Processing",
        "dataset": "mimic-iii",
        "notebook": "notebook_py_marimo",
        "source": "rendered",
        "auto_named": False,
        "superseded": False,
        "file": f"graphs/EHR-Dataset-Processing/mimic-iii/{stem}.png",
        "svg": f"graphs/EHR-Dataset-Processing/mimic-iii/{stem}.svg",
        "thumb": f"graphs/EHR-Dataset-Processing/mimic-iii/thumbs/{stem}_thumb.png",
        **meta,
    })
    print(f"    saved {stem}.png")


def attempt(fn, *args, **kwargs):
    """Call a plot function, and shrug if it blows up.

    Blanket except is deliberate. Roughly a third of the (aggregation, filter)
    combinations have incomplete caches, and the plotting functions fail on those
    in a dozen different ways — KeyError, shape mismatch, empty-array warnings
    escalated to errors. None of that should stop the other few hundred figures
    from rendering, and the failures print so you can spot a real regression.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"      [attempt failed] {fn.__name__}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Section A — how many measurements actually exist, per vital per hour.
#
# This is the data-availability picture that motivates the whole filtering
# story: a nurse charts heart rate continuously and temperature every few hours,
# so the raw observation counts are wildly uneven before you do anything else.
# ─────────────────────────────────────────────────────────────────────────────

def render_length_plots(manifest: list):
    print("\n=== Section A: Length plots ===")
    print("Loading processed_record_ehr.pkl ...", flush=True)
    import numpy as np
    import pandas as pd

    pkl_path = DATA_ROOT / "processed_record_ehr.pkl"
    if not pkl_path.exists():
        print(f"  [SKIP] {pkl_path} not found")
        return

    records = load_pkl(pkl_path)
    print(f"  Loaded {len(records)} RecordEHR objects")

    # Each cell of record.timeseries holds a *list* of that hour's measurements,
    # not a single number — so counting them gives observation density. The
    # hasattr/isinstance dance is because older pickles stored numpy arrays and
    # newer ones store plain lists; empty and missing cells both count as 0.
    print("  Building length matrices ...", flush=True)
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
        print("  [SKIP] no length DataFrames built")
        return

    # concat then groupby(level=0) stacks every record's 24-hour frame and
    # collapses back down by hour, so we end up with one 24×7 matrix per
    # aggregation method across the whole cohort.
    print(f"  Aggregating across {len(length_dfs)} records ...")
    try:
        combined = pd.concat(length_dfs)
        aggregated = {m: combined.groupby(level=0).agg(m) for m in LENGTH_METHODS}
    except Exception as e:
        print(f"  [ERROR] aggregation failed: {e}")
        return

    for method in LENGTH_METHODS:
        df = aggregated[method]
        stem_s = f"length_{method}_surface"
        stem_h = f"length_{method}_heatmap"
        meta_base = {
            "category": "Timeseries Length",
            "plot_type": "surface",
            "label": None, "aggregation": None, "vital": None,
            "scenario": None, "filter": None,
            "description": f"3D surface plot of {method} observation-count length per vital per hour across all mimic-iii ICU stays.",
        }
        meta_hm = {**meta_base,
                   "plot_type": "heatmap",
                   "description": f"Heatmap of {method} observation-count length per vital per hour across all mimic-iii ICU stays.",
                   }
        meta_s = {**meta_base, "title": f"{method.capitalize()} Observation Length 3D Surface (mimic-iii)"}
        meta_h = {**meta_hm, "title": f"{method.capitalize()} Observation Length Heatmap (mimic-iii)"}

        fig_s = attempt(surface_plot, df,
                        title=f"{method.capitalize()} 3D Surface",
                        y_title="Hour",
                        z_title=f"{method.capitalize()} Length",
                        background_colour=PLOT_THEME)
        save_fig(fig_s, stem_s, manifest, meta_s)

        fig_h = attempt(heatmap, df,
                        title=f"{method.capitalize()} Heatmap",
                        y_title="Hour",
                        background_colour=PLOT_THEME)
        save_fig(fig_h, stem_h, manifest, meta_h)


# ─────────────────────────────────────────────────────────────────────────────
# Sections B and C — did each filter actually help, and is the difference real?
#
# Kept in one function because they read from the same cached results tuple and
# splitting them would mean loading each pickle twice.
# ─────────────────────────────────────────────────────────────────────────────

def render_impact_plots(manifest: list):
    print("\n=== Section B+C: Filter impact / McNemar plots ===")

    for label in LABELS:
        for agg in AGGREGATION_METHODS:
            pkl_path = DATA_ROOT / agg / f"{label}_filter_impact.pkl"
            results = load_pkl(pkl_path)
            if results is None:
                print(f"  [SKIP] {pkl_path}")
                continue

            print(f"  {label}/{agg} ...", flush=True)
            agg_slug = agg.replace(" ", "_")
            label_display = label.upper()

            # results is a positional tuple, and the indices are the only thing
            # telling you what's what: 0/2 are the training metrics, 1/3 the
            # testing ones. We only ever plot test accuracy and test F1 —
            # training numbers are in the pickle but aren't interesting here.
            stem = f"filter_impact_{label}_{agg_slug}_test_accuracy"
            fig = attempt(filter_impact_plot,
                          results[1], list(FILTER_NAMES),
                          label_display, "Testing Accuracy", agg,
                          background=PLOT_THEME)
            save_fig(fig, stem, manifest, {
                "title": f"{label_display} Filter Impact — Testing Accuracy Deviation ({agg.title()} Agg)",
                "description": f"Bar chart of test accuracy deviation from raw baseline for each filter, label={label}, aggregation={agg}.",
                "category": "Filter Impact",
                "plot_type": "bar",
                "label": label, "aggregation": agg,
                "vital": None, "scenario": None, "filter": None,
            })

            stem = f"filter_impact_{label}_{agg_slug}_test_f1"
            fig = attempt(filter_impact_plot,
                          results[3], list(FILTER_NAMES),
                          label_display, "Testing F1", agg,
                          background=PLOT_THEME)
            save_fig(fig, stem, manifest, {
                "title": f"{label_display} Filter Impact — Testing F1 Deviation ({agg.title()} Agg)",
                "description": f"Bar chart of test F1 deviation from raw baseline for each filter, label={label}, aggregation={agg}.",
                "category": "Filter Impact",
                "plot_type": "bar",
                "label": label, "aggregation": agg,
                "vital": None, "scenario": None, "filter": None,
            })

            try:
                # Dropping index 0 because that entry is raw-vs-raw — comparing
                # the baseline against itself always gives p=1 and would waste a
                # bar. Everything downstream assumes p_values lines up with
                # FILTER_NAMES after that shift.
                p_values = [x[1] for x in results[-1]][1:]
            except Exception as e:
                print(f"    [SKIP McNemar] {label}/{agg}: {e}")
                continue

            stem = f"mcnemar_{label}_{agg_slug}"
            fig = attempt(mcnemar_plot,
                          p_values, list(FILTER_NAMES),
                          label_display, agg,
                          background=PLOT_THEME)
            save_fig(fig, stem, manifest, {
                "title": f"{label_display} McNemar Significance ({agg.title()} Aggregation)",
                "description": f"−log₁₀(p) bar chart of McNemar test for each filter vs raw baseline, label={label}, aggregation={agg}.",
                "category": "McNemar Significance",
                "plot_type": "bar",
                "label": label, "aggregation": agg,
                "vital": None, "scenario": None, "filter": None,
            })


# ─────────────────────────────────────────────────────────────────────────────
# Section D — where each filter moves the class centroids.
#
# This is the "did filtering change the population, or just the noise?" view.
# A filter that improves accuracy while dragging the centroids a long way is
# suspicious; one that improves accuracy without moving them is doing real work.
# ─────────────────────────────────────────────────────────────────────────────

def load_all_centroids():
    """Slurp every centroid cache into [label][agg][filter][sign] dicts.

    Each pickle holds a (centroid, points) pair, which is why this returns two
    parallel structures rather than one. "raw" is prepended to the filter list
    because it's the unfiltered baseline every other filter is measured against
    — it isn't one of the 11 filters, but it is stored the same way.

    Missing files are simply absent from the dict rather than being stored as
    None, so callers detect gaps with a KeyError. Nothing is validated here; the
    get_*_pairs helpers below are where that shows up.
    """
    centroids = {}
    points = {}

    for label in LABELS:
        centroids[label] = {}
        points[label] = {}
        for agg in AGGREGATION_METHODS:
            centroids[label][agg] = {}
            points[label][agg] = {}
            cent_dir = DATA_ROOT / agg / "centroids"
            all_filters = ["raw"] + FILTER_NAMES

            for filter_name in all_filters:
                centroids[label][agg][filter_name] = {}
                points[label][agg][filter_name] = {}
                for sign in ["pos", "neg"]:
                    pkl_path = cent_dir / f"{label}_{filter_name}_{sign}.pkl"
                    data = load_pkl(pkl_path)
                    if data is not None:
                        c, p = data
                        centroids[label][agg][filter_name][sign] = c
                        points[label][agg][filter_name][sign] = p

    return centroids, points


def get_centroid_pairs(centroids, agg, filter_name):
    """Pair each filtered centroid with its raw counterpart, four groups' worth.

    The four groups (ICU positive/negative, mortality positive/negative) and this
    exact ordering are what centroid_shift_plot expects — it labels the bar groups
    positionally from the `categories` list at the call site. Returns None on any
    missing piece so the caller can skip the whole figure rather than draw a
    half-empty one.
    """
    try:
        return [
            (centroids["icu"][agg]["raw"]["pos"],  centroids["icu"][agg][filter_name]["pos"]),
            (centroids["icu"][agg]["raw"]["neg"],  centroids["icu"][agg][filter_name]["neg"]),
            (centroids["mortality"][agg]["raw"]["pos"], centroids["mortality"][agg][filter_name]["pos"]),
            (centroids["mortality"][agg]["raw"]["neg"], centroids["mortality"][agg][filter_name]["neg"]),
        ]
    except KeyError:
        return None


def get_point_pairs(points, agg, filter_name):
    """Same four groups, but the underlying point clouds instead of the centroids.

    Kept separate from get_centroid_pairs because the density plots need both and
    the bar chart needs only the centroids — a record can have centroids cached
    without points.
    """
    try:
        return [
            (points["icu"][agg]["raw"]["pos"],  points["icu"][agg][filter_name]["pos"]),
            (points["icu"][agg]["raw"]["neg"],  points["icu"][agg][filter_name]["neg"]),
            (points["mortality"][agg]["raw"]["pos"], points["mortality"][agg][filter_name]["pos"]),
            (points["mortality"][agg]["raw"]["neg"], points["mortality"][agg][filter_name]["neg"]),
        ]
    except KeyError:
        return None


def render_centroid_plots(manifest: list):
    print("\n=== Section D: Centroid plots ===")
    print("Loading centroid caches ...", flush=True)
    centroids, points = load_all_centroids()
    print("  Done loading centroids")

    categories = ["icu pos", "icu neg", "mortality pos", "mortality neg"]
    centroid_plot_categories = [
        "Raw ICU Pos", "Raw ICU Neg", "Raw Mort Pos", "Raw Mort Neg",
        "Filt ICU Pos", "Filt ICU Neg", "Filt Mort Pos", "Filt Mort Neg",
    ]

    for agg in AGGREGATION_METHODS:
        agg_slug = agg.replace(" ", "_")
        for filter_name in FILTER_NAMES:
            filter_slug = filter_name.replace(" ", "_")
            print(f"  {agg}/{filter_name} ...", flush=True)

            cp = get_centroid_pairs(centroids, agg, filter_name)
            pp = get_point_pairs(points, agg, filter_name)

            if cp is None:
                print(f"    [SKIP] missing centroid data")
                continue

            stem = f"centroid_shift_{agg_slug}_{filter_slug}"
            fig = attempt(centroid_shift_plot,
                          cp, categories, filter_name, VITAL_NAMES,
                          background=PLOT_THEME)
            save_fig(fig, stem, manifest, {
                "title": f"Centroid Deviations — {filter_name.title()} Filter ({agg.title()} Agg)",
                "description": f"Grouped bar chart of centroid deviations per vital for ICU+Mortality × pos+neg groups, filter={filter_name}, aggregation={agg}.",
                "category": "Centroid Shift",
                "plot_type": "bar",
                "label": None, "aggregation": agg,
                "vital": None, "scenario": None, "filter": filter_name,
            })

            if pp is None:
                continue

            # centroid_plot wants one flat list of 8 series, not 4 pairs — hence
            # the flatten, and hence centroid_plot_categories being ordered
            # all-raw-then-all-filtered to match what the flatten produces.
            all_c = [x for pair in cp for x in pair]
            all_p = [x for pair in pp for x in pair]

            # Two versions of every density plot on purpose. Max-range shows the
            # true extent including outliers, p95 clips them — with vitals the
            # outliers are often charting errors that squash the interesting part
            # of the distribution into a single pixel, but sometimes they're the
            # actual signal, so it's worth having both to eyeball.
            stem = f"centroid_density_maxrange_{agg_slug}_{filter_slug}"
            fig = attempt(centroid_plot,
                          all_c, all_p, VITAL_NAMES, VITAL_UNITS,
                          centroid_plot_categories,
                          use_percentile=False,
                          background=PLOT_THEME)
            save_fig(fig, stem, manifest, {
                "title": f"Centroid Distribution (Max Range) — {filter_name.title()} Filter ({agg.title()} Agg)",
                "description": f"8-panel density violin plot of raw vs filtered centroids for ICU+Mortality × pos+neg, max-distance limit, filter={filter_name}, agg={agg}.",
                "category": "Centroid Distribution",
                "plot_type": "violin",
                "label": None, "aggregation": agg,
                "vital": None, "scenario": None, "filter": filter_name,
            })

            stem = f"centroid_density_p95_{agg_slug}_{filter_slug}"
            fig = attempt(centroid_plot,
                          all_c, all_p, VITAL_NAMES, VITAL_UNITS,
                          centroid_plot_categories,
                          use_percentile=True,
                          background=PLOT_THEME)
            save_fig(fig, stem, manifest, {
                "title": f"Centroid Distribution (95th Pct) — {filter_name.title()} Filter ({agg.title()} Agg)",
                "description": f"8-panel density violin plot (95th-percentile distance limit) for filter={filter_name}, agg={agg}.",
                "category": "Centroid Distribution",
                "plot_type": "violin",
                "label": None, "aggregation": agg,
                "vital": None, "scenario": None, "filter": filter_name,
            })


# ─────────────────────────────────────────────────────────────────────────────
# Section E — the unfiltered baseline on its own.
#
# Section D always draws raw and filtered together, which is the right call for
# judging a filter but useless if what you want is a clean picture of the raw
# cohort for a paper. These are that: raw only, white background rather than the
# Aurora dark theme, so they drop into a printed figure without inverting.
# ─────────────────────────────────────────────────────────────────────────────

def render_raw_baseline_centroid_plots(manifest: list):
    print("\n=== Section E: Raw baseline centroid plots ===")
    print("Loading centroid caches ...", flush=True)
    centroids, points = load_all_centroids()
    print("  Done loading centroids")

    raw_categories = ["Raw ICU Pos", "Raw ICU Neg", "Raw Mort Pos", "Raw Mort Neg"]

    for idx, agg in enumerate(AGGREGATION_METHODS):
        agg_slug = agg.replace(" ", "_")
        print(f"  {agg} ...", flush=True)

        try:
            raw_c = [
                centroids["icu"][agg]["raw"]["pos"],
                centroids["icu"][agg]["raw"]["neg"],
                centroids["mortality"][agg]["raw"]["pos"],
                centroids["mortality"][agg]["raw"]["neg"],
            ]
            raw_p = [
                points["icu"][agg]["raw"]["pos"],
                points["icu"][agg]["raw"]["neg"],
                points["mortality"][agg]["raw"]["pos"],
                points["mortality"][agg]["raw"]["neg"],
            ]
        except KeyError:
            print(f"    [SKIP] missing raw centroid data for {agg}")
            continue

        meta_base = {
            "category": "Centroid Density",
            "plot_type": "kde",
            "label": None,
            "aggregation": agg,
            "vital": None,
            "scenario": None,
            "filter": None,
        }

        stem = f"raw_baseline_{idx}_{agg_slug}_maxrange"
        fig = attempt(centroid_plot,
                      raw_c, raw_p, VITAL_NAMES, VITAL_UNITS,
                      raw_categories,
                      use_percentile=False,
                      background="white")
        save_fig(fig, stem, manifest, {
            **meta_base,
            "title": f"Centroid Density — Raw Baseline ({agg.title()})",
        })

        stem = f"raw_baseline_{idx}_{agg_slug}_p95"
        fig = attempt(centroid_plot,
                      raw_c, raw_p, VITAL_NAMES, VITAL_UNITS,
                      raw_categories,
                      use_percentile=True,
                      background="white")
        save_fig(fig, stem, manifest, {
            **meta_base,
            "title": f"Centroid Density — Raw Baseline ({agg.title()}) Percentile",
        })


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", default="all",
                    choices=["all", "length", "impact", "centroid", "baseline"],
                    help="Which section to render (default: all)")
    args = ap.parse_args()

    manifest = []

    if args.section in ("all", "length"):
        render_length_plots(manifest)

    if args.section in ("all", "impact"):
        render_impact_plots(manifest)

    if args.section in ("all", "centroid"):
        render_centroid_plots(manifest)

    if args.section in ("all", "baseline"):
        render_raw_baseline_centroid_plots(manifest)

    # Written unconditionally, including under --section — so a targeted re-run
    # leaves a fragment covering only that section. Do a full run before calling
    # build_manifest.py or the gallery loses everything you skipped.
    frag_path = TOOLBOX_ROOT / "figures" / "marimo_manifest_fragment.json"
    frag_path.parent.mkdir(parents=True, exist_ok=True)
    with open(frag_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest fragment written: {frag_path}  ({len(manifest)} records)")


if __name__ == "__main__":
    main()
