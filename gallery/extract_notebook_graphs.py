#!/usr/bin/env python3
"""
Rips the base64 PNG/SVG blobs back out of executed Jupyter notebooks so the
gallery has real image files to serve.

Everything hinges on one fragile assumption: a figure is identified by its
position in the notebook. CURATED_CATALOG is keyed on (notebook slug, Nth image
output in the notebook), because notebook outputs carry no stable identity of
their own — no name, no id, nothing that survives a re-execute. So if you insert
a cell that plots something in the middle of a notebook, every curated title
after it shifts by one and starts describing the wrong figure. It fails quietly,
too: you just get plausible-looking wrong captions. After editing a notebook,
re-run with --check and eyeball the counts before trusting the output.

Anything not in the catalog falls back to guessing a title from the cell source,
and gets flagged auto_named=True so the UI can put a `*` next to it.

Output is ../figures/extracted_manifest_fragment.json, which build_manifest.py
merges with the Marimo fragment into the real manifest.

    python extract_notebook_graphs.py            # extract everything
    python extract_notebook_graphs.py --check    # count only, touch nothing
    python extract_notebook_graphs.py --only pipeline_comparison
"""

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Optional

try:
    from PIL import Image
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False

# ─── Paths ────────────────────────────────────────────────────────────────────
GALLERY = Path(__file__).parent.resolve()
TOOLBOX_ROOT = GALLERY.parent
PIPELINES = TOOLBOX_ROOT / "pipelines"
COMPARISON = TOOLBOX_ROOT / "comparison"
STATIC_GRAPHS = TOOLBOX_ROOT / "figures"

# ─── Hand-written titles for the figures that actually matter ─────────────────
# Keyed on (slug, Nth image in the notebook) — see the module docstring for why
# that's brittle. The _c() helper and the loops below exist because these titles
# are overwhelmingly "same plot, once per vital" or "once per aggregation
# method"; spelling out all ~40 entries by hand was unmaintainable and I kept
# getting the ordering wrong. The loop order has to match the notebook's own
# loop order exactly.
CURATED_CATALOG = {}


def _c(slug: str, idx: int, **kw):
    CURATED_CATALOG[(slug, idx)] = kw


# ── Experiment_Statistics ─────────────────────────────────────────────────────
_NB = "experiment_statistics"
_c(_NB, 0,
   title="ICU Stays per Scenario",
   description="Bar chart of ICU stay counts across all 13 MIMIC_Extract cohort scenarios, sorted descending.",
   category="Cohort Statistics", plot_type="bar",
   dataset="mimic-iii", pipeline="MIMIC_Extract", label=None, aggregation=None,
   vital=None, scenario=None, filter=None)
# Heads up: Temperature and Respiratory Rate are swapped relative to _VITALS_LENS
# further down. Not a typo — the two notebooks emit their per-vital plots in
# different orders, and these lists mirror whatever each notebook does. Sort them
# into a common order and the captions stop matching the images.
for _i, _v in enumerate(["Heart Rate", "Systolic Blood Pressure", "Diastolic Blood Pressure",
                          "Mean Blood Pressure", "Temperature", "Respiratory Rate", "Oxygen Saturation"]):
    _c(_NB, 1 + _i,
       title=f"Vital Mean ± Std with Median across Scenarios — {_v}",
       description=f"Error-bar plot of mean ± 1 std dev and median of {_v} for each of the 13 MIMIC_Extract cohort scenarios.",
       category="Vital Statistics", plot_type="error-bar",
       dataset="mimic-iii", pipeline="MIMIC_Extract", label=None, aggregation=None,
       vital=_v.lower(), scenario=None, filter=None)
_c(_NB, 8,
   title="Z-Scored Variable Means Heatmap across Scenarios",
   description="Row-normalised (z-score) heatmap of mean values for all 7 vitals across the 13 MIMIC_Extract cohort scenarios.",
   category="Vital Statistics", plot_type="heatmap",
   dataset="mimic-iii", pipeline="MIMIC_Extract", label=None, aggregation=None,
   vital=None, scenario=None, filter=None)

# ── curated_mimic_iii_analysis (copy 2) ───────────────────────────────────────
_NB = "curated_mimic_iii_analysis_copy2"
_VITALS_LENS = ["Heart Rate", "Systolic Blood Pressure", "Diastolic Blood Pressure",
                "Mean Blood Pressure", "Respiratory Rate", "Temperature", "Oxygen Saturation"]
for _i, _v in enumerate(_VITALS_LENS):
    _c(_NB, _i,
       title=f"Lens Plot — {_v} (Aggregation × Hour Heatmap)",
       description=f"Heatmap of {_v} values across 5 aggregation methods (rows) and 24 hours (columns) for the baseline cohort.",
       category="Lens Plots", plot_type="heatmap",
       dataset="mimic-iii", pipeline="MIMIC_Extract", label=None, aggregation=None,
       vital=_v.lower(), scenario=None, filter=None)
_AGG = ["mean", "median", "standard deviation", "mean deviation", "maximum deviation"]
_LABELS_ME = ["icu_los", "mortality"]
_idx = 7
for _a in _AGG:
    for _l in _LABELS_ME:
        _c(_NB, _idx,
           title=f"McNemar Significance Heatmap — {_l.upper()} / {_a.title()}",
           description=f"Single-row heatmap of −log₁₀(p) values for McNemar test comparing each scenario to the baseline, for label '{_l}' with '{_a}' aggregation.",
           category="McNemar Significance", plot_type="heatmap",
           dataset="mimic-iii", pipeline="MIMIC_Extract", label=_l, aggregation=_a,
           vital=None, scenario=None, filter=None)
        _idx += 1
for _a in _AGG:
    _c(_NB, _idx,
       title=f"Centroid Shift PCA — {_a.title()} Aggregation",
       description=f"PCA scatter plot (2 panels: ICU-LOS and Mortality) showing baseline centroid (red star) vs scenario centroids with connector lines, using '{_a}' aggregation.",
       category="Centroid Shift PCA", plot_type="pca-scatter",
       dataset="mimic-iii", pipeline="MIMIC_Extract", label=None, aggregation=_a,
       vital=None, scenario=None, filter=None)
    _idx += 1
_c(_NB, _idx,
   title="Positive-Class Prevalence by Scenario",
   description="Grouped bar chart of ICU-LOS and hospital mortality positive-class prevalence for baseline and all 12 variant scenarios.",
   category="Cohort Statistics", plot_type="bar",
   dataset="mimic-iii", pipeline="MIMIC_Extract", label=None, aggregation=None,
   vital=None, scenario=None, filter=None)
_idx += 1
_c(_NB, _idx,
   title="Distribution of Record Lengths across Scenarios (KDE)",
   description="Overlaid KDE density plot of ICU stay lengths for each cohort scenario, showing how record-length distribution shifts across variants.",
   category="Cohort Statistics", plot_type="kde",
   dataset="mimic-iii", pipeline="MIMIC_Extract", label=None, aggregation=None,
   vital=None, scenario=None, filter=None)

# ── pipeline_comparison ───────────────────────────────────────────────────────
_NB = "pipeline_comparison"
_c(_NB, 0,
   title="Pipeline Performance by Aggregation Method — Grouped Bar",
   description="Grouped bar chart (Accuracy, F1, AUC-ROC) comparing MIMIC_Extract vs EHR-Dataset-Processing for each of the 5 aggregation methods.",
   category="Pipeline Comparison", plot_type="bar",
   dataset="mimic-iii", pipeline="Comparison", label="mortality", aggregation=None,
   vital=None, scenario=None, filter=None)
_c(_NB, 1,
   title="Pipeline Score Deviation Heatmap (EHR − MIMIC)",
   description="Annotated heatmap showing EHR minus MIMIC percentage-point difference across metrics (rows) and aggregation methods (columns).",
   category="Pipeline Comparison", plot_type="heatmap",
   dataset="mimic-iii", pipeline="Comparison", label="mortality", aggregation=None,
   vital=None, scenario=None, filter=None)
_c(_NB, 2,
   title="McNemar Significance — MIMIC_Extract vs EHR (by Aggregation)",
   description="McNemar significance bars for paired MIMIC_Extract vs EHR predictions, broken down by aggregation method.",
   category="McNemar Significance", plot_type="bar",
   dataset="mimic-iii", pipeline="Comparison", label="mortality", aggregation=None,
   vital=None, scenario=None, filter=None)
_idx = 3
for _a in _AGG:
    _c(_NB, _idx,
       title=f"Per-Vital Centroid Shift — MIMIC_Extract vs EHR ({_a.title()} Aggregation)",
       description=f"Bar chart of relative centroid shift per vital (7 vitals) comparing MIMIC_Extract and EHR-Dataset-Processing with '{_a}' aggregation.",
       category="Centroid Shift", plot_type="bar",
       dataset="mimic-iii", pipeline="Comparison", label=None, aggregation=_a,
       vital=None, scenario=None, filter=None)
    _idx += 1
_c(_NB, _idx,
   title="MIMIC_Extract Vital Means by Scenario (Heatmap)",
   description="Annotated heatmap of z-scored vital means across all 13 MIMIC_Extract cohort scenarios.",
   category="MIMIC_Extract Scenario Analysis", plot_type="heatmap",
   dataset="mimic-iii", pipeline="Comparison", label=None, aggregation=None,
   vital=None, scenario=None, filter=None)
_idx += 1
_c(_NB, _idx,
   title="MIMIC_Extract Accuracy Δ vs Baseline by Scenario (Heatmap)",
   description="Heatmap of test accuracy change vs baseline_nofilters, for each aggregation method × scenario combination.",
   category="MIMIC_Extract Scenario Analysis", plot_type="heatmap",
   dataset="mimic-iii", pipeline="Comparison", label=None, aggregation=None,
   vital=None, scenario=None, filter=None)
_idx += 1
_c(_NB, _idx,
   title="MIMIC_Extract McNemar −log₁₀(p) by Scenario (Heatmap)",
   description="Heatmap of −log₁₀(McNemar p-value) vs baseline_nofilters for each aggregation method × scenario combination.",
   category="MIMIC_Extract Scenario Analysis", plot_type="heatmap",
   dataset="mimic-iii", pipeline="Comparison", label=None, aggregation=None,
   vital=None, scenario=None, filter=None)
_idx += 1
_c(_NB, _idx,
   title="EHR Filter-Step Progression — Accuracy and AUC (Line Chart)",
   description="Line chart of test accuracy and AUC-ROC across all 13 EHR filter-pipeline steps, one line per aggregation method.",
   category="EHR Filter Pipeline", plot_type="line",
   dataset="mimic-iii", pipeline="Comparison", label=None, aggregation=None,
   vital=None, scenario=None, filter=None)
_idx += 1
_c(_NB, _idx,
   title="EHR AUC-ROC by Filter Step and Aggregation (Heatmap)",
   description="Heatmap of AUC-ROC values for each aggregation method × filter step in the EHR-Dataset-Processing pipeline.",
   category="EHR Filter Pipeline", plot_type="heatmap",
   dataset="mimic-iii", pipeline="Comparison", label=None, aggregation=None,
   vital=None, scenario=None, filter=None)
_idx += 1
for _a in _AGG:
    _c(_NB, _idx,
       title=f"Cross-Pipeline Deviation Bars — {_a.title()} Aggregation",
       description=f"Side-by-side diverging bar chart: MIMIC_Extract scenario Δ accuracy (left) and EHR filter-step Δ accuracy (right) relative to each pipeline's baseline, using '{_a}' aggregation.",
       category="Cross-Pipeline Deviation", plot_type="bar",
       dataset="mimic-iii", pipeline="Comparison", label=None, aggregation=_a,
       vital=None, scenario=None, filter=None)
    _idx += 1
_c(_NB, _idx,
   title="Summary Dashboard — Accuracy and F1 by Aggregation Method",
   description="Grouped bar chart of mean accuracy and F1 (with error bars) for MIMIC_Extract vs EHR-Dataset-Processing across all 5 aggregation methods.",
   category="Pipeline Comparison", plot_type="bar",
   dataset="mimic-iii", pipeline="Comparison", label=None, aggregation=None,
   vital=None, scenario=None, filter=None)

# ─── Which notebooks to scrape ────────────────────────────────────────────────
# Entries are (path_or_git_spec, slug, pipeline, dataset, superseded), where a
# git spec is "git:<repo_dir>:<branch>:<path>". Reading straight out of git lets
# us pull figures from notebooks that were never checked out here, without
# carrying another copy on disk.
#
# The catch, and it's a live one: EHR-Dataset-Processing lost its nested .git
# when everything was consolidated into this toolbox on 2026-07-01. Every
# `git:` entry below therefore fails until you clone the upstream repo
# (https://github.com/noahsub/EHR-Dataset-Processing) somewhere and repoint
# EHR_REPO at it. The failure is a plain git error and the notebook just gets
# skipped, so it's easy to miss that a chunk of the gallery went missing.
#
# `superseded=True` means "keep extracting it, but hide it behind Show All" —
# these are older duplicates I didn't want to delete but don't want cluttering
# the default view either.
EHR_REPO = str(PIPELINES / "EHR-Dataset-Processing")
BRANCH = "origin/optimization_refactor"

NOTEBOOKS = [
    # ── MIMIC_Extract key notebooks ──────────────────────────────────────────
    (str(PIPELINES / "MIMIC_Extract/notebooks/Experiment_Statistics.ipynb"),
     "experiment_statistics", "MIMIC_Extract", "mimic-iii", False),
    (str(PIPELINES / "MIMIC_Extract/notebooks/curated_mimic_iii_analysis_executed copy 2.ipynb"),
     "curated_mimic_iii_analysis_copy2", "MIMIC_Extract", "mimic-iii", False),
    (str(PIPELINES / "MIMIC_Extract/notebooks/mimic_extract_analysis.ipynb"),
     "mimic_extract_analysis", "MIMIC_Extract", "mimic-iii", False),
    # ── MIMIC_Extract upstream baselines ─────────────────────────────────────
    (str(PIPELINES / "MIMIC_Extract/notebooks/Baselines for Mortality and LOS prediction - GRU-D.ipynb"),
     "baseline_grud", "MIMIC_Extract", "mimic-iii", False),
    (str(PIPELINES / "MIMIC_Extract/notebooks/Baselines for Mortality and LOS prediction - Sklearn.ipynb"),
     "baseline_sklearn", "MIMIC_Extract", "mimic-iii", False),
    (str(PIPELINES / "MIMIC_Extract/notebooks/Baselines for Intervention Prediction - Mechanical Ventilation.ipynb"),
     "baseline_ventilation", "MIMIC_Extract", "mimic-iii", False),
    (str(PIPELINES / "MIMIC_Extract/notebooks/Baselines for Intervention Prediction - Vasopressor.ipynb"),
     "baseline_vasopressor", "MIMIC_Extract", "mimic-iii", False),
    (str(PIPELINES / "MIMIC_Extract/notebooks/Summary Stats.ipynb"),
     "summary_stats", "MIMIC_Extract", "mimic-iii", False),
    (str(PIPELINES / "MIMIC_Extract/notebooks/Testing mimic_direct_extract.ipynb"),
     "testing_extract", "MIMIC_Extract", "mimic-iii", False),
    # ── MIMIC_Extract old / superseded ───────────────────────────────────────
    (str(PIPELINES / "MIMIC_Extract/notebooks/curated_mimic_iii_analysis_executed copy.ipynb"),
     "curated_mimic_iii_analysis_copy1", "MIMIC_Extract", "mimic-iii", True),
    (str(PIPELINES / "MIMIC_Extract/notebooks/curated_mimic_iii_analysis.ipynb"),
     "curated_mimic_iii_analysis_source", "MIMIC_Extract", "mimic-iii", True),
    (str(PIPELINES / "MIMIC_Extract/notebooks/old/Experiment_Statistics.ipynb"),
     "old_experiment_statistics", "MIMIC_Extract", "mimic-iii", True),
    (str(PIPELINES / "MIMIC_Extract/notebooks/old/curated_mimic_iii_analysis_executed copy.ipynb"),
     "old_curated_copy", "MIMIC_Extract", "mimic-iii", True),
    # ── Pipeline comparison ───────────────────────────────────────────────────
    (str(COMPARISON / "pipeline_comparison.ipynb"),
     "pipeline_comparison", "Comparison", "mimic-iii", False),
    # ── EHR per-dataset analysis (on optimization_refactor branch) ────────────
    (f"git:{EHR_REPO}:{BRANCH}:Experiments/mimic-iii-analysis.ipynb",
     "ehr_mimic_iii_analysis", "EHR-Dataset-Processing", "mimic-iii", False),
    (f"git:{EHR_REPO}:{BRANCH}:Experiments/mimic-iv-analysis.ipynb",
     "ehr_mimic_iv_analysis", "EHR-Dataset-Processing", "mimic-iv", False),
    (f"git:{EHR_REPO}:{BRANCH}:Experiments/eicu_analysis.ipynb",
     "ehr_eicu_analysis", "EHR-Dataset-Processing", "eicu", False),
    (f"git:{EHR_REPO}:{BRANCH}:Experiments/mimic_iii_distribution.ipynb",
     "ehr_mimic_iii_distribution", "EHR-Dataset-Processing", "mimic-iii", False),
    # ── EHR sandbox ──────────────────────────────────────────────────────────
    (f"git:{EHR_REPO}:{BRANCH}:Experiments/sandbox/mimic-iii_filter_optimization.ipynb",
     "ehr_sandbox_mimic_iii_opt", "EHR-Dataset-Processing", "mimic-iii", False),
    (f"git:{EHR_REPO}:{BRANCH}:Experiments/sandbox/mimic-iv_filter_optimization.ipynb",
     "ehr_sandbox_mimic_iv_opt", "EHR-Dataset-Processing", "mimic-iv", False),
    (f"git:{EHR_REPO}:{BRANCH}:Experiments/sandbox/test3.ipynb",
     "ehr_sandbox_test3", "EHR-Dataset-Processing", "mimic-iii", False),
]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_notebook(spec: str) -> dict:
    """Read notebook JSON from disk, or out of a git object if the spec says so.

    The split has maxsplit=3 so Windows-style or colon-containing paths in the
    final component survive intact. Git failures are deliberately loud (stderr is
    swallowed but the CalledProcessError isn't) — a missing branch is a config
    problem worth stopping for, unlike a missing local file, which just returns
    None and gets skipped.
    """
    if spec.startswith("git:"):
        _, repo, branch, nb_path = spec.split(":", 3)
        data = subprocess.check_output(
            ["git", "-C", repo, "show", f"{branch}:{nb_path}"],
            stderr=subprocess.DEVNULL
        )
        return json.loads(data)
    else:
        p = Path(spec)
        if not p.exists():
            return None
        with open(p, "rb") as f:
            return json.load(f)


def make_thumb(png_path: Path, thumb_path: Path, size=(300, 200)):
    """Shrink a figure for the grid view, or punt and copy it full-size.

    Pillow is optional because this has to run on nibi, where the venv is
    whatever CVMFS gives you. Without it every "thumbnail" is a full-resolution
    copy — the gallery still works, it just pushes a few hundred MB at the
    browser and the grid crawls. Worth installing Pillow if you're doing a full
    re-extract.
    """
    if HAS_PILLOW:
        try:
            img = Image.open(png_path)
            img.thumbnail(size, Image.LANCZOS)
            img.save(thumb_path, "PNG")
            return
        except Exception:
            pass
    if not thumb_path.exists():
        import shutil
        shutil.copy2(png_path, thumb_path)


def auto_name_from_cell(cell_src: str, prev_markdown: str, out_idx_in_cell: int) -> dict:
    """Best-effort title for anything CURATED_CATALOG doesn't cover.

    Order of preference is the plot's own title string, then the last heading in
    whatever markdown cell came before it, then nothing. That ordering is
    empirical: a hardcoded plt.title() is nearly always right, a preceding
    heading is right maybe half the time, and past that you may as well give up.
    Callers mark everything from here as auto_named so the UI can flag it.

    Only literal string titles get picked up — an f-string or a variable leaves
    no title in the source to match, which is exactly the case CURATED_CATALOG
    exists to cover.
    """
    title_match = re.search(
        r"(?:plt\.title|ax\.set_title|suptitle|set_title)\s*\(\s*['\"]([^'\"]+)['\"]",
        cell_src
    )
    if title_match:
        title = title_match.group(1).strip()
    elif prev_markdown:
        # Last heading, not the first — markdown cells in these notebooks tend to
        # open with a section header and end with the specific thing being plotted.
        headings = re.findall(r"#{1,4}\s+(.+)", prev_markdown)
        title = headings[-1].strip() if headings else ""
    else:
        title = ""

    if not title:
        title = "Plot"
    if out_idx_in_cell > 0:
        title = f"{title} ({out_idx_in_cell + 1})"

    # Ordering is load-bearing. `plot(` matches almost any matplotlib call so it
    # has to come last, and `scatter|pca` before it because PCA plots are drawn
    # with scatter. Bar is the default because it's the most common thing in
    # these notebooks, so guessing it is wrong least often.
    plot_type = "bar"
    if re.search(r"heatmap|imshow", cell_src, re.I):
        plot_type = "heatmap"
    elif re.search(r"scatter|PCA|pca", cell_src):
        plot_type = "pca-scatter"
    elif re.search(r"kdeplot|kde\b", cell_src, re.I):
        plot_type = "kde"
    elif re.search(r"errorbar|error_bar", cell_src, re.I):
        plot_type = "error-bar"
    elif re.search(r"surface|plot_surface", cell_src, re.I):
        plot_type = "surface"
    elif re.search(r"plot\(|lineplot", cell_src, re.I):
        plot_type = "line"

    return {"title": title, "plot_type": plot_type, "description": ""}


def extract_notebook(spec: str, slug: str, pipeline: str, dataset: str, superseded: bool,
                     dry_run: bool = False) -> list:
    """Walk one notebook's cells and pull out every image output.

    Two counters, and mixing them up is the easy mistake here: `global_out_idx`
    runs across the entire notebook and is what CURATED_CATALOG is keyed on and
    what names the file on disk; `cell_png_idx` resets per cell and only exists
    so auto-named titles from a cell that emits several figures get a "(2)",
    "(3)" suffix instead of colliding.
    """
    nb = load_notebook(spec)
    if nb is None:
        print(f"  [SKIP] {slug}: notebook not found at {spec}", file=sys.stderr)
        return []

    out_dir = STATIC_GRAPHS / pipeline / slug
    thumb_dir = out_dir / "thumbs"
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        thumb_dir.mkdir(parents=True, exist_ok=True)

    cells = nb.get("cells", [])
    records = []
    global_out_idx = 0
    prev_markdown = ""

    for cell in cells:
        cell_type = cell.get("cell_type", "")
        cell_src = "".join(cell.get("source", []))

        if cell_type == "markdown":
            prev_markdown = cell_src
            continue

        if cell_type != "code":
            continue

        cell_outputs = cell.get("outputs", [])
        cell_png_idx = 0

        for output in cell_outputs:
            data = output.get("data", {})

            png_b64 = data.get("image/png", "")
            svg_b64 = data.get("image/svg+xml", "")

            if not png_b64 and not svg_b64:
                continue

            cat_key = (slug, global_out_idx)
            if cat_key in CURATED_CATALOG:
                meta = CURATED_CATALOG[cat_key].copy()
                auto_named = False
            else:
                meta = auto_name_from_cell(cell_src, prev_markdown, cell_png_idx)
                meta.setdefault("category", "Exploratory")
                meta.setdefault("label", None)
                meta.setdefault("aggregation", None)
                meta.setdefault("vital", None)
                meta.setdefault("scenario", None)
                meta.setdefault("filter", None)
                meta.setdefault("dataset", dataset)
                meta.setdefault("pipeline", pipeline)
                auto_named = True

            record_id = f"{slug}_{global_out_idx:04d}"
            rel_png = f"graphs/{pipeline}/{slug}/{record_id}.png"
            rel_svg = f"graphs/{pipeline}/{slug}/{record_id}.svg" if svg_b64 else None
            rel_thumb = f"graphs/{pipeline}/{slug}/thumbs/{record_id}_thumb.png"

            if not dry_run:
                # nbformat lets an output payload be either one string or a list
                # of lines, depending on which tool wrote the notebook, so both
                # image branches have to handle the list case.
                png_path = out_dir / f"{record_id}.png"
                if png_b64:
                    raw = base64.b64decode(png_b64 if isinstance(png_b64, str) else "".join(png_b64))
                    png_path.write_bytes(raw)
                elif svg_b64:
                    # SVG-only output: the block below writes it. Nothing to do
                    # here, but the branch documents that this isn't a case we
                    # forgot about.
                    pass

                if svg_b64:
                    svg_path = out_dir / f"{record_id}.svg"
                    svg_content = svg_b64 if isinstance(svg_b64, str) else "".join(svg_b64)
                    # SVG is *supposed* to be stored as plain text, but some
                    # notebooks base64 it like the PNGs. Sniffing for a leading
                    # '<' is cheap and covers both.
                    if not svg_content.strip().startswith("<"):
                        svg_content = base64.b64decode(svg_content).decode("utf-8")
                    svg_path.write_text(svg_content)

                # SVG-only figures get no thumbnail — Pillow can't rasterise SVG
                # and the grid falls back to the full image for those.
                if png_b64:
                    make_thumb(out_dir / f"{record_id}.png", thumb_dir / f"{record_id}_thumb.png")

            record = {
                "id": record_id,
                "pipeline": meta.get("pipeline", pipeline),
                "dataset": meta.get("dataset", dataset),
                "notebook": slug,
                "category": meta.get("category", "Exploratory"),
                "plot_type": meta.get("plot_type", "bar"),
                "label": meta.get("label"),
                "aggregation": meta.get("aggregation"),
                "vital": meta.get("vital"),
                "scenario": meta.get("scenario"),
                "filter": meta.get("filter"),
                "title": meta.get("title", f"Plot {global_out_idx}"),
                "description": meta.get("description", ""),
                "file": rel_png if png_b64 else None,
                "svg": rel_svg,
                "thumb": rel_thumb if png_b64 else None,
                "auto_named": auto_named,
                "superseded": superseded,
                "source": "extracted",
            }
            records.append(record)

            global_out_idx += 1
            cell_png_idx += 1

    return records


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="Count images without writing anything — use this after "
                         "editing a notebook to see if the curated indices shifted")
    ap.add_argument("--only", help="Only process notebook slug (comma-separated)")
    args = ap.parse_args()

    only_set = set(args.only.split(",")) if args.only else None
    dry_run = args.check

    all_records = []
    total = 0

    for spec, slug, pipeline, dataset, superseded in NOTEBOOKS:
        if only_set and slug not in only_set:
            continue
        print(f"Processing {slug} ...", end=" ", flush=True)
        records = extract_notebook(spec, slug, pipeline, dataset, superseded, dry_run=dry_run)
        print(f"{len(records)} images")
        all_records.extend(records)
        total += len(records)

    print(f"\nTotal: {total} images extracted")

    if not dry_run:
        # Note this rewrites the whole fragment even under --only, so a targeted
        # re-run drops every notebook you didn't ask for. Re-run without --only
        # before calling build_manifest.py.
        frag_path = STATIC_GRAPHS / "extracted_manifest_fragment.json"
        frag_path.parent.mkdir(parents=True, exist_ok=True)
        with open(frag_path, "w") as f:
            json.dump(all_records, f, indent=2)
        print(f"Fragment written to {frag_path}")

    return total


if __name__ == "__main__":
    main()
