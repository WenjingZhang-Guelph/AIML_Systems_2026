"""
Retrofits titles and metadata onto EHR graphs that were scraped out of the Marimo
HTML exports with nothing but a sequence number.

This is archaeology, and it should be treated as such. extract_marimo_html.py
gives us numbered PNGs and no context whatsoever, so the only way back to "which
plot is this" is to replay the notebook's output order in our heads:

  0–9    length analysis     — 5 stat methods × (surface, heatmap)
  10–24  ICU filter impact   — 5 aggregations × (accuracy, F1, McNemar)
  25–39  mortality, same shape
  40+    centroid analysis   — 5 aggregations × 12 filters × 3 plot types
  last 10  raw-baseline centroid density, appended by a later notebook cell

Every one of those boundaries is a hardcoded constant derived from counting
figures by hand. Change the notebook and this file is wrong, with no error —
just confidently mislabelled graphs. It's also why classify() resorts to
guessing plot type from the image's aspect ratio, which is exactly as fragile as
it sounds and only works because each plot type happens to use a distinct
figsize.

The right fix is to stop scraping HTML and render through
render_marimo_mimic_iii.py, which emits proper metadata at source. This script
exists for eICU and MIMIC-IV, which don't have a renderer yet.

Run after build_manifest.py — it edits graphs_manifest.json in place.
"""

import json
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).parent.parent
MANIFEST_PATH = REPO_ROOT / "graphs_manifest.json"
GRAPHS_DIR = REPO_ROOT.parent / "figures"

AGGREGATION_METHODS = [
    "mean",
    "median",
    "standard deviation",
    "mean deviation",
    "maximum deviation",
]
FILTERS = [
    "heart rate",
    "systolic blood pressure",
    "diastolic blood pressure",
    "mean blood pressure",
    "respiration rate",
    "temperature",
    "oxygen saturation",
    "fill missing data",
    "long missing segment",
    "long gap",
    "high invalid data",
    "all vitals",
]
LENGTH_STAT_METHODS = ["mean", "median", "std", "max", "min"]
RAW_BASELINE_COUNT = len(AGGREGATION_METHODS) * 2  # each agg gets a max-range and a p95 version


def classify(width: int, height: int) -> str:
    """Guess the plot type from the image's shape. Yes, really.

    Each plot type in visualization_manager_v2 uses a different figsize, so
    aspect ratio is a usable fingerprint when you have no other information. The
    thresholds are empirical — I measured the actual outputs and picked
    midpoints between the clusters. They're tight (2.2 vs 2.52!), so anyone who
    changes a figsize upstream breaks this silently.

    Only used as a sanity signal; build_metadata() trusts the sequence position
    over this.
    """
    ratio = width / height
    if ratio < 1.3:
        return "length"
    if ratio < 1.9:
        return "filter_impact"
    if ratio < 2.2:
        return "mcnemar"
    if ratio < 2.52:
        return "centroid_shift"
    return "centroid_density"


def build_metadata(idx: int, ptype: str, dataset: str) -> dict:
    """Work out what image number `idx` must be, from the notebook's cell order.

    The arithmetic reads oddly but it's all the same shape: subtract the offset of
    the current block, then divide and modulo by however many plots that block
    emits per iteration. See the module docstring for the block boundaries.
    """
    # Alternating surface/heatmap, hence the //2 and %2.
    if idx < 10:
        method_idx = idx // 2
        is_surface = (idx % 2 == 0)
        stat = LENGTH_STAT_METHODS[method_idx]
        label = "3D Surface" if is_surface else "Heatmap"
        return {
            "category": "Length Analysis",
            "plot_type": "surface" if is_surface else "heatmap",
            "aggregation": stat,
            "label": None,
            "filter": None,
            "title": f"{label} — Observation Count by Hour ({stat.title()})",
            "auto_named": False,
        }

    # Each aggregation emits a triplet: accuracy bar, F1 bar, McNemar heatmap.
    if idx < 25:
        j = idx - 10
        agg_idx = j // 3
        plot_in_triplet = j % 3
        agg = AGGREGATION_METHODS[agg_idx]

        if plot_in_triplet == 2:
            return {
                "category": "McNemar Significance",
                "plot_type": "heatmap",
                "aggregation": agg,
                "label": "icu",
                "filter": None,
                "title": f"McNemar Significance — ICU ({agg.title()})",
                "auto_named": False,
            }
        metric = "Accuracy" if plot_in_triplet == 0 else "F1"
        return {
            "category": "Filter Impact",
            "plot_type": "bar",
            "aggregation": agg,
            "label": "icu",
            "filter": None,
            "title": f"Filter Impact — ICU Testing {metric} ({agg.title()})",
            "auto_named": False,
        }

    # Identical block to the one above, just the mortality label instead of ICU.
    if idx < 40:
        j = idx - 25
        agg_idx = j // 3
        plot_in_triplet = j % 3
        agg = AGGREGATION_METHODS[agg_idx]

        if plot_in_triplet == 2:
            return {
                "category": "McNemar Significance",
                "plot_type": "heatmap",
                "aggregation": agg,
                "label": "mortality",
                "filter": None,
                "title": f"McNemar Significance — Mortality ({agg.title()})",
                "auto_named": False,
            }
        metric = "Accuracy" if plot_in_triplet == 0 else "F1"
        return {
            "category": "Filter Impact",
            "plot_type": "bar",
            "aggregation": agg,
            "label": "mortality",
            "filter": None,
            "title": f"Filter Impact — Mortality Testing {metric} ({agg.title()})",
            "auto_named": False,
        }

    # Centroid block: every (aggregation, filter) pair emits shift bar, density,
    # density-at-p95, in that order.
    centroid_i = idx - 40
    group_idx = centroid_i // 3
    subtype = centroid_i % 3

    category = "Centroid Shift" if subtype == 0 else "Centroid Density"
    plot_type = "bar" if subtype == 0 else "kde"

    # And here's where the positional trick finally breaks down. Working out
    # which aggregation and filter a group belongs to assumes no group failed to
    # render — every failure shifts everything after it. MIMIC-III and IV lose
    # about 3 of 60 groups, which is tolerable. eICU loses 27 of 60, at which
    # point the numbering is meaningless, so we leave agg/filter null there
    # rather than publish confident nonsense. Those graphs are still browsable,
    # just not filterable.
    if dataset in ("mimic-iii", "mimic-iv"):
        agg_idx = group_idx // len(FILTERS)
        filter_idx = group_idx % len(FILTERS)
        agg = AGGREGATION_METHODS[agg_idx] if agg_idx < len(AGGREGATION_METHODS) else None
        filter_name = FILTERS[filter_idx] if filter_idx < len(FILTERS) else None
    else:
        agg = None
        filter_name = None

    parts = [category]
    if filter_name:
        parts.append(filter_name.title())
    if agg:
        parts.append(f"({agg.title()})")
    suffix = " Percentile" if subtype == 2 else ""

    return {
        "category": category,
        "plot_type": plot_type,
        "aggregation": agg,
        "label": None,
        "filter": filter_name,
        "title": " — ".join(parts) + suffix,
        "auto_named": False,
    }


def assign_dataset(dataset: str, records_by_id: dict[str, dict]) -> int:
    ehr_dir = GRAPHS_DIR / "EHR-Dataset-Processing" / dataset
    pngs = sorted([p for p in ehr_dir.glob("*.png") if "thumbs" not in str(p)])
    total = len(pngs)

    # The raw-baseline plots come from a cell added to the notebook later, so
    # they're bolted onto the end of the sequence and have to be counted
    # backwards from the total rather than forwards from an offset. eICU's export
    # predates that cell, so setting raw_start past the end disables the branch.
    has_raw_baseline = dataset in ("mimic-iii", "mimic-iv")
    raw_start = total - RAW_BASELINE_COUNT if has_raw_baseline else total + 1

    updated = 0
    for idx, png_path in enumerate(pngs):
        with Image.open(png_path) as img:
            ptype = classify(*img.size)

        if idx >= raw_start:
            raw_idx = idx - raw_start
            agg_idx = raw_idx // 2
            is_percentile = bool(raw_idx % 2)
            agg = AGGREGATION_METHODS[agg_idx] if agg_idx < len(AGGREGATION_METHODS) else None
            suffix = " Percentile" if is_percentile else ""
            meta = {
                "category": "Centroid Density",
                "plot_type": "kde",
                "aggregation": agg,
                "label": None,
                "filter": None,
                "title": (f"Centroid Density — Raw Baseline ({agg.title()}){suffix}"
                          if agg else f"Centroid Density — Raw Baseline{suffix}"),
                "auto_named": False,
            }
        else:
            meta = build_metadata(idx, ptype, dataset)

        rec_id = png_path.stem
        rec = records_by_id.get(rec_id)
        if rec is None:
            print(f"  Warning: no manifest record for {rec_id}")
            continue
        rec.update(meta)
        updated += 1

    return updated


def main():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    records = manifest["records"]
    records_by_id = {r["id"]: r for r in records}

    total = 0
    for dataset in ("eicu", "mimic-iii", "mimic-iv"):
        print(f"Assigning metadata for EHR {dataset}...")
        n = assign_dataset(dataset, records_by_id)
        print(f"  {n} records updated")
        total += n

    # We just flipped a few hundred records from auto_named to curated and
    # changed their categories, so the counts and the dropdown tree that
    # build_manifest.py computed are now stale. Rebuilding both here rather than
    # asking people to re-run build_manifest.py, which would undo this script's work.
    curated = sum(1 for r in records if not r["auto_named"])
    auto = sum(1 for r in records if r["auto_named"])
    manifest["stats"]["curated"] = curated
    manifest["stats"]["auto_named"] = auto

    # Duplicated from build_manifest.build_option_tree — if you change the tree
    # shape, change it in both places.
    NULL = "—"
    tree: dict = {}
    for r in records:
        if r["superseded"]:
            continue
        ds = r["dataset"] or "unknown"
        pl = r["pipeline"] or "unknown"
        cat = r["category"] or "Uncategorized"
        lbl = r["label"] or NULL
        agg = r["aggregation"] or NULL
        tree.setdefault(ds, {}).setdefault(pl, {}).setdefault(cat, {}).setdefault(lbl, {}).setdefault(agg, []).append(r["id"])
    manifest["option_tree"] = tree

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(f"\nDone. {total} EHR records updated. curated={curated}, auto_named={auto}")


if __name__ == "__main__":
    main()
