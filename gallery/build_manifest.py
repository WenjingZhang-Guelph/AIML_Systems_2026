#!/usr/bin/env python3
"""
Last step of the figure pipeline: stitch the per-extractor fragments into the one
graphs_manifest.json the Flask app actually reads.

Run order matters — extract_notebook_graphs.py and render_marimo_mimic_iii.py both
have to have written their fragment to ../figures/ before this runs, otherwise you
silently get a manifest that's missing half the gallery. Missing fragments are a
skip-with-warning rather than an error, because during development you often only
re-run one of the two extractors.

The interesting work here is suggested_pairs, which is what powers the "you might
also want to look at this" panel in the sidebar. Three kinds of pairing, in
decreasing order of how useful they turned out to be in practice: the same plot
from the other pipeline, the same plot for the other label, and (for the
comparison notebook's own plots) the single-pipeline plots it was derived from.
"""

import json
import os
from pathlib import Path
from typing import Optional

GALLERY = Path(__file__).parent.resolve()
STATIC = GALLERY.parent / "figures"

FRAGMENT_PATHS = [
    STATIC / "extracted_manifest_fragment.json",
    STATIC / "marimo_manifest_fragment.json",
]

OUT_PATH = GALLERY / "graphs_manifest.json"


# ─── Pairing heuristics ───────────────────────────────────────────────────────

def _key_cross_pipeline(r: dict) -> Optional[tuple]:
    """What has to line up for two graphs from different pipelines to be "the same graph".

    Deliberately strict — every facet has to match, including the ones that are
    usually None. A looser key (just dataset + category) pairs e.g. every
    per-vital plot with every other per-vital plot, and the sidebar fills up with
    junk suggestions. Records with no dataset or category can't be matched
    meaningfully at all, so they opt out entirely.
    """
    if not r.get("dataset") or not r.get("category"):
        return None
    return (
        r.get("dataset"),
        r.get("category"),
        r.get("label"),
        r.get("aggregation"),
        r.get("vital"),
        r.get("plot_type"),
    )


def _key_cross_label(r: dict) -> Optional[tuple]:
    """Same idea, but holding the pipeline fixed and letting the label vary.

    Note `pipeline` is in the key and `label` isn't — that's the whole point, we
    want the ICU version of a plot to suggest its mortality twin. Unlabelled
    records have no twin to find.
    """
    if not r.get("label"):
        return None
    return (
        r.get("dataset"),
        r.get("pipeline"),
        r.get("category"),
        r.get("aggregation"),
        r.get("vital"),
        r.get("plot_type"),
    )


def compute_suggestions(records: list) -> dict:
    """Build the {id: [{"id", "why"}, ...]} map the sidebar renders.

    The `why` strings are user-facing — they show up verbatim as the caption on
    each suggestion chip, so keep them short and phrased from the reader's side.

    Superseded records are excluded from both indexes on purpose: an old duplicate
    still matches on every facet, so leaving them in means the top suggestion for
    any graph is usually its own stale predecessor.
    """
    by_cross = {}
    for r in records:
        if r.get("superseded"):
            continue
        k = _key_cross_pipeline(r)
        if k:
            by_cross.setdefault(k, []).append(r)

    by_label = {}
    for r in records:
        if r.get("superseded"):
            continue
        k = _key_cross_label(r)
        if k:
            by_label.setdefault(k, []).append(r)

    suggestions: dict = {}

    # The headline pairing — this is the one the whole comparison story rests on.
    for k, group in by_cross.items():
        pipelines = {r["pipeline"] for r in group}
        if "MIMIC_Extract" in pipelines and "EHR-Dataset-Processing" in pipelines:
            mimic_ids = [r["id"] for r in group if r["pipeline"] == "MIMIC_Extract"]
            ehr_ids = [r["id"] for r in group if r["pipeline"] == "EHR-Dataset-Processing"]
            for mid in mimic_ids:
                for eid in ehr_ids:
                    suggestions.setdefault(mid, []).append({"id": eid, "why": "Same plot type from EHR-Dataset-Processing"})
                    suggestions.setdefault(eid, []).append({"id": mid, "why": "Same plot type from MIMIC_Extract"})

    # Both labels have to actually be present in the group, otherwise there's
    # nothing to flip between and we'd emit a one-sided suggestion.
    for k, group in by_label.items():
        labels = {r["label"] for r in group if r.get("label")}
        if "icu" in labels and "mortality" in labels:
            icu_ids = [r["id"] for r in group if r.get("label") == "icu"]
            mort_ids = [r["id"] for r in group if r.get("label") == "mortality"]
            for iid in icu_ids:
                for mid in mort_ids:
                    suggestions.setdefault(iid, []).append({"id": mid, "why": "Same plot for Mortality label"})
                    suggestions.setdefault(mid, []).append({"id": iid, "why": "Same plot for ICU label"})

    # Comparison plots get a link back to the per-pipeline plots they came from,
    # so you can go from "the two pipelines disagree here" to "here's each one on
    # its own" in a click. Self-matches get filtered since a Comparison record
    # lands in its own by_cross bucket.
    comp_records = [r for r in records if r.get("pipeline") == "Comparison" and not r.get("superseded")]
    for cr in comp_records:
        k = _key_cross_pipeline(cr)
        if k and k in by_cross:
            for peer in by_cross[k]:
                if peer["id"] != cr["id"] and peer["pipeline"] != "Comparison":
                    suggestions.setdefault(cr["id"], []).append({
                        "id": peer["id"],
                        "why": f"Single-pipeline source from {peer['pipeline']}"
                    })

    # A graph can pick up the same peer from more than one rule above, and the
    # sidebar only has room for a handful anyway.
    for rid in suggestions:
        seen = set()
        deduped = []
        for s in suggestions[rid]:
            if s["id"] not in seen:
                seen.add(s["id"])
                deduped.append(s)
        suggestions[rid] = deduped[:6]  # sidebar gets cramped past ~6

    return suggestions


# ─── Dropdown option tree ─────────────────────────────────────────────────────

def build_option_tree(records: list) -> dict:
    """Pre-compute the cascade for the six Browse dropdowns.

    Shape is dataset → pipeline → category → label → aggregation → [record_ids].
    The nesting order is not arbitrary: it's the order the dropdowns appear in the
    UI, so the frontend can just walk one level deeper per selection instead of
    re-filtering ~2000 records on every change.

    The "—" placeholders matter. Plenty of graphs have no label or no aggregation,
    and if those levels collapse to None the JSON round-trip turns them into null
    keys that the frontend then has to special-case. A visible em-dash keeps the
    tree uniformly string-keyed and doubles as the dropdown text.
    """
    tree: dict = {}
    for r in records:
        if r.get("superseded") or not r.get("file"):
            continue
        d = r.get("dataset") or "unknown"
        p = r.get("pipeline") or "unknown"
        c = r.get("category") or "Misc"
        lbl = r.get("label") or "—"
        agg = r.get("aggregation") or "—"

        tree.setdefault(d, {}).setdefault(p, {}).setdefault(c, {}).setdefault(lbl, {}).setdefault(agg, []).append(r["id"])

    return tree


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    all_records = []
    seen_ids: set = set()

    for frag_path in FRAGMENT_PATHS:
        if not frag_path.exists():
            print(f"[skip] {frag_path} not found")
            continue
        with open(frag_path) as f:
            frag = json.load(f)
        added = 0
        for r in frag:
            if r["id"] not in seen_ids:
                # Fragments outlive the figures they describe — re-running an
                # extractor with different settings leaves entries pointing at
                # PNGs that no longer exist. Those render as broken images in the
                # gallery, so drop them here rather than debugging it in the browser.
                if r.get("file"):
                    img_path = GALLERY / "static" / r["file"]
                    if not img_path.exists():
                        continue
                seen_ids.add(r["id"])
                all_records.append(r)
                added += 1
        print(f"Loaded {added} records from {frag_path.name}")

    print(f"\nTotal records: {len(all_records)}")

    suggestions = compute_suggestions(all_records)
    for r in all_records:
        r["suggested_pairs"] = suggestions.get(r["id"], [])

    option_tree = build_option_tree(all_records)

    manifest = {
        "records": all_records,
        "option_tree": option_tree,
        "stats": {
            "total": len(all_records),
            "curated": sum(1 for r in all_records if not r.get("auto_named")),
            "auto_named": sum(1 for r in all_records if r.get("auto_named")),
            "superseded": sum(1 for r in all_records if r.get("superseded")),
            "pipelines": sorted({r.get("pipeline") for r in all_records if r.get("pipeline")}),
            "datasets": sorted({r.get("dataset") for r in all_records if r.get("dataset")}),
        },
    }

    with open(OUT_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest written to {OUT_PATH}")
    print(f"  curated: {manifest['stats']['curated']}, auto_named: {manifest['stats']['auto_named']}")
    print(f"  datasets: {manifest['stats']['datasets']}")
    print(f"  pipelines: {manifest['stats']['pipelines']}")

    # Belt-and-braces: the loop above already dropped anything missing, so this
    # should always come back clean. If it doesn't, the two paths have drifted
    # apart and the manifest is lying about what it can serve.
    errors = 0
    for r in all_records:
        if r.get("file"):
            p = GALLERY / "static" / r["file"]
            if not p.exists():
                print(f"  [WARN] missing file: {r['file']}")
                errors += 1
    if errors:
        print(f"\n  {errors} missing image files")
    else:
        print("\n  All image files present ✓")


if __name__ == "__main__":
    main()
