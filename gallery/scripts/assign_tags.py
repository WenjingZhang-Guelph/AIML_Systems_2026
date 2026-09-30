#!/usr/bin/env python3
"""
Bulk-tags every graph by which research question it speaks to, so the gallery can
be browsed by "what am I trying to argue" rather than by which notebook happened
to produce a figure.

Tagging is derived purely from each record's category — no per-graph curation.
That's a deliberate trade: it's coarse, but it stays correct as figures get
regenerated, whereas hand-tagged IDs would rot the moment the extraction indices
shift.

Re-running replaces tags.json wholesale, so any tags edited in the UI are lost.
Run it after build_manifest.py, then don't run it again unless you mean it.
"""

import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.resolve()
MANIFEST_PATH = REPO_ROOT / "graphs_manifest.json"
TAGS_PATH = REPO_ROOT / "tags.json"

TAGS = [
    "filtering-effects",
    "data-distribution",
    "ml-task-impact",
    "preprocessing-shift",
    "pipeline-transparency",
]

# Categories map onto the research questions roughly one-to-one. Two exceptions
# worth knowing about:
#
#   - Centroid Density sits under data-distribution rather than
#     filtering-effects, because those plots show what the cohort looks like, not
#     what a filter did to it.
#   - McNemar Significance gets two tags. It's the only evidence that a
#     difference in scores is real rather than noise, so it belongs both to the
#     "does preprocessing change task difficulty" question and to the "can we
#     detect a preprocessing shift" one.
#
# Anything not listed here goes untagged and gets counted in the unmapped
# warning — new categories show up there rather than being silently dropped.
CATEGORY_TAGS = {
    # Q1 — what filtering does to the statistical structure
    "Filter Impact":                   ["filtering-effects"],
    "Length Analysis":                 ["filtering-effects"],
    "EHR Filter Pipeline":             ["filtering-effects"],
    "MIMIC_Extract Scenario Analysis": ["filtering-effects"],
    "Centroid Density":                ["data-distribution"],
    # Q2 — does any of it change how hard the ML task is
    "Pipeline Comparison":             ["ml-task-impact"],
    "McNemar Significance":            ["ml-task-impact", "preprocessing-shift"],
    # Q3 — can a preprocessing shift be detected at all
    "Centroid Shift":                  ["preprocessing-shift"],
    "Centroid Shift PCA":              ["preprocessing-shift"],
    "Cross-Pipeline Deviation":        ["preprocessing-shift"],
    # Q4 — descriptive material, for explaining what the pipeline did
    "Exploratory":                     ["pipeline-transparency"],
    "Vital Statistics":                ["pipeline-transparency"],
    "Cohort Statistics":               ["pipeline-transparency"],
    "Lens Plots":                      ["pipeline-transparency"],
}


def main():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    records = manifest.get("records", [])
    assignments = {}
    unmapped = defaultdict(int)

    for record in records:
        graph_id = record["id"]
        category = record.get("category", "")
        tags = CATEGORY_TAGS.get(category)
        if tags:
            assignments[graph_id] = tags
        else:
            unmapped[category] += 1

    if unmapped:
        print("WARNING — unmapped categories (no tag assigned):")
        for cat, count in sorted(unmapped.items()):
            print(f"  {cat!r}: {count} records")
    else:
        print("All categories mapped.")

    tag_counts = defaultdict(int)
    for tags in assignments.values():
        for t in tags:
            tag_counts[t] += 1

    print(f"\nAssigned tags to {len(assignments)}/{len(records)} records:")
    for tag in TAGS:
        print(f"  {tag}: {tag_counts[tag]}")

    output = {"tags": TAGS, "assignments": assignments}
    with open(TAGS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {TAGS_PATH}")


if __name__ == "__main__":
    main()
