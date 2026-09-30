# ICU-EHR Graph Gallery

The pipelines produce around 2000 figures, spread across three datasets, two preprocessing pipelines, and a comparison notebook. Finding the one you half-remember by digging through `figures/` is miserable, so this is a browser front-end that lets you filter down to it — and, more usefully, put two of them side by side.

**Not standalone.** It reads images from the sibling `../figures/` directory, so it only works from inside a full checkout of `clinical_preprocessing_toolbox/`. Copying just this folder somewhere else gets you an empty gallery.

## Running the application

Via Docker (bind-mounts `../figures/` read-only, see `docker-compose.yml`):

```bash
cd clinical_preprocessing_toolbox/gallery
docker compose up --build
```

Or directly, without Docker:

```bash
cd clinical_preprocessing_toolbox/gallery
./run.sh
```

Then open **http://localhost:8000** in your browser.

To use a different port, edit `docker-compose.yml` before running:

```yaml
ports:
  - "9000:8000"   # change 9000 to any port you want
```

To stop:

```bash
docker compose down
```

## Regenerating the figures

The gallery serves whatever `build_manifest.py` last wrote. After a pipeline changes, re-run the three steps in order — the manifest builder needs both extractor fragments on disk before it runs, and it won't complain if one is missing, it'll just quietly serve half a gallery:

```bash
python extract_notebook_graphs.py     # Jupyter notebooks → figures + fragment
python render_marimo_mimic_iii.py     # EHR Marimo plots  → figures + fragment
python build_manifest.py              # merge into graphs_manifest.json
```

`extract_notebook_graphs.py --check` is worth running first if you've edited a notebook: figure titles are keyed on each figure's *position* in the notebook, so inserting a plot shifts every title after it onto the wrong image without any error.

The scripts under `scripts/` are earlier versions of the above plus one-off repair tools. You shouldn't need them.

## Using the gallery

### Browsing graphs

The **Browse Graphs** panel on the left narrows things down. The dropdowns cascade, each one filtering the options in the next:

1. **Dataset** — MIMIC-III, MIMIC-IV, or eICU
2. **Pipeline** — which preprocessing pipeline produced the graph
3. **Category** — the type of analysis (e.g. Cohort Statistics, Filter Impact)
4. **Label** — the prediction target (ICU length of stay or mortality), if applicable
5. **Aggregation** — the feature aggregation method, if applicable
6. **Graph** — the specific graph to display

A live counter under the filters tells you how many graphs still match.

A `*` on a title means nobody wrote that title — it was guessed from the notebook's cell source. Those are worth double-checking against the image before you quote them anywhere, since a guessed title is occasionally just wrong.

### Side-by-side comparison

**Side-by-Side** in the header splits the view into two independently-filtered panes. **⇄ Swap**, **Copy →**, and **← Copy** shuffle things between them, and **Hide Controls** collapses the dropdowns once you've got what you want on screen.

This is really what the gallery is for — the whole point of the project is comparing two pipelines, and eyeballing them together beats flipping between browser tabs.

### Suggested comparisons

When the manifest can find a counterpart for the graph you're looking at, a **Suggested Comparisons** panel shows up in the sidebar; clicking one opens side-by-side with the pair loaded. Suggestions come from `build_manifest.py`, which matches on dataset, category, label, aggregation, vital and plot type — so a graph with sparse metadata may get no suggestions even though a counterpart exists.

### Show All

Older duplicates are hidden by default. **Show All** brings them back, which is occasionally useful when you want to see how a figure changed between runs.
