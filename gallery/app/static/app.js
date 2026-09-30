/* ─── Graph Gallery App ─────────────────────────────────────────────────── */
"use strict";

let MANIFEST = null;       // full manifest JSON from /api/manifest
let RECORDS_BY_ID = {};    // id -> record
let COMPARE_MODE = false;
let TAGS_DATA = { tags: [], assignments: {} };
let CURRENT_MAIN_RECORD = null;  // track which record is displayed in main pane

// ─── Bootstrap ───────────────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", async () => {
  await loadManifest();
  await loadTags();
  initDropdowns("sel-", ["sel-dataset", "sel-pipeline", "sel-category", "sel-label", "sel-agg", "sel-graph"]);
  initDropdowns("cmp-", ["cmp-dataset-left", "cmp-pipeline-left", "cmp-category-left",
                         "cmp-label-left", "cmp-agg-left", "cmp-graph-left"], "left");
  initDropdowns("cmp-", ["cmp-dataset-right", "cmp-pipeline-right", "cmp-category-right",
                         "cmp-label-right", "cmp-agg-right", "cmp-graph-right"], "right");
  populateDataset("sel-dataset");
  populateDataset("cmp-dataset-left");
  populateDataset("cmp-dataset-right");

  document.getElementById("btn-help").addEventListener("click", () => showModal(true));
  document.getElementById("btn-close-help").addEventListener("click", () => showModal(false));
  document.getElementById("help-modal").addEventListener("click", e => {
    if (e.target === document.getElementById("help-modal")) showModal(false);
  });
  ["sel-dataset", "sel-pipeline", "sel-category", "sel-label", "sel-agg"].forEach(id =>
    document.getElementById(id).addEventListener("change", updateMatchCount)
  );

  ["left", "right"].forEach(side =>
    ["dataset", "pipeline", "category", "label", "agg"].forEach(field =>
      document.getElementById(`cmp-${field}-${side}`)
        .addEventListener("change", () => updateCompareMatchCount(side))
    )
  );

  document.getElementById("btn-compare").addEventListener("click", toggleCompare);
  document.getElementById("btn-toggle-controls").addEventListener("click", toggleControls);
  document.getElementById("btn-swap-left").addEventListener("click",    swapPanes);
  document.getElementById("btn-copy-to-right").addEventListener("click", () => copyPane("left",  "right"));
  document.getElementById("btn-copy-to-left").addEventListener("click",  () => copyPane("right", "left"));

  // Tag filter: re-cascade from dataset when tag selection changes
  document.getElementById("sel-tag").addEventListener("change", () => {
    populateDataset("sel-dataset");
    document.getElementById("sel-dataset").dispatchEvent(new Event("change"));
    updateMatchCount();
  });

  // Tag manager toggle
  document.getElementById("btn-toggle-tag-mgr").addEventListener("click", () => {
    const body = document.getElementById("tag-manager-body");
    const btn  = document.getElementById("btn-toggle-tag-mgr");
    const open = body.style.display === "none";
    body.style.display = open ? "" : "none";
    btn.textContent    = open ? "▼" : "▶";
  });

  // Create tag
  document.getElementById("btn-create-tag").addEventListener("click", createTag);
  document.getElementById("tag-new-name").addEventListener("keydown", e => {
    if (e.key === "Enter") createTag();
  });

  // Add tag to current graph
  document.getElementById("sel-add-tag-main").addEventListener("change", e => {
    const tag = e.target.value;
    if (tag && CURRENT_MAIN_RECORD) {
      addTagToGraph(CURRENT_MAIN_RECORD.id, tag);
    }
    e.target.value = "";
  });

  updateStats();
});

async function loadManifest() {
  try {
    const res = await fetch("/api/manifest");
    MANIFEST = await res.json();
    RECORDS_BY_ID = {};
    for (const r of MANIFEST.records) RECORDS_BY_ID[r.id] = r;
  } catch (e) {
    console.error("Failed to load manifest:", e);
    document.getElementById("graph-placeholder").innerHTML =
      "<p style='color:#bd3140'>⚠ Could not load manifest. Run build_manifest.py first.</p>";
  }
}

// ─── Dropdown population helpers ─────────────────────────────────────────────

function visibleRecords() {
  if (!MANIFEST) return [];
  let recs = MANIFEST.records.filter(r => !!r.file);
  const tagFilter = document.getElementById("sel-tag")?.value;
  if (tagFilter) {
    recs = recs.filter(r => (TAGS_DATA.assignments[r.id] || []).includes(tagFilter));
  }
  return recs;
}

function uniqueSorted(arr) {
  return [...new Set(arr.filter(Boolean))].sort();
}

function setOptions(selId, values, placeholder = "— any —") {
  const sel = document.getElementById(selId);
  const cur = sel.value;
  sel.innerHTML = `<option value="">${placeholder}</option>`;
  for (const v of values) {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    sel.appendChild(opt);
  }
  if (values.includes(cur)) sel.value = cur;
  sel.disabled = values.length === 0;
}

function populateDataset(selId) {
  const records = visibleRecords();
  const datasets = uniqueSorted(records.map(r => r.dataset));
  setOptions(selId, datasets, "— select —");
}

// ─── Cascading dropdown logic (shared for main + compare panes) ───────────────

function initDropdowns(prefix, ids, pane = null) {
  // pane: null = main, "left" | "right" = compare pane
  const selIds = {
    dataset:  `${prefix}${pane ? "dataset-" + pane  : "dataset"}`,
    pipeline: `${prefix}${pane ? "pipeline-" + pane : "pipeline"}`,
    category: `${prefix}${pane ? "category-" + pane : "category"}`,
    label:    `${prefix}${pane ? "label-" + pane    : "label"}`,
    agg:      `${prefix}${pane ? "agg-" + pane      : "agg"}`,
    graph:    `${prefix}${pane ? "graph-" + pane    : "graph"}`,
  };

  const $ = id => document.getElementById(id);

  function filtered(upTo) {
    let recs = visibleRecords();
    if (upTo !== "dataset"  && $(selIds.dataset).value)  recs = recs.filter(r => r.dataset   === $(selIds.dataset).value);
    if (upTo !== "pipeline" && $(selIds.pipeline).value) recs = recs.filter(r => r.pipeline  === $(selIds.pipeline).value);
    if (upTo !== "category" && $(selIds.category).value) recs = recs.filter(r => r.category  === $(selIds.category).value);
    if (upTo !== "label"    && $(selIds.label).value)    recs = recs.filter(r => r.label     === $(selIds.label).value);
    if (upTo !== "agg"      && $(selIds.agg).value)      recs = recs.filter(r => r.aggregation === $(selIds.agg).value);
    return recs;
  }

  function refreshPipeline() {
    const recs = filtered("pipeline");
    setOptions(selIds.pipeline, uniqueSorted(recs.map(r => r.pipeline)), "— select —");
    refreshCategory();
  }
  function refreshCategory() {
    const recs = filtered("category");
    setOptions(selIds.category, uniqueSorted(recs.map(r => r.category)), "— select —");
    refreshLabel();
  }
  function refreshLabel() {
    const recs = filtered("label");
    const labels = uniqueSorted(recs.map(r => r.label).filter(Boolean));
    setOptions(selIds.label, labels, "— any —");
    refreshAgg();
  }
  function refreshAgg() {
    const recs = filtered("agg");
    const aggs = uniqueSorted(recs.map(r => r.aggregation).filter(Boolean));
    setOptions(selIds.agg, aggs, "— any —");
    refreshGraph();
  }
  function refreshGraph() {
    const recs = filtered("graph");
    const sortedRecs = [...recs].sort((a, b) => a.title.localeCompare(b.title));
    const sel = $(selIds.graph);
    sel.innerHTML = `<option value="">— select graph —</option>`;
    for (const r of sortedRecs) {
      const opt = document.createElement("option");
      opt.value = r.id;
      opt.textContent = r.title + (r.auto_named ? " *" : "");
      sel.appendChild(opt);
    }
    sel.disabled = sortedRecs.length === 0;
    // Auto-select if only one option
    if (sortedRecs.length === 1) {
      sel.value = sortedRecs[0].id;
      onGraphSelected(selIds.graph, pane);
    } else {
      onGraphSelected(selIds.graph, pane);  // clear if nothing selected
    }
  }

  $(selIds.dataset).addEventListener("change",  refreshPipeline);
  $(selIds.pipeline).addEventListener("change", refreshCategory);
  $(selIds.category).addEventListener("change", refreshLabel);
  $(selIds.label).addEventListener("change",    refreshAgg);
  $(selIds.agg).addEventListener("change",      refreshGraph);
  $(selIds.graph).addEventListener("change",    () => onGraphSelected(selIds.graph, pane));
}

// ─── Graph display ────────────────────────────────────────────────────────────

function onGraphSelected(graphSelId, pane = null) {
  const sel = document.getElementById(graphSelId);
  const id = sel ? sel.value : null;
  const record = id ? RECORDS_BY_ID[id] : null;

  if (pane === "left")  renderPane("left",  record);
  else if (pane === "right") renderPane("right", record);
  else                  renderMainPane(record);
}

function renderMainPane(record) {
  CURRENT_MAIN_RECORD = record;
  const img    = document.getElementById("graph-img-main");
  const title  = document.getElementById("graph-title-main");
  const desc   = document.getElementById("graph-desc-main");
  const tags   = document.getElementById("graph-tags-main");
  const ph     = document.getElementById("graph-placeholder");
  const dlLink = document.getElementById("link-download-main");
  const svgLink= document.getElementById("link-svg-main");
  const sugPan = document.getElementById("suggestions-panel");
  const sugList= document.getElementById("suggestions-list");

  if (!record) {
    img.style.display = "none";
    ph.style.display = "";
    dlLink.style.display = "none";
    svgLink.style.display = "none";
    sugPan.style.display = "none";
    title.textContent = "Select a graph from the sidebar";
    desc.textContent = "";
    tags.innerHTML = "";
    renderCustomTags(null);
    return;
  }

  ph.style.display = "none";
  title.textContent = record.title;
  desc.textContent = record.description || "";
  tags.innerHTML = buildTags(record);
  renderCustomTags(record);

  img.style.display = "";
  img.src = "/" + record.file;
  img.alt = record.title;

  dlLink.style.display = "";
  dlLink.href = "/" + record.file;
  dlLink.download = record.id + ".png";

  if (record.svg) {
    svgLink.style.display = "";
    svgLink.href = "/" + record.svg;
  } else {
    svgLink.style.display = "none";
  }

  // Suggestions
  const suggestions = findCrossPipelineSuggestions(record);
  if (suggestions.length > 0) {
    sugList.innerHTML = "";
    for (const s of suggestions) {
      const peer = RECORDS_BY_ID[s.id];
      if (!peer) continue;
      const li = document.createElement("li");
      li.innerHTML = `<button>
        <div>${escHtml(peer.title)}</div>
        <div class="why">${escHtml(s.why)} · ${escHtml(peer.pipeline)}</div>
      </button>`;
      li.querySelector("button").addEventListener("click", () => {
        const currentId = document.getElementById("sel-graph").value;
        if (!COMPARE_MODE) toggleCompare();
        if (currentId) _applyRecordToPane("left",  RECORDS_BY_ID[currentId]);
        _applyRecordToPane("right", peer);
      });
      sugList.appendChild(li);
    }
    sugPan.style.display = "";
  } else {
    sugPan.style.display = "none";
  }
}

function renderPane(side, record) {
  const img   = document.getElementById(`graph-img-${side}`);
  const title = document.getElementById(`graph-title-${side}`);
  const ph    = document.getElementById(`placeholder-${side}`);
  const dl    = document.getElementById(`dl-${side}`);
  const badge = document.getElementById(`compare-badge-${side}`);

  if (!record) {
    img.style.display = "none";
    ph.style.display = "";
    dl.style.display = "none";
    title.textContent = "—";
    if (badge) badge.style.display = "none";
    return;
  }

  ph.style.display = "none";
  img.style.display = "";
  img.src = "/" + record.file;
  img.alt = record.title;
  title.textContent = record.title;
  dl.style.display = "";
  dl.href = "/" + record.file;
  dl.download = record.id + ".png";

}

function buildTags(r) {
  const items = [
    r.dataset    ? `<span class="tag">${escHtml(r.dataset)}</span>` : "",
    r.pipeline   ? `<span class="tag">${escHtml(r.pipeline)}</span>` : "",
    r.label      ? `<span class="tag">label: ${escHtml(r.label)}</span>` : "",
    r.aggregation? `<span class="tag">agg: ${escHtml(r.aggregation)}</span>` : "",
    r.vital      ? `<span class="tag">vital: ${escHtml(r.vital)}</span>` : "",
    r.filter     ? `<span class="tag">filter: ${escHtml(r.filter)}</span>` : "",
    r.plot_type  ? `<span class="tag">${escHtml(r.plot_type)}</span>` : "",
    r.auto_named ? `<span class="tag" style="color:#888">auto-named</span>` : "",
  ];
  return items.filter(Boolean).join("");
}

// ─── Tag management ───────────────────────────────────────────────────────────

async function loadTags() {
  try {
    const res = await fetch("/api/tags");
    TAGS_DATA = await res.json();
  } catch (e) {
    console.error("Failed to load tags:", e);
  }
  refreshTagFilter();
  refreshTagManagerList();
}

async function saveTags() {
  try {
    await fetch("/api/tags", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(TAGS_DATA),
    });
  } catch (e) {
    console.error("Failed to save tags:", e);
  }
}

function refreshTagFilter() {
  const sel = document.getElementById("sel-tag");
  if (!sel) return;
  const cur = sel.value;
  sel.innerHTML = `<option value="">— any —</option>`;
  for (const t of TAGS_DATA.tags) {
    const opt = document.createElement("option");
    opt.value = t;
    opt.textContent = t;
    sel.appendChild(opt);
  }
  if (TAGS_DATA.tags.includes(cur)) sel.value = cur;
}

function refreshTagManagerList() {
  const ul = document.getElementById("tag-manager-list");
  if (!ul) return;
  ul.innerHTML = "";
  if (TAGS_DATA.tags.length === 0) {
    ul.innerHTML = `<li style="color:var(--muted);font-size:11px;border:none;background:none;padding:4px 0">No tags defined yet.</li>`;
    return;
  }
  for (const t of TAGS_DATA.tags) {
    const li = document.createElement("li");
    li.innerHTML = `<span>${escHtml(t)}</span><button class="tag-delete" title="Delete tag" data-tag="${escHtml(t)}">×</button>`;
    li.querySelector(".tag-delete").addEventListener("click", () => deleteTag(t));
    ul.appendChild(li);
  }
}

function renderCustomTags(record) {
  const section = document.getElementById("user-tags-main");
  const list    = document.getElementById("user-tag-list-main");
  const addSel  = document.getElementById("sel-add-tag-main");
  if (!section) return;

  if (!record || TAGS_DATA.tags.length === 0) {
    section.style.display = "none";
    return;
  }

  section.style.display = "";
  list.innerHTML = "";
  const assigned = TAGS_DATA.assignments[record.id] || [];
  for (const t of assigned) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="user-tag">${escHtml(t)}<button class="user-tag-remove" title="Remove tag" data-tag="${escHtml(t)}">×</button></span>`;
    li.querySelector(".user-tag-remove").addEventListener("click", () => removeTagFromGraph(record.id, t));
    list.appendChild(li);
  }

  // Populate add-tag select with unassigned tags
  addSel.innerHTML = `<option value="">+ add tag…</option>`;
  const unassigned = TAGS_DATA.tags.filter(t => !assigned.includes(t));
  for (const t of unassigned) {
    const opt = document.createElement("option");
    opt.value = t;
    opt.textContent = t;
    addSel.appendChild(opt);
  }
  addSel.style.display = unassigned.length > 0 ? "" : "none";
}

function addTagToGraph(graphId, tag) {
  if (!TAGS_DATA.assignments[graphId]) TAGS_DATA.assignments[graphId] = [];
  if (!TAGS_DATA.assignments[graphId].includes(tag)) {
    TAGS_DATA.assignments[graphId].push(tag);
    saveTags();
    renderCustomTags(CURRENT_MAIN_RECORD);
  }
}

function removeTagFromGraph(graphId, tag) {
  const arr = TAGS_DATA.assignments[graphId];
  if (!arr) return;
  const idx = arr.indexOf(tag);
  if (idx !== -1) {
    arr.splice(idx, 1);
    if (arr.length === 0) delete TAGS_DATA.assignments[graphId];
    saveTags();
    renderCustomTags(CURRENT_MAIN_RECORD);
  }
}

function createTag() {
  const input = document.getElementById("tag-new-name");
  const name  = input.value.trim();
  if (!name || TAGS_DATA.tags.includes(name)) return;
  TAGS_DATA.tags.push(name);
  saveTags();
  input.value = "";
  refreshTagFilter();
  refreshTagManagerList();
  if (CURRENT_MAIN_RECORD) renderCustomTags(CURRENT_MAIN_RECORD);
}

function deleteTag(name) {
  const idx = TAGS_DATA.tags.indexOf(name);
  if (idx === -1) return;
  TAGS_DATA.tags.splice(idx, 1);
  for (const id of Object.keys(TAGS_DATA.assignments)) {
    const arr = TAGS_DATA.assignments[id];
    const i   = arr.indexOf(name);
    if (i !== -1) {
      arr.splice(i, 1);
      if (arr.length === 0) delete TAGS_DATA.assignments[id];
    }
  }
  saveTags();
  // If the tag filter was set to the deleted tag, clear it and re-cascade
  const sel = document.getElementById("sel-tag");
  if (sel && sel.value === name) {
    sel.value = "";
    populateDataset("sel-dataset");
    document.getElementById("sel-dataset").dispatchEvent(new Event("change"));
    updateMatchCount();
  }
  refreshTagFilter();
  refreshTagManagerList();
  if (CURRENT_MAIN_RECORD) renderCustomTags(CURRENT_MAIN_RECORD);
}

function openInMain(id) {
  const record = RECORDS_BY_ID[id];
  if (!record) return;
  // Set the graph selector in the main view to this record's id, then render
  const sel = document.getElementById("sel-graph");
  if (sel) {
    // Try to directly set and trigger
    for (const opt of sel.options) {
      if (opt.value === id) { sel.value = id; break; }
    }
  }
  renderMainPane(record);
}

function prefillComparePane(side, id) {
  const record = RECORDS_BY_ID[id];
  if (!record) return;
  renderPane(side, record);
  // Optionally update the selects in the compare pane to reflect the record
  const graphSel = document.getElementById(`cmp-graph-${side}`);
  if (graphSel) {
    for (const opt of graphSel.options) {
      if (opt.value === id) { graphSel.value = id; return; }
    }
    // Option not present — add it and select
    const opt = document.createElement("option");
    opt.value = id; opt.textContent = record.title;
    graphSel.appendChild(opt);
    graphSel.value = id;
  }
}

// ─── Compare / ShowAll toggles ─────────────────────────────────────────────

function toggleCompare() {
  const mainId = document.getElementById("sel-graph").value;
  const leftId = document.getElementById("cmp-graph-left").value;

  COMPARE_MODE = !COMPARE_MODE;
  document.getElementById("single-view").style.display  = COMPARE_MODE ? "none" : "";
  document.getElementById("compare-view").style.display = COMPARE_MODE ? "" : "none";
  document.getElementById("sidebar").style.display      = COMPARE_MODE ? "none" : "";
  document.getElementById("btn-toggle-controls").style.display = COMPARE_MODE ? "" : "none";
  const btn = document.getElementById("btn-compare");
  btn.classList.toggle("active", COMPARE_MODE);
  btn.textContent = COMPARE_MODE ? "⬛ Single View" : "⬛ Side-by-Side";
  if (COMPARE_MODE) {
    // Reset controls to visible each time compare mode is entered
    CONTROLS_VISIBLE = true;
    document.getElementById("compare-controls-left").style.display  = "";
    document.getElementById("compare-controls-right").style.display = "";
    document.getElementById("btn-toggle-controls").textContent = "Hide Controls";
    populateDataset("cmp-dataset-left");
    populateDataset("cmp-dataset-right");
    updateCompareMatchCount("left");
    updateCompareMatchCount("right");
    if (mainId) _applyRecordToPane("left", RECORDS_BY_ID[mainId]);
  } else {
    if (leftId) _applyRecordToMain(RECORDS_BY_ID[leftId]);
  }
}

let CONTROLS_VISIBLE = true;

function toggleControls() {
  CONTROLS_VISIBLE = !CONTROLS_VISIBLE;
  const display = CONTROLS_VISIBLE ? "" : "none";
  document.getElementById("compare-controls-left").style.display  = display;
  document.getElementById("compare-controls-right").style.display = display;
  document.getElementById("btn-toggle-controls").textContent = CONTROLS_VISIBLE ? "Hide Controls" : "Show Controls";
}

function swapPanes() {
  const leftId  = document.getElementById("cmp-graph-left").value;
  const rightId = document.getElementById("cmp-graph-right").value;
  const leftRec  = leftId  ? RECORDS_BY_ID[leftId]  : null;
  const rightRec = rightId ? RECORDS_BY_ID[rightId] : null;
  _applyRecordToPane("left",  rightRec);
  _applyRecordToPane("right", leftRec);
}

function copyPane(from, to) {
  const id = document.getElementById(`cmp-graph-${from}`).value;
  if (!id) return;
  const record = RECORDS_BY_ID[id];
  if (!record) return;
  _applyRecordToPane(to, record);
}

// Drive the full cascading dropdown sequence for a compare pane so all
// selectors reflect the given record (or clear everything if record is null).
function _applyRecordToPane(side, record) {
  const ds  = document.getElementById(`cmp-dataset-${side}`);
  const pl  = document.getElementById(`cmp-pipeline-${side}`);
  const cat = document.getElementById(`cmp-category-${side}`);
  const lbl = document.getElementById(`cmp-label-${side}`);
  const agg = document.getElementById(`cmp-agg-${side}`);
  const gr  = document.getElementById(`cmp-graph-${side}`);

  if (!record) {
    // Pre-clear downstream selectors so setOptions's "restore cur value" logic
    // finds "" and leaves them at the placeholder instead of restoring old values.
    if (pl)  pl.value  = "";
    if (cat) cat.value = "";
    if (lbl) lbl.value = "";
    if (agg) agg.value = "";
    if (gr)  gr.value  = "";
    if (ds)  { ds.value = ""; ds.dispatchEvent(new Event("change")); }
    return;
  }

  // Each dispatchEvent triggers the registered cascade listener which
  // repopulates every downstream selector synchronously.
  if (ds)  { ds.value  = record.dataset;            ds.dispatchEvent(new Event("change")); }
  if (pl)  { pl.value  = record.pipeline;            pl.dispatchEvent(new Event("change")); }
  if (cat) { cat.value = record.category;           cat.dispatchEvent(new Event("change")); }
  if (lbl) { lbl.value = record.label        || ""; lbl.dispatchEvent(new Event("change")); }
  if (agg) { agg.value = record.aggregation  || ""; agg.dispatchEvent(new Event("change")); }
  if (gr)  { gr.value  = record.id;                  gr.dispatchEvent(new Event("change")); }
}

// Drive the full cascading dropdown sequence for the main pane.
function _applyRecordToMain(record) {
  const ds  = document.getElementById("sel-dataset");
  const pl  = document.getElementById("sel-pipeline");
  const cat = document.getElementById("sel-category");
  const lbl = document.getElementById("sel-label");
  const agg = document.getElementById("sel-agg");
  const gr  = document.getElementById("sel-graph");

  if (!record) {
    if (pl)  pl.value  = "";
    if (cat) cat.value = "";
    if (lbl) lbl.value = "";
    if (agg) agg.value = "";
    if (gr)  gr.value  = "";
    if (ds)  { ds.value = ""; ds.dispatchEvent(new Event("change")); }
    return;
  }

  if (ds)  { ds.value  = record.dataset;           ds.dispatchEvent(new Event("change")); }
  if (pl)  { pl.value  = record.pipeline;           pl.dispatchEvent(new Event("change")); }
  if (cat) { cat.value = record.category;          cat.dispatchEvent(new Event("change")); }
  if (lbl) { lbl.value = record.label       || ""; lbl.dispatchEvent(new Event("change")); }
  if (agg) { agg.value = record.aggregation || ""; agg.dispatchEvent(new Event("change")); }
  if (gr)  { gr.value  = record.id;                gr.dispatchEvent(new Event("change")); }
}

// ─── Help modal ────────────────────────────────────────────────────────────
function showModal(show) {
  document.getElementById("help-modal").style.display = show ? "flex" : "none";
}

// ─── Stats ─────────────────────────────────────────────────────────────────
function updateStats() {
  if (!MANIFEST) return;
  document.getElementById("total-count").textContent =
    `${MANIFEST.stats.total} graphs in gallery`;
  updateMatchCount();
}

function updateMatchCount() {
  if (!MANIFEST) return;
  const $ = id => document.getElementById(id);
  let recs = visibleRecords();
  if ($("sel-dataset").value)  recs = recs.filter(r => r.dataset      === $("sel-dataset").value);
  if ($("sel-pipeline").value) recs = recs.filter(r => r.pipeline     === $("sel-pipeline").value);
  if ($("sel-category").value) recs = recs.filter(r => r.category     === $("sel-category").value);
  if ($("sel-label").value)    recs = recs.filter(r => r.label        === $("sel-label").value);
  if ($("sel-agg").value)      recs = recs.filter(r => r.aggregation  === $("sel-agg").value);
  const n = recs.length;
  $("match-count").textContent = `${n} graph${n !== 1 ? "s" : ""} match current filters`;
}

function updateCompareMatchCount(side) {
  if (!MANIFEST) return;
  const $ = id => document.getElementById(id);
  let recs = visibleRecords();
  if ($(`cmp-dataset-${side}`).value)  recs = recs.filter(r => r.dataset      === $(`cmp-dataset-${side}`).value);
  if ($(`cmp-pipeline-${side}`).value) recs = recs.filter(r => r.pipeline     === $(`cmp-pipeline-${side}`).value);
  if ($(`cmp-category-${side}`).value) recs = recs.filter(r => r.category     === $(`cmp-category-${side}`).value);
  if ($(`cmp-label-${side}`).value)    recs = recs.filter(r => r.label        === $(`cmp-label-${side}`).value);
  if ($(`cmp-agg-${side}`).value)      recs = recs.filter(r => r.aggregation  === $(`cmp-agg-${side}`).value);
  const n = recs.length;
  const el = $(`match-count-${side}`);
  if (el) el.textContent = `${n} graph${n !== 1 ? "s" : ""} match current filters`;
}

// ─── Cross-pipeline suggestions ────────────────────────────────────────────

function findCrossPipelineSuggestions(record) {
  if (!record) return [];

  // Candidates must be from a different pipeline but share dataset and category.
  const candidates = visibleRecords().filter(r =>
    r.id       !== record.id       &&
    r.pipeline !== record.pipeline &&
    r.dataset  === record.dataset  &&
    r.category === record.category
  );

  if (candidates.length === 0) return [];

  // Score by how many additional fields match. null === null counts as a match.
  function scoreMatch(r) {
    let s = 0;
    if (r.plot_type   === record.plot_type)   s += 4;
    if (r.label       === record.label)       s += 3;
    if (r.aggregation === record.aggregation) s += 3;
    if (r.vital       === record.vital)       s += 2;
    if (r.filter      === record.filter)      s += 2;
    return s;
  }

  return candidates
    .map(r => ({ id: r.id, score: scoreMatch(r), why: _suggestionWhy(record, r) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 6);
}

function _suggestionWhy(from, to) {
  const shared = [];
  if (from.aggregation && from.aggregation === to.aggregation) shared.push(from.aggregation);
  if (from.vital       && from.vital       === to.vital)       shared.push(from.vital);
  if (from.label       && from.label       === to.label)       shared.push(from.label + " label");
  if (from.filter      && from.filter      === to.filter)      shared.push(from.filter + " filter");
  return shared.length > 0 ? "Same " + shared.join(" · ") : "Same " + from.category;
}

// ─── Utility ───────────────────────────────────────────────────────────────
function escHtml(s) {
  return String(s || "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
