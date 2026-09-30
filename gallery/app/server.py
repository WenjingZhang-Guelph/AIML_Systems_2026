#!/usr/bin/env python3
"""
Flask server for the Graph Gallery. Deliberately thin — all the real work
(figure extraction, metadata, pairing) happens offline in build_manifest.py, and
this just hands the resulting JSON and PNGs to the frontend, which does the
filtering client-side.

Two things worth knowing. The static route is remapped to /app-static so it
doesn't collide with the figures served under /graphs. And tags.json is the only
mutable state in the whole app: POST /api/tags replaces it wholesale, no merge,
no locking, so two browser tabs editing tags will happily clobber each other.
Fine for a single-user research tool, not fine if this ever gets shared.
"""

import json
import os
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

GALLERY_ROOT = Path(__file__).parent.parent.resolve()
GRAPHS_DIR = GALLERY_ROOT.parent / "figures"
MANIFEST_PATH = GALLERY_ROOT / "graphs_manifest.json"
TAGS_PATH = GALLERY_ROOT / "tags.json"

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static"),
    static_url_path="/app-static",
)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/manifest")
def manifest():
    if not MANIFEST_PATH.exists():
        abort(503, "graphs_manifest.json not found — run build_manifest.py first")
    with open(MANIFEST_PATH) as f:
        data = json.load(f)
    return jsonify(data)


@app.route("/api/tags", methods=["GET"])
def get_tags():
    # Empty shape rather than a 404 — the frontend treats "no tags yet" and "tags
    # exist" identically, so there's nothing useful for it to do with an error.
    if not TAGS_PATH.exists():
        return jsonify({"tags": [], "assignments": {}})
    with open(TAGS_PATH) as f:
        return jsonify(json.load(f))


@app.route("/api/tags", methods=["POST"])
def save_tags():
    data = request.get_json(force=True)
    if not isinstance(data, dict) or "tags" not in data or "assignments" not in data:
        abort(400, "Invalid tags payload")
    with open(TAGS_PATH, "w") as f:
        json.dump(data, f, indent=2)
    return jsonify({"ok": True})


@app.route("/graphs/<path:filepath>")
def serve_graph(filepath):
    # figures/ lives outside the Flask app dir (it's shared with the pipelines and
    # is ~845MB), so it can't be a normal static folder. send_from_directory does
    # the path-traversal check for us, which matters since filepath comes
    # straight from the manifest.
    return send_from_directory(str(GRAPHS_DIR), filepath)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
