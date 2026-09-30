"""
Older, simpler cousin of ../extract_notebook_graphs.py.

Superseded — use the top-level extractor for anything new. This one has no
curated title catalog and no git support, so everything it produces comes out
auto-named. It survives only because build_new_manifest.py consumes the
extracted_jupyter_metadata.json sidecar it writes.

Same caveat as the other legacy script here: SOURCE_DIR is a hardcoded Windows
path from the original author's machine and needs changing before this runs.
"""

import base64
import json
import sys
from pathlib import Path

import nbformat
from PIL import Image

REPO_ROOT = Path(__file__).parent.parent
SOURCE_DIR = Path(r"C:\Users\legoc\school\research\Data pre-processing html")
GRAPHS_DIR = REPO_ROOT.parent / "figures"
THUMB_WIDTH = 512

NOTEBOOKS = [
    {
        "file": "curated_mimic_iii_analysis_executed copy 2.ipynb",
        "pipeline": "MIMIC_Extract",
        "dataset": "mimic-iii",
        "notebook": "curated_mimic_iii_analysis",
    },
    {
        "file": "Experiment_Statistics.ipynb",
        "pipeline": "MIMIC_Extract",
        "dataset": "mimic-iii",
        "notebook": "experiment_statistics",
    },
    {
        "file": "mimic_extract_analysis.ipynb",
        "pipeline": "MIMIC_Extract",
        "dataset": "mimic-iii",
        "notebook": "mimic_extract_analysis",
    },
    {
        "file": "pipeline_comparison.ipynb",
        "pipeline": "Comparison",
        "dataset": "mimic-iii",
        "notebook": "pipeline_comparison",
    },
]


def make_thumbnail(src_path: Path, thumb_path: Path) -> None:
    thumb_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as img:
        w, h = img.size
        new_h = int(h * THUMB_WIDTH / w)
        thumb = img.resize((THUMB_WIDTH, new_h), Image.LANCZOS)
        thumb.save(thumb_path, "PNG", optimize=True)


def extract_notebook(cfg: dict) -> list[dict]:
    nb_path = SOURCE_DIR / cfg["file"]
    if not nb_path.exists():
        print(f"  MISSING: {nb_path}", file=sys.stderr)
        return []

    nb = nbformat.read(str(nb_path), as_version=4)
    pipeline = cfg["pipeline"]
    dataset = cfg["dataset"]
    notebook = cfg["notebook"]

    out_dir = GRAPHS_DIR / pipeline / notebook
    thumb_dir = out_dir / "thumbs"
    out_dir.mkdir(parents=True, exist_ok=True)
    thumb_dir.mkdir(parents=True, exist_ok=True)

    records = []
    img_index = 0

    for cell_idx, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            png_data = None
            if output.get("output_type") in ("display_data", "execute_result"):
                data = output.get("data", {})
                if "image/png" in data:
                    png_data = data["image/png"]
            elif output.get("output_type") == "stream":
                continue

            if png_data is None:
                continue

            # png_data may include line breaks from nbformat encoding
            png_bytes = base64.b64decode(png_data.replace("\n", ""))

            slug = f"{notebook}_{img_index:04d}"
            png_path = out_dir / f"{slug}.png"
            thumb_path = thumb_dir / f"{slug}_thumb.png"

            png_path.write_bytes(png_bytes)
            make_thumbnail(png_path, thumb_path)

            records.append({
                "id": slug,
                "pipeline": pipeline,
                "dataset": dataset,
                "notebook": notebook,
                "cell_index": cell_idx,
                "output_index": img_index,
                "file": f"graphs/{pipeline}/{notebook}/{slug}.png",
                "thumb": f"graphs/{pipeline}/{notebook}/thumbs/{slug}_thumb.png",
            })

            img_index += 1

    print(f"  {notebook}: {img_index} images extracted")
    return records


def main():
    all_records = []
    for cfg in NOTEBOOKS:
        print(f"Processing {cfg['file']} ...")
        records = extract_notebook(cfg)
        all_records.extend(records)

    metadata_path = Path(__file__).parent / "extracted_jupyter_metadata.json"
    metadata_path.write_text(json.dumps(all_records, indent=2))
    print(f"\nTotal: {len(all_records)} images")
    print(f"Metadata written to {metadata_path}")


if __name__ == "__main__":
    main()
