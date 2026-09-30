"""
One-off scraper for the self-contained Marimo WASM HTML exports.

Marimo's "export to HTML" bundles every cell output as a base64 PNG inside a
__MARIMO_MOUNT_CONFIG__ blob of JavaScript. There's no manifest, no ordering
metadata, no titles — just images buried in a JS object, so the approach here is
frankly crude: regex the whole file for anything that looks like a PNG and write
them out numbered. Figures come out in document order, which is the only thing
we get for free.

Superseded by render_marimo_mimic_iii.py, which drives the plotting functions
directly and produces real metadata. Kept because the HTML exports cover eICU
and MIMIC-IV, which the renderer doesn't.

Heads up before running: SOURCE_DIR is a hardcoded Windows path pointing at the
original author's machine. You'll need to change it.
"""

import base64
import re
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).parent.parent
SOURCE_DIR = Path(r"C:\Users\legoc\school\research\Data pre-processing html")
GRAPHS_DIR = REPO_ROOT.parent / "figures"
THUMB_WIDTH = 512

HTML_FILES = [
    ("eicu_analysis.html",                "eicu"),
    ("mimic_iii_analysis_uptpdate.html",  "mimic-iii"),
    ("mimic_iv_analysis.html",            "mimic-iv"),
]

PIPELINE = "EHR-Dataset-Processing"

# Every PNG starts with the same magic bytes, which base64-encode to "iVBOR".
# Grabbing that plus every base64-legal character after it means we never have to
# understand the surrounding JS quoting, which is the only reason this approach
# is viable at all. It does mean a stray "iVBOR" in the JS would produce a
# garbage image — safe_b64decode catches most of that.
PNG_B64_RE = re.compile(r"iVBOR([A-Za-z0-9+/=]*)")


def safe_b64decode(b64str: str) -> bytes | None:
    """Decode, re-padding first, and give up quietly rather than raise.

    The regex above stops at the first non-base64 character, which can land
    mid-quartet and leave a string Python refuses to decode. Re-padding fixes the
    legitimate cases; the ones that still fail are junk matches we want to drop.
    """
    pad = len(b64str) % 4
    if pad:
        b64str += "=" * (4 - pad)
    try:
        return base64.b64decode(b64str)
    except Exception:
        return None


def make_thumbnail(src_path: Path, thumb_path: Path) -> None:
    thumb_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as img:
        w, h = img.size
        new_h = int(h * THUMB_WIDTH / w)
        thumb = img.resize((THUMB_WIDTH, new_h), Image.LANCZOS)
        thumb.save(thumb_path, "PNG", optimize=True)


def extract_html(html_file: str, dataset: str) -> int:
    src = SOURCE_DIR / html_file
    if not src.exists():
        print(f"  MISSING: {src}", file=sys.stderr)
        return 0

    out_dir = GRAPHS_DIR / PIPELINE / dataset
    thumb_dir = out_dir / "thumbs"
    out_dir.mkdir(parents=True, exist_ok=True)
    thumb_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Reading {html_file} ({src.stat().st_size / 1e6:.1f} MB)...")
    with open(src, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # `saved` numbers the files, not `i` — so a failed decode doesn't leave a gap
    # in the sequence. These exports run 50-200MB, hence reading the whole thing
    # into memory being worth a mention; it's fine on a workstation, less so on a
    # login node.
    raw_matches = PNG_B64_RE.findall(content)
    saved = 0
    for i, suffix in enumerate(raw_matches):
        png_bytes = safe_b64decode("iVBOR" + suffix)
        if png_bytes is None:
            print(f"    Warning: skipping image {i} (decode error)")
            continue

        slug = f"{dataset.replace('-', '_')}_{saved:04d}"
        png_path = out_dir / f"{slug}.png"
        thumb_path = thumb_dir / f"{slug}_thumb.png"

        png_path.write_bytes(png_bytes)
        make_thumbnail(png_path, thumb_path)
        saved += 1

    return saved


def main():
    total = 0
    for html_file, dataset in HTML_FILES:
        print(f"Processing {html_file} -> {dataset}...")
        n = extract_html(html_file, dataset)
        print(f"  {dataset}: {n} images saved")
        total += n
    print(f"\nTotal: {total} Marimo images extracted")


if __name__ == "__main__":
    main()
