"""
download_models.py  —  Download CC0 weapon .obj models for BlenderProc rendering
=================================================================================
Downloads placeholder/proxy weapon meshes from publicly accessible CC0 / public-
domain sources that work without browser login.

Strategy
--------
1. Try direct OBJ downloads from sources with programmatic access.
2. For models that require a Sketchfab account, print a clear manual instruction.

After this script finishes you should have at least the rifle models ready to
render.  The knife models are hardest to find as CC0 OBJs — fallback primitives
are generated procedurally if nothing downloads.

Usage:
    python scripts/blenderproc_pipeline/download_models.py

Output:
    scripts/blenderproc_pipeline/models/knife_01.obj   (or fallback primitive)
    scripts/blenderproc_pipeline/models/knife_02.obj
    scripts/blenderproc_pipeline/models/knife_03.obj
    scripts/blenderproc_pipeline/models/rifle_ak47.obj
    scripts/blenderproc_pipeline/models/rifle_ar15.obj
    scripts/blenderproc_pipeline/models/rifle_bolt.obj
"""

import sys
import urllib.request
import zipfile
import tarfile
import io
import os
import shutil
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ── Download helper ───────────────────────────────────────────────────────────

def download_bytes(url: str, label: str) -> bytes | None:
    print(f"  Downloading {label} ...", end="", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        print(f" {len(data)//1024} KB")
        return data
    except Exception as e:
        print(f" FAILED ({e})")
        return None


def save(data: bytes, path: Path):
    path.write_bytes(data)
    print(f"    → saved {path.name}  ({path.stat().st_size//1024} KB)")


# ── Procedural fallback OBJ generator ────────────────────────────────────────
# These are *extremely* simple meshes — a stretched box for a rifle and a thin
# elongated box for a knife.  They give BlenderProc something to project a
# bounding box from, and the renders will look bad, but the label geometry is
# correct.  Replace with real meshes ASAP.

def _box_obj(sx: float, sy: float, sz: float, name: str) -> str:
    """Generate a unit-cube scaled to (sx, sy, sz), centred at origin."""
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    verts = [
        (-hx, -hy, -hz), ( hx, -hy, -hz), ( hx,  hy, -hz), (-hx,  hy, -hz),
        (-hx, -hy,  hz), ( hx, -hy,  hz), ( hx,  hy,  hz), (-hx,  hy,  hz),
    ]
    faces = [
        (1,2,3,4), (5,8,7,6), (1,5,6,2),
        (2,6,7,3), (3,7,8,4), (4,8,5,1),
    ]
    lines = [f"# {name} (procedural fallback)", f"o {name}"]
    for v in verts:
        lines.append(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}")
    for f in faces:
        lines.append("f " + " ".join(str(i) for i in f))
    return "\n".join(lines) + "\n"


def write_fallback(path: Path, sx: float, sy: float, sz: float):
    name = path.stem
    path.write_text(_box_obj(sx, sy, sz, name))
    print(f"  [FALLBACK] Written procedural box mesh → {path.name}")
    print(f"             Replace with a real .obj for better render quality.")


# ── Model definitions ─────────────────────────────────────────────────────────
# Each entry: (dest_filename, [(try_url, inner_path_in_archive_or_None), ...])
# inner_path=None means the URL is a raw .obj file.
# inner_path="auto" means look for any .obj inside the zip/tar.

# We use Poly Haven's free prop models where available.
# Poly Haven does not host weapon models, so we fall through to procedural.
# The NASA / Smithsonian 3D programmes host some props as GLB/OBJ CC0.

MODELS = [
    # ── Knives ────────────────────────────────────────────────────────────────
    # No free-to-download-programmatically CC0 knife OBJs found without auth.
    # The script falls through to procedural fallbacks for all three knives
    # unless you add URLs here.
    {
        "dest":  "knife_01.obj",
        "label": "knife (fixed-blade)",
        "urls":  [],   # add download URLs here if you find CC0 sources
        "fallback_dims": (0.25, 0.03, 0.02),  # approx 25cm blade
    },
    {
        "dest":  "knife_02.obj",
        "label": "knife (folding / slightly smaller)",
        "urls":  [],
        "fallback_dims": (0.20, 0.025, 0.015),
    },
    {
        "dest":  "knife_03.obj",
        "label": "knife (hunting, wider blade)",
        "urls":  [],
        "fallback_dims": (0.28, 0.04, 0.02),
    },

    # ── Rifles ────────────────────────────────────────────────────────────────
    # Similarly, CC0 rifle OBJs without auth are rare.  Procedural fallbacks
    # are elongated boxes that approximate long-gun proportions.
    {
        "dest":  "rifle_ak47.obj",
        "label": "AK-style rifle",
        "urls":  [],
        "fallback_dims": (0.87, 0.06, 0.18),  # ~87cm long
    },
    {
        "dest":  "rifle_ar15.obj",
        "label": "AR-style rifle (shorter)",
        "urls":  [],
        "fallback_dims": (0.76, 0.055, 0.16),
    },
    {
        "dest":  "rifle_bolt.obj",
        "label": "bolt-action rifle (long)",
        "urls":  [],
        "fallback_dims": (1.10, 0.05, 0.14),
    },
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n  Weapon 3D model downloader")
    print(f"  Target dir: {MODELS_DIR}\n")

    n_downloaded = 0
    n_fallback   = 0
    n_skipped    = 0

    for m in MODELS:
        dest = MODELS_DIR / m["dest"]
        if dest.exists():
            print(f"  [cached] {dest.name}")
            n_skipped += 1
            continue

        print(f"\n  {m['label']} → {dest.name}")
        downloaded = False

        for url, inner in m.get("urls", []):
            data = download_bytes(url, url.split("/")[-1])
            if data is None:
                continue

            # Raw OBJ
            if inner is None:
                save(data, dest)
                downloaded = True
                break

            # ZIP archive
            if url.endswith(".zip") or data[:2] == b"PK":
                try:
                    with zipfile.ZipFile(io.BytesIO(data)) as zf:
                        if inner == "auto":
                            obj_names = [n for n in zf.namelist()
                                         if n.lower().endswith(".obj")]
                            if not obj_names:
                                print("    No .obj inside archive")
                                continue
                            inner = obj_names[0]
                        with zf.open(inner) as f:
                            save(f.read(), dest)
                    downloaded = True
                    break
                except Exception as e:
                    print(f"    ZIP extract failed: {e}")

        if not downloaded:
            sx, sy, sz = m["fallback_dims"]
            write_fallback(dest, sx, sy, sz)
            n_fallback += 1
        else:
            n_downloaded += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Downloaded : {n_downloaded}")
    print(f"  Fallbacks  : {n_fallback}  (procedural box meshes)")
    print(f"  Cached     : {n_skipped}")
    total = len(list(MODELS_DIR.glob("*.obj")))
    print(f"  Total OBJs : {total} / {len(MODELS)}")
    print(f"{'─'*60}")

    if n_fallback > 0:
        print(f"""
  IMPORTANT — Replace fallback meshes with real weapon models.
  ─────────────────────────────────────────────────────────────
  The procedural boxes give correct bounding-box labels but
  renders will look unrealistic.  For better synthetic data:

  Option A — Sketchfab (best quality, free CC0 filter):
    1. Go to https://sketchfab.com
    2. Search: "knife" or "rifle"
    3. Filter: Free license → CC Attribution or CC0
    4. Download → OBJ format
    5. Rename & place in:
         scripts/blenderproc_pipeline/models/

  Option B — Smithsonian 3D (all CC0, good quality):
    https://3d.si.edu/collections  (search "knife", "rifle")
    Download GLB → convert with:
      blenderproc run -  -- (or use Blender GUI: File→Import→glTF)

  Option C — CGTrader free section:
    https://www.cgtrader.com/free-3d-models/military/gun
    Filter: Free → OBJ format

  Required filenames:
    knife_01.obj  knife_02.obj  knife_03.obj
    rifle_ak47.obj  rifle_ar15.obj  rifle_bolt.obj

  You can run the pipeline with fallback meshes for a quick
  label-correctness smoke test, then swap in real models.
""")

    print(f"  Run the pipeline:")
    print(f"    python scripts/blenderproc_pipeline/generate_dataset.py --n 10 --dry-run")
    print(f"    python scripts/blenderproc_pipeline/generate_dataset.py --n 50   # smoke test")
    print(f"    python scripts/blenderproc_pipeline/generate_dataset.py          # full 4,500\n")


if __name__ == "__main__":
    main()
