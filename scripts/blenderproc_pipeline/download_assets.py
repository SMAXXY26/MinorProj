"""
download_assets.py  —  Download free CC0 HDRI + ground textures for rendering
==============================================================================
Downloads:
  - 5 outdoor HDRI environment maps from Poly Haven (CC0)
  - 5 ground textures (grass, gravel, concrete, sand, tarmac) from Poly Haven (CC0)

These are placed in:
  scripts/blenderproc_pipeline/hdri/       ← .exr files for environment lighting
  scripts/blenderproc_pipeline/textures/   ← .jpg files for ground plane material

Usage:
    python scripts/blenderproc_pipeline/download_assets.py

Poly Haven is CC0 — no attribution required, free for any use including training data.
"""

import urllib.request
import sys
from pathlib import Path

SCRIPT_DIR  = Path(__file__).resolve().parent
HDRI_DIR    = SCRIPT_DIR / "hdri"
TEX_DIR     = SCRIPT_DIR / "textures"
HDRI_DIR.mkdir(exist_ok=True)
TEX_DIR.mkdir(exist_ok=True)

# ── Poly Haven HDRI — outdoor environments, 1K resolution (fast download) ────
# All CC0. 1K is enough for background lighting in our renders.
HDRI_ASSETS = [
    ("sunflowers_puresky_1k.exr",
     "https://dl.polyhaven.org/file/ph-assets/HDRIs/exr/1k/sunflowers_puresky_1k.exr"),
    ("kloofendal_48d_partly_cloudy_1k.exr",
     "https://dl.polyhaven.org/file/ph-assets/HDRIs/exr/1k/kloofendal_48d_partly_cloudy_1k.exr"),
    ("wasteland_clouds_1k.exr",
     "https://dl.polyhaven.org/file/ph-assets/HDRIs/exr/1k/wasteland_clouds_1k.exr"),
    ("overcast_soil_1k.exr",
     "https://dl.polyhaven.org/file/ph-assets/HDRIs/exr/1k/overcast_soil_1k.exr"),
    ("evening_road_01_1k.exr",
     "https://dl.polyhaven.org/file/ph-assets/HDRIs/exr/1k/evening_road_01_1k.exr"),
]

# ── Poly Haven ground textures — 1K diffuse maps ──────────────────────────────
TEXTURE_ASSETS = [
    ("grass_meadow_diff_1k.jpg",
     "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/brown_mud_leaves_01/brown_mud_leaves_01_diff_1k.jpg"),
    ("gravel_concrete_diff_1k.jpg",
     "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/gravel_concrete/gravel_concrete_diff_1k.jpg"),
    ("concrete_floor_02_diff_1k.jpg",
     "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/concrete_floor_02/concrete_floor_02_diff_1k.jpg"),
    ("sandy_gravel_02_diff_1k.jpg",
     "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/sandy_gravel_02/sandy_gravel_02_diff_1k.jpg"),
    ("asphalt_02_diff_1k.jpg",
     "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/asphalt_02/asphalt_02_diff_1k.jpg"),
]


def download(url: str, dest: Path):
    if dest.exists():
        print(f"  [cached] {dest.name}")
        return True
    print(f"  Downloading {dest.name} ...", end="", flush=True)
    try:
        def _prog(block, block_size, total):
            if total > 0:
                pct = min(block * block_size / total * 100, 100)
                print(f"\r  Downloading {dest.name} ... {pct:.0f}%", end="", flush=True)
        urllib.request.urlretrieve(url, dest, reporthook=_prog)
        size_mb = dest.stat().st_size / 1e6
        print(f"\r  Downloaded  {dest.name}  ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"\r  FAILED: {dest.name} — {e}")
        if dest.exists():
            dest.unlink()
        return False


def main():
    print("\n  Downloading HDRI environment maps (Poly Haven CC0)...")
    for fname, url in HDRI_ASSETS:
        download(url, HDRI_DIR / fname)

    print("\n  Downloading ground textures (Poly Haven CC0)...")
    for fname, url in TEXTURE_ASSETS:
        download(url, TEX_DIR / fname)

    print(f"\n  HDRIs   → {HDRI_DIR}  ({len(list(HDRI_DIR.iterdir()))} files)")
    print(f"  Textures→ {TEX_DIR}  ({len(list(TEX_DIR.iterdir()))} files)")
    print("\n  Done. Now download your 3D weapon models and place them in:")
    print(f"  {SCRIPT_DIR / 'models'}/")
    print("\n  Free CC0 weapon 3D models:")
    print("    Sketchfab (filter: Free + CC):  https://sketchfab.com/search?q=pistol&licenses=322a749bcfa841b29dff1e8a1bb74b0b&type=models")
    print("    Poly Haven props (some weapons): https://polyhaven.com/models")
    print("\n  Recommended files to download:")
    print("    pistol.obj  → place in models/pistol.obj")
    print("    rifle.obj   → place in models/rifle.obj")
    print("    knife.obj   → place in models/knife.obj")


if __name__ == "__main__":
    main()
