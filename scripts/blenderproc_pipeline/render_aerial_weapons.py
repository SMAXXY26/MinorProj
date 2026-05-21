import blenderproc as bproc  # MUST be first import — BlenderProc requirement

"""
render_aerial_weapons.py  —  BlenderProc aerial weapon renderer
================================================================
Renders a single weapon 3D model from randomised drone-perspective camera
angles and outputs YOLO-format labels + RGB images.

This script runs INSIDE BlenderProc's managed Blender environment:
    blenderproc run render_aerial_weapons.py -- \\
        --model    models/pistol.obj \\
        --class-id 1 \\
        --n        500 \\
        --out      ../../data/synthetic

Do NOT run with plain python — it will fail because blenderproc
patches the Python environment at startup.

Camera model
------------
The weapon sits at the world origin on a flat ground plane.
The camera orbits it at:
  - elevation  : 40–90° above the horizon  (40° = steep oblique, 90° = nadir)
  - azimuth    : 0–360° (all yaw angles)
  - distance   : sampled to give a weapon apparent size of 40–200 px in 416px image
                 (matches the size distribution seen in our val set)

Randomisation applied per frame
--------------------------------
  - Camera elevation, azimuth, distance (see above)
  - Weapon yaw about its own vertical axis   (0–360°)
  - Weapon scale jitter                      (±15%)
  - Ground plane material: one of grass / gravel / concrete / sand / tarmac
    selected from textures in scripts/blenderproc_pipeline/textures/
    OR solid colour if no textures available
  - HDRI environment light from scripts/blenderproc_pipeline/hdri/
    OR sun + ambient fallback if no HDRI available
  - Sun azimuth and elevation randomised
  - Slight camera tilt noise (±3°) to break perfect alignment

Output
------
  <out>/images/<class>_XXXXX.jpg   — 416×416 RGB
  <out>/labels/<class>_XXXXX.txt   — YOLO row: class_id cx cy w h (normalised)
"""

import numpy as np
import argparse
import json
import math
import os
import sys
from pathlib import Path

import bpy          # available inside BlenderProc environment
import mathutils    # Euler/Vector/Matrix moved from bpy.types to mathutils in Blender 4.x


# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model",    required=True,  help="Path to .obj/.fbx/.glb weapon model")
parser.add_argument("--class-id", required=True,  type=int,  help="0=knife 1=pistol 2=rifle")
parser.add_argument("--n",        default=500,    type=int,  help="Number of images to render")
parser.add_argument("--out",      default="../../data/synthetic", help="Output root dir")
parser.add_argument("--img-size", default=416,    type=int,  help="Square output resolution")
parser.add_argument("--seed",     default=42,     type=int,  help="RNG seed")
# BlenderProc may or may not strip the "--" separator from sys.argv:
# - When "--" is present: raw Blender invocation — args are after "--"
# - When "--" is absent:  BlenderProc already stripped it — args are sys.argv[1:]
if "--" in sys.argv:
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
else:
    args = parser.parse_args(sys.argv[1:])

CLASS_NAMES = {0: "knife", 1: "pistol", 2: "rifle"}
CLASS_NAME  = CLASS_NAMES[args.class_id]
IMG_SIZE    = args.img_size
OUT_ROOT    = Path(args.out)
IMG_DIR     = OUT_ROOT / "images"
LBL_DIR     = OUT_ROOT / "labels"
IMG_DIR.mkdir(parents=True, exist_ok=True)
LBL_DIR.mkdir(parents=True, exist_ok=True)

SCRIPT_DIR  = Path(__file__).resolve().parent
HDRI_DIR    = SCRIPT_DIR / "hdri"
TEX_DIR     = SCRIPT_DIR / "textures"

rng = np.random.default_rng(args.seed)


# =============================================================================
#  Helpers
# =============================================================================

def _count_existing() -> int:
    return len(list(IMG_DIR.glob(f"{CLASS_NAME}_*.jpg")))


def _get_hdri_paths():
    if HDRI_DIR.exists():
        return sorted(HDRI_DIR.glob("*.hdr")) + sorted(HDRI_DIR.glob("*.exr"))
    return []


def _get_texture_paths():
    if TEX_DIR.exists():
        return sorted(TEX_DIR.glob("*.jpg")) + sorted(TEX_DIR.glob("*.png"))
    return []


def _set_ground_material(obj, tex_paths):
    """Apply a random ground texture or a flat earthy colour."""
    mat = bpy.data.materials.new(name="GroundMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]

    if tex_paths:
        tex_path = str(rng.choice(tex_paths))
        tex_node = mat.node_tree.nodes.new("ShaderNodeTexImage")
        tex_node.image = bpy.data.images.load(tex_path)
        # Tile to cover 20m² ground plane
        mapping = mat.node_tree.nodes.new("ShaderNodeMapping")
        mapping.inputs["Scale"].default_value = (4, 4, 4)
        coord = mat.node_tree.nodes.new("ShaderNodeTexCoord")
        mat.node_tree.links.new(coord.outputs["UV"], mapping.inputs["Vector"])
        mat.node_tree.links.new(mapping.outputs["Vector"], tex_node.inputs["Vector"])
        mat.node_tree.links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        # Random earthy tone
        r = rng.uniform(0.25, 0.60)
        g = rng.uniform(0.22, 0.55)
        b = rng.uniform(0.10, 0.35)
        bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)

    bsdf.inputs["Roughness"].default_value = rng.uniform(0.7, 1.0)
    # "Specular" was renamed to "Specular IOR Level" in Blender 4.x
    spec_key = "Specular IOR Level" if "Specular IOR Level" in bsdf.inputs else "Specular"
    bsdf.inputs[spec_key].default_value = rng.uniform(0.0, 0.1)

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def _get_weapon_bbox_px(scene, cam_obj, weapon_obj, img_size):
    """
    Project the weapon object's bounding box corners into 2D camera space.
    Returns (cx, cy, w, h) in normalised [0,1] coords, or None if off-screen.
    """
    scene.camera = cam_obj

    from bpy_extras.object_utils import world_to_camera_view
    xs, ys = [], []
    # weapon_obj.bound_box gives 8 local-space corner tuples
    for corner_local in weapon_obj.bound_box:
        corner_world = weapon_obj.matrix_world @ mathutils.Vector(corner_local)
        co_2d = world_to_camera_view(scene, cam_obj, corner_world)
        xs.append(co_2d.x)
        ys.append(1.0 - co_2d.y)  # flip Y (Blender Y is bottom-up)

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Clip to [0,1] — if entirely outside, skip
    if x_max < 0 or x_min > 1 or y_max < 0 or y_min > 1:
        return None

    x_min = max(0.0, x_min)
    x_max = min(1.0, x_max)
    y_min = max(0.0, y_min)
    y_max = min(1.0, y_max)

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    w  = x_max - x_min
    h  = y_max - y_min

    # Reject if weapon is too small (<8px) or implausibly large (>90% of frame)
    if w * img_size < 8 or h * img_size < 8:
        return None
    if w > 0.9 or h > 0.9:
        return None

    return cx, cy, w, h


# =============================================================================
#  Main render loop
# =============================================================================

def main():
    bproc.init()

    # ── Load weapon model ─────────────────────────────────────────────────────
    model_path = str(Path(args.model).resolve())
    print(f"  Loading model: {model_path}")
    loaded = bproc.loader.load_obj(model_path) if model_path.endswith(".obj") else \
             bproc.loader.load_blend(model_path) if model_path.endswith(".blend") else \
             bproc.loader.load_obj(model_path)  # fallback

    if not loaded:
        raise RuntimeError(f"Failed to load model: {model_path}")

    weapon = loaded[0]
    weapon.set_cp("category_id", args.class_id)

    # Normalise weapon scale so its longest dimension = 1 Blender unit
    dims   = weapon.blender_obj.dimensions
    max_dim = max(dims.x, dims.y, dims.z)
    if max_dim > 0:
        scale = 1.0 / max_dim
        weapon.blender_obj.scale = (scale, scale, scale)
    bpy.context.view_layer.update()

    # Centre weapon at origin
    weapon.blender_obj.location = (0, 0, 0)

    # ── Ground plane ─────────────────────────────────────────────────────────
    bpy.ops.mesh.primitive_plane_add(size=40)
    ground = bpy.context.active_object
    ground.location = (0, 0, -0.01)   # just below weapon origin
    tex_paths = _get_texture_paths()
    _set_ground_material(ground, tex_paths)

    # ── Camera ────────────────────────────────────────────────────────────────
    bpy.ops.object.camera_add()
    cam_obj  = bpy.context.active_object
    cam_data = cam_obj.data
    cam_data.lens_unit = "FOV"
    cam_data.angle     = math.radians(70)  # ~70° HFOV — typical drone camera

    bpy.context.scene.camera = cam_obj
    bproc.camera.set_resolution(IMG_SIZE, IMG_SIZE)

    # ── Lighting ──────────────────────────────────────────────────────────────
    hdri_paths = _get_hdri_paths()
    if hdri_paths:
        hdri = str(rng.choice(hdri_paths))
        bproc.world.set_world_background_hdr_img(hdri)
    else:
        # Fallback: sun + ambient
        bpy.ops.object.light_add(type="SUN")
        sun = bpy.context.active_object
        sun.data.energy = rng.uniform(2.0, 5.0)

    # ── Render settings ───────────────────────────────────────────────────────
    bproc.renderer.set_output_format("PNG")
    bproc.renderer.set_max_amount_of_samples(32)   # fast; raise for quality

    scene = bpy.context.scene

    # ── Render loop ───────────────────────────────────────────────────────────
    start_idx = _count_existing()
    n_success = 0
    n_attempt = 0
    MAX_ATTEMPTS = args.n * 3   # allow 3× attempts for rejected frames

    print(f"\n  Rendering {args.n} images of class={CLASS_NAME} ({args.class_id})")
    print(f"  Output → {OUT_ROOT}")
    print(f"  Existing: {start_idx} images already in output dir")

    while n_success < args.n and n_attempt < MAX_ATTEMPTS:
        n_attempt += 1

        # ── Randomise weapon pose ─────────────────────────────────────────────
        weapon_yaw   = rng.uniform(0, 360)
        weapon_scale = rng.uniform(0.85, 1.15)
        weapon.blender_obj.rotation_euler = (0, 0, math.radians(weapon_yaw))
        s = scale * weapon_scale
        weapon.blender_obj.scale = (s, s, s)

        # ── Randomise camera position ─────────────────────────────────────────
        elevation_deg = rng.uniform(40, 90)       # 40° oblique → 90° nadir
        azimuth_deg   = rng.uniform(0, 360)

        # Distance chosen so weapon subtends 40–200px in 416px frame
        # weapon is ~1 BU long; pixel_size ≈ 1 / (img_size / (2*tan(hfov/2)))
        target_px   = rng.uniform(40, 200)
        px_per_bu   = IMG_SIZE / (2 * math.tan(math.radians(70) / 2))
        # At nadir the apparent size scales with cos(elevation)
        dist = px_per_bu / target_px  # simplification; close enough
        dist = max(dist, 0.5)

        el_rad = math.radians(elevation_deg)
        az_rad = math.radians(azimuth_deg)

        cam_x = dist * math.cos(el_rad) * math.cos(az_rad)
        cam_y = dist * math.cos(el_rad) * math.sin(az_rad)
        cam_z = dist * math.sin(el_rad)

        cam_obj.location = (cam_x, cam_y, cam_z)

        # Point camera at origin (weapon centre) + tiny tilt noise
        dx = -cam_x + rng.uniform(-0.05, 0.05)
        dy = -cam_y + rng.uniform(-0.05, 0.05)
        dz = -cam_z
        cam_obj.rotation_euler = mathutils.Euler(
            (math.atan2(math.sqrt(dx**2 + dy**2), -dz),
             0,
             math.atan2(dy, dx) + math.pi / 2),
            "XYZ",
        )

        # ── Randomise lighting per frame ──────────────────────────────────────
        if not hdri_paths:
            sun.rotation_euler = (
                math.radians(rng.uniform(30, 80)),
                0,
                math.radians(rng.uniform(0, 360)),
            )
            sun.data.energy = rng.uniform(1.5, 6.0)

        bpy.context.view_layer.update()

        # ── Compute 2D bounding box ───────────────────────────────────────────
        bbox = _get_weapon_bbox_px(scene, cam_obj, weapon.blender_obj, IMG_SIZE)
        if bbox is None:
            continue   # weapon off-screen or too small — try again

        cx, cy, w, h = bbox

        # ── Render ────────────────────────────────────────────────────────────
        img_idx  = start_idx + n_success
        img_name = f"{CLASS_NAME}_{img_idx:06d}"
        tmp_path = str(OUT_ROOT / f"_tmp_{img_name}")

        scene.render.filepath = tmp_path
        bpy.ops.render.render(write_still=True)

        # Move rendered PNG to final location and convert to JPEG
        tmp_png = Path(tmp_path + ".png")
        if not tmp_png.exists():
            continue

        import PIL.Image
        img = PIL.Image.open(tmp_png).convert("RGB")
        jpg_path = IMG_DIR / f"{img_name}.jpg"
        img.save(jpg_path, "JPEG", quality=92)
        tmp_png.unlink()

        # ── Write YOLO label ──────────────────────────────────────────────────
        lbl_path = LBL_DIR / f"{img_name}.txt"
        lbl_path.write_text(f"{args.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        n_success += 1
        if n_success % 50 == 0 or n_success == args.n:
            print(f"  [{n_success}/{args.n}]  attempts={n_attempt}  "
                  f"last: elev={elevation_deg:.0f}° az={azimuth_deg:.0f}°  "
                  f"bbox=({cx:.2f},{cy:.2f},{w:.2f},{h:.2f})")

    print(f"\n  Done. {n_success} images rendered to {OUT_ROOT}")
    if n_success < args.n:
        print(f"  WARNING: only {n_success}/{args.n} succeeded after {n_attempt} attempts.")
        print(f"  Increase --n or check that the model has visible geometry.")


if __name__ == "__main__":
    main()
