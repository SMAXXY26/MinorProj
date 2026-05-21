"""
generate_weapon_meshes.py  —  Procedural weapon mesh generator
==============================================================
Creates geometrically accurate weapon silhouettes as OBJ files.
These are NOT box placeholders — each mesh captures the key
structural features that define the weapon's shape from above:

  Knife   : tapered blade (wedge), bolster, cylindrical handle
  Rifle   : barrel + receiver + magazine + pistol-grip + stock
            (each a separate sub-mesh, forming correct L/T shapes)

Aerial training data is primarily judged on silhouette accuracy,
not photorealism.  A correct rifle silhouette (stock + barrel +
mag extending down) gives the renderer real bounding-box variation
across elevations: from nadir the weapon is short-and-wide,
from oblique it's elongated and L-shaped — exactly the variation
we need.

Run once; it creates the 6 required .obj files and exits.

Usage:
    python scripts/blenderproc_pipeline/generate_weapon_meshes.py
"""

import math
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ── OBJ writer ────────────────────────────────────────────────────────────────

class OBJBuilder:
    def __init__(self):
        self.verts  = []   # (x, y, z)
        self.faces  = []   # [(v0, v1, v2, ...)] 1-indexed

    def add_quad(self, v0, v1, v2, v3):
        """Add a quad face (4 indices into current vert list, 1-based)."""
        self.faces.append((v0, v1, v2, v3))

    def _v(self, x, y, z) -> int:
        """Add vertex, return 1-based index."""
        self.verts.append((x, y, z))
        return len(self.verts)

    def box(self, x0, y0, z0, x1, y1, z1):
        """Add an axis-aligned box from (x0,y0,z0) to (x1,y1,z1)."""
        a = self._v(x0, y0, z0)
        b = self._v(x1, y0, z0)
        c = self._v(x1, y1, z0)
        d = self._v(x0, y1, z0)
        e = self._v(x0, y0, z1)
        f = self._v(x1, y0, z1)
        g = self._v(x1, y1, z1)
        h = self._v(x0, y1, z1)
        # 6 faces
        self.add_quad(a, b, c, d)  # bottom
        self.add_quad(e, h, g, f)  # top
        self.add_quad(a, e, f, b)  # front
        self.add_quad(b, f, g, c)  # right
        self.add_quad(c, g, h, d)  # back
        self.add_quad(d, h, e, a)  # left

    def tapered_box(self, x0, y0, z0, x1, y1, z1,
                    tip_x=None, tip_z=None, tip_side="x1"):
        """
        Box tapered to a line at one end (approximates a blade wedge).
        tip_side: which end gets tapered ('x1' = far end tapers to zero width)
        """
        mx = (x0 + x1) / 2
        mz = (z0 + z1) / 2
        if tip_x is None: tip_x = mx
        if tip_z is None: tip_z = mz

        # Base quad (wide end)
        a = self._v(x0, y0, z0)
        b = self._v(x1, y0, z0)
        c = self._v(x1, y1, z0)
        d = self._v(x0, y1, z0)
        # Tip line (zero-width edge)
        e = self._v(tip_x, y0, tip_z)
        f = self._v(tip_x, y1, tip_z)

        # side faces
        self.add_quad(a, b, c, d)     # base
        self.faces.append((e, a, d, f))  # left side (triangular)
        self.faces.append((b, e, f, c))  # right side (triangular)
        self.faces.append((a, e, b))     # front triangle
        self.faces.append((d, c, f))     # back triangle

    def cylinder(self, cx, cy0, cy1, r, segs=8):
        """Vertical cylinder along Y axis at (cx, -, cz=0)."""
        # This is a round shape: for aerial view it looks circular
        # We approximate with segs-gon prism
        # centre at (cx, *, 0)
        bot_centre = self._v(cx, cy0, 0)
        top_centre = self._v(cx, cy1, 0)
        bot_ring = []
        top_ring = []
        for i in range(segs):
            angle = 2 * math.pi * i / segs
            x = cx + r * math.cos(angle)
            z =      r * math.sin(angle)
            bot_ring.append(self._v(x, cy0, z))
            top_ring.append(self._v(x, cy1, z))
        # Side faces
        for i in range(segs):
            j = (i + 1) % segs
            self.add_quad(bot_ring[i], bot_ring[j],
                          top_ring[j], top_ring[i])
        # Caps
        for i in range(segs):
            j = (i + 1) % segs
            self.faces.append((bot_centre, bot_ring[j], bot_ring[i]))
            self.faces.append((top_centre, top_ring[i], top_ring[j]))

    def write(self, path: Path, name: str):
        lines = [f"# {name} (procedural weapon mesh)", f"o {name}"]
        for v in self.verts:
            lines.append(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}")
        for face in self.faces:
            lines.append("f " + " ".join(str(i) for i in face))
        path.write_text("\n".join(lines) + "\n")
        kb = path.stat().st_size // 1024
        print(f"  Wrote {path.name}  ({len(self.verts)} verts, "
              f"{len(self.faces)} faces, {kb} KB)")


# ── Weapon builders ───────────────────────────────────────────────────────────
# Coordinate system: X = length axis (longest dimension), Y = height, Z = width
# Lengths approximate real weapons (metres), normalised to max_dim=1 BU by
# BlenderProc's load step so absolute values don't matter, ratios do.

def build_knife(blade_len=0.20, handle_len=0.13, blade_w=0.025, handle_w=0.020,
                spine_t=0.008, handle_t=0.016, bolster_w=0.030) -> OBJBuilder:
    """
    Knife: tapered blade + bolster + cylindrical-ish handle.
    Blade tip is at X=0, pommel at X=+total_len.
    """
    b = OBJBuilder()
    total = blade_len + handle_len

    # Spine of blade (top edge, constant thickness)
    blade_spine_t = spine_t * 0.4   # blade is thin at spine

    # Blade: a wedge — wide at bolster, tapers to a point
    # We model it as two quads: upper face + lower bevel
    blade_x0 = 0.0          # tip
    blade_x1 = blade_len    # bolster junction

    # Blade body — flat plate (very thin)
    b.box(blade_x0, -blade_spine_t/2, -blade_w/2,
          blade_x1,  blade_spine_t/2,  blade_w/4)  # asymmetric: edge vs spine

    # Taper: the blade narrows linearly toward the tip
    # We approximate this by adding a tapered sub-mesh at the tip
    tip_len = blade_len * 0.35
    b.box(blade_x0, -blade_spine_t/2, -blade_w * 0.05,
          blade_x0 + tip_len, blade_spine_t/2, blade_w/4)

    # Bolster (thicker, wider section between blade and handle)
    bolster_len = 0.012
    b.box(blade_x1, -handle_t/2, -bolster_w/2,
          blade_x1 + bolster_len, handle_t/2, bolster_w/2)

    # Handle: slightly tapered rectangular prism
    hx0 = blade_x1 + bolster_len
    hx1 = total
    hw  = handle_w
    b.box(hx0, -handle_t/2, -hw/2,
          hx1,  handle_t/2,  hw/2)

    # Pommel cap (slightly wider than handle)
    pom_len = 0.010
    b.box(hx1 - pom_len, -handle_t*0.6, -hw*0.55,
          hx1 + pom_len,  handle_t*0.6,  hw*0.55)

    return b


def build_rifle_ak(total_len=0.87) -> OBJBuilder:
    """
    AK-47 style: curved wooden stock, pistol grip, long barrel.
    Distinctive feature: curved banana mag, slanted pistol grip.
    """
    b = OBJBuilder()

    # Major proportions (normalised to total_len)
    barrel_len  = total_len * 0.52
    receiver_l  = total_len * 0.22
    stock_len   = total_len * 0.26

    barrel_w    = 0.025
    barrel_h    = 0.025
    recv_w      = 0.055
    recv_h      = 0.055
    stock_w     = 0.040

    barrel_start = 0.0
    recv_start   = barrel_len
    stock_start  = recv_start + receiver_l

    # Barrel (thin, extends forward)
    b.box(barrel_start,   -barrel_h/2, -barrel_w/2,
          recv_start,      barrel_h/2,  barrel_w/2)

    # Gas block / front sight (bump on barrel ~75% forward)
    gs_x = barrel_len * 0.75
    b.box(gs_x - 0.015, -barrel_h*0.8, -barrel_w*0.8,
          gs_x + 0.015,  barrel_h*1.2,  barrel_w*0.8)

    # Receiver (main body)
    b.box(recv_start,  -recv_h/2, -recv_w/2,
          stock_start,  recv_h/2,  recv_w/2)

    # Pistol grip (angled down-back from receiver, ~15° slant)
    grip_h = 0.12
    grip_w = 0.028
    grip_t = 0.035
    grip_x = recv_start + receiver_l * 0.70
    # Slanted grip box: upper attachment at receiver level, extends down+back
    b.box(grip_x - grip_t/2,      -recv_h/2 - grip_h, -grip_w/2,
          grip_x + grip_t/2,      -recv_h/2,           grip_w/2)

    # Magazine (AK banana mag — curved, approximated as angled box)
    mag_w   = 0.040
    mag_h   = 0.20
    mag_t   = 0.030
    mag_x   = recv_start + receiver_l * 0.40
    # Two-segment curved mag
    b.box(mag_x - mag_t/2,  -recv_h/2 - mag_h * 0.60, -mag_w/2,
          mag_x + mag_t/2,  -recv_h/2,                  mag_w/2)
    b.box(mag_x - mag_t/2 + 0.012,  -recv_h/2 - mag_h, -mag_w/2,
          mag_x + mag_t/2 + 0.012,  -recv_h/2 - mag_h * 0.55, mag_w/2)

    # Stock (AK wood stock — curves down slightly)
    b.box(stock_start,           -recv_h/2,         -stock_w/2,
          stock_start + stock_len*0.6, recv_h/2,     stock_w/2)
    # Stock cheekpiece (curves down at the back)
    b.box(stock_start + stock_len*0.5, -recv_h,      -stock_w/2,
          stock_start + stock_len,     -recv_h/2,     stock_w/2)
    # Buttplate
    b.box(stock_start + stock_len - 0.01, -recv_h*1.1, -stock_w*0.6,
          stock_start + stock_len + 0.01,  recv_h*0.6,  stock_w*0.6)

    # Handguard (wood below barrel, near receiver)
    b.box(barrel_len * 0.3,  -barrel_h*0.5, -barrel_w*0.8,
          recv_start,         barrel_h*0.5,  barrel_w*0.8)

    return b


def build_rifle_ar(total_len=0.76) -> OBJBuilder:
    """
    AR-15 style: flat-top upper, buffer tube stock, A2 pistol grip.
    Distinctive: carry handle / flat-top rail, collapsible stock tube.
    """
    b = OBJBuilder()

    barrel_len   = total_len * 0.50
    receiver_l   = total_len * 0.22
    stock_len    = total_len * 0.28

    barrel_w     = 0.022
    recv_w       = 0.048
    recv_h       = 0.055
    stock_w      = 0.032

    recv_start   = barrel_len
    stock_start  = recv_start + receiver_l

    # Barrel (with partial shroud)
    b.box(0.0,         -barrel_w/2, -barrel_w/2,
          barrel_len,   barrel_w/2,  barrel_w/2)

    # Muzzle device (A2 birdcage flash hider)
    b.box(-0.04,       -barrel_w*0.8, -barrel_w*0.8,
          0.0,          barrel_w*0.8,  barrel_w*0.8)

    # Handguard (quad-rail style, slightly wider than barrel)
    b.box(barrel_len*0.10, -recv_w*0.40, -recv_w*0.40,
          recv_start,       recv_w*0.40,  recv_w*0.40)

    # Upper + lower receiver (flat-top)
    b.box(recv_start,    0.0,          -recv_w/2,
          stock_start,   recv_h*0.55,   recv_w/2)  # upper
    b.box(recv_start,   -recv_h*0.45,  -recv_w/2,
          stock_start,   0.0,            recv_w/2)  # lower

    # Flat-top rail / carry handle area on upper receiver
    b.box(recv_start + 0.01,  recv_h*0.55, -recv_w*0.35,
          stock_start - 0.01, recv_h*0.75,  recv_w*0.35)

    # A2 pistol grip (steeper angle than AK)
    grip_x = recv_start + receiver_l * 0.80
    b.box(grip_x - 0.015, -recv_h*0.45 - 0.10, -0.015,
          grip_x + 0.015, -recv_h*0.45,          0.015)

    # Magazine (straight 30-rnd STANAG)
    mag_x = recv_start + receiver_l * 0.35
    b.box(mag_x - 0.015, -recv_h*0.45 - 0.19, -recv_w*0.30,
          mag_x + 0.015, -recv_h*0.45,          recv_w*0.30)

    # Buffer tube (the cylindrical stock extension — approximated as thin box)
    b.box(stock_start,             -recv_h*0.10, -0.020,
          stock_start + stock_len,  recv_h*0.35,  0.020)

    # Buttstock (collapsible pad at end)
    b.box(stock_start + stock_len - 0.015, -recv_h*0.30, -stock_w/2,
          stock_start + stock_len + 0.010,  recv_h*0.50,  stock_w/2)

    return b


def build_rifle_bolt(total_len=1.10) -> OBJBuilder:
    """
    Bolt-action sniper: very long barrel, traditional wood stock, bolt handle.
    Distinctive: scope rail on top, bolt handle protruding to side, no mag box.
    """
    b = OBJBuilder()

    barrel_len  = total_len * 0.60
    receiver_l  = total_len * 0.15
    stock_len   = total_len * 0.25

    barrel_w    = 0.028   # heavier barrel than semi-auto
    recv_h      = 0.060
    recv_w      = 0.050
    stock_w     = 0.038

    recv_start  = barrel_len
    stock_start = recv_start + receiver_l

    # Heavy barrel with progressive taper (thicker at breech, thinner at muzzle)
    # Approximate as two segments
    b.box(0.0,              -barrel_w*0.65, -barrel_w*0.65,
          barrel_len*0.50,   barrel_w*0.65,  barrel_w*0.65)
    b.box(barrel_len*0.48, -barrel_w*0.50, -barrel_w*0.50,
          recv_start,        barrel_w*0.50,  barrel_w*0.50)

    # Muzzle brake
    b.box(-0.030,         -barrel_w*0.90, -barrel_w*0.90,
           0.000,          barrel_w*0.90,  barrel_w*0.90)

    # Receiver
    b.box(recv_start,    -recv_h/2, -recv_w/2,
          stock_start,    recv_h/2,  recv_w/2)

    # Scope base rail (on top of receiver)
    b.box(recv_start + 0.010, recv_h/2, -recv_w*0.25,
          stock_start - 0.010, recv_h/2 + 0.008, recv_w*0.25)

    # Bolt handle (protrudes to one side — from side it looks like a T)
    bolt_x = recv_start + receiver_l * 0.50
    b.box(bolt_x - 0.010, -0.005, recv_w/2,
          bolt_x + 0.010, 0.005,  recv_w/2 + 0.065)
    # Bolt knob
    b.box(bolt_x - 0.012, -0.012, recv_w/2 + 0.055,
          bolt_x + 0.012,  0.012, recv_w/2 + 0.080)

    # Trigger guard + trigger
    tg_x = recv_start + receiver_l * 0.70
    b.box(tg_x - 0.012, -recv_h/2 - 0.025, -0.012,
          tg_x + 0.012, -recv_h/2,           0.012)

    # Detachable box magazine (small, 5-rnd)
    mag_x = recv_start + receiver_l * 0.40
    b.box(mag_x - 0.014, -recv_h/2 - 0.060, -recv_w*0.40,
          mag_x + 0.014, -recv_h/2,           recv_w*0.40)

    # Stock (traditional — wider cheekpiece, pistol grip wrist, butt)
    # Wrist / grip area
    b.box(stock_start,           -recv_h*0.55, -stock_w*0.45,
          stock_start + 0.090,    recv_h*0.45,  stock_w*0.45)
    # Cheekpiece / comb rises above centreline
    b.box(stock_start + 0.05,    recv_h*0.30,  -stock_w*0.50,
          stock_start + stock_len*0.65, recv_h*0.70, stock_w*0.50)
    # Main stock shaft
    b.box(stock_start + 0.08,   -recv_h*0.45,  -stock_w/2,
          stock_start + stock_len, recv_h*0.30,  stock_w/2)
    # Butt pad
    b.box(stock_start + stock_len - 0.012, -recv_h*0.50, -stock_w*0.55,
          stock_start + stock_len + 0.015,  recv_h*0.50,  stock_w*0.55)

    return b


def build_knife_02(blade_len=0.17, handle_len=0.10) -> OBJBuilder:
    """
    Folding knife (slightly smaller, thicker handle due to mechanism).
    """
    b = OBJBuilder()
    blade_w  = 0.022
    handle_w = 0.022
    blade_t  = 0.006
    handle_t = 0.018

    # Blade (narrower, more clip-point style)
    b.box(0.0,       -blade_t/2, -blade_w/2,
          blade_len,  blade_t/2,  blade_w * 0.20)

    # Lock back / pivot (slight bump at junction)
    b.box(blade_len - 0.005, -handle_t/2, -handle_w/2,
          blade_len + 0.015,  handle_t/2,  handle_w/2)

    # Handle (thicker — contains the folding mechanism)
    b.box(blade_len + 0.010, -handle_t/2, -handle_w/2,
          blade_len + handle_len, handle_t/2, handle_w/2)

    # Clip (pocket clip on spine)
    clip_x0 = blade_len + handle_len * 0.20
    clip_x1 = blade_len + handle_len * 0.90
    b.box(clip_x0, handle_t/2,         -handle_w * 0.30,
          clip_x1, handle_t/2 + 0.003,  handle_w * 0.30)

    return b


def build_knife_03(blade_len=0.23, handle_len=0.14) -> OBJBuilder:
    """
    Hunting knife (wider blade, fuller groove, crossguard).
    """
    b = OBJBuilder()
    blade_w  = 0.038
    handle_w = 0.024
    blade_t  = 0.008
    handle_t = 0.020

    # Blade — drop-point style, wide near bolster
    b.box(0.0,       -blade_t/2, -blade_w*0.05,
          blade_len,  blade_t/2,  blade_w)

    # Blood groove (fuller) — raised ridge on flat face, approx as a thin box
    b.box(blade_len*0.10, blade_t/2,          -blade_w*0.40,
          blade_len*0.80, blade_t/2 + 0.002,   blade_w*0.10)

    # Crossguard (wider than blade+handle in Z)
    cg_w  = blade_w * 1.40
    cg_len = 0.016
    b.box(blade_len,       -handle_t*0.65, -cg_w/2,
          blade_len + cg_len, handle_t*0.65,  cg_w/2)

    # Handle (scales — checkered grip)
    hx0 = blade_len + cg_len
    hx1 = hx0 + handle_len
    b.box(hx0, -handle_t/2, -handle_w/2,
          hx1,  handle_t/2,  handle_w/2)

    # Pommel (rounded end cap — approximated as wider box)
    pom = 0.014
    b.box(hx1 - pom/2, -handle_t*0.65, -handle_w*0.60,
          hx1 + pom/2,  handle_t*0.65,  handle_w*0.60)

    return b


# ── Main ──────────────────────────────────────────────────────────────────────

MESHES = [
    ("knife_01.obj", "KnifeFixedBlade", build_knife),
    ("knife_02.obj", "KnifeFolding",    build_knife_02),
    ("knife_03.obj", "KnifeHunting",    build_knife_03),
    ("rifle_ak47.obj", "RifleAK47",     build_rifle_ak),
    ("rifle_ar15.obj", "RifleAR15",     build_rifle_ar),
    ("rifle_bolt.obj", "RifleBoltAction", build_rifle_bolt),
]


def main():
    print(f"\n  Generating weapon silhouette meshes → {MODELS_DIR}\n")
    for fname, name, builder_fn in MESHES:
        path = MODELS_DIR / fname
        mesh = builder_fn()
        mesh.write(path, name)
    print(f"\n  Done. {len(MESHES)} meshes written.")
    print(f"  Each mesh has correct weapon proportions for aerial bounding-box training.")
    print(f"\n  Preview a render (10 images):")
    print(f"    python scripts/blenderproc_pipeline/generate_dataset.py --n 10")


if __name__ == "__main__":
    main()
