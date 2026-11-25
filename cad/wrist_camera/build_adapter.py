"""Wrist-camera adapter for tablet_tap: raises the D435 off the kit's 20-degree wrist mount, with a lockable tilt.

    python build_adapter.py <out_dir> [tilt_deg]        (needs cadquery-ocp, trimesh, numpy; tilt only for the check)

The kit mount stays on the arm: the base is screwed on where the camera was (two M3, 45 mm apart), the cradle carries
the camera. Two M4 screws hold the tilt: one is the pivot, the other runs in a slot; both tight = fixed.
Frame (mm) of the base: the D435's own frame on the kit mount. X = its left, Y = up, Z = optical axis, origin = middle
of the front glass. Tilt = optical axis below the gripper axis, 20 on the kit mount.
"""
import pathlib
import sys

import numpy as np
import trimesh
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_Transform
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf
from OCP.GProp import GProp_GProps
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.StlAPI import StlAPI_Writer

KIT_TILT, KIT_U, KIT_F = 20.0, 71.0, 41.0  # kit mount: lens 71 above the gripper axis, 41 in front of the flange
T_MIN, T_MAX = 0.0, 35.0
PIV_Y, PIV_Z, D = 28.0, -12.0, 12.0  # pivot axis; back of the cradle plate this far in front of it
LOCK_R, LUG, CHEEK, PLAY = 13.0, 12.0, 6.0, 0.15
REAR_Z, BASE_T, PLATE_T, M3_PITCH, M3_D, M4_D = -25.0, 6.0, 4.0, 45.0, 3.4, 4.3
NUT_AF, NUT_T = 5.7, 2.6


def P(*a):
    return gp_Pnt(*map(float, a))


def box(p0, p1):
    return BRepPrimAPI_MakeBox(P(*p0), P(*p1)).Shape()


def hole(p, d, r, h):
    return BRepPrimAPI_MakeCylinder(gp_Ax2(P(*p), gp_Dir(*map(float, d))), r, h).Shape()


def wire(pts):
    w = BRepBuilderAPI_MakePolygon()
    for p in pts:
        w.Add(P(*p))
    w.Close()
    return w.Wire()


def loft(a, b):
    s = BRepOffsetAPI_ThruSections(True, True)
    s.AddWire(wire(a))
    s.AddWire(wire(b))
    s.Build()
    return s.Shape()


def prism_x(zy, x0, x1):
    return loft([(x0, y, z) for z, y in zy], [(x1, y, z) for z, y in zy])


def fuse(a, *more):
    for b in more:
        a = BRepAlgoAPI_Fuse(a, b).Shape()
    return a


def cut(a, *more):
    for b in more:
        a = BRepAlgoAPI_Cut(a, b).Shape()
    return a


def volume(s):
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(s, g)
    return g.Mass()


def overlap(a, b):
    return volume(BRepAlgoAPI_Common(a, b).Shape())


def xcyl(y, z, r, x0, x1):
    return hole((x0, y, z), (1, 0, 0), r, x1 - x0)


def arc(r, t_deg):  # (z, y) of a point behind the pivot for the given tilt
    d = np.radians(KIT_TILT - t_deg)
    return PIV_Z - r * np.cos(d), PIV_Y - r * np.sin(d)


def tilt_frame(t_deg):  # rotation and origin of the cradle frame (origin at the pivot)
    d = np.radians(KIT_TILT - t_deg)
    return np.array([[1, 0, 0], [0, np.cos(d), np.sin(d)], [0, -np.sin(d), np.cos(d)]]), np.array([0, PIV_Y, PIV_Z])


def placed(s, t_deg):
    R, o = tilt_frame(t_deg)
    t = gp_Trsf()
    t.SetValues(*(float(v) for row in np.column_stack([R, o]) for v in row))
    return BRepBuilderAPI_Transform(s, t, True).Shape()


# base: plate on the kit mount, two cheeks, a web between them
xi, xo = LUG / 2 + PLAY, LUG / 2 + PLAY + CHEEK
base = box((-30, -9, REAR_Z), (30, 8, REAR_Z + BASE_T))
outline = [(-25, -9), (-17, -9), (-9, 12), (-4, 22), (-4, 33), (-8, 37), (-31, 37), (-31, 20), (-25, 9)]
base = fuse(base, prism_x(outline, -xo, -xi), prism_x(outline, xi, xo), prism_x([(-25, -9), (-17, -9), (-13, 2), (-13, 13), (-25, 9)], -xi - 1, xi + 1))
base = cut(base, xcyl(PIV_Y, PIV_Z, M4_D / 2, -30, 30))
for t in np.arange(T_MIN - 1, T_MAX + 1.01, 1.0):
    z, y = arc(LOCK_R, t)
    base = cut(base, xcyl(y, z, M4_D / 2, -30, 30))
for t in np.arange(T_MIN, T_MAX + 0.1, 5.0):  # marks every 5 degrees, longer at 0, 10, 20, 30
    r0, r1 = LOCK_R + 2.8, LOCK_R + (5.6 if t % 10 == 0 else 4.4)
    d = np.radians(0.8)
    pts = []
    for r, s in ((r0, -1), (r1, -1), (r1, 1), (r0, 1)):
        a = np.radians(KIT_TILT - t) + s * 0.25 / r
        pts.append((PIV_Z - r * np.cos(a), PIV_Y - r * np.sin(a)))
    base = cut(base, prism_x(pts, xo - 0.6, xo + 1))
for x in (-M3_PITCH / 2, M3_PITCH / 2):
    base = cut(base, hole((x, 0, REAR_Z - 1), (0, 0, 1), M3_D / 2, BASE_T + 2))
    c = [(x + NUT_AF / 3**0.5 * np.cos(a), NUT_AF / 3**0.5 * np.sin(a)) for a in np.radians(np.arange(0, 360, 60))]
    z0 = REAR_Z + BASE_T - NUT_T
    base = cut(base, loft([(p, q, z0) for p, q in c], [(p, q, z0 + NUT_T + 1) for p, q in c]))
base = cut(base, box((-29, -3, REAR_Z - 1), (-26.6, 3, REAR_Z + 10)))  # zip tie for the USB cable

# cradle, in its own frame: origin on the pivot, Z = optical axis
plate = box((-32, -10, D), (32, 10, D + PLATE_T))
lug = prism_x([(D + 1, -9), (D + 1, 9), (0, 8), (-LOCK_R, 5.2), (-LOCK_R - 5, 3), (-LOCK_R - 5, -3), (-LOCK_R, -5.2), (0, -8)], -LUG / 2, LUG / 2)
cradle = cut(fuse(plate, lug), xcyl(0, 0, M4_D / 2, -20, 20), xcyl(0, -LOCK_R, M4_D / 2, -20, 20))
for x in (-M3_PITCH / 2, M3_PITCH / 2):
    cradle = cut(cradle, hole((x, 0, D - 2), (0, 0, 1), M3_D / 2, PLATE_T + 4))
d435 = box((-45, -12.5, D + PLATE_T), (45, 12.5, D + PLATE_T + 25))

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
lines = []
for name, part in (("base", base), ("cradle", cradle)):
    w = STEPControl_Writer()
    w.Transfer(part, STEPControl_AsIs)
    w.Write(str(out / f"wrist_adapter_{name}.step"))
    BRepMesh_IncrementalMesh(part, 0.03, False, 0.15, True)
    f = out / f"wrist_adapter_{name}.stl"
    StlAPI_Writer().Write(part, str(f))
    m = trimesh.load(f)
    lines.append(f"{name}: solid valid {BRepCheck_Analyzer(part).IsValid()}, watertight {m.is_watertight}, bodies {m.body_count}, size {np.round(m.extents, 1).tolist()} mm, {volume(part) / 1000:.1f} cm3")
kit_screws = [hole((x, 0, -29 - 3), (0, 0, 1), 3.0, 3.0) for x in (-22.5, 22.5)] + [hole((x, 0, -29), (0, 0, 1), 1.5, 10) for x in (-22.5, 22.5)]
lines.append(f"base vs kit screws (M3x10): {max(overlap(base, s) for s in kit_screws):.2f} mm3")
for t in (0, 5, 10, 15, 20, 25, 30, 35):
    c, cam = placed(cradle, t), placed(d435, t)
    z, y = arc(LOCK_R, t)
    bolts = [xcyl(PIV_Y, PIV_Z, 2.0, -20, 20), xcyl(y, z, 2.0, -20, 20)]
    heads = [xcyl(yy, zz, 4.5, a, b) for yy, zz in ((PIV_Y, PIV_Z), (y, z)) for a, b in ((-18.5, -xo), (xo, 20.0))]
    R, o = tilt_frame(t)
    lens = R @ np.array([32.5, 0, D + PLATE_T + 21]) + o
    a = np.radians(KIT_TILT)
    dy, dz = lens[1], lens[2] + 4
    U, F = KIT_U + dy * np.cos(a) - dz * np.sin(a), KIT_F + dy * np.sin(a) + dz * np.cos(a)
    M3drv = [hole(tuple(R @ np.array([x, 0, D]) + o), tuple(-R[:, 2]), 2.5, 100) for x in (-22.5, 22.5)]
    lines.append(
        f"tilt {t:2d}: lens {U:.0f} above the gripper axis, {F:.0f} ahead of the flange; overlap mm3 cradle/base {overlap(c, base):.2f}, camera/base {overlap(cam, base):.2f}, "
        f"M4/base {max(overlap(b, base) for b in bolts):.2f}, M4/cradle {max(overlap(b, c) for b in bolts):.2f}, M4 heads+nuts/camera+cradle {max(max(overlap(h, cam), overlap(h, c)) for h in heads):.2f}, "
        f"camera-screw driver/base {max(overlap(d_, base) for d_ in M3drv):.2f}"
    )
(out / "check_report.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
