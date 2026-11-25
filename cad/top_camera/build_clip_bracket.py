"""Clip-on top-camera bracket: holds a D435 low and 50 degrees down, with the old bracket left where it is above.

    python build_clip_bracket.py <out_dir>        (needs cadquery-ocp, trimesh, numpy)

Two parts close around the 30 mm profile from the front and the back and are pulled together by four M4x16 screws put
in from behind, so nothing has to slide in from the end of the post and the profile's slots are not needed. The front
part also has the kit bracket's two M4 holes: with two drop-in T-nuts it can be used alone.
Frame (mm): X = robot's left, Y = up, Z = forward. Origin: middle of the clamp plate, on the post's front face.
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

PITCH_DEG = 50.0  # optical axis below horizontal
LENS_FORWARD = 62.0  # colour lens in front of the post's front face; 75 would show the camera at the bottom of the old one's picture
CLAMP_W, CLAMP_H, CLAMP_T, M4_PITCH, M4_D = 66.0, 64.0, 8.0, 39.1, 4.3
POST, FIT, GAP = 30.0, 0.2, 1.0  # profile width, side clearance, gap the screws close
WALL, WEB, FLANGE_T = 5.0, 6.0, 7.0
BOLT_X, BOLT_Y = 27.0, 20.0
NUT_AF, NUT_T = 7.3, 3.2  # M4 nut pocket, open to the front
PLATE_W, PLATE_H, PLATE_T, M3_PITCH, M3_D = 68.0, 20.0, 4.0, 45.0, 3.3
LENS_LOCAL = np.array([32.5, 0.0, -4.0])  # colour optical centre in the D435 part frame (origin: front-glass centre)
REAR_Z = -25.0  # D435 rear face

a = np.radians(PITCH_DEG)
fwd, up, left = np.array([0, -np.sin(a), np.cos(a)]), np.array([0, np.cos(a), np.sin(a)]), np.array([1.0, 0, 0])
R = np.column_stack([left, up, fwd])
# clamp centre is level with the middle of the plate's back face; camera body centred on the post
back = R @ np.array([0, 0, REAR_Z - PLATE_T])
lens = R @ LENS_LOCAL
EYE = np.array([0.0, -back[1], LENS_FORWARD - lens[2]])  # D435 part origin


def cam(p):
    return R @ np.asarray(p, float) + EYE


def placed(shape):
    t = gp_Trsf()
    t.SetValues(*(float(v) for row in np.column_stack([R, EYE]) for v in row))
    return BRepBuilderAPI_Transform(shape, t, True).Shape()


def box(p0, p1):
    return BRepPrimAPI_MakeBox(gp_Pnt(*map(float, p0)), gp_Pnt(*map(float, p1))).Shape()


def wire(points):
    poly = BRepBuilderAPI_MakePolygon()
    for p in points:
        poly.Add(gp_Pnt(*map(float, p)))
    poly.Close()
    return poly.Wire()


def hole(p, d, r, h):
    return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(*map(float, p)), gp_Dir(*map(float, d))), r, h).Shape()


def volume(s):
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(s, g)
    return g.Mass()


def cut(s, t):
    return BRepAlgoAPI_Cut(s, t).Shape()


def prism(points, y0, y1):
    """A closed XZ outline swept from y0 to y1."""
    s = BRepOffsetAPI_ThruSections(True, True)
    for y in (y0, y1):
        s.AddWire(wire([(x, y, z) for x, z in points]))
    s.Build()
    return s.Shape()


def hexagon(x, y, z0, z1):
    s = BRepOffsetAPI_ThruSections(True, True)
    c = [(x + NUT_AF / 3**0.5 * np.cos(t), y + NUT_AF / 3**0.5 * np.sin(t)) for t in np.radians(np.arange(30, 390, 60))]
    for z in (z0, z1):
        s.AddWire(wire([(p, q, z) for p, q in c]))
    s.Build()
    return s.Shape()


bolts = [(sx * BOLT_X, sy * BOLT_Y) for sx in (-1, 1) for sy in (-1, 1)]
tnuts = [(0.0, -M4_PITCH / 2), (0.0, M4_PITCH / 2)]
H = CLAMP_H / 2

# front part
clamp = box((-CLAMP_W / 2, -H, 0), (CLAMP_W / 2, H, CLAMP_T))
plate = placed(box((-PLATE_W / 2, -PLATE_H / 2, REAR_Z - PLATE_T), (PLATE_W / 2, PLATE_H / 2, REAR_Z)))
sec = BRepOffsetAPI_ThruSections(True, True)
sec.AddWire(wire([(-13, -13, CLAMP_T - 0.5), (13, -13, CLAMP_T - 0.5), (13, 13, CLAMP_T - 0.5), (-13, 13, CLAMP_T - 0.5)]))
zb = REAR_Z - PLATE_T + 0.5
sec.AddWire(wire([cam((-15, -9, zb)), cam((15, -9, zb)), cam((15, 9, zb)), cam((-15, 9, zb))]))
sec.Build()
front = BRepAlgoAPI_Fuse(BRepAlgoAPI_Fuse(clamp, sec.Shape()).Shape(), plate).Shape()
for x, y in tnuts:
    front = cut(front, hole((x, y, -1), (0, 0, 1), M4_D / 2, CLAMP_T + 2))
    front = cut(front, hole((x, y, 4), (0, 0, 1), 4.0, CLAMP_T))  # 4 mm left under the head, as in the kit bracket
for x, y in bolts:
    front = cut(front, hole((x, y, -1), (0, 0, 1), M4_D / 2, CLAMP_T + 2))
    front = cut(front, hexagon(x, y, CLAMP_T - NUT_T, CLAMP_T + 1))
for x in (-M3_PITCH / 2, M3_PITCH / 2):
    front = cut(front, hole(cam((x, 0, REAR_Z - PLATE_T - 2)), fwd, M3_D / 2, PLATE_T + 4))
front = cut(front, box((-20, -2.2, 16), (20, 0.0, 21)))  # zip tie for the USB cable

# rear part: a U around the profile with a flange on each side
xi, zi = POST / 2 + FIT, -POST
xo, xf = xi + WALL, CLAMP_W / 2
half = [(xo, zi - WEB), (xo, -GAP - FLANGE_T - 2.5), (xo + 2.5, -GAP - FLANGE_T), (xf, -GAP - FLANGE_T), (xf, -GAP), (xi, -GAP), (xi, zi)]
rear = prism(half + [(-x, z) for x, z in half[::-1]], -H, H)
for x, y in bolts:
    rear = cut(rear, hole((x, y, -20), (0, 0, 1), M4_D / 2, 25))

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
ov = lambda s, b: volume(BRepAlgoAPI_Common(s, b).Shape())  # noqa: E731
d435 = placed(box((-45, -12.5, REAR_Z), (45, 12.5, 0)))
post = box((-POST / 2, -100, -POST), (POST / 2, 100, 0))
m4 = []
for x, y in bolts:  # head 4 mm, shank 16 mm, nut
    z = -GAP - FLANGE_T
    m4.append(hole((x, y, z - 4), (0, 0, 1), 3.5, 4.0))
    m4.append(hole((x, y, z), (0, 0, 1), 2.0, 16.0))
cam_heads = [hole(cam((x, 0, REAR_Z - PLATE_T)), -fwd, 3.0, 3.5) for x in (-M3_PITCH / 2, M3_PITCH / 2)]
rear_driver = [hole((x, y, -GAP - FLANGE_T - 4), (0, 0, -1), 2.5, 120) for x, y in bolts]
cam_driver = [hole(cam((x, 0, REAR_Z - PLATE_T)), -fwd, 2.5, 120) for x in (-M3_PITCH / 2, M3_PITCH / 2)]
lines = []
for name, part in (("front", front), ("rear", rear)):
    w = STEPControl_Writer()
    w.Transfer(part, STEPControl_AsIs)
    w.Write(str(out / f"clip_bracket_{name}.step"))
    BRepMesh_IncrementalMesh(part, 0.03, False, 0.15, True)
    f = out / f"clip_bracket_{name}.stl"
    StlAPI_Writer().Write(part, str(f))
    m = trimesh.load(f)
    lines.append(f"{name}: solid valid {BRepCheck_Analyzer(part).IsValid()}, STL watertight {m.is_watertight}, bodies {m.body_count}, size {np.round(m.extents, 1).tolist()} mm, {volume(part) / 1000:.1f} cm3")
    lines.append(f"  overlap mm3: D435 {ov(part, d435):.2f}, post {ov(part, post):.2f}, M4 screws {max(ov(part, b) for b in m4):.2f}, M3 heads {max(ov(part, b) for b in cam_heads):.2f}, M4 driver from behind {max(ov(part, b) for b in rear_driver):.2f}, M3 driver {max(ov(part, b) for b in cam_driver):.2f}")
L = cam(LENS_LOCAL)
box_c = np.array([cam((x, y, z)) for x in (-45, 45) for y in (-12.5, 12.5) for z in (REAR_Z, 0)])
lines += [
    f"front/rear overlap {ov(front, rear):.2f} mm3; D435 vs post {ov(d435, post):.2f}, vs M4 screws {max(ov(d435, b) for b in m4):.2f}, vs drivers from behind {max(ov(d435, b) for b in rear_driver):.2f}, M3 driver vs post {max(ov(post, b) for b in cam_driver):.2f}",
    f"colour lens: {L[2]:.1f} in front of the post, {L[0]:.1f} to the robot's left of the post centre, {-L[1]:.1f} below the clamp centre",
    f"D435 reaches {box_c[:, 2].max():.1f} in front of the post, {-box_c[:, 1].min():.1f} below / {box_c[:, 1].max():.1f} above the clamp centre",
    f"mounting: clamp centre {240 - 4.18 - L[1]:.0f} mm above the base-plate top (colour lens {240 - 4.18:.0f} mm); lower edge {240 - 4.18 - L[1] - H:.0f}, upper edge {240 - 4.18 - L[1] + H:.0f}",
]
(out / "clip_check_report.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
