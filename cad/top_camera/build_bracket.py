"""Top-camera bracket for tablet_tap: holds the D435 low and 50 degrees down on the bench's 30 mm post.

    python build_bracket.py <out_dir>        (needs cadquery-ocp, trimesh, numpy)

Frame (mm): X = robot's left, Y = up, Z = forward. Origin: middle of the clamp plate, on the post's front face.
Interfaces are those of the kit's 45-degree bracket, so its M4 screws / T-nuts and M3 screws are reused.
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
LENS_FORWARD = 75.0  # colour lens in front of the post's front face
CLAMP_W, CLAMP_H, CLAMP_T, M4_PITCH, M4_D = 30.0, 64.1, 4.0, 39.1, 4.3
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


clamp = box((-CLAMP_W / 2, -CLAMP_H / 2, 0), (CLAMP_W / 2, CLAMP_H / 2, CLAMP_T))
plate = placed(box((-PLATE_W / 2, -PLATE_H / 2, REAR_Z - PLATE_T), (PLATE_W / 2, PLATE_H / 2, REAR_Z)))
sec = BRepOffsetAPI_ThruSections(True, True)
sec.AddWire(wire([(-13, -13, CLAMP_T - .5), (13, -13, CLAMP_T - .5), (13, 13, CLAMP_T - .5), (-13, 13, CLAMP_T - .5)]))
zb = REAR_Z - PLATE_T + .5
sec.AddWire(wire([cam((-15, -9, zb)), cam((15, -9, zb)), cam((15, 9, zb)), cam((-15, 9, zb))]))
sec.Build()
part = BRepAlgoAPI_Fuse(BRepAlgoAPI_Fuse(clamp, sec.Shape()).Shape(), plate).Shape()
for dy in (-M4_PITCH / 2, M4_PITCH / 2):
    part = BRepAlgoAPI_Cut(part, hole((0, dy, -1), (0, 0, 1), M4_D / 2, CLAMP_T + 2)).Shape()
for x in (-M3_PITCH / 2, M3_PITCH / 2):
    part = BRepAlgoAPI_Cut(part, hole(cam((x, 0, REAR_Z - PLATE_T - 2)), fwd, M3_D / 2, PLATE_T + 4)).Shape()
# zip-tie slot across the arm: the USB cable is tied here, so that pulling it does not turn the camera
part = BRepAlgoAPI_Cut(part, box((-20, -2.2, 16), (20, 0.0, 21))).Shape()

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
w = STEPControl_Writer()
w.Transfer(part, STEPControl_AsIs)
w.Write(str(out / "top_camera_bracket_50deg.step"))
BRepMesh_IncrementalMesh(part, 0.03, False, 0.15, True)
stl = out / "top_camera_bracket_50deg.stl"
StlAPI_Writer().Write(part, str(stl))

m = trimesh.load(stl)
d435 = placed(box((-45, -12.5, REAR_Z), (45, 12.5, 0)))
post = box((-15, -60, -30), (15, 60, 0))
head = [hole((0, dy, CLAMP_T), (0, 0, 1), 3.6, 4.0) for dy in (-M4_PITCH / 2, M4_PITCH / 2)]
head += [hole(cam((x, 0, REAR_Z - PLATE_T)), -fwd, 3.0, 3.5) for x in (-M3_PITCH / 2, M3_PITCH / 2)]
driver = [hole((0, dy, CLAMP_T), (0, 0, 1), 2.5, 120) for dy in (-M4_PITCH / 2, M4_PITCH / 2)]
driver += [hole(cam((x, 0, REAR_Z - PLATE_T)), -fwd, 2.5, 120) for x in (-M3_PITCH / 2, M3_PITCH / 2)]
ov = lambda s, b: volume(BRepAlgoAPI_Common(s, b).Shape())  # noqa: E731
L = cam(LENS_LOCAL)
lines = [
    f"solid valid {BRepCheck_Analyzer(part).IsValid()}, STL watertight {m.is_watertight}, bodies {m.body_count}",
    f"size {np.round(m.extents, 1).tolist()} mm, volume {volume(part) / 1000:.1f} cm3",
    f"colour lens: {L[2]:.1f} in front of the post, {L[0]:.1f} to the robot's left of the post centre, {-L[1]:.1f} below the clamp centre",
    f"overlap mm3: D435 box {ov(part, d435):.2f}, post {ov(part, post):.2f}, screw heads {[round(ov(part, h), 2) for h in head]}",
    f"screwdriver path (d5 x 120) vs bracket {[round(ov(part, h), 1) for h in driver]}, vs D435 {[round(ov(d435, h), 1) for h in driver]}, vs post {[round(ov(post, h), 1) for h in driver]}",
]
(out / "check_report.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
