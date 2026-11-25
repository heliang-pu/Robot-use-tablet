"""One-piece wrist-camera mount for tablet_tap (left arm): replaces the kit's 20-degree mount M-D088-180006-A.

    python build_one_piece.py <out_dir> [<kit arm .brep>]      (needs cadquery-ocp, trimesh, numpy; the arm is only for the check)

The saddle, strap grooves and screw block are the kit part's own (inputs/kit_wrist_mount_left.brep, cut at the top of
the screw block); column and camera plate are new. Nothing on it moves: the tilt is fixed. The method and the gripper
frame are those of the sim-view mount (step_work/tools/make_wrist.py on the user's Mac).

Colour lens: LENS = above the gripper axis, to the robot's left, ahead of the flange face (mm); optical axis TILT
degrees below the gripper axis. Chosen with tablet_tap.core.look over tablet places 30-38 cm from the arm's base.
"""

import pathlib
import sys

import numpy as np
from OCP.BRep import BRep_Builder
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_Transform
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCP.BRepTools import BRepTools
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf
from OCP.GProp import GProp_GProps
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.StlAPI import StlAPI_Writer
from OCP.TopoDS import TopoDS_Shape
import trimesh

HERE = pathlib.Path(__file__).resolve().parent
LENS = (100.0, 2.5, 50.0)
TILT = 15.0
RGB_LOCAL = np.array([32.5, 0.0, -4.0])  # colour optical centre in the D435 part frame (X left, Y up, Z axis)
REAR_Z, PLATE_T, M3_X, M3_D = -25.0, 4.0, 22.5, 3.4

# the kit CAD's gripper (left arm, CAD mm: X robot's left, Y up, Z forward)
AXIS_P = np.array([302.08, 200.11, 69.70])
AXIS_D = np.array([0.010, -0.004, 1.000]) / np.linalg.norm([0.010, -0.004, 1.000])
FLANGE_Z = 66.7
TAP_HOLES = (np.array([297.0, 236.1, 106.9]), np.array([308.9, 235.9, 106.8]))
Y_CUT = 240.3  # top of the screw block

origin = AXIS_P + AXIS_D * (FLANGE_Z - AXIS_P[2]) / AXIS_D[2]
mid = (TAP_HOLES[0] + TAP_HOLES[1]) / 2
up = mid - (AXIS_P + AXIS_D * (mid[2] - AXIS_P[2]) / AXIS_D[2])
up -= up.dot(AXIS_D) * AXIS_D
up /= np.linalg.norm(up)
left = np.cross(up, AXIS_D)
lens = origin + LENS[0] * up + LENS[1] * left + LENS[2] * AXIS_D
t = np.radians(TILT)
fwd, cam_up = np.cos(t) * AXIS_D - np.sin(t) * up, np.cos(t) * up + np.sin(t) * AXIS_D
R = np.column_stack([np.cross(cam_up, fwd), cam_up, fwd])
eye = lens - R @ RGB_LOCAL


def cam(p):
    return R @ np.asarray(p, float) + eye


def P(p):
    return gp_Pnt(*map(float, p))


def box(a, b):
    return BRepPrimAPI_MakeBox(P(a), P(b)).Shape()


def trsf(rot, shift):
    out = gp_Trsf()
    out.SetValues(*(float(v) for row in np.column_stack([rot, shift]) for v in row))
    return out


def moved(s, rot, shift):
    return BRepBuilderAPI_Transform(s, trsf(rot, shift), True).Shape()


def rod(p, d, r, h):
    return BRepPrimAPI_MakeCylinder(gp_Ax2(P(p), gp_Dir(*map(float, d))), r, h).Shape()


def wire(points):
    w = BRepBuilderAPI_MakePolygon()
    for p in points:
        w.Add(P(p))
    w.Close()
    return w.Wire()


def loft(a, b):
    s = BRepOffsetAPI_ThruSections(True, True)
    s.AddWire(wire(a))
    s.AddWire(wire(b))
    s.Build()
    return s.Shape()


def volume(s):
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(s, g)
    return g.Mass()


def fuse(a, b):
    return BRepAlgoAPI_Fuse(a, b).Shape()


def cut(a, b):
    return BRepAlgoAPI_Cut(a, b).Shape()


def common(a, b):
    return volume(BRepAlgoAPI_Common(a, b).Shape())


def read(path):
    s = TopoDS_Shape()
    BRepTools.Read_s(s, str(path), BRep_Builder())
    return s


kit = read(HERE / "inputs" / "kit_wrist_mount_left.brep")
base = BRepAlgoAPI_Common(kit, box((269.0, 150.0, 60.0), (336.0, Y_CUT, 125.0))).Shape()
x0, x1 = 279.8, 325.8
lx0, lx1 = x0 - eye[0], x1 - eye[0]
zb = REAR_Z - PLATE_T
plate = moved(box((-37.0, -28.0, zb), (37.0 + 10.0, 12.5, REAR_Z)), R, eye)
column = loft([(x0, 239.0, 69.5), (x1, 239.0, 69.5), (x1, 239.0, 83.3), (x0, 239.0, 83.3)],
              [cam((lx0, -14.0, zb - 8.0)), cam((lx1, -14.0, zb - 8.0)), cam((lx1, -14.0, REAR_Z)), cam((lx0, -14.0, REAR_Z))])  # fmt: skip
# two ribs up the back of the plate, between and outside the camera screws, carry the column's stiffness on to its top
mount = fuse(fuse(base, column), plate)
for xc in (-10.0, 8.0, 41.0):
    a = [cam((xc - 2.5, -14.5, zb - 8.0)), cam((xc + 2.5, -14.5, zb - 8.0)), cam((xc + 2.5, -14.5, zb + 1)), cam((xc - 2.5, -14.5, zb + 1))]
    b = [cam((xc - 2.5, 11.0, zb - 1.5)), cam((xc + 2.5, 11.0, zb - 1.5)), cam((xc + 2.5, 11.0, zb + 1)), cam((xc - 2.5, 11.0, zb + 1))]
    mount = fuse(mount, loft(a, b))
for hp in TAP_HOLES:
    mount = cut(mount, rod(hp + AXIS_D * (40.0 - hp[2]) / AXIS_D[2], AXIS_D, 4.0, 44.0))
for sx in (-M3_X, M3_X):
    mount = cut(mount, rod(cam((sx, 0, zb - 2)), R[:, 2], M3_D / 2, PLATE_T + 4))
for xs in (-32.0, -26.0):  # zip tie for the USB cable, at the plate's free lower corner
    mount = cut(mount, moved(box((xs, -25.0, zb - 1), (xs + 2.4, -18.0, REAR_Z + 1)), R, eye))

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
to_gripper = np.column_stack([left, up, AXIS_D])
local = moved(mount, to_gripper.T, -to_gripper.T @ origin)  # origin: flange centre, x left, y up, z forward
w = STEPControl_Writer()
w.Transfer(local, STEPControl_AsIs)
w.Write(str(out / "wrist_mount_left.step"))
BRepMesh_IncrementalMesh(local, 0.04, False, 0.15, True)
StlAPI_Writer().Write(local, str(out / "wrist_mount_left.stl"))
m = trimesh.load(out / "wrist_mount_left.stl")
d435 = moved(box((-45, -12.5, REAR_Z), (45, 12.5, 0)), R, eye)
heads = [rod(cam((sx, 0, zb)), -R[:, 2], 3.0, 3.5) for sx in (-M3_X, M3_X)]
drivers = [rod(cam((sx, 0, zb)), -R[:, 2], 2.5, 110) for sx in (-M3_X, M3_X)]
keys = [rod(hp + AXIS_D * (84.0 - hp[2]) / AXIS_D[2], -AXIS_D, 2.0, 44.0) for hp in TAP_HOLES]
tip = origin + (143.0 + 85.0) * AXIS_D
q = R.T @ (tip - lens)
lines = [
    f"solid valid {BRepCheck_Analyzer(mount).IsValid()}, STL watertight {m.is_watertight}, bodies {m.body_count}, size {np.round(m.extents, 1).tolist()} mm, {volume(mount) / 1000:.1f} cm3",
    f"colour lens {LENS[0]:g} above the gripper axis, {LENS[1]:g} to the left, {LENS[2]:g} ahead of the flange; axis {np.degrees(np.arcsin(-fwd @ up)):.2f} deg below the gripper axis",
    f"overlap mm3: D435 {common(mount, d435):.2f}, camera screw heads {max(common(mount, h) for h in heads):.2f}, camera screwdriver {max(common(mount, h) for h in drivers):.2f}, hex key to the gripper screws {max(common(mount, h) for h in keys):.2f}",
    f"gripper screws: kit holes kept: {[round(common(mount, rod(hp - 4 * AXIS_D, AXIS_D, 1.4, 4)), 2) for hp in TAP_HOLES]}",
    f"pen tip (85 mm pen) in the 1080p picture: column {960 - 1372.6 * q[0] / q[2]:.0f}, row {540 - 1370.5 * q[1] / q[2]:.0f} of 1080",
]
if len(sys.argv) > 2:
    arm = read(sys.argv[2])
    lines.append(f"against the kit CAD arm and gripper: mount {common(mount, arm):.2f} mm3 (the kit mount itself {common(kit, arm):.2f}), D435 {common(d435, arm):.2f}")
(out / "check_report.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
