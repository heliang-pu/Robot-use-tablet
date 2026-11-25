"""Adapter flange, version 2: AgileX's own flange ring (stays on the PiPER) -> Joyson TeleHand Lite wrist spigot (left hand).

    python build_hand_flange.py <out_dir>      (needs cadquery-ocp, trimesh, numpy, scipy, shapely, matplotlib, rtree)

Writes hand_flange.step/.stl (CNC 6061, nominal sizes), hand_flange_print.step/.stl (printing allowances, nut slots
instead of threads), check_report.txt and render.png. The specification is 设计任务书.md ("第二版"); the meshes in
inputs/ and the hand's URDF in ../../telehand_lite are only used by the checks and the picture. Version 1 (bolted
straight on the arm's disc, without the ring) is build_hand_flange_v1.py.

Part frame (mm): origin on the wrist axis at the face of the arm's inner disc (= the ring's bottom), Z towards the
hand (hand +Y), X = hand +Z, Y = hand +X (back of the hand, cable window). Angles "al" run from +Y towards +X.

Lower end = the gripper base's interface: dia 40 spigot in the ring, shoulder on the ring's top face, four radial
M3 countersunk screws from outside the ring into threads in the spigot. Upper end: the hand's plate sits on two
raised pads; two long M3 screws through oversize holes draw the spigot's thick walls sideways (side A against the
bore, side B against two jack screws), with disc springs under the heads to keep the preload.
"""

import pathlib
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon, BRepBuilderAPI_Transform
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCone, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakePrism, BRepPrimAPI_MakeRevol, BRepPrimAPI_MakeSphere
from OCP.BRepTools import BRepTools
from OCP.GeomAbs import GeomAbs_Cone, GeomAbs_Cylinder, GeomAbs_Plane
from OCP.gp import gp_Ax1, gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec
from OCP.GProp import GProp_GProps
from OCP.Message import Message, Message_PrinterOStream
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
from OCP.StlAPI import StlAPI_Writer
from OCP.TopAbs import TopAbs_FACE, TopAbs_IN, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
import shapely
import shapely.geometry as sg
import shapely.ops as so
import trimesh

HERE = pathlib.Path(__file__).resolve().parent
HAND_DIR = HERE.parents[1] / "telehand_lite"  # URDF and MJCF of the whole hand, for the finger and thumb check
X, Y, Z = (1, 0, 0), (0, 1, 0), (0, 0, 1)

# ---- hand side (设计任务书.md; where the manual and the model disagree the worse value is used) ----
AXIS_X, AXIS_Z, FACE_Y = 1.0218, 10.0178, -52.0243  # wrist axis and plate face in the hand's own frame, fitted on the mesh (spec: 1.02, 10.02, -52.02)
TUBE_D, TUBE_LEN = 26.0, 8.0  # spigot of the hand
TUBE_ID_MODEL, TUBE_ID_MANUAL = 20.0, 21.0  # screw tip limit / thread length
HOLE_BELOW = 3.10  # radial thread centre below the plate face: not measured on the real hand, the part allows +-0.3
TIP_INDEX = (-30.02, 0.0)  # index fingertip from the wrist axis (X, Y), at hand y = 127.7
TIP_Y, HAND_KG = 127.7, 0.408
# the hand's radial thread is inferred from the model (M3 x 0.5): check with a real screw. Cap screw, washer and disc springs under its head
RAD = dict(name="M3", d=3.0, pitch=0.5, clear=3.6, head_d=5.5, head_k=3.0, key=2.5, length=25.0, tol=0.42)
SPRING = dict(n=3, de=6.0, di=3.2, t=0.3, h0=0.15, use=0.70)  # DIN 2093 6 x 3.2 x 0.3, n nested the same way, compressed to 70 % of h0
WASHER = dict(d=6.0, t=0.5)
JACK = dict(name="M3", d=3.0, length=8.0, key=1.5)  # flat-point set screws on side B

# ---- AgileX flange ring M-JXB2-130103-D, read off inputs/agilex_flange_ring.step (the check reads them again) ----
RING_H, RING_R, RING_BORE_R, RING_FLOOR = 10.5, 28.5, 20.0, 3.5
RING_HOLE_Z, RING_HOLE_D, RING_CSK_R, RING_FLAT = 7.0, 3.4, 3.36, 18.0  # 4 radial holes with 90 degree countersinks; key flat at 45 degrees
DISC_PROUD = 1.0  # the arm's inner disc stands this much above the J6 housing
RING_SCREW = dict(length=14.0, tol=0.35, dk=6.0, k=(1.5, 1.7, 1.86), key=2.0)  # M3 countersunk, from outside the ring

# ---- the part ----
H = 29.0  # disc face to the hand's plate face (18.5 above the ring)
SPIG_BOT, FLAT_D = 4.0, 17.0  # spigot from here up to the ring's top (6.5 long, 0.5 off the ring's floor); four flats at this distance from the axis
RING_AL, FLAT_AL = (45, 135, 225, 315), (0, 90, 180, 270)  # threads for the ring screws; flats (one faces the ring's key flat)
E_RING = 0.08  # the threads are this much higher than the ring's holes: the countersinks can only push the shoulder down
FLANGE_TOP, NECK_R, TABLE_BOT = 13.0, 17.5, 21.0  # shoulder flange up to here; neck around the cable chamber; table for the hand
TABLE_Y, TAB_Y = 14.5, 4.0  # table half width (the neck comes up through it); half width of its end faces (seats of the draw screws)
SEAT_REL, SEAT_C, SEAT_R, SEAT_IN, SEAT_X = 0.3, 16.6, 10.8, 14.0, 10.0  # table top is lower than the pads by this (also clears a fillet at the root of the hand's spigot); pads = stadium (centres, radius) outside this radius and this |X|
TIP_NOM = 10.70  # radius of a draw screw's tip: never inside r = 10.0 with all tolerances
STACK = WASHER["t"] + SPRING["t"] * SPRING["n"] + SPRING["h0"] * (1 - SPRING["use"])  # washer and springs under the head, tightened
TAB_X = round(TIP_NOM + RAD["length"] - STACK, 2)  # end faces of the table
JACK_AL, JACK_R1 = 20.0, 22.0  # jack screws: degrees either side of -X; threaded from the bore out to this radius
WIN_W, WIN_FLOOR, WIN_R = 7.25, 12.0, 2.0  # cable window half width, floor (a gutter 1 deep across the flange), corner radius; its top is the table
TIE_AL, TIE_Z = 50.0, 16.5  # cable-tie holes through the neck either side of the window
THIN_GAP, THIN_X = 0.15, 6.75  # the bore stands back from the hand spigot's two thin walls
FITS = {  # bore = measured hand spigot + 0.02; spigot = measured ring bore - 0.02
    "cnc": dict(bore=26.02, spigot=39.98, nut=False),
    "print": dict(bore=26.20, spigot=39.85, nut=True),
}
RHO = {"6061 铝": 2.70, "PA12": 1.01, "PETG-CF": 1.30}
E_MOD = {"6061 铝": 68900.0, "PA12": 1700.0, "PETG-CF": 4500.0}


def P(p):
    return gp_Pnt(*map(float, p))


def box(a, b):
    return BRepPrimAPI_MakeBox(P(a), P(b)).Shape()


def cyl(p, d, r, h):
    return BRepPrimAPI_MakeCylinder(gp_Ax2(P(p), gp_Dir(*map(float, d))), r, h).Shape()


def cone(p, d, r1, r2, h):
    return BRepPrimAPI_MakeCone(gp_Ax2(P(p), gp_Dir(*map(float, d))), r1, r2, h).Shape()


def ball(p, r):
    return BRepPrimAPI_MakeSphere(P(p), r).Shape()


def fuse(a, *more):
    for b in more:
        a = BRepAlgoAPI_Fuse(a, b).Shape()
    return a


def cut(a, *more):
    for b in more:
        a = BRepAlgoAPI_Cut(a, b).Shape()
    return a


def both(a, b):
    return BRepAlgoAPI_Common(a, b).Shape()


def props(s):
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(s, g)
    c = g.CentreOfMass()
    return g.Mass(), np.array([c.X(), c.Y(), c.Z()])


def volume(s):
    return props(s)[0]


def common(a, b):
    return volume(both(a, b))


def gap(a, b):
    return BRepExtrema_DistShapeShape(a, b).Value()


def pol(r, al):
    return r * np.sin(np.radians(al)), r * np.cos(np.radians(al))


def turn(s, al):
    """Rotate about Z so that +Y goes to the direction al."""
    t = gp_Trsf()
    t.SetRotation(gp_Ax1(P((0, 0, 0)), gp_Dir(0, 0, 1)), -np.radians(al))
    return BRepBuilderAPI_Transform(s, t, True).Shape()


def face(points):
    w = BRepBuilderAPI_MakePolygon()
    for p in points:
        w.Add(P(p))
    w.Close()
    return BRepBuilderAPI_MakeFace(w.Wire()).Face()


def revolve(profile):
    return BRepPrimAPI_MakeRevol(face([(r, 0, z) for r, z in profile]), gp_Ax1(P((0, 0, 0)), gp_Dir(0, 0, 1))).Shape()


def prism(xy, z0, z1):
    return BRepPrimAPI_MakePrism(face([(x, y, z0) for x, y in xy]), gp_Vec(0, 0, z1 - z0)).Shape()


def radial(al, z, r0, r1, rad):
    """Cylinder along the radius at angle al, from r0 to r1."""
    return cyl((*pol(r0, al), z), (*pol(1, al), 0), rad, r1 - r0)


def table_outline():
    c, s = np.cos(np.radians(JACK_AL)), np.sin(np.radians(JACK_AL))
    xk = TAB_X - (TABLE_Y - TAB_Y) * s / c  # the corner cuts are square to the jack screws
    return [(TAB_X, -TAB_Y), (TAB_X, TAB_Y), (xk, TABLE_Y), (-xk, TABLE_Y), (-TAB_X, TAB_Y), (-TAB_X, -TAB_Y), (-xk, -TABLE_Y), (xk, -TABLE_Y)]


def build(fit):
    rb, rs = fit["bore"] / 2, fit["spigot"] / 2
    zh = H - HOLE_BELOW
    # turned shape: spigot, shoulder flange, neck
    s = revolve([(0, SPIG_BOT), (rs - 0.4, SPIG_BOT), (rs, SPIG_BOT + 0.4), (rs, RING_H), (RING_R, RING_H), (RING_R, FLANGE_TOP - 0.3), (RING_R - 0.3, FLANGE_TOP),
                 (NECK_R, FLANGE_TOP), (NECK_R, H - SEAT_REL), (0, H - SEAT_REL)])
    s = fuse(s, prism(table_outline(), TABLE_BOT, H - SEAT_REL))
    pads = fuse(box((-SEAT_C, -SEAT_R, H - SEAT_REL), (SEAT_C, SEAT_R, H)), cyl((SEAT_C, 0, H - SEAT_REL), Z, SEAT_R, SEAT_REL), cyl((-SEAT_C, 0, H - SEAT_REL), Z, SEAT_R, SEAT_REL))
    s = fuse(s, cut(pads, cyl((0, 0, H - 1), Z, SEAT_IN, 2), box((-SEAT_X, -20, H - 1), (SEAT_X, 20, H + 1))))
    for al in FLAT_AL:
        s = cut(s, turn(box((-25, FLAT_D, SPIG_BOT - 1), (25, 25, RING_H)), al))
    # one straight bore: socket for the hand's spigot at the top, cable chamber below, open towards the ring's floor
    s = cut(s, cyl((0, 0, 0), Z, rb, H + 1), both(cyl((0, 0, 0), Z, rb + THIN_GAP, H + 1), box((-THIN_X, -30, 0), (THIN_X, 30, 60))))
    win = fuse(box((-WIN_W, 5, WIN_FLOOR), (WIN_W, 19, TABLE_BOT - WIN_R)), box((-WIN_W + WIN_R, 5, WIN_FLOOR), (WIN_W - WIN_R, 19, TABLE_BOT)),
               cyl((WIN_W - WIN_R, 5, TABLE_BOT - WIN_R), Y, WIN_R, 14), cyl((-WIN_W + WIN_R, 5, TABLE_BOT - WIN_R), Y, WIN_R, 14),
               box((-WIN_W, 12, WIN_FLOOR), (WIN_W, 30, FLANGE_TOP + 1)))
    s = cut(s, win)
    for al in (TIE_AL, -TIE_AL):
        s = cut(s, radial(al, TIE_Z, 11, 19, 1.7))
    for al in RING_AL:
        if fit["nut"]:  # square nut M3 (5.5 x 1.8) pushed in from below
            s = cut(s, radial(al, RING_HOLE_Z + E_RING, 11, 21, 1.7), turn(box((-2.85, 15.2, SPIG_BOT - 1), (2.85, 17.3, RING_HOLE_Z + E_RING + 2.9)), al))
        else:  # tap M3 right through the wall
            s = cut(s, radial(al, RING_HOLE_Z + E_RING, 11, 21, 1.25))
    for al in (90, 270):
        s = cut(s, radial(al, zh, 11, TAB_X + 2, RAD["clear"] / 2))
    for al in (270 - JACK_AL, 270 + JACK_AL):
        s = cut(s, radial(al, zh, 11, JACK_R1 + 0.2, 1.25), radial(al, zh, JACK_R1, TAB_X + 2, 1.7))
    s = cut(s, cyl((TAB_X - 2.6, 3.2, H - SEAT_REL - 0.4), Z, 1.0, 1))  # dimple: side A, tightened first
    u = ShapeUpgrade_UnifySameDomain(s, True, True, True)
    u.Build()
    return u.Shape()


def ring_screw_seat(e=E_RING, k=RING_SCREW["k"][1]):
    """Radius where a ring screw's head cone starts, and of its tip, when the head rides on the upper side of the ring's countersink."""
    rj = RING_R - (RING_CSK_R - 1.5) + e
    return rj, rj + k - RING_SCREW["length"]


def fasteners():
    """Screws at their nominal tightened places, for the checks and the picture."""
    out = {}
    zh = H - HOLE_BELOW
    rj, tip = ring_screw_seat()
    k = RING_SCREW["k"][1]
    for al in RING_AL:
        d = (*pol(1, al), 0)
        z = RING_HOLE_Z + E_RING
        out[f"ring{al}"] = fuse(cyl((*pol(tip, al), z), d, 1.5, rj - tip), cone((*pol(rj, al), z), d, 1.5, RING_SCREW["dk"] / 2, 1.5),
                                cyl((*pol(rj + 1.5, al), z), d, RING_SCREW["dk"] / 2, k - 1.5))
    for sx, name in ((1, "A"), (-1, "B")):
        d = (sx, 0, 0)
        x0 = TAB_X + STACK
        out["draw" + name] = fuse(cyl((sx * TIP_NOM, 0, zh), d, RAD["d"] / 2, x0 - TIP_NOM), cyl((sx * x0, 0, zh), d, RAD["head_d"] / 2, RAD["head_k"]))
        out["stack" + name] = cut(cyl((sx * TAB_X, 0, zh), d, SPRING["de"] / 2, STACK), cyl((sx * (TAB_X - 1), 0, zh), d, SPRING["di"] / 2, STACK + 2))
    for al, name in ((270 - JACK_AL, "jack1"), (270 + JACK_AL, "jack2")):
        out[name] = radial(al, zh, TUBE_D / 2, TUBE_D / 2 + JACK["length"], JACK["d"] / 2)
    return out


def read_step(path):
    r = STEPControl_Reader()
    r.ReadFile(str(path))
    r.TransferRoots()
    return r.OneShape()


def ring_solid():
    """AgileX's ring rebuilt from the sizes read off its STEP, in its own frame (radial holes at 0/90/180/270 degrees, key flat at 45).

    The STEP itself closes the 0 degree hole with a face of no thickness at the bore, which upsets booleans; the check compares this with their mesh."""
    R, rf = RING_BORE_R, 5.0
    n, t = np.array([1, 1]) / np.sqrt(2), np.array([-1, 1]) / np.sqrt(2)
    w = np.sqrt((R - rf) ** 2 - (RING_FLAT - rf) ** 2)
    c1, c2 = (RING_FLAT - rf) * n + w * t, (RING_FLAT - rf) * n - w * t  # centres of the two R5 corners of the flat
    wedge = prism([(0, 0), tuple(60 * c2 / np.linalg.norm(c2)), tuple(60 * c1 / np.linalg.norm(c1))], 0, 20)
    z0, z1, zt = RING_FLOOR, RING_H - 0.5, RING_H + 0.01
    void = fuse(cut(cyl((0, 0, z0), Z, R, zt - z0), wedge), prism([(0, 0), tuple(c2), tuple(c2 + rf * n), tuple(c1 + rf * n), tuple(c1)], z0, zt),
                cyl((*c1, z0), Z, rf, zt - z0), cyl((*c2, z0), Z, rf, zt - z0))
    tri = face([(*(c2 + (rf - 0.001) * n), z1), (*(c2 + (rf + 0.51) * n), z1 + 0.51), (*(c2 + (rf - 0.001) * n), z1 + 0.51)])  # 0.5 chamfer at the mouth
    void = fuse(void, cut(cone((0, 0, z1), Z, R, R + 0.51, 0.51), wedge), BRepPrimAPI_MakePrism(tri, gp_Vec(*(2 * w * t), 0)).Shape(),
                cone((*c1, z1), Z, rf, rf + 0.51, 0.51), cone((*c2, z1), Z, rf, rf + 0.51, 0.51))
    s = cut(cyl((0, 0, 0), Z, RING_R, RING_H), void)
    for a in range(30, 360, 60):  # countersunk screws into the arm's disc
        x, y = 13.5 * np.cos(np.radians(a)), 13.5 * np.sin(np.radians(a))
        s = cut(s, cyl((x, y, -1), Z, 1.7, 3), cone((x, y, RING_FLOOR - 1.66), Z, 1.7, 3.37, 1.67))
    for a in range(0, 360, 90):  # radial holes with 90 degree countersinks from outside
        d = (np.cos(np.radians(a)), np.sin(np.radians(a)), 0)
        r0 = RING_R - RING_CSK_R + RING_HOLE_D / 2
        s = cut(s, cyl((19 * d[0], 19 * d[1], RING_HOLE_Z), d, RING_HOLE_D / 2, 9), cone((r0 * d[0], r0 * d[1], RING_HOLE_Z), d, RING_HOLE_D / 2, RING_CSK_R + 0.5, RING_CSK_R - RING_HOLE_D / 2 + 0.5))
    # groove in the bottom (the J6 stop pin, if there is one, runs in it): 240 degrees, from 315 through 0 and 180 to 195
    groove = turn(BRepPrimAPI_MakeRevol(face([(22, 0, -1), (27.75, 0, -1), (27.75, 0, 4), (22, 0, 4)]), gp_Ax1(P((0, 0, 0)), gp_Dir(0, 0, 1)), np.radians(240)).Shape(), 45)
    for a in (315, 195):
        groove = fuse(groove, cyl((24.875 * np.cos(np.radians(a)), 24.875 * np.sin(np.radians(a)), -1), Z, 2.875, 5))
    return cut(s, groove)


def ring_model(step=0):
    """The ring in the part frame: one of four positions, its key flat facing one of the spigot's flats."""
    return turn(ring_solid(), -45 + 90 * step)


def ring_measure():
    """What the interface is, read from the ring's STEP as delivered (its own frame)."""
    s = read_step(HERE / "inputs" / "agilex_flange_ring.step")
    out = dict(top=0.0, holes=[], csk=[], coax=[], patch=[])
    ex = TopExp_Explorer(s, TopAbs_FACE)
    while ex.More():
        a = BRepAdaptor_Surface(TopoDS.Face(ex.Current()))
        if a.GetType() == GeomAbs_Plane:
            pl = a.Plane()
            n, o = pl.Axis().Direction(), pl.Location()
            d = n.X() * o.X() + n.Y() * o.Y() + n.Z() * o.Z()
            if abs(n.Z()) > 0.99:
                out["top"] = max(out["top"], abs(d))
                if 1 < abs(d) < 6 and n.Z() > 0:
                    out["floor"] = abs(d)
            elif abs(n.Z()) < 0.01 and 10 < abs(d) < 20:
                out["flat"] = (abs(d), np.degrees(np.arctan2(n.Y() * np.sign(d), n.X() * np.sign(d))) % 360)
        elif a.GetType() == GeomAbs_Cylinder:
            c = a.Cylinder()
            n, o = c.Axis().Direction(), c.Location()
            if abs(n.Z()) > 0.99 and abs(o.X()) + abs(o.Y()) < 1e-6:
                out["coax"].append(round(c.Radius(), 4))
                g = GProp_GProps()
                BRepGProp.SurfaceProperties_s(ex.Current(), g)
                if abs(c.Radius() - RING_BORE_R) < 1e-6 and g.Mass() < 20:  # a lid over a hole's mouth
                    q = g.CentreOfMass()
                    out["patch"].append((g.Mass(), round(np.degrees(np.arctan2(q.Y(), q.X()))) % 360))
            elif abs(n.Z()) < 0.01 and abs(c.Radius() - 1.7) < 0.01:
                out["holes"].append((c.Radius(), o.Z(), round(np.degrees(np.arctan2(o.Y(), o.X()))) % 360))
        elif a.GetType() == GeomAbs_Cone:
            c = a.Cone()
            if abs(c.Axis().Direction().Z()) < 0.01:
                out["csk"].append((c.RefRadius(), np.degrees(c.SemiAngle()), np.hypot(c.Location().X(), c.Location().Y())))
        ex.Next()
    return out


def tess(shape, defl=0.02):
    BRepTools.Clean_s(shape)
    BRepMesh_IncrementalMesh(shape, defl, False, 0.15, True)
    with tempfile.TemporaryDirectory() as d:
        StlAPI_Writer().Write(shape, d + "/t.stl")
        return trimesh.load(d + "/t.stl")


def to_part(v):
    """Points in the hand's own frame (mm) -> part frame."""
    return np.column_stack([v[:, 2] - AXIS_Z, v[:, 0] - AXIS_X, v[:, 1] - FACE_Y + H])


def load_hand():
    m = trimesh.load(HERE / "inputs" / "hand_root.stl")
    m.vertices = to_part(m.vertices * 1000.0)
    return m


def load_arm():
    """J6 housing and inner disc; how it is turned about the axis relative to the ring is not known."""
    m = trimesh.load(HERE / "inputs" / "piper_end.stl")
    c, s = np.cos(np.radians(45)), np.sin(np.radians(45))
    v = m.vertices
    m.vertices = np.column_stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1], v[:, 2] - DISC_PROUD])
    return m


def hand_links():
    """Finger and thumb links of the whole hand (URDF), with the joint couplings of the MJCF. None if the model is not there."""
    urdf, mjcf = HAND_DIR / "urdf" / "telehand_lite.urdf", HAND_DIR / "mjcf" / "telehand_lite.xml"
    if not (urdf.exists() and mjcf.exists()):
        return None

    def rpy(r, p, y):
        cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
        return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])

    root = ET.parse(urdf).getroot()
    joints, pts = {}, {}
    for j in root.findall("joint"):
        o, lim = j.find("origin"), j.find("limit")
        joints[j.get("name")] = dict(parent=j.find("parent").get("link"), child=j.find("child").get("link"), xyz=1000 * np.array(o.get("xyz").split(), float),
                                     rot=rpy(*map(float, o.get("rpy").split())), axis=np.array(j.find("axis").get("xyz").split(), float),
                                     lim=(float(lim.get("lower")), float(lim.get("upper"))) if lim is not None else None)
    for link in root.findall("link"):
        if link.get("name") != "root":
            v = trimesh.load(urdf.parent / link.find("visual/geometry/mesh").get("filename")).vertices * 1000.0
            pts[link.get("name")] = v[:: max(1, len(v) // 6000)]
    couple = {e.get("joint1"): (e.get("joint2"), [float(c) for c in e.get("polycoef").split()]) for e in ET.parse(mjcf).getroot().find("equality").findall("joint")}
    return joints, pts, couple


def pose_hand(model, q, meshes=False):
    """Points of every moving link in the part frame (or their meshes), for the driven joints q (the coupled ones follow)."""
    joints, pts, couple = model
    q = dict(q)
    for _ in range(3):
        for j1, (j2, c) in couple.items():
            if j2 in q:
                q[j1] = sum(ci * q[j2] ** i for i, ci in enumerate(c))
    frames, out = {"root": (np.eye(3), np.zeros(3))}, {}
    todo = dict(joints)
    while todo:
        for name, j in list(todo.items()):
            if j["parent"] in frames:
                rp, tp = frames[j["parent"]]
                a = j["axis"] / np.linalg.norm(j["axis"])
                k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
                ang = q.get(name, 0.0)
                frames[j["child"]] = (rp @ j["rot"] @ (np.eye(3) + np.sin(ang) * k + (1 - np.cos(ang)) * k @ k), tp + rp @ j["xyz"])
                if meshes:
                    m = trimesh.load(HAND_DIR / "urdf" / "meshes" / (j["child"] + ".stl"))
                    m.vertices = to_part(m.vertices * 1000.0 @ frames[j["child"]][0].T + frames[j["child"]][1])
                    out[j["child"]] = m
                else:
                    out[j["child"]] = to_part(pts[j["child"]] @ frames[j["child"]][0].T + frames[j["child"]][1])
                del todo[name]
    return out


def flat_region(mesh, z, nz):
    """Union of the mesh's triangles that lie in the plane at height z with normal nz, as a polygon in (X, Y)."""
    k = (np.abs(mesh.triangles_center[:, 2] - z) < 0.01) & (mesh.face_normals[:, 2] * nz > 0.99)
    return so.unary_union([sg.Polygon(t[:, :2]).buffer(1e-4) for t in mesh.triangles[k]])


def area_moments(poly, step=0.05):
    """Area, centroid, second moments about the centroid (of x*x and of y*y) and largest |x|, |y| from it."""
    x0, y0, x1, y1 = poly.bounds
    gx, gy = np.meshgrid(np.arange(x0, x1, step) + step / 2, np.arange(y0, y1, step) + step / 2)
    k = shapely.contains_xy(poly, gx, gy)
    x, y = gx[k], gy[k]
    c = np.array([x.mean(), y.mean()])
    return k.sum() * step**2, c, ((x - c[0]) ** 2).sum() * step**2, ((y - c[1]) ** 2).sum() * step**2, np.abs(x - c[0]).max(), np.abs(y - c[1]).max()


def loops_polygon(loops):
    out = sg.Polygon()
    for p in sorted((sg.Polygon(q) for q in loops if len(q) > 3), key=lambda p: -p.area):
        out = out.symmetric_difference(p.buffer(0))
    return out


def half_section(mesh, al, solid=True):
    """Cut by the half plane through the axis at angle al: polygon (solid; "closed" = only from loops that close) or lines, in (radius, z)."""
    d = np.array([*pol(1.0, al), 0.0])
    sec = mesh.section(plane_origin=(0, 0, 0), plane_normal=(d[1], -d[0], 0))
    if sec is None:
        return sg.Polygon()
    loops = [np.column_stack([q @ d, q[:, 2]]) for q in sec.discrete]
    half = sg.box(0, -200, 200, 300)
    if solid == "closed":
        loops = [q for q in loops if np.allclose(q[0], q[-1], atol=1e-6)]
    if solid:
        return loops_polygon(loops).intersection(half)
    seg = trimesh.intersections.mesh_plane(mesh, (d[1], -d[0], 0), (0, 0, 0))  # every cut triangle, also where the mesh is open
    return so.unary_union([sg.LineString(np.column_stack([q @ d, q[:, 2]])) for q in seg]).intersection(half)


def spring_force(s):
    """Force of one disc spring at deflection s (Almen-Laszlo, steel)."""
    de, di, t, h0 = SPRING["de"], SPRING["di"], SPRING["t"], SPRING["h0"]
    dl = de / di
    k1 = ((dl - 1) / dl) ** 2 / (np.pi * ((dl + 1) / (dl - 1) - 2 / np.log(dl)))
    return 4 * 206000 / (1 - 0.3**2) * t**4 / (k1 * de**2) * (s / t) * ((h0 / t - s / t) * (h0 / t - s / (2 * t)) + 1)


def check(part, fit, out):
    """All the numerical checks of the nominal part. Returns the report lines and what the picture needs."""
    L = []
    rb, rs, zh = fit["bore"] / 2, fit["spigot"] / 2, H - HOLE_BELOW
    fine, coarse = tess(part, 0.005), tess(part, 0.03)
    hand, arm, fast = load_hand(), load_arm(), fasteners()
    ring = ring_model()
    ring_mesh = tess(ring, 0.005)
    theirs = trimesh.load(HERE / "inputs" / "agilex_flange_ring.stl")
    vol, com = props(part)

    # 1 solids
    n_solid = 0
    ex = TopExp_Explorer(part, TopAbs_SOLID)
    while ex.More():
        n_solid += 1
        ex.Next()
    stl = trimesh.load(out / "hand_flange.stl")
    back = read_step(out / "hand_flange.step")
    L += ["一、实体",
          f"  B-rep 有效 {BRepCheck_Analyzer(part).IsValid()}，实体数 {n_solid}；STL 封闭 {stl.is_watertight}，连通体 {stl.body_count}，三角形 {len(stl.faces)}；"
          f"体积 B-rep {vol / 1000:.3f} cm3 / STL {stl.volume / 1000:.3f} cm3；外形 {np.round(stl.extents, 2).tolist()} mm，Z 从 {stl.bounds[0, 2]:.2f} 到 {stl.bounds[1, 2]:.2f}",
          f"  STEP 文件读回：有效 {BRepCheck_Analyzer(back).IsValid()}，体积 {volume(back) / 1000:.3f} cm3"]

    # 2 the ring's interface, read from its STEP, against the part
    rm = ring_measure()
    raw_mesh = tess(ring_solid(), 0.01)
    tv = np.vstack([theirs.vertices, trimesh.sample.sample_surface(theirs, 100000, seed=4)[0]])
    lid = (np.abs(tv[:, 1]) < 1.75) & (np.abs(tv[:, 2] - RING_HOLE_Z) < 1.75) & (np.abs(np.hypot(tv[:, 0], tv[:, 1]) - RING_BORE_R) < 0.05) & (tv[:, 0] > 0)
    dev = max(trimesh.proximity.closest_point(raw_mesh, tv[~lid])[1].max(), trimesh.proximity.closest_point(theirs, raw_mesh.vertices)[1].max())
    same = (abs(rm["top"] - RING_H) < 1e-6 and abs(max(rm["coax"]) - RING_R) < 1e-6 and abs(min(rm["coax"]) - RING_BORE_R) < 1e-6 and abs(rm["floor"] - RING_FLOOR) < 1e-6
            and abs(rm["flat"][0] - RING_FLAT) < 1e-6 and len(rm["holes"]) == 4 and all(abs(h[1] - RING_HOLE_Z) < 1e-6 for h in rm["holes"]) and all(abs(c[0] - RING_CSK_R) < 1e-6 for c in rm["csk"]))
    fits = []
    for step in range(4):
        holes = {(h[2] - 45 + 90 * step) % 360 for h in rm["holes"]}  # ring angle (from +x towards +y) -> al of the part
        holes = {(90 - a) % 360 for a in holes}
        flat = (90 - (rm["flat"][1] - 45 + 90 * step)) % 360
        fits.append((sorted(holes) == sorted(RING_AL), round(flat) % 360 in FLAT_AL))
    probe = max(common(part, radial(al, RING_HOLE_Z + E_RING, 11, 21, 1.2)) for al in RING_AL)
    L += ["二、法兰环的接口（从 agilex_flange_ring.step 读出）与新件对照",
          f"  环：顶面 Z={rm['top']:.3f}，外圆 Φ{2 * max(rm['coax']):.2f}，孔 Φ{2 * min(rm['coax']):.3f}、孔底 Z={rm['floor']:.2f}，钥匙面离轴 {rm['flat'][0]:.3f}（{rm['flat'][1]:.0f}° 方向）；"
          f"径向孔 {len(rm['holes'])} 个 Φ{2 * rm['holes'][0][0]:.2f}，在 {sorted(h[2] for h in rm['holes'])}°，轴线 Z={rm['holes'][0][1]:.3f}（离顶面 {rm['top'] - rm['holes'][0][1]:.2f}）；"
          f"沉孔 90°、外圆处 Φ{2 * rm['csk'][0][0]:.2f}。与脚本里用的数一致：{same}",
          f"  松灵的 STEP 有一处瑕疵：{rm['patch'][0][1] if rm['patch'] else '-'}° 方向那个径向孔在 Φ40 孔壁处被一张 {rm['patch'][0][0] if rm['patch'] else 0:.1f} mm2 的零厚度面封住（另三个是通孔），布尔运算会算错。"
          f"所以检查用按上面尺寸重建的环：体积 {volume(ring) / 1000:.3f} cm3（STEP {volume(read_step(HERE / 'inputs' / 'agilex_flange_ring.step')) / 1000:.3f}），"
          f"与松灵的网格互量表面最大偏差 {dev:.2f} mm（重建时省略了外圆上 0.2 深的刻痕和槽口 0.2 倒角；封口面不算）",
          f"  新件：止口 Φ{fit['spigot']:.2f}（单边间隙 {RING_BORE_R - rs:.3f}），长 {RING_H - SPIG_BOT:g}，离孔底 {SPIG_BOT - RING_FLOOR:g}；肩面 Z={RING_H:g}；4 个 M3 螺纹在 al={list(RING_AL)}，"
          f"轴线 Z={RING_HOLE_Z + E_RING:.2f}，比环的孔高 {E_RING:.2f}（有意）；4 个平面在 al={list(FLAT_AL)}，离轴 {FLAT_D:g}。Φ2.4 探针沿螺纹轴线与新件重叠 {probe:.4f} mm3",
          "  四个安装位置（每 90°）：" + "；".join(f"第{i + 1}位 螺纹对上环的孔 {a}，平面对着钥匙面 {b}" for i, (a, b) in enumerate(fits))]

    # 3 the ring and the arm against the part
    sp = np.vstack([fine.vertices, trimesh.sample.sample_surface(fine, 400000, seed=3)[0]])
    sp = sp[sp[:, 2] < RING_H - 0.02]  # the spigot's surface
    d_ring = trimesh.proximity.closest_point(ring_mesh, sp)[1]
    land = np.hypot(sp[:, 0], sp[:, 1]) > rs - 0.002
    rp = np.vstack([ring_mesh.vertices, trimesh.sample.sample_surface(ring_mesh, 300000, seed=5)[0]])
    rp = rp[(np.hypot(rp[:, 0], rp[:, 1]) < RING_BORE_R - 0.05) & (rp[:, 2] > RING_FLOOR + 0.05) & (rp[:, 2] < RING_H - 0.6)]  # the ring's key flat and its two rounded corners
    d_flat = trimesh.proximity.closest_point(fine, rp)[1]
    i = np.argmin(d_flat)
    shoulder = flat_region(fine, RING_H, -1).intersection(flat_region(ring_mesh, RING_H, 1))
    pin = cut(cyl((0, 0, -5), Z, 26.5, 5 + 5 - DISC_PROUD), cyl((0, 0, -6), Z, 23.5, 12))  # where a J6 stop pin could travel
    av = arm.vertices[arm.vertices[:, 2] > -8]
    L += ["三、与法兰环、与臂",
          "  与环的重叠体积（四个位置）：" + "、".join(f"{common(part, ring_model(k)):.4f}" for k in range(4)) + " mm3（应为 0）",
          f"  预定接触 1：肩面压在环顶面上，接触面积 {shoulder.area:.0f} mm2（半径 {RING_BORE_R + 0.5:g}..{RING_R:g}）",
          f"  预定接触 2：止口四段圆柱面在 Φ40 孔里，单边间隙 {d_ring[land].min():.3f}（网格量得，设计 {RING_BORE_R - rs:.3f}）",
          f"  环的钥匙面和它两端的 R5 圆角离新件最近 {d_flat[i]:.2f} mm，在 X={rp[i][0]:.1f} Y={rp[i][1]:.1f} Z={rp[i][2]:.1f}（圆角起点处）；平面正对钥匙面处间隙 {RING_FLAT - FLAT_D:.2f}；止口底面离孔底 {SPIG_BOT - RING_FLOOR:.2f}",
          f"  新件最低点 Z={stl.bounds[0, 2]:.2f}（在环的孔里）；环以外新件最低是肩面 Z={RING_H:g}。臂（piper_end.stl，内盘面 Z=0、外壳端面 Z={-DISC_PROUD:g}）最高点 Z={av[:, 2].max():.2f}："
          f"新件离臂最近 {trimesh.proximity.closest_point(coarse, av)[1].min():.2f} mm",
          f"  J6 限位销若存在（外壳 r=25 处 Φ3，最高到外壳以上 5 即 Z={5 - DISC_PROUD:g}），它扫过的环形区域离新件最近 {gap(part, pin):.2f} mm"]

    # 4 the hand's palm mesh against the part
    low = hand.submesh([np.nonzero(hand.triangles[:, :, 2].min(axis=1) < H + 12)[0]], append=True)
    pts = np.vstack([low.vertices, trimesh.sample.sample_surface(low, 400000, seed=0)[0]])
    under = pts[pts[:, 2] < H - 5e-3]  # below the seat plane only the hand's spigot may exist (the mesh is noisy by 0.001)
    ru = np.hypot(under[:, 0], under[:, 1])
    near_pt, dist, tri = trimesh.proximity.closest_point(fine, under)
    behind = int((((under - near_pt) * fine.face_normals[tri]).sum(1) < -1e-6).sum())  # on the wrong side of the nearest face
    exact = np.vstack([low.vertices[low.vertices[:, 2] < H - 5e-3], under[np.random.default_rng(0).choice(len(under), 3000, replace=False)]])
    cls = BRepClass3d_SolidClassifier(part)
    inside = 0
    for p in exact:  # the B-rep itself; slow, so the mesh vertices and a sample
        cls.Perform(P(p), 1e-7)
        inside += cls.State() == TopAbs_IN
    lands = (ru > TUBE_D / 2 - 0.3) & (np.abs(under[:, 0]) > THIN_X + 0.5)
    pads, plate = flat_region(fine, H, 1), flat_region(low, H, -1)
    contact = pads.intersection(plate)
    above = pts[(pts[:, 2] >= H - 5e-3) & (pts[:, 2] < H + 4)]
    table = sg.Polygon(table_outline()).difference(sg.Point(0, 0).buffer(rb + 0.6))
    off = above[shapely.contains_xy(table, above[:, 0], above[:, 1]) & ~shapely.contains_xy(pads.buffer(0.02), above[:, 0], above[:, 1])]  # hand over the table but not over a pad
    j = np.argmin(off[:, 2])
    L += ["四、与手掌网格（hand_root.stl，顶点加 40 万个表面采样点）",
          f"  座面以下的手的点 {len(under)} 个，最大半径 {ru.max():.3f}（只有止口，外径 {TUBE_D:g}）；在新件表面内侧的 {behind} 个（按最近面的法向，全部点），"
          f"用 B-rep 精确判断其中 {len(exact)} 个（全部网格顶点加抽样）在实体内的 {inside} 个（都应为 0）",
          f"  预定接触 1：手底面贴在两块凸台上，接触面积 {contact.area:.0f} mm2（凸台共 {pads.area:.0f} mm2，其余是手底板上螺钉的沉孔）",
          f"  预定接触 2：止口外圆在孔 Φ{fit['bore']:.2f} 里，±X 两段孔壁处单边间隙 {rb - ru[lands].max():.3f}（网格量得最小 {dist[lands].min():.3f}）；±Y 薄壁处让开，间隙 {rb + THIN_GAP - TUBE_D / 2:.2f}",
          f"  凸台以外的台面比凸台低 {SEAT_REL:g}：手在它上方最低 {off[j][2] - (H - SEAT_REL):.2f} mm（在 X={off[j][0]:.1f} Y={off[j][1]:.1f}，是手底面本身）；手的止口最低点 Z={under[:, 2].min():.2f}，下面是线腔"]

    # 5 the hand's two threads against the holes
    L.append("五、手的螺纹孔与新件的孔")
    hv = hand.vertices
    for sx, name in ((1, "A(+X，手的 +Z)"), (-1, "B(−X，手的 −Z)")):
        q = hv[(np.abs(hv[:, 1]) < 1.9) & (np.abs(hv[:, 2] - zh) < 1.9) & (sx * hv[:, 0] > 10.2) & (sx * hv[:, 0] < 12.8)][:, 1:]
        cx, cz, c0 = np.linalg.lstsq(np.column_stack([2 * q, np.ones(len(q))]), (q**2).sum(1), rcond=None)[0]
        L.append(f"  {name}：网格上螺纹轴线 Y={cx:+.3f}，Z={cz:.3f}，半径 {np.sqrt(c0 + cx * cx + cz * cz):.2f}；新件过孔 Φ{RAD['clear']:g}，轴线 Y=0，Z={zh:.3f}："
                 f"螺钉四周间隙 {(RAD['clear'] - RAD['d']) / 2 - np.hypot(cx, cz - zh):.2f}，即孔距 {HOLE_BELOW:g} 差 ±{(RAD['clear'] - RAD['d']) / 2:g} 都装得上，不会把手顶起")

    # 6 heads and drivers, as swept cylinders
    links = hand_links()
    grid = []
    if links:
        for a in np.linspace(*links[0]["thumb_swing"]["lim"], 8):
            for b in np.linspace(*links[0]["thumb_rot"]["lim"], 5):
                grid.append((dict(thumb_swing=a, thumb_rot=b), "thumb"))
        for f in np.linspace(0, 1, 7):
            grid.append(({k: f * links[0][k]["lim"][1] for k in ("index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp")}, "finger"))
    moving = []  # (pose, link, points) of the links that move in each pose
    for q, kind in grid:
        for name, v in pose_hand(links, q).items():
            if name.startswith("thumb") == (kind == "thumb"):
                moving.append((q, name, v))
    every = np.vstack([pts] + [v for _, _, v in moving]) if moving else pts

    def line_gap(p0, d, r0, r1, rad):
        """Smallest distance from the whole hand (palm and all poses) to a cylinder along d from p0 + r0 d to p0 + r1 d."""
        w = every - np.asarray(p0, float)
        t = np.clip(w @ np.asarray(d, float), r0, r1)
        return np.linalg.norm(w - np.outer(t, d), axis=1).min() - rad

    L.append("六、螺钉头和扳手通道（扫掠圆柱与新件的重叠体积 mm3；离手 = 离手掌网格和手指、拇指全部姿态的最小距离）")
    rj, tip = ring_screw_seat()
    rows = []
    for al in RING_AL:
        d = (*pol(1, al), 0)
        z = RING_HOLE_Z + E_RING
        sweep = radial(al, z, RING_R + 0.05, 150, RING_SCREW["dk"] / 2 + 0.1)
        rows.append((common(part, sweep), gap(part, sweep), common(ring, fast[f"ring{al}"]), gap(ring, fast[f"ring{al}"]), line_gap((0, 0, z), d, RING_R, 150, RING_SCREW["dk"] / 2 + 0.1)))
    L.append(f"  环的 4 颗径向螺钉（M3x{RING_SCREW['length']:g} 沉头，新件坐到环上后从环外侧拧，{RING_SCREW['key']:g} 扳手）：钉头通道 Φ{RING_SCREW['dk'] + 0.2:g} 从环外圆到半径 150 与新件重叠最大 {max(r[0] for r in rows):.4f}，"
             f"离新件最近 {min(r[1] for r in rows):.2f}（上方是肩面法兰）；就位的螺钉与环重叠 {max(r[2] for r in rows):.4f}、距离 {max(r[3] for r in rows):.3f}（钉头贴在沉孔上侧）；"
             f"通道最低 Z={RING_HOLE_Z + E_RING - RING_SCREW['dk'] / 2 - 0.1:.1f}，臂在 Z≤0；离手 {min(r[4] for r in rows):.1f} mm")
    for sx, name in ((1, "A"), (-1, "B")):
        d = (sx, 0, 0)
        sweep = cyl((sx * (TAB_X + 0.01), 0, zh), d, SPRING["de"] / 2 + 0.1, 120)
        seated = fuse(fast["draw" + name], fast["stack" + name])
        near = pts[(sx * pts[:, 0] > TUBE_D / 2 + 0.2) & (np.abs(pts[:, 1]) < 9) & (np.abs(pts[:, 2] - zh) < 9)]  # beside the screw, outside the hand's own thread
        L.append(f"  拉紧螺钉 {name}（{RAD['name']}x{RAD['length']:g} 圆柱头 + 垫圈 + {SPRING['n']} 片碟簧，沿 X 从台面端面拧，{RAD['key']:g} 扳手）：Φ{SPRING['de'] + 0.2:g} 通道与新件重叠 {common(part, sweep):.4f}；"
                 f"就位的螺钉、垫圈、碟簧与新件重叠 {common(part, seated):.4f}；它们离手掌网格最近 {trimesh.proximity.closest_point(tess(seated, 0.01), near)[1].min():.2f} mm；"
                 f"通道离手 {line_gap((0, 0, zh), d, TAB_X, 150, SPRING['de'] / 2 + 0.1):.1f} mm")
    rows = []
    for al in (270 - JACK_AL, 270 + JACK_AL):
        d = (*pol(1, al), 0)
        rows.append((common(part, radial(al, zh, JACK_R1 + 0.3, 150, JACK["key"] * 0.58 + 0.1)), line_gap((0, 0, zh), d, TAB_X, 150, JACK["key"] * 0.58 + 0.1)))
    L.append(f"  B 侧 2 颗顶丝（{JACK['name']}x{JACK['length']:g} 平端紧定，{JACK['key']:g} 扳手从台面斜角面的孔里进）：扳手通道 Φ{JACK['key'] * 1.16 + 0.2:.1f} 与新件重叠最大 {max(r[0] for r in rows):.4f}，离手 {min(r[1] for r in rows):.1f} mm")

    # 7 threads
    L.append("七、螺纹啮合")
    rg = RING_SCREW
    tips = [RING_R - (RING_CSK_R - 1.5) + e + k - rg["length"] - t for e, k, t in ((0.0, rg["k"][0], rg["tol"]), (E_RING, rg["k"][1], 0), (2 * E_RING, rg["k"][2], -rg["tol"]))]
    pull_r = 0.8 * 1000 / (0.26 * 3.0)
    as_n = np.pi * (3.0 - 0.25 * 0.5) * (0.5 + 0.57735 * 0.2)  # internal thread shear area per mm of M3 x 0.5
    L.append(f"  环的螺钉 M3x{rg['length']:g} 沉头：钉尖在半径 {tips[1]:.2f}（头高、长度、偏置的公差叠加 {tips[0]:.2f}..{tips[2]:.2f}），线腔壁在 {rb:.2f} → 钉尖离线腔最少 {tips[0] - rb:.2f}，不伸进线腔；"
             f"旋入新件 {rs - tips[1]:.2f}（{rs - tips[2]:.2f}..{rs - tips[0]:.2f}）。0.8 N·m（沉头 K≈0.26）→ 轴力 {pull_r:.0f} N，6061 螺纹（抗剪 207 MPa）拉脱 {as_n * (rs - tips[2] - 0.5) * 207:.0f} N，安全系数 {as_n * (rs - tips[2] - 0.5) * 207 / pull_r:.1f}。不要用 M3x16（钉尖会进线腔 {rb - (tips[0] - 2):.1f}）")
    free = RING_R - 0.8 - rs
    play = 0.02 / (rs - tips[1]) * free  # head movement that the smallest M3 6H/6g clearance allows
    L.append(f"  偏置：新件螺纹比环的孔高 {E_RING:g}，螺钉在环的 Φ{RING_HOLE_D:g} 孔里还有 {(RING_HOLE_D - 3) / 2 - E_RING:.2f} 的间隙；钉头只可能压在沉孔上侧，把肩面往下压，环的孔位差 ±{E_RING:g} 以内都不会把新件顶离环。"
             f"螺钉自由长约 {free:.1f}，偏摆 {np.degrees(E_RING / free):.2f}°，一般由螺纹间隙吸收；螺纹间隙最小时要弯 {E_RING - play:.3f}，根部应力约 {3 * 200000 * 2.39 * (E_RING - play) / (2 * free**2):.0f} MPa，"
             f"所以用 10.9 级以上的螺钉，偏置也不取更大")
    s_use = SPRING["h0"] * SPRING["use"]
    pull, flat_f = SPRING["n"] * spring_force(s_use), SPRING["n"] * spring_force(SPRING["h0"])
    tips = [TIP_NOM + t for t in (-RAD["tol"] - 0.15, 0.0, RAD["tol"] + 0.13)]
    engs = [TUBE_D / 2 - max(t, TUBE_ID_MANUAL / 2) for t in tips]
    strip = [as_n * (e - RAD["pitch"]) * 150.0 for e in (engs[2], engs[1])]  # one pitch lost to lead-ins; 150 MPa for an unknown cast alloy
    L.append(f"  手的拉紧螺钉 {RAD['name']}x{RAD['length']:g}：钉尖在半径 {tips[1]:.2f}（螺钉长度 ±{RAD['tol']:g}、垫圈碟簧、端面位置叠加 {tips[0]:.2f}..{tips[2]:.2f}），不得小于 {TUBE_ID_MODEL / 2:g} → 余量 {tips[0] - TUBE_ID_MODEL / 2:.2f}；"
             f"按手册壁厚 {(TUBE_D - TUBE_ID_MANUAL) / 2:g} 算旋入 {engs[1]:.2f}（{engs[2]:.2f}..{engs[0]:.2f}）")
    L.append(f"  预紧：{SPRING['n']} 片碟簧 {SPRING['de']:g}×{SPRING['di']:g}×{SPRING['t']:g} 同向叠放，压到自由高度的 {100 * SPRING['use']:.0f}%（{s_use:.3f} mm，约 {360 * s_use / RAD['pitch']:.0f}° 转角）时轴力 {pull:.0f} N，约 {0.2 * RAD['d'] * pull / 1000:.2f} N·m；"
             f"压平 {flat_f:.0f} N。手上 M3 内螺纹（铝，抗剪按 150 MPa、扣一个螺距）拉脱 {strip[0]:.0f}..{strip[1]:.0f} N：安全系数 {strip[0] / pull:.1f}..{strip[1] / pull:.1f}，压平时 {strip[0] / flat_f:.1f}..{strip[1] / flat_f:.1f}。"
             f"碟簧有 {1000 * s_use:.0f} µm 行程，贴合面沉陷几 µm 不掉预紧（没有碟簧时螺钉只伸长 {1000 * pull * (TAB_X - TUBE_D / 2) / (200000 * 5.03):.0f} µm）")
    L.append(f"  顶丝 {JACK['name']}x{JACK['length']:g}：螺纹段半径 {TUBE_D / 2:g}..{JACK_R1:g}（{JACK_R1 - TUBE_D / 2:g} 长），顶丝全在螺纹里；B 侧拉紧后每颗受 {pull / (2 * np.cos(np.radians(JACK_AL))):.0f} N")

    # 8 minimum wall
    c, n = coarse.triangles_center, coarse.face_normals
    p2, f2 = trimesh.sample.sample_surface(coarse, 40000, seed=1)
    o, nn = np.vstack([c, p2]), np.vstack([n, n[f2]])
    loc, ray, tri = coarse.ray.intersects_location(o - nn * 1e-4, -nn, multiple_hits=False)
    t, at = np.linalg.norm(loc - o[ray], axis=1), o[ray]
    opp = (nn[ray] * n[tri]).sum(1) < -0.5  # only between faces that face each other
    t, at = t[opp], at[opp]
    k = np.argmin(t)
    web = 2 * rb * np.sin(np.radians(JACK_AL / 2)) - JACK["d"] / 2 - RAD["clear"] / 2
    L += ["八、最小壁厚（沿法向向内射线，只算两面大致相对的地方）",
          f"  最薄 {t[k]:.2f} mm，在 X={at[k][0]:.1f} Y={at[k][1]:.1f} Z={at[k][2]:.1f}；厚度 < 1.0 的采样点占 {100 * (t < 0.995).mean():.2f}%，< 1.5 的占 {100 * (t < 1.5).mean():.1f}%",
          f"  算出的关键壁：拉紧螺钉过孔到台面（凸台以外）{HOLE_BELOW - RAD['clear'] / 2 - SEAT_REL:.2f}、到凸台面 {HOLE_BELOW - RAD['clear'] / 2:.2f}、到台底 {zh - RAD['clear'] / 2 - TABLE_BOT:.2f}；"
          f"顶丝螺纹与过孔之间（孔壁处）{web:.2f}；出线槽底 {WIN_FLOOR - RING_H:.2f}；肩面法兰 {FLANGE_TOP - RING_H:.2f}；颈部壁 {NECK_R - rb:.2f}；止口在平面处 {FLAT_D - rb:.2f}；"
          f"环螺钉螺纹孔到止口底 {RING_HOLE_Z + E_RING - 1.5 - SPIG_BOT:.2f}"]

    # 9 cable path
    zc = WIN_FLOOR + 3.6
    path = [(0, 0, H + 1.0), (0, 0, zc + 6)] + [(0, 5 - 5 * np.cos(a), zc + 1 + 5 * (1 - np.sin(a))) for a in np.radians((15, 30, 45, 60, 75, 90))] + [(0, 12, zc), (0, 34, zc)]
    cable = ball(path[0], 3.5)
    for a, b in zip(path[:-1], path[1:]):
        v = np.subtract(b, a)
        cable = fuse(cable, cyl(a, v, 3.5, np.linalg.norm(v)), ball(b, 3.5))
    plug = fuse(box((-6, -7.5, WIN_FLOOR + 0.5), (6, 40, WIN_FLOOR + 8.5)), box((-6, -7.5, WIN_FLOOR + 0.5), (6, 7.5, H)))
    inner = pts[(np.hypot(pts[:, 0], pts[:, 1]) < 12.5) & (pts[:, 2] < H + 1.0)]
    L += ["九、走线（扫掠体）",
          f"  线束 Φ7（6 根线，以后加 5 根也够）：从手的止口中心向下、半径 5 的弯、穿出线口、沿法兰上的槽出去。与新件重叠 {common(part, cable):.3f} mm3，最近 {gap(part, cable):.2f} mm（线贴着槽底，模型里留 0.1）；"
          f"离环最近 {gap(ring, cable):.2f} mm（线始终在环顶面 Z={RING_H:g} 以上）；离手的止口内壁 {(TUBE_ID_MODEL / 2 - 3.5):.1f} mm（采样最小内半径 {np.hypot(inner[:, 0], inner[:, 1]).min():.2f}）",
          f"  插头 12×8×15：平着穿过出线口再竖直进止口，与新件重叠 {common(part, plug):.3f} mm3；出线口宽 {2 * WIN_W:g}、高 {TABLE_BOT - WIN_FLOOR:g}（12×8 的余量：宽 {WIN_W - 6:.2f}/边，高 {(TABLE_BOT - WIN_FLOOR - 8) / 2:.2f}/边）",
          f"  线腔：Φ{fit['bore']:.2f}，从手的止口端面 Z={H - TUBE_LEN:g} 向下到止口底 Z={SPIG_BOT:g}，高 {H - TUBE_LEN - SPIG_BOT:g}，底下敞开对着环的孔底（Z={RING_FLOOR:g}）；环的螺钉不进线腔（见七）。"
          f"扎带孔 Φ3.4 两个（al=±{TIE_AL:g}，Z={TIE_Z:g}）：扎带穿孔和出线口，把线勒在颈部外壁上"]

    # 10 mass
    m_al = vol * RHO["6061 铝"] / 1000
    m_fast = 4 * 0.62 + 2 * 1.45 + 2 * 0.1 + 6 * 0.04 + 2 * 0.35
    m_ring = volume(ring) * 2.70 / 1000
    L += ["十、质量和质心（原点在臂内盘端面中心，Z 朝手）",
          "  " + "；".join(f"{k} {vol * v / 1000:.1f} g" for k, v in RHO.items()) + f"；质心 X={com[0]:.2f} Y={com[1]:.2f} Z={com[2]:.2f} mm",
          f"  螺钉、垫圈、碟簧约 {m_fast:.1f} g；松灵的环（铝，按模型体积）{m_ring:.0f} g；铝件 + 紧固件 + 环 + 手 {HAND_KG * 1000:.0f} g（手册值，不含线）= {m_al + m_fast + m_ring + HAND_KG * 1000:.0f} g，臂的末端负载 1500 g"]

    # 11 the whole hand, posed
    L.append("十一、整只手摆姿态（telehand_lite 的 URDF，联动关系取自 MJCF）")
    worst = None
    if links:
        obstacle = trimesh.util.concatenate([coarse, tess(ring, 0.05)] + [tess(f, 0.05) for f in fast.values()])
        low_z = {"thumb": 1e9, "finger": 1e9}
        for q, name, v in moving:
            kind = "thumb" if name.startswith("thumb") else "finger"
            low_z[kind] = min(low_z[kind], v[:, 2].min() - H)
            near = v[np.argsort(v[:, 2])[:150]]  # everything in the way is below the seat plane, so the lowest points decide
            if near[0, 2] < H + 40:
                dmin = trimesh.proximity.closest_point(obstacle, near)[1].min()
                if worst is None or dmin < worst[0]:
                    worst = (dmin, name, q)
        L += [f"  拇指 2 个主动关节（摆动 {np.degrees(links[0]['thumb_swing']['lim'][0]):.0f}..{np.degrees(links[0]['thumb_swing']['lim'][1]):.0f}°，旋转 {np.degrees(links[0]['thumb_rot']['lim'][0]):.0f}..0°，"
              f"屈曲两节按联动）取 8×5 个姿态；四指从伸直到握紧 7 个姿态。共 {len(moving)} 个连杆姿态",
              f"  拇指各节最低到座面以上 {low_z['thumb']:.1f} mm，四指最低到座面以上 {low_z['finger']:.1f} mm：都到不了座面以下，新件全部在座面以下",
              f"  离新件、环、螺钉最近 {worst[0]:.1f} mm（{worst[1]}，" + "，".join(f"{k}={np.degrees(a):.0f}°" for k, a in worst[2].items()) + "）；三种扳手通道离手的距离见六" if worst else "  没有连杆进入新件上方 25 mm 以内"]
    else:
        L.append(f"  没有找到 {HAND_DIR}，这一项没有检查")

    # 12 holding force and compliance, by hand
    zt = TIP_Y - FACE_Y + H
    A_s, c_s, ixx_s, iyy_s, xm_s, ym_s = area_moments(contact)
    A_r, c_r, ixx_r, iyy_r, _, _ = area_moments(shoulder)
    zs = np.arange(RING_H + 0.5, H, 1.0)
    inv = np.zeros(2)
    for z in zs:  # bending of the body between the ring's top and the seat
        _, _, ixx, iyy, _, _ = area_moments(loops_polygon([q[:, :2] for q in coarse.section(plane_origin=(0, 0, z), plane_normal=Z).discrete]), 0.1)
        inv += 1.0 / np.array([iyy, ixx])
    mu = 0.15
    hold = 4 * mu * pull
    arm_y = ym_s * 0.65  # from the screw line to the resultant on the loaded half of the pads
    m_x, m_y = hold * arm_y / 1000, mu * pull * sum(abs(x + xm_s) for x in (-TAB_X, -rb, rb, TAB_X)) / 1000
    m_tap, m_g = 2 * (zt - H) / 1000, HAND_KG * 9.81 * 0.06
    L += ["十二、夹持力和指尖位移（手算估计；不含手本身、松灵的环和它与内盘的结合）",
          f"  手与新件：靠摩擦，没有楔紧。两颗拉紧螺钉各 {pull:.0f} N，摩擦面 4 处（A 侧孔壁、B 侧顶丝尖、两端垫圈），摩擦系数按 {mu:g}：轴向能抗 {hold:.0f} N（手重 {HAND_KG * 9.81:.0f} N）；"
          f"绕 X 轴（指腹方向受力）能抗 {m_x:.2f} N·m，绕 Y 轴 {m_y:.1f} N·m，绕腕轴 {mu * pull * 2 * (rb + TAB_X) / 1000:.1f} N·m",
          f"  载荷：点击 2 N × {zt - H:.0f} mm = {m_tap:.2f} N·m，手自重力矩最大约 {m_g:.2f} N·m（质心按离座面 60 mm 估），绕腕轴 {2 * abs(TIP_INDEX[0]) / 1000:.2f} N·m。"
          f"最弱方向（绕 X）安全系数：只算点击 {m_x / m_tap:.1f}，点击加自重 {m_x / (m_tap + m_g):.1f}；摩擦系数若是 0.3 则翻倍。万一打滑，螺钉在 Φ{RAD['clear']:g} 孔里最多走 {(RAD['clear'] - RAD['d']) / 2:g} 就顶住，手掉不下来，但要重新标定",
          f"  新件与环：4 颗沉头螺钉各 {pull_r:.0f} N 坐在环的 90° 沉孔里，位置和转角由锥面定死；肩面接触 {A_r:.0f} mm2，∫y² {iyy_r:.0f}、∫x² {ixx_r:.0f} mm4",
          f"  食指尖：离座面 {zt - H:.1f}，离环顶面 {zt - RING_H:.1f}，离内盘端面 {zt:.1f}，偏离腕轴 {abs(TIP_INDEX[0]):.1f}。座面接触 {A_s:.0f} mm2，∫y² {iyy_s:.0f}、∫x² {ixx_s:.0f} mm4（座面没有预紧，只算受压的一半）"]
    for label, kn_seat, kn_ring in (("保守", 50.0, 200.0), ("一般", 200.0, 800.0)):  # contact stiffness per area, N/mm3
        row = []
        for name, i_s, i_r, ib in (("沿手的 X（指腹方向）", iyy_s, iyy_r, inv[0]), ("沿手的 Z（侧向）", ixx_s, ixx_r, inv[1])):
            d_seat = 2 * (zt - H) ** 2 / (kn_seat * i_s / 2)
            d_ring2 = 2 * (zt - RING_H) ** 2 / (kn_ring * i_r)
            d_body = 2 * (zt - (RING_H + H) / 2) ** 2 * ib / E_MOD["6061 铝"]
            row.append(f"{name} {1000 * (d_seat + d_ring2 + d_body):.0f} µm（座面 {1000 * d_seat:.0f}，环顶面 {1000 * d_ring2:.1f}，本体 {1000 * d_body:.2f}）")
        L.append(f"  2 N 时指尖位移，{label}（结合面刚度 座面 {kn_seat:g}、环顶面 {kn_ring:g} N/mm3）：" + "；".join(row))
    L.append(f"  打印件本体项乘 {E_MOD['6061 铝'] / E_MOD['PA12']:.0f}（PA12）/ {E_MOD['6061 铝'] / E_MOD['PETG-CF']:.0f}（PETG-CF），即 {1000 * 2 * (zt - (RING_H + H) / 2) ** 2 * inv[0] / E_MOD['PA12']:.0f} / {1000 * 2 * (zt - (RING_H + H) / 2) ** 2 * inv[0] / E_MOD['PETG-CF']:.0f} µm")
    L.append(f"  几何误差：凸台面在窄方向（{2 * ym_s:.0f} mm）每歪 0.01，指尖走 {0.01 / (2 * ym_s) * (zt - H):.3f} mm；在长方向（{2 * xm_s:.0f} mm）每歪 0.01，指尖走 {0.01 / (2 * xm_s) * (zt - H):.3f} mm；"
             f"肩面在 Φ{2 * RING_R:g} 上每歪 0.01，指尖走 {0.01 / (2 * RING_R) * (zt - RING_H):.3f} mm；手绕腕轴每转 0.1°，指尖走 {np.radians(0.1) * abs(TIP_INDEX[0]):.3f} mm")
    thumb = {k: m for k, m in pose_hand(links, worst[2], meshes=True).items() if k.startswith("thumb")} if links and worst else {}
    num = dict(seat=contact.area, shoulder=shoulder.area, pull=pull, torque=0.2 * RAD["d"] * pull / 1000, sf_tap=m_x / m_tap, sf_all=m_x / (m_tap + m_g), eng_ring=rs - ring_screw_seat()[1], pad_x=2 * xm_s, pad_y=2 * ym_s)
    return L, dict(hand=hand, arm=arm, fast=fast, ring=ring, contact=contact, pads=pads, plate=plate, shoulder=shoulder, cable=cable, coarse=coarse, thumb=thumb, num=num)


def render(part, fit, extra, out):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path
    import matplotlib.pyplot as plt

    for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"):
        if pathlib.Path(f).exists():
            font_manager.fontManager.addfont(f)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    rb, rs, zh, zr = fit["bore"] / 2, fit["spigot"] / 2, H - HOLE_BELOW, RING_HOLE_Z + E_RING
    hand, arm, fast, ring = extra["hand"], extra["arm"], extra["fast"], extra["ring"]
    hand_low = hand.submesh([np.nonzero(hand.triangles[:, :, 2].min(axis=1) < H + 30)[0]], append=True)
    arm_top = arm.submesh([np.nonzero(arm.triangles[:, :, 2].max(axis=1) > -30)[0]], append=True)
    BLUE, GREY, DARK, RING, STEEL, JACK_C, ORANGE = (0.36, 0.56, 0.86), (0.84, 0.84, 0.82), (0.42, 0.42, 0.45), (0.72, 0.74, 0.78), (0.9, 0.72, 0.25), (0.85, 0.3, 0.3), (0.95, 0.5, 0.1)
    colour = lambda k: JACK_C if k.startswith("jack") else STEEL
    solids = [(part, BLUE), (ring, RING), (extra["cable"], ORANGE)] + [(s, colour(k)) for k, s in fast.items()]
    shells = [(hand_low, GREY, True), (arm_top, DARK, True)] + [(m, (0.78, 0.78, 0.76), True) for m in extra["thumb"].values()]

    def shot(ax, items, v, centre, half, title, px=800, keep=None):
        """Orthographic picture by splatting surface samples into a depth buffer."""
        v = np.array(v, float) / np.linalg.norm(v)
        right = np.cross(Z, v) / np.linalg.norm(np.cross(Z, v))
        R = np.array([right, np.cross(v, right), v])
        l1, l2 = np.array([-0.45, 0.55, 0.7]) / np.linalg.norm([-0.45, 0.55, 0.7]), np.array([0.75, -0.1, 0.65]) / np.linalg.norm([0.75, -0.1, 0.65])
        scale = px / (2 * half)
        F, D, C, N = [], [], [], []
        for mesh, rgb, shell in items:
            near = np.nonzero((np.abs((mesh.triangles_center - centre) @ R[:2].T) < half + 8).all(axis=1))[0]  # only what is in the picture
            if not len(near):
                continue
            w = mesh.area_faces[near]
            f = near[np.random.default_rng(2).choice(len(near), int(w.sum() * 4 * scale**2) + 100, p=w / w.sum())]
            u = np.random.default_rng(3).random((len(f), 2))
            flip = u.sum(1) > 1
            u[flip] = 1 - u[flip]
            t = mesh.triangles[f]
            p = t[:, 0] + u[:, :1] * (t[:, 1] - t[:, 0]) + u[:, 1:] * (t[:, 2] - t[:, 0])
            n = mesh.face_normals[f]
            if keep is not None:
                k = keep(p)
                p, n = p[k], n[k]
            q, nc = (p - centre) @ R.T, n @ R.T
            if shell:  # open mesh: light both sides
                nc = nc * np.sign(nc[:, 2:3] + 1e-9)
            iu, iv = np.floor((q[:, 0] + half) * scale).astype(int), np.floor((half - q[:, 1]) * scale).astype(int)
            ok = (iu >= 0) & (iu < px) & (iv >= 0) & (iv < px) & (nc[:, 2] > -0.02)
            sh = 0.30 + 0.55 * np.clip(nc[ok] @ l1, 0, 1) + 0.22 * np.clip(nc[ok] @ l2, 0, 1)
            F.append(iv[ok] * px + iu[ok]), D.append(q[ok, 2]), C.append(np.outer(sh, rgb)), N.append(nc[ok])
        F, D, C, N = np.concatenate(F), np.concatenate(D), np.concatenate(C), np.concatenate(N)
        o = np.argsort(D)
        img, zb, nb = np.ones((px * px, 3)), np.full(px * px, -np.inf), np.zeros((px * px, 3))
        img[F[o]], zb[F[o]], nb[F[o]] = C[o], D[o], N[o]
        img, zb, nb = img.reshape(px, px, 3), zb.reshape(px, px), nb.reshape(px, px, 3)
        for _ in range(2):  # pixels no sample fell on take their nearest neighbour
            best, bi, bn, near = zb.copy(), img.copy(), nb.copy(), np.zeros((px, px))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    z2 = np.roll(zb, (dy, dx), (0, 1))
                    near += z2 > zb + 0.6
                    k = z2 > best
                    best[k], bi[k], bn[k] = z2[k], np.roll(img, (dy, dx), (0, 1))[k], np.roll(nb, (dy, dx), (0, 1))[k]
            k = near >= 5
            img[k], zb[k], nb[k] = bi[k], best[k], bn[k]
        edge = np.zeros((px, px), bool)
        for axis in (0, 1):
            z2, n2 = np.roll(zb, 1, axis), np.roll(nb, 1, axis)
            fin = np.isfinite(zb) & np.isfinite(z2)
            step = np.abs(np.where(fin, zb, 0) - np.where(fin, z2, 0))
            edge |= (np.isfinite(zb) ^ np.isfinite(z2)) | (step > 0.9) | (fin & ((nb * n2).sum(-1) < 0.55))  # outline, depth step, crease
        edge[0], edge[:, 0] = False, False
        img[edge] *= 0.35
        ax.imshow(img, interpolation="lanczos")
        ax.set_title(title, fontsize=13)
        ax.axis("off")

        def note(text, p, at, **kw):
            q = (np.array(p, float) - centre) @ R.T
            ax.annotate(text, xy=((q[0] + half) * scale, (half - q[1]) * scale), xytext=(at[0] * px, at[1] * px), fontsize=10, ha="center", va="center",
                        arrowprops=dict(arrowstyle="-", lw=0.8, color="k"), bbox=dict(fc="white", ec="0.5", lw=0.5, pad=1.5), **kw)

        return note

    def meshes(cutter=None):
        got = [(tess(s if cutter is None else cut(s, cutter), 0.03), c, False) for s, c in solids]
        return [g for g in got if isinstance(g[0], trimesh.Trimesh)]  # a screw wholly inside the cut-away quarter leaves nothing

    fig = plt.figure(figsize=(23, 22.5), dpi=100)
    gs = fig.add_gridspec(3, 3, left=0.02, right=0.99, top=0.955, bottom=0.03, wspace=0.08, hspace=0.12)
    fig.suptitle(f"转接法兰第二版：松灵法兰环 → TeleHand Lite（左手）。内盘端面到手底面 {H:g} mm（环 {RING_H:g} + 新件 {H - RING_H:g}），一件，6061 铝 {volume(part) * RHO['6061 铝'] / 1000:.0f} g", fontsize=17)

    # --- 3D: assembly, quarter cut away, the part alone from above and from below ---
    whole = meshes()
    note = shot(fig.add_subplot(gs[0, 0]), [w for w in whole if w[1] != ORANGE] + shells, (0.5, 0.8, 0.22), (0, 0, 20), 58, "装配：手腕（拇指在最低的姿态）、新件、松灵的环、J6 末端",
                keep=lambda p: (p[:, 2] < H + 30) & (p[:, 2] > -28))
    note("手的底板", (0, 14, H + 2), (0.14, 0.33))
    note("拉紧螺钉 A\nM3x25 + 碟簧", (TAB_X + 3, 0, zh), (0.9, 0.42))
    note("新件", (-10, 16, 17), (0.12, 0.5))
    note("出线口", (0, NECK_R, 16), (0.8, 0.88))
    note("松灵的环（不动）", (-8, 27, 5), (0.14, 0.64))
    note("环的径向螺钉 ×4\nM3x14 沉头", (*pol(RING_R, 45), zr), (0.88, 0.66))
    note("J6 外壳", (0, 28, -12), (0.14, 0.8))
    quarter = box((0, 0, -60), (100, 100, 120))
    note = shot(fig.add_subplot(gs[0, 1]), meshes(quarter) + shells, (0.62, 0.72, 0.36), (0, 0, 15), 42, "切掉四分之一（X>0、Y>0）",
                keep=lambda p: ~((p[:, 0] > 1e-6) & (p[:, 1] > 1e-6)) & (p[:, 2] < H + 30) & (p[:, 2] > -20))  # the cut faces themselves stay
    note("线束 Φ7", (0, 14, WIN_FLOOR + 3.6), (0.9, 0.62))
    note(f"线腔\nΦ26 × {H - TUBE_LEN - SPIG_BOT:g}", (0, -rb, 12), (0.12, 0.55))
    note("手的止口", (0, -TUBE_D / 2, H - 5), (0.1, 0.36))
    note("拉紧螺钉 A", (24, 0, zh), (0.14, 0.2))
    note("止口 Φ40 在环里", (rs * 0.7, 0, 7), (0.14, 0.72))
    note("肩面压环顶面", (0, 24, RING_H), (0.9, 0.8))
    alone = [(tess(part, 0.03), BLUE, False)]
    note = shot(fig.add_subplot(gs[0, 2]), alone, (0.5, 0.62, 0.6), (0, 0, 17), 40, "零件：从手一侧看")
    note("两块凸台（手底面的座）", (-18, 0, H), (0.2, 0.1))
    note("拉紧螺钉过孔 Φ3.6\n端面放垫圈和碟簧", (TAB_X, 0, zh), (0.83, 0.2))
    note("B 侧顶丝孔 ×2", (*pol(TAB_X - 2, 270 - JACK_AL), zh), (0.12, 0.36))
    note("出线口 14.5×9 和槽", (0, NECK_R, 16), (0.84, 0.9))
    note("扎带孔", (*pol(NECK_R, TIE_AL), TIE_Z), (0.92, 0.66))
    note("肩面法兰 Φ57", (-20, 18, FLANGE_TOP), (0.12, 0.72))
    note("凹坑 = A 侧", (TAB_X - 2.6, 3.2, H - SEAT_REL), (0.9, 0.34))
    note = shot(fig.add_subplot(gs[2, 2]), alone, (0.45, 0.6, -0.66), (0, 0, 17), 40, "零件：从臂一侧看")
    note(f"止口 Φ{fit['spigot']:.2f} × {RING_H - SPIG_BOT:g}\n四段圆柱面 + 四个平面", (*pol(rs, 315), 6), (0.2, 0.86))
    note("M3 螺纹 ×4（接环的螺钉）", (*pol(rs, 45), zr), (0.8, 0.9))
    note("肩面（倾角基准）", (0, -24, RING_H), (0.2, 0.2))
    note("线腔，底下敞开", (0, 0, SPIG_BOT), (0.84, 0.7))
    note("手的台面", (-TAB_X + 4, 0, TABLE_BOT), (0.12, 0.5))

    # --- 2D sections ---
    coarse = extra["coarse"]
    fm = {k: tess(s, 0.02) for k, s in fast.items()}
    ring_mesh = tess(ring, 0.02)

    def draw(ax, g, sign=1, **kw):
        for p in getattr(g, "geoms", [g]):
            if p.geom_type == "Polygon" and not p.is_empty:
                verts, codes = [], []
                for loop in (p.exterior, *p.interiors):
                    xy = np.asarray(loop.coords) * (sign, 1)
                    verts += list(xy)
                    codes += [Path.MOVETO] + [Path.LINETO] * (len(xy) - 2) + [Path.CLOSEPOLY]
                ax.add_patch(PathPatch(Path(verts, codes), **kw))
            elif p.geom_type == "LineString":
                xy = np.asarray(p.coords)
                ax.plot(sign * xy[:, 0], xy[:, 1], **{"color": "0.25", "lw": 0.6, **kw})

    def dim(ax, a, b, text, at=None, fs=10, color="k", rot=0):
        ax.annotate("", xy=a, xytext=b, arrowprops=dict(arrowstyle="<|-|>", lw=0.8, color=color, shrinkA=0, shrinkB=0, mutation_scale=7))
        at = at or ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        ax.text(*at, text, fontsize=fs, color=color, ha="center", va="center", rotation=rot, bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.9))

    def lead(ax, text, p, at, fs=10, color="k"):
        ax.annotate(text, xy=p, xytext=at, fontsize=fs, color=color, ha="center", va="center", arrowprops=dict(arrowstyle="->", lw=0.7, color=color),
                    bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.9))

    def section(ax, right, left, title, xlim, ylim):
        for al, sign in ((right, 1), (left, -1)):
            draw(ax, half_section(arm_top, al, solid=False), sign, color="0.3", lw=0.9)
            draw(ax, half_section(ring_mesh, al), sign, fc=RING, ec="k", lw=0.7, hatch="\\\\\\", alpha=0.9)
            draw(ax, half_section(coarse, al), sign, fc=BLUE, ec="k", lw=0.8, hatch="///", alpha=0.9)
            for k, m in fm.items():
                draw(ax, half_section(m, al), sign, fc=colour(k), ec="k", lw=0.6)
            draw(ax, half_section(hand_low, al, solid="closed"), sign, fc="0.88", ec="none")
            draw(ax, half_section(hand_low, al, solid=False), sign)
        ax.axvline(0, color="k", lw=0.5, ls="-.")
        ax.set_aspect("equal")
        ax.set_xlim(*xlim), ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=13)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25, lw=0.4)

    ax = fig.add_subplot(gs[1, 0])
    section(ax, 0, 180, "剖面 Y–Z（过出线口；右 = 手背 +Y，新件的平面对着环的钥匙面）", (-44, 46), (-22, 42))
    dim(ax, (-38, 0), (-38, H), f"{H:g}", rot=90)
    ax.plot([-39, -TABLE_Y], [H, H], "k", lw=0.4), ax.plot([-39, -RING_R], [0, 0], "k", lw=0.4)
    dim(ax, (-33, 0), (-33, RING_H), f"环 {RING_H:g}", rot=90)
    dim(ax, (-33, RING_H), (-33, H), f"新件 {H - RING_H:g}", rot=90)
    dim(ax, (-4, SPIG_BOT), (-4, H - TUBE_LEN), f"线腔 {H - TUBE_LEN - SPIG_BOT:g}", rot=90)
    dim(ax, (-4, H - TUBE_LEN), (-4, H), f"手的止口 {TUBE_LEN:g}", rot=90)
    dim(ax, (-rb, 15.5), (rb, 15.5), f"Φ{fit['bore']:.2f}", at=(5, 15.5))
    dim(ax, (39, WIN_FLOOR), (39, TABLE_BOT), f"出线口 {TABLE_BOT - WIN_FLOOR:g}", at=(39, TABLE_BOT + 2))
    dim(ax, (33, RING_H), (33, WIN_FLOOR), "", color="r")
    ax.text(40, RING_H - 1.5, f"槽底厚 {WIN_FLOOR - RING_H:g}", fontsize=10, color="r", ha="center")
    for z, r, t in ((-11, RING_BORE_R, "Φ40 止口"), (-14.5, RING_R, f"Φ{2 * RING_R:g} 环和肩面法兰"), (-18, TAB_X, f"台面长 {2 * TAB_X:.1f}（在 X 方向）")):
        dim(ax, (-r, z), (r, z), t)
    lead(ax, "环的钥匙面，间隙 1.0", (RING_FLAT - 0.5, 6.5), (37, 6.5))
    lead(ax, "环底的限位槽\n（J6 限位销在里面走）", (25, 2), (38, -6))
    lead(ax, f"薄壁处让开 {THIN_GAP:g}", (-rb - 0.07, H - 4), (-27, 36))
    lead(ax, "内盘（端面 Z=0）", (-6, -0.5), (-10, -6))
    lead(ax, "J6 外壳", (-24, -1.5), (-38, -6))
    ax.text(0, 39, "手", fontsize=12, ha="center"), ax.text(-24.5, 5, "环", fontsize=11, ha="center")

    ax = fig.add_subplot(gs[1, 1])
    section(ax, 90, 270, "剖面 X–Z（过两颗拉紧螺钉；右 = A 侧，手的 +Z）", (-46, 46), (-10, 44))
    ax.plot([-44, 44], [zh, zh], "k", lw=0.5, ls="-.")
    dim(ax, (43, zh), (43, H), f"{HOLE_BELOW:.2f}", at=(43, H + 1.8))
    dim(ax, (0, -5), (TAB_X, -5), f"台面端面 {TAB_X:g}", at=(TAB_X / 2, -7))
    dim(ax, (-SEAT_C - SEAT_R, 34), (-SEAT_IN, 34), "凸台", at=(-21, 35.8))
    lead(ax, f"A：M3x{RAD['length']:g} + 垫圈 + {SPRING['n']} 片碟簧\n先拧，约 0.2 N·m（碟簧将平）", (TAB_X + 2.5, zh - 1), (32, 16))
    lead(ax, "B：两颗顶丝碰上后再拧", (-TAB_X - 2.5, zh - 1), (-32, 16))
    lead(ax, f"过孔 Φ{RAD['clear']:g}：手上孔距差 ±0.3 也行", (24, zh + 1.6), (27, 41))
    lead(ax, f"钉尖 r{TIP_NOM:g}，不过内壁", (TIP_NOM, zh - 1.4), (0, 16.5))
    lead(ax, "A 侧孔壁是基准", (rb, H - 6), (0, 12))
    lead(ax, f"台面比凸台低 {SEAT_REL:g}", (-31, H - SEAT_REL), (-30, 41))
    ax.text(0, 40.5, "手", fontsize=12, ha="center"), ax.text(-24.5, 5, "环", fontsize=11, ha="center")

    ax = fig.add_subplot(gs[1, 2])
    section(ax, 45, 225, "剖面 al = 45° / 225°（过环的两颗径向螺钉和止口的圆柱面）", (-44, 44), (-20, 40))
    lead(ax, f"M3x{RING_SCREW['length']:g} 沉头\n从环外侧拧\n2.0 扳手，0.8 N·m", (RING_R + 1, zr), (36, 19))
    lead(ax, "肩面压在环顶面上", (24.5, RING_H), (31, 28))
    lead(ax, f"止口 Φ{fit['spigot']:.2f}\n在环的 Φ40 孔里", (-rs, 8.6), (-34, 20))
    lead(ax, f"离孔底 {SPIG_BOT - RING_FLOOR:g}", (-16, SPIG_BOT - 0.25), (-30, -8))
    lead(ax, "线腔（钉尖不进来）", (8, 8), (6, -8))
    ax.text(0, 37, "手", fontsize=12, ha="center")

    # detail of one ring screw
    ax = fig.add_subplot(gs[2, 0])
    section(ax, 45, 225, f"放大：环的径向螺钉（新件的螺纹比环的孔高 {E_RING:g}，钉头只压沉孔上侧）", (11.5, 31.5), (1.5, 15.5))
    tip = ring_screw_seat()[1]
    ax.plot([11.5, 31.5], [zr, zr], "k", lw=0.6, ls="-.")
    ax.plot([19.5, 31.5], [RING_HOLE_Z, RING_HOLE_Z], "r", lw=0.6, ls="-.")
    ax.text(31.3, zr + 0.25, "螺纹轴线", fontsize=9, ha="right"), ax.text(31.3, RING_HOLE_Z - 0.65, "环的孔轴线", fontsize=9, ha="right", color="r")
    dim(ax, (30.3, RING_HOLE_Z), (30.3, RING_H), "3.50", at=(30.3, RING_H + 0.8), color="r")
    dim(ax, (rb, 11.6), (tip, 11.6), "", color="g")
    ax.text(15.4, 12.4, f"钉尖离线腔 {tip - rb:.2f}", fontsize=9, color="g", ha="center", bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.9))
    dim(ax, (tip, 14.2), (rs, 14.2), f"旋入 {rs - tip:.2f}", at=((tip + rs) / 2, 14.9))
    lead(ax, "肩面（倾角基准）", (25, RING_H), (26.5, 14.5))
    lead(ax, "向下压", (27.55, zr + 2.25), (24.2, 12.4), color="r")
    lead(ax, "环（松灵，不改）", (24, 4.4), (27, 2.4))
    lead(ax, "新件的止口", (18.5, 9.4), (20.5, 12.9))
    lead(ax, "线腔", (12.3, 6.5), (12.6, 9.6))
    lead(ax, f"离孔底 {SPIG_BOT - RING_FLOOR:g}", (16, SPIG_BOT - 0.25), (16, 2.2))

    # plan view
    ax = fig.add_subplot(gs[2, 1])
    t = np.radians(np.arange(361))
    top = sg.Polygon(table_outline()).union(sg.Point(0, 0).buffer(NECK_R, 128)).difference(sg.Point(0, 0).buffer(rb, 128))
    ax.plot(RING_R * np.sin(t), RING_R * np.cos(t), color="0.45", lw=0.8)
    draw(ax, top, fc=(0.75, 0.83, 0.95), ec="k", lw=0.8)
    draw(ax, extra["pads"], fc="none", ec="k", lw=0.8)
    draw(ax, extra["contact"], fc=(0.3, 0.75, 0.4), ec="none", alpha=0.75)
    for g in getattr(extra["plate"], "geoms", [extra["plate"]]):
        ax.plot(*np.asarray(g.exterior.coords).T, color="k", lw=1.0, ls="--")
    ax.add_patch(plt.Polygon([(-WIN_W, NECK_R - 1), (-WIN_W, RING_R), (WIN_W, RING_R), (WIN_W, NECK_R - 1)], fc=(0.95, 0.85, 0.7), ec="k", lw=0.6))
    for sx, name in ((1, "A"), (-1, "B")):
        ax.annotate("", xy=(sx * (TAB_X + 5), 0), xytext=(sx * 47, 0), arrowprops=dict(arrowstyle="->", lw=1.3, color="r"))
        ax.text(sx * 41, 2.4, f"{name} 侧 2.5 扳手", fontsize=10, color="r", ha="center")
    for al in (270 - JACK_AL, 270 + JACK_AL):
        ax.annotate("", xy=pol(JACK_R1 + 1, al), xytext=pol(45, al), arrowprops=dict(arrowstyle="->", lw=1.0, color=JACK_C))
        ax.plot(*np.array([pol(TUBE_D / 2, al), pol(JACK_R1, al)]).T, color=JACK_C, lw=2.5)
    ax.text(-37, -19.5, "顶丝 ×2\n1.5 扳手", fontsize=10, color=JACK_C, ha="center")
    for al in RING_AL:
        ax.annotate("", xy=pol(RING_R + 0.5, al), xytext=pol(44, al), arrowprops=dict(arrowstyle="->", lw=1.0, color="0.3", ls="--"))
        ax.text(*pol(48, al), f"环螺钉\nal={al}°", fontsize=9, ha="center", va="center", color="0.3")
    for al in (TIE_AL, -TIE_AL):
        ax.plot(*np.array([pol(rb, al), pol(NECK_R, al)]).T, color="k", lw=1.5)
    ax.annotate("", xy=(0, 40), xytext=(0, rb), arrowprops=dict(arrowstyle="->", lw=1.3, color=ORANGE))
    ax.text(0, 42, "出线（al = 0，手背）", fontsize=10, color=ORANGE, ha="center")
    ax.text(0, -35.5, f"虚线：手底板外形    绿：凸台上实际接触（{extra['contact'].area:.0f} mm²）", fontsize=10, ha="center")
    ax.text(0, -40, "灰圆：环和肩面法兰 Φ57（环螺钉在法兰下面 Z=7 处）", fontsize=10, ha="center", color="0.3")
    ax.set_aspect("equal")
    ax.set_xlim(-54, 54), ax.set_ylim(-44, 46)
    ax.set_xlabel("X = 手的 +Z（小指侧）→", fontsize=10), ax.set_ylabel("Y = 手的 +X（手背）→", fontsize=10)
    ax.set_title("从手一侧看：凸台、台面、各螺钉的方向（al 从 +Y 向 +X 量）", fontsize=13)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, lw=0.4)
    fig.savefig(out / "render.png")
    plt.close(fig)


HEAD = """转接法兰第二版：松灵法兰环 → TeleHand Lite（左手）检查报告
由 build_hand_flange.py 生成；尺寸 mm。坐标：原点在臂内盘端面中心（也是松灵法兰环的底面），Z 沿腕轴朝手，X = 手的 +Z，Y = 手的 +X（手背，出线口）。
角度 al 从 +Y 向 +X 量。图见 render.png。第一版（不用环，直接锁在内盘上）在 out_v1/ 和 build_hand_flange_v1.py。

结构（一件，6061 铝）
  内盘端面到手底面 {H:g}：松灵的环 {ring_h:g} + 新件 {above:g}。环留在臂上不动（它底面的槽多半是 J6 的机械限位）。换工具只动环上 4 颗径向螺钉。
  下端照夹爪底座的接口：Φ{spig:.2f} 止口插进环的 Φ40 孔（长 {spig_l:g}，离孔底 {spig_gap:g}）；Φ{flange:g} 肩面法兰压在环顶面上；
  4 颗 M3x{ring_l:g} 沉头螺钉从环外侧穿过环的沉孔，拧进止口上的 M3 螺纹。止口有四个平面（离轴 {flat:g}），任意一个对着环的钥匙面，所以每 90° 一档。
  中间是 Φ{neck:g} 的颈，Φ{bore:.2f} 直孔上下贯通：最上面 {tube:g} mm 套手的止口，下面 {cham:g} mm 是线腔（底下敞开，对着环的孔底）；
  +Y 侧开 {winw:g}×{winh:g} 出线口，口外法兰上铣 1 深的槽，旁边两个扎带孔。
  上端是手的台面（长 {tab_l:.1f}、宽 {tab_w:g}、厚 {tab_t:g}）：两块凸台托住手底面；两颗 M3x{rad_l:g} 螺钉沿 X 穿过 Φ{clear:g} 过孔，拧进手止口的两个螺纹。
  除了插在环孔里的止口，新件全部在环顶面（Z={ring_h:g}）以上，线也在环顶面以上，碰不到 J6 外壳和限位销。

怎么消除间隙
  环一侧
  1 倾角：肩面压在环顶面上（接触 {shoulder:.0f} mm2，外径 57）。新件的螺纹故意比环的孔高 {e_ring:g}，所以 4 颗沉头螺钉的头只可能压在环沉孔的上侧，
    把肩面往下压，不会把新件顶离环。环本身不改。
  2 位置和转角：4 颗螺钉的 90° 头坐在环的沉孔里，锥面定心，没有间隙；钢螺钉拧在 {eng_ring:.1f} mm 深的铝螺纹里，可以正常拧紧（0.8 N·m）。
  手一侧
  3 倾角：手底面压在两块凸台上（接触 {seat:.0f} mm2，跨度长 {pad_x:.0f}、宽 {pad_y:.0f}，比第一版宽）。
  4 径向：A 侧螺钉把止口的厚壁拉到 A 侧孔壁上，这是基准。B 侧先把两颗顶丝拧到刚碰上止口，再拧 B 侧螺钉，把厚壁夹在螺钉和两个顶丝尖之间。
    两处都是就地夹住厚壁，力不经过止口的薄壁；孔在薄壁处还让开 {thin:g}。
  5 轴向和转角：靠 4 处摩擦面（A 孔壁、B 顶丝尖、两端垫圈）夹住。每颗螺钉头下垫 {n_spring} 片碟簧，贴合面沉陷后预紧还在；
    手上的 M3 只有约 2.5 mm 螺纹，预紧只取 {pull:.0f} N（约 {torque:.1f} N·m，看碟簧将平未平即可）。
    过孔 Φ{clear:g}：手上螺纹孔的高度（{below:g}，没有在实物上量过）差 ±{slack:g} 都照样装，螺钉不会把手顶离凸台。

手一侧三种办法的比较
  a 第一版的偏置锥座（沉头螺钉把手楔紧在座面上）：有真正的轴向预紧；但依赖 {below:g} 这个没量过的尺寸，差 0.25 就失效甚至把手顶起，
    锥座上方的唇只有 0.6，力矩只能 0.15 N·m。不用。
  b 开口夹紧止口的两段厚壁（一颗切向 M3）：不依赖 {below:g}。但夹紧力要经过止口这个环才能到对面，1 mm 的薄壁按估算要分担约一半；
    按薄壁受得了的夹紧力（约 600 N）算，摩擦力并不比 c 大；零件上还要开缝、留铰链和让位槽，座面得另做。不用。
  c 两颗径向螺钉各自就地夹住厚壁，B 侧用顶丝补间隙（本设计）：不依赖 {below:g}，薄壁不受力，零件上只有孔。
    代价：轴向靠摩擦，没有楔紧；最弱方向的安全系数只算点击是 {sf_tap:.1f}，点击加手的自重是 {sf_all:.1f}（摩擦系数按 0.15，见十二）。

螺钉清单
  4 × M3x{ring_l:g} 内六角沉头（GB/T 70.3 / DIN 7991，10.9 级），2.0 扳手，0.8 N·m —— 环到新件。不要用 M3x16（会伸进线腔）
  2 × M3x{rad_l:g} 内六角圆柱头（GB/T 70.1，12.9 级或 A2-70），2.5 扳手，约 {torque:.1f} N·m —— 拉紧手的止口
  2 × 小平垫圈 Φ6×Φ3.2×0.5（GB/T 848），垫在碟簧和台面端面之间
  {n_all} × 碟簧 6×3.2×0.3（GB/T 1972 / DIN 2093），每颗螺钉 {n_spring} 片同向叠放，凸面朝钉头
  2 × M3x{jack_l:g} 内六角平端紧定螺钉（GB/T 77），1.5 扳手 —— B 侧顶丝
  打印件另加 4 × M3 方螺母（DIN 562，5.5×1.8），从止口底面的槽里推入，代替铝螺纹

装配顺序
  先在桌上把手装到新件上（以后换工具不用再拆）
  1 线插到手上，另一头从 Φ26 孔里向下、再从出线口穿出。
  2 手的止口插进 Φ26 孔，手背朝出线口（也可以转 180°），手底面压实在两块凸台上。
  3 A 侧（台面上有小凹坑的一端）：M3x{rad_l:g} 套上 {n_spring} 片碟簧（凸面朝钉头）和垫圈，沿 X 拧入，拧到碟簧将平未平（从碟簧贴上起约 1/4 圈）。
  4 B 侧两颗顶丝：从台面斜角面上的孔里拧入，用扳手短柄捻到刚碰上止口就停。不要用力，用力会挤止口。
  5 B 侧 M3x{rad_l:g} 同样拧紧。然后用 0.02 塞尺查手底面与两块凸台之间，塞不进才对。
  6 扎带穿过出线口旁的扎带孔和出线口，把线勒在颈部外壁上。
  再装到臂上（换工具只做这两步）
  7 止口对准环的 Φ40 孔放下，任选一个平面对着环的钥匙面（90° 一档），肩面贴住环顶面。
  8 4 颗 M3x{ring_l:g} 沉头从环外侧拧入：先都带上，再对角分两次拧到 0.8 N·m。
  拆：松开环上 4 颗螺钉，手连新件一起取下。手的两颗螺钉和顶丝只有要把手从新件上拆下时才动。
"""

TODO = """加工前必须在实物上量的（括号里是现在用的值，都是脚本开头的参数）
  松灵的环和臂
  1 环的 Φ40 孔实际尺寸：止口按实测 −0.02 加工。
  2 环的径向孔中心到环顶面的距离（{ring_drop:.2f}，模型值）：新件螺纹比它高 {e_ring:g}；实测差 {e_ring:g} 以上就改参数 RING_HOLE_Z 重出图。
    四个径向孔是不是都通（松灵的 STEP 里 0° 那个被一张面封住，按建模瑕疵处理）。
  3 环的孔底有没有螺钉头高出来（模型里 6 颗沉头与孔底齐平；止口离孔底只有 {spig_gap:g}）。
  4 夹爪底座实物：止口长度、径向螺钉的规格和长度（本设计按 M3x{ring_l:g} 沉头）。
  5 环在臂上的角度、J6 限位销有没有：只影响手朝哪边，不影响零件。
  手（第一版列的仍然有效）
  6 手的止口外径和圆度：Φ26 孔按实测 +0.02~+0.04 加工。有了顶丝，孔大一点（到 +0.1）也能用，只是 A 侧贴合面会窄。
  7 径向螺纹是不是 {rad}×{pitch:g}，能拧进几圈（壁厚 2.5 还是 3.0：手册内孔 Φ21，模型 Φ20）。螺纹规格是参数 RAD。
  8 螺纹孔中心到手底面的距离（{below:g}）：现在 ±{slack:g} 以内都行，量一下确认在范围内。
  9 手底面平不平，6 颗固定螺钉的头是否都不高出底面；薄壁槽宽（手册 7，模型 9.6；孔让开的宽度 {thin_w:g}）；插座个数（手册 4，模型 3）；
    插头实际尺寸，能否过 {winw:g}×{winh:g} 的出线口（12×8 的插头每边余 0.5）。
  10 手的底座材料：若是塑料，拉紧螺钉每颗只放 2 片碟簧。手的重量（手册 408 g，模型 395 g）。
  11 买到的 M3x{rad_l:g} 实长（{rad_l:g}±{rad_tol:g}）：拧紧后从止口里面看，钉尖不应伸出内壁超过 0.5。

加工要求（给加工方）
  先车：止口 Φ{spig:.2f}、肩面、Φ{neck:g} 颈、Φ{bore:.2f} 通孔；再铣：台面外形和两块凸台、止口四个平面、出线口和槽；最后钻各径向孔。颈是车出来的外槽，零件内部没有倒扣。
  肩面与两块凸台面平行 0.02，两块凸台共面 0.01；止口、Φ26 孔对肩面垂直 0.02。止口根部圆角 ≤ R0.3（环的孔口倒角 0.5）。
  4 个接环的 M3 螺纹攻通，轴线离肩面 {thread_drop:.2f} ±0.03。顶丝孔从外向里：Φ3.4 钻到半径 {jack_r:g}，再攻 M3 到孔壁。出线口上角 R2。
  阳极氧化的话，止口和 Φ26 孔按氧化后的尺寸控制；肩面、凸台面不喷漆。
  打印件（hand_flange_print.*）：Φ26 孔做 Φ{pbore:.2f}，止口做 Φ{pspig:.2f}，接环的螺纹改成 Φ3.4 过孔加方螺母槽；台面朝下打印，法兰下面和凸台周围加支撑；
  只用于试装和走线检查，手的拉紧螺钉每颗 1 片碟簧，不能用来验证精度。
"""


def main():
    Message.DefaultMessenger_s().RemovePrinters(Message_PrinterOStream.get_type_descriptor_s())  # the STEP writer's chatter
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "out")
    out.mkdir(parents=True, exist_ok=True)
    parts = {}
    for name, fit in FITS.items():
        s = build(fit)
        stem = "hand_flange" if name == "cnc" else "hand_flange_print"
        w = STEPControl_Writer()
        w.Transfer(s, STEPControl_AsIs)
        w.Write(str(out / f"{stem}.step"))
        BRepMesh_IncrementalMesh(s, 0.01, False, 0.1, True)
        stl = StlAPI_Writer()
        stl.ASCIIMode = False
        stl.Write(s, str(out / f"{stem}.stl"))
        parts[name] = s
    fit = FITS["cnc"]
    lines, extra = check(parts["cnc"], fit, out)
    pm = trimesh.load(out / "hand_flange_print.stl")
    lines += ["十三、打印件", f"  B-rep 有效 {BRepCheck_Analyzer(parts['print']).IsValid()}，STL 封闭 {pm.is_watertight}，连通体 {pm.body_count}，体积 {pm.volume / 1000:.3f} cm3；"
              f"与环的重叠体积 {common(parts['print'], extra['ring']):.4f} mm3"]
    v = dict(H=H, ring_h=RING_H, above=H - RING_H, spig=fit["spigot"], spig_l=RING_H - SPIG_BOT, spig_gap=SPIG_BOT - RING_FLOOR, flange=2 * RING_R, ring_l=RING_SCREW["length"], flat=FLAT_D,
             neck=2 * NECK_R, bore=fit["bore"], tube=TUBE_LEN, cham=H - TUBE_LEN - SPIG_BOT, winw=2 * WIN_W, winh=TABLE_BOT - WIN_FLOOR, tab_l=2 * TAB_X, tab_w=2 * TABLE_Y, tab_t=H - TABLE_BOT,
             rad_l=RAD["length"], clear=RAD["clear"], e_ring=E_RING, thin=THIN_GAP, n_spring=SPRING["n"], n_all=2 * SPRING["n"], below=HOLE_BELOW, slack=(RAD["clear"] - RAD["d"]) / 2,
             jack_l=JACK["length"], ring_drop=RING_H - RING_HOLE_Z, rad=RAD["name"], pitch=RAD["pitch"], thin_w=2 * THIN_X, rad_tol=RAD["tol"], thread_drop=RING_H - RING_HOLE_Z - E_RING,
             jack_r=JACK_R1, pbore=FITS["print"]["bore"], pspig=FITS["print"]["spigot"], **extra["num"])
    text = HEAD.format(**v) + "\n检查结果\n" + "\n".join(lines) + "\n\n" + TODO.format(**v)
    (out / "check_report.txt").write_text(text)
    print(text)
    render(parts["cnc"], fit, extra, out)


if __name__ == "__main__":
    main()
