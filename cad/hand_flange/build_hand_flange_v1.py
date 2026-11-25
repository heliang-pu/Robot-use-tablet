"""Adapter flange, one piece: AgileX PiPER end flange -> Joyson TeleHand Lite wrist spigot (left hand).

    python build_hand_flange.py <out_dir>      (needs cadquery-ocp, trimesh, numpy, scipy, shapely, matplotlib, rtree)

Writes hand_flange.step/.stl (CNC 6061, nominal sizes), hand_flange_print.step/.stl (same part with printing
allowances), check_report.txt and render.png. The specification is 设计任务书.md; inputs/hand_root.stl is only used
by the checks and the picture.

Part frame (mm): origin = centre of the arm's flange face, Z = wrist axis towards the hand (hand +Y),
X = hand +Z, Y = hand +X (back of the hand, where the cable window is). Angles "al" run from +Y towards +X.

How it holds the hand: the plate face of the hand sits on the top face (this sets the tilt). Two M3 countersunk
screws go radially through the socket wall into the spigot's two threads; their 90 degree seats are cut FITS[..]["e"] lower
than the threads, so each head rides on the upper side of its seat and pushes the hand down on the top face while
it draws the spigot wall against the bore. Nothing in the part is tapped.
"""

import pathlib
import sys
import tempfile

import numpy as np
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCone, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeRevol, BRepPrimAPI_MakeSphere
from OCP.BRepTools import BRepTools
from OCP.Bnd import Bnd_Box
from OCP.gp import gp_Ax1, gp_Ax2, gp_Dir, gp_Pnt
from OCP.Message import Message, Message_PrinterOStream
from OCP.GProp import GProp_GProps
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
from OCP.StlAPI import StlAPI_Writer
from OCP.TopAbs import TopAbs_IN, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
import shapely
import shapely.geometry as sg
import shapely.ops as so
import trimesh

HERE = pathlib.Path(__file__).resolve().parent
X, Y, Z = (1, 0, 0), (0, 1, 0), (0, 0, 1)

# ---- hand side (设计任务书.md; where the manual and the model disagree the worse value is used) ----
AXIS_X, AXIS_Z, FACE_Y = 1.0218, 10.0178, -52.0243  # wrist axis and plate face in the hand's own frame, fitted on the mesh (spec: 1.02, 10.02, -52.02)
TUBE_D, TUBE_LEN = 26.0, 8.0  # spigot
TUBE_ID_MODEL, TUBE_ID_MANUAL = 20.0, 21.0  # screw tip limit / thread length
HOLE_BELOW = 3.10  # radial thread centre below the plate face (4.90 from the spigot end)
# the radial thread is inferred from the model (M3 x 0.5): check with a real screw. Countersunk screw: head dia, head height (min, usual, max), key, length
RAD = dict(name="M3", d=3.0, pitch=0.5, clear=3.3, dk=6.0, k=(1.5, 1.7, 1.86), key=2.0, length=5.0, tol=0.24)
TIP_INDEX = (-30.02, 0.0)  # index fingertip from the wrist axis (X, Y), at hand y = 127.7
TIP_Y, HAND_KG = 127.7, 0.408

# ---- arm side ----
ARM_PCD, ARM_DEPTH, PIN_PCD, DISC_D, RING_D = 27.0, 5.0, 24.3, 35.0, 57.0
ARM_AL = (30, 150, 210, 330)  # the four of the six M3 that are used (the other two would sit under the radial screws)
PIN_AL = (60, 120, 240, 300)  # pin holes; any 60 degree step finds two of the arm's three pin holes
ARM_SCREW = dict(length=8.0, tol=0.29, head_d=5.5, head_k=3.0)

# ---- the part ----
H = 25.0  # flange face to the hand's plate face
BODY_R, PILOT_R, RELIEF_H = 20.0, 17.0, 1.5  # outside dia 35 nothing is lower than RELIEF_H
FLOOR = 4.0  # under the arm screw heads
ARM_CLEAR, CHAN_D = 3.4, 6.2  # arm screw hole; channel for head and key through the socket ring
PIN_DEPTH, PIN_OUT = 5.0, 3.0  # pin hole depth from the flange face; pin length 8, 3 into the arm
WIN_W, WIN_TOP, WIN_R = 7.25, 16.0, 2.0  # cable window half width, top, corner radius (bottom = FLOOR)
TON_W, TON_TIP, TON_F, TIE_Y = WIN_W, 30.0, 2.0, 23.5  # cable-tie tongue: half width (= window), tip, root fillet, where the tie goes
THIN_GAP, THIN_X = 0.15, 6.75  # the bore stands back from the spigot's two thin walls
NOTCH_R, LIP = RAD["dk"] / 2 + 0.25, 0.6  # access notch for head and key; thinnest lip left over the screw seat
TIP_NOM = 10.65  # radius of the radial screw's tip: never inside r = 10.0 with all tolerances
FITS = {  # bore = measured spigot + 0.02; e = how much lower the screw seats are than the hand's threads
    "cnc": dict(bore=26.02, e=0.25, pin=4.0),
    "print": dict(bore=26.20, e=0.35, pin=3.9),
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


def cut(a, b):
    return BRepAlgoAPI_Cut(a, b).Shape()


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


def bounds(s):
    b = Bnd_Box()
    BRepBndLib.AddOptimal_s(s, b, False, False)
    lo, hi = b.CornerMin(), b.CornerMax()
    return np.array([[lo.X(), lo.Y(), lo.Z()], [hi.X(), hi.Y(), hi.Z()]])


def pol(r, al):
    return r * np.sin(np.radians(al)), r * np.cos(np.radians(al))


def revolve(profile):
    w = BRepBuilderAPI_MakePolygon()
    for r, z in profile:
        w.Add(P((r, 0, z)))
    w.Close()
    return BRepPrimAPI_MakeRevol(BRepBuilderAPI_MakeFace(w.Wire()).Face(), gp_Ax1(P((0, 0, 0)), gp_Dir(0, 0, 1))).Shape()


def seat(fit):
    """Radial screw seat: thread axis height, cone axis height, radius where the cone starts (shank size), where its lip is cut off, where the notch starts."""
    ah = H - HOLE_BELOW
    ac = ah - fit["e"]
    r0 = TIP_NOM + RAD["length"] - RAD["k"][1] - fit["e"]
    return ah, ac, r0, r0 + (H - ac - LIP) - RAD["d"] / 2, r0 + NOTCH_R - RAD["d"] / 2


def build(fit):
    rb = fit["bore"] / 2
    s = revolve([(0, 0), (PILOT_R - 0.3, 0), (PILOT_R, 0.3), (PILOT_R, RELIEF_H), (BODY_R, RELIEF_H), (BODY_R, H - 0.5), (BODY_R - 0.5, H), (0, H)])
    # cable-tie tongue under the window, with the cutter's radius at its root
    tongue = fuse(box((-TON_W, 15, RELIEF_H), (TON_W, TON_TIP - 3, FLOOR)), box((-TON_W + 3, 15, RELIEF_H), (TON_W - 3, TON_TIP, FLOOR)))
    xc = TON_W + TON_F
    yc = np.sqrt((BODY_R + TON_F) ** 2 - xc**2)
    for sx in (-1, 1):
        corner = cyl((sx * (TON_W - 3), TON_TIP - 3, RELIEF_H), Z, 3.0, FLOOR - RELIEF_H)
        web = cut(box((sx * TON_W, 15, RELIEF_H), (sx * xc * BODY_R / (BODY_R + TON_F), yc, FLOOR)), cyl((sx * xc, yc, 0), Z, TON_F, 6))
        slot = fuse(box((sx * (TON_W - 1.5), TIE_Y - 0.5, 0), (sx * (TON_W + 1), TIE_Y + 0.5, 6)), cyl((sx * (TON_W + 0.5), TIE_Y - 0.5, 0), Z, 2.0, 6), cyl((sx * (TON_W + 0.5), TIE_Y + 0.5, 0), Z, 2.0, 6))
        tongue = cut(fuse(tongue, corner, web), slot)
    s = fuse(s, tongue)
    # socket bore = cable chamber, straight down to the floor; relieved where the spigot wall is thin
    void = fuse(cyl((0, 0, FLOOR), Z, rb, H), both(cyl((0, 0, FLOOR), Z, rb + THIN_GAP, H), box((-THIN_X, -30, 0), (THIN_X, 30, 60))))
    for al in ARM_AL:
        x, y = pol(ARM_PCD / 2, al)
        void = fuse(void, cyl((x, y, FLOOR), Z, CHAN_D / 2, H))
        s = cut(s, cyl((x, y, -1), Z, ARM_CLEAR / 2, FLOOR + 2))
    win = fuse(box((-WIN_W, 5, FLOOR), (WIN_W, 25, WIN_TOP - WIN_R)), box((-(WIN_W - WIN_R), 5, FLOOR), (WIN_W - WIN_R, 25, WIN_TOP)))
    for sx in (-1, 1):
        win = fuse(win, cyl((sx * (WIN_W - WIN_R), 5, WIN_TOP - WIN_R), Y, WIN_R, 20))
    s = cut(s, fuse(void, win))
    s = cut(s, cone((0, 0, H - 0.5), Z, rb, rb + 0.501, 0.501))
    for al in PIN_AL:
        x, y = pol(PIN_PCD / 2, al)
        s = cut(s, cyl((x, y, -1), Z, fit["pin"] / 2, PIN_DEPTH + 1))
    s = cut(s, cyl((0, 0, -1), Z, 9.0, 1.3))  # the flange face bears on a ring, not on its middle
    ah, ac, r0, rlip, rnotch = seat(fit)
    rd, rc = RAD["d"] / 2, RAD["clear"] / 2
    for sx in (-1, 1):
        d = (sx, 0, 0)
        t = cyl((sx * 11.5, 0, ah), d, rc, r0 + rc + fit["e"] - rd + 0.2 - 11.5)  # around the hand's thread axis
        t = fuse(t, cone((sx * r0, 0, ac), d, rd, NOTCH_R, rnotch - r0))  # 90 degree seat, lower by e
        t = fuse(t, cyl((sx * rnotch, 0, ac), d, NOTCH_R, BODY_R + 2 - rnotch))
        t = fuse(t, box((sx * rlip, -NOTCH_R, ac), (sx * (BODY_R + 2), NOTCH_R, H + 1)))
        s = cut(s, t)
    s = cut(s, cyl((BODY_R - 0.4, 0, 12.0), X, 1.0, 2))  # dimple: side A, tightened first
    u = ShapeUpgrade_UnifySameDomain(s, True, True, True)
    u.Build()
    return u.Shape()


def fasteners(fit, pins=(120, 240)):
    """Screws and pins at their nominal seated places, for the checks and the picture."""
    out = {}
    for al in ARM_AL:
        x, y = pol(ARM_PCD / 2, al)
        out[f"arm{al}"] = fuse(cyl((x, y, FLOOR), Z, ARM_SCREW["head_d"] / 2, ARM_SCREW["head_k"]), cyl((x, y, FLOOR - ARM_SCREW["length"]), Z, 1.5, ARM_SCREW["length"]))
    for al in pins:
        x, y = pol(PIN_PCD / 2, al)
        out[f"pin{al}"] = cyl((x, y, -PIN_OUT), Z, 2.0, 8.0)
    ah, ac, r0, _, _ = seat(fit)
    rj = r0 + fit["e"]  # where the head's cone starts when it rides on the upper side of the seat
    rd, rh, k = RAD["d"] / 2, RAD["dk"] / 2, RAD["k"][1]
    for sx, name in ((1, "radA"), (-1, "radB")):
        d = (sx, 0, 0)
        tip = rj + k - RAD["length"]
        out[name] = fuse(cyl((sx * tip, 0, ah), d, rd, rj - tip), cone((sx * rj, 0, ah), d, rd, rh, rh - rd), cyl((sx * (rj + rh - rd), 0, ah), d, rh, k - rh + rd))
    return out


def arm_model(proud, slit=0.0):
    disc = cyl((0, 0, -9), Z, DISC_D / 2, 9)
    for al in range(0, 360, 60):
        x, y = pol(ARM_PCD / 2, al + 30)
        disc = cut(disc, cyl((x, y, -ARM_DEPTH), Z, 1.25, ARM_DEPTH + 1))
    for al in (0, 120, 240):
        x, y = pol(PIN_PCD / 2, al)
        disc = cut(disc, cyl((x, y, -5), Z, 2.02, 6))
    ring = cut(cyl((0, 0, -40), Z, RING_D / 2, 40 - proud), cyl((0, 0, -41), Z, DISC_D / 2 + slit, 42))
    return disc, ring


def tess(shape, defl=0.02):
    BRepTools.Clean_s(shape)
    BRepMesh_IncrementalMesh(shape, defl, False, 0.15, True)
    with tempfile.TemporaryDirectory() as d:
        StlAPI_Writer().Write(shape, d + "/t.stl")
        return trimesh.load(d + "/t.stl")


def load_hand():
    m = trimesh.load(HERE / "inputs" / "hand_root.stl")
    v = m.vertices * 1000.0
    m.vertices = np.column_stack([v[:, 2] - AXIS_Z, v[:, 0] - AXIS_X, v[:, 1] - FACE_Y + H])
    return m


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
    half = sg.box(0, -100, 100, 300)
    if solid == "closed":
        loops = [q for q in loops if np.allclose(q[0], q[-1], atol=1e-6)]
    if solid:
        return loops_polygon(loops).intersection(half)
    return so.unary_union([sg.LineString(q) for q in loops if len(q) > 1]).intersection(half)


def check(part, fit, out):
    """All the numerical checks of the nominal part. Returns the report lines and what the picture needs."""
    L = []
    rb = fit["bore"] / 2
    ah, ac, r0, rlip, rnotch = seat(fit)
    rd, rc = RAD["d"] / 2, RAD["clear"] / 2
    fine = tess(part, 0.005)
    hand = load_hand()
    fast = fasteners(fit)
    vol, com = props(part)

    # 1 solids
    n_solid = 0
    ex = TopExp_Explorer(part, TopAbs_SOLID)
    while ex.More():
        n_solid += 1
        ex.Next()
    stl = trimesh.load(out / "hand_flange.stl")
    rd_step = STEPControl_Reader()
    rd_step.ReadFile(str(out / "hand_flange.step"))
    rd_step.TransferRoots()
    back = rd_step.OneShape()
    L += ["一、实体",
          f"  B-rep 有效 {BRepCheck_Analyzer(part).IsValid()}，实体数 {n_solid}；STL 封闭 {stl.is_watertight}，连通体 {stl.body_count}，"
          f"三角形 {len(stl.faces)}；体积 B-rep {vol / 1000:.3f} cm3 / STL {stl.volume / 1000:.3f} cm3；外形 {np.round(stl.extents, 2).tolist()} mm",
          f"  STEP 文件读回：有效 {BRepCheck_Analyzer(back).IsValid()}，体积 {volume(back) / 1000:.3f} cm3"]

    # 2 hand mesh against the part
    low = hand.submesh([np.nonzero(hand.triangles[:, :, 2].min(axis=1) < H + 12)[0]], append=True)
    pts = np.vstack([low.vertices, trimesh.sample.sample_surface(low, 300000, seed=0)[0]])
    under = pts[pts[:, 2] < H - 5e-3]  # below the seat plane only the spigot may exist (the mesh is noisy by 0.001)
    ru = np.hypot(under[:, 0], under[:, 1])
    cls = BRepClass3d_SolidClassifier(part)
    inside = 0
    for p in under:
        cls.Perform(P(p), 1e-7)
        inside += cls.State() == TopAbs_IN
    dist = trimesh.proximity.closest_point(fine, under)[1]
    wall = ru > TUBE_D / 2 - 0.3
    lands = wall & (np.abs(under[:, 0]) > THIN_X + CHAN_D / 2)
    top_part, plate = flat_region(fine, H, 1), flat_region(low, H, -1)
    contact = top_part.intersection(plate)
    on_seat = pts[(np.abs(pts[:, 2] - H) < 5e-3)]
    on_seat = on_seat[shapely.contains_xy(top_part, on_seat[:, 0], on_seat[:, 1])]
    L += ["二、与手的网格（hand_root.stl，顶点加 30 万个表面采样点）",
          f"  座面以下的手的点 {len(under)} 个，最大半径 {ru.max():.3f}（只有止口，外径 {TUBE_D:g}）；落在法兰实体内部的点 {inside} 个（干涉 0 才对）",
          f"  预定接触 1：手底面贴法兰顶面，间隙 0，接触面积 {contact.area:.0f} mm2（采样点 {len(on_seat)} 个在顶面上）",
          f"  预定接触 2：止口外圆在孔 Φ{fit['bore']:.2f} 里，±X 两段孔壁处径向间隙 {rb - ru[lands].max():.3f}（网格量得最小 {dist[lands].min():.3f}）；"
          f"±Y 薄壁处让开，径向间隙 {rb + THIN_GAP - TUBE_D / 2:.2f}",
          f"  止口最低点 Z={under[:, 2].min():.2f}：离腔底 {under[:, 2].min() - FLOOR:.2f} mm，离臂螺钉头顶 {under[:, 2].min() - FLOOR - ARM_SCREW['head_k']:.2f} mm"]

    # 3 the arm's outer ring
    outside = both(part, cut(cyl((0, 0, -5), Z, 60, 40), cyl((0, 0, -6), Z, DISC_D / 2, 42)))
    zmin = bounds(outside)[0, 2]
    L.append("三、与 PiPER 外圈（Φ35 以外，可能不转）")
    for proud in (0.0, 4.0):
        disc, ring = arm_model(proud)
        L.append(f"  内盘{'与外圈齐平' if proud == 0 else '凸出 4 mm'}：重叠体积 {common(part, ring):.3f} mm3；Φ35 以外法兰最低点离外圈端面 {zmin + proud:.2f} mm"
                 f"（要求 ≥ 1）；三维最近距离 {gap(part, ring):.2f} mm" + ("（是 Φ34 定位台与 Φ35 圆的径向间隙，平面上不重叠）" if proud == 0 else ""))
    L.append(f"  法兰压在内盘上的重叠体积 {common(part, disc):.3f} mm3；捆线舌头底面高 {RELIEF_H:g}，宽 {2 * TON_W:g}，伸到半径 {np.hypot(TON_TIP, TON_W - 3):.1f}")

    # 4 hole positions
    L.append("四、孔位")
    rows = []
    for k in range(6):
        psi = 30 + 60 * k
        arm_m3 = {(b - psi) % 360 for b in range(0, 360, 60)}
        arm_pin = {(b - psi) % 360 for b in (30, 150, 270)}
        rows.append((psi, set(ARM_AL) <= arm_m3, sorted(arm_pin & set(PIN_AL))))
    probe_m3 = max(common(part, cyl((*pol(ARM_PCD / 2, al), -6), Z, 1.5, 6 + FLOOR + 30)) for al in ARM_AL)
    probe_pin = max(common(part, cyl((*pol(PIN_PCD / 2, al), -6), Z, fit["pin"] / 2 - 0.005, 6 + PIN_DEPTH - 0.01)) for al in PIN_AL)
    L += [f"  臂：4 个 M3 过孔在 Φ{ARM_PCD:g} 圆上 al = {list(ARM_AL)}，销孔在 Φ{PIN_PCD:g} 圆上 al = {list(PIN_AL)}。Φ3 螺钉探针与法兰重叠 {probe_m3:.4f} mm3，"
          f"Φ{fit['pin'] - 0.01:.2f} 销探针重叠 {probe_pin:.4f} mm3（孔确实在那里）",
          "  六个安装角（出线口相对臂上 0 度方向）：" + "；".join(f"{p}°→M3 全对上 {ok}，销用 al={pins}" for p, ok, pins in rows)]
    hv = hand.vertices
    for sx, name in ((1, "A(+X，手的 +Z)"), (-1, "B(−X，手的 −Z)")):
        q = hv[(np.abs(hv[:, 1]) < 1.9) & (np.abs(hv[:, 2] - ah) < 1.9) & (sx * hv[:, 0] > 10.2) & (sx * hv[:, 0] < 12.8)][:, 1:]
        cx, cz, c0 = np.linalg.lstsq(np.column_stack([2 * q, np.ones(len(q))]), (q**2).sum(1), rcond=None)[0]
        L.append(f"  手 {name}：网格上螺纹孔轴线 Y={cx:+.3f}，Z={cz:.3f}，半径 {np.sqrt(c0 + cx * cx + cz * cz):.2f}；法兰过孔轴线 Y=0，Z={ah:.3f}；"
                 f"锥座轴线 Z={ac:.3f}，比螺纹低 {cz - ac:.3f}（设计 {fit['e']:g}，必须为正）")

    # 5 heads and drivers
    L.append("五、螺钉头和扳手通道（扫掠圆柱与法兰的重叠体积，mm3）")
    reach = max(common(part, cyl((*pol(ARM_PCD / 2, al), FLOOR), Z, ARM_SCREW["head_d"] / 2 + 0.1, 120)) for al in ARM_AL)
    seated = max(common(part, f) for k, f in fast.items() if k.startswith("arm"))
    pinned = max(common(part, f) for k, f in fast.items() if k.startswith("pin"))
    L.append(f"  臂螺钉 M3x{ARM_SCREW['length']:g}（4 颗，手未装时从上方直插）：Φ{ARM_SCREW['head_d'] + 0.2:g} 圆柱从座面到顶面以上 95 mm：{reach:.4f}；就位的螺钉本身：{seated:.4f}；"
             f"通道 Φ{CHAN_D:g}、深 {H - FLOOR:g}，2.5 内六角扳手长臂够得着。销（紧配合，面重合）：{pinned:.4f}")
    hk = np.hypot(pts[:, 1], pts[:, 2] - ah)
    for sx, name in ((1, "A"), (-1, "B")):
        d = (sx, 0, 0)
        head = common(part, cyl((sx * (rnotch + 0.01), 0, ah), d, RAD["dk"] / 2 + 0.05, 60))
        key = common(part, cyl((sx * (r0 + 0.8), 0, ah), d, RAD["key"] * 0.58 + 0.1, 120))
        shank = common(part, fast["rad" + name])
        touch = gap(part, fast["rad" + name])
        sel = (sx * pts[:, 0] > TUBE_D / 2 + 0.2) & (sx * pts[:, 0] < 130)
        near = pts[sel & (hk < RAD["dk"]) & (pts[:, 2] > ah)]
        head_gap = trimesh.proximity.closest_point(tess(fast["rad" + name], 0.005), near)[1].min()
        L.append(f"  径向螺钉 {name}（{RAD['name']}x{RAD['length']:g} 沉头，手装上后从侧面拧）：钉头通道 Φ{RAD['dk'] + 0.1:g}：{head:.4f}；{RAD['key']:g} 内六角扳手 Φ{RAD['key'] * 1.16 + 0.2:.1f}×120："
                 f"{key:.4f}；就位的螺钉本身：{shank:.4f}，它与锥座上侧的距离 {touch:.3f}（接触），下侧间隙 {2 * fit['e']:.2f}；"
                 f"扳手离手的网格最近 {hk[sel].min() - RAD['key'] * 0.58:.2f} mm，钉头离手的网格最近 {head_gap:.2f} mm（上方是手的底板）")
    L.append(f"  钉头按最大 Φ{RAD['dk']:g} 画：头的上缘离手底面只有 {HOLE_BELOW - RAD['dk'] / 2:.2f} mm（由手自己的尺寸 {HOLE_BELOW:g} 决定，偏紧，买头径 ≤ {RAD['dk']:g} 的）；"
             f"臂螺钉头顶 Z={FLOOR + ARM_SCREW['head_k']:g}，止口端面 Z={H - TUBE_LEN:g}，相隔 {H - TUBE_LEN - FLOOR - ARM_SCREW['head_k']:g}")
    L.append("  只检查了手掌根部网格：拇指在 −Z 侧，拧 B 侧螺钉时若挡扳手，先把拇指张开")

    # 6 thread engagement
    L.append("六、螺纹啮合")
    a = ARM_SCREW
    eng = a["length"] - FLOOR
    L.append(f"  臂 M3x{a['length']:g}：座面下 {FLOOR:g}，旋入 {eng:.2f}（{eng - a['tol']:.2f}..{eng + a['tol']:.2f}），臂上螺纹深 {ARM_DEPTH:g}，钉尖离孔底 {ARM_DEPTH - eng:.2f}"
             f"（最少 {ARM_DEPTH - eng - a['tol']:.2f}）→ 不顶底。不要换 M3x10")
    tips = [r0 + e + k - RAD["length"] - t for e, k, t in ((fit["e"] - 0.15, RAD["k"][0], RAD["tol"]), (fit["e"], RAD["k"][1], 0), (fit["e"] + 0.15, RAD["k"][2], -RAD["tol"]))]
    engs = [TUBE_D / 2 - max(t, TUBE_ID_MANUAL / 2) for t in tips]
    L.append(f"  径向 {RAD['name']}x{RAD['length']:g} 沉头：钉尖在半径 {tips[1]:.2f}（公差叠加 {tips[0]:.2f}..{tips[2]:.2f}），不得小于 {TUBE_ID_MODEL / 2:g}（模型内孔）→ 余量 {tips[0] - TUBE_ID_MODEL / 2:.2f}；"
             f"按手册壁厚 {(TUBE_D - TUBE_ID_MANUAL) / 2:g} 算旋入 {engs[1]:.2f}（{engs[2]:.2f}..{engs[0]:.2f}）")
    pt = RAD["pitch"]
    as_n = np.pi * (RAD["d"] - 0.25 * pt) * (0.5 + 0.57735 * (0.45 * pt - 0.25 * pt) / pt)  # internal thread shear area per mm (6H/6g limits, roughly)
    strip = [as_n * max(e - RAD["pitch"], 0) * 150.0 for e in (engs[2], engs[1])]  # one pitch lost to lead-ins; 150 MPa for an unknown cast alloy
    torque, kf, mu = 0.15, 0.26, 0.15
    pull = torque * 1000 / (kf * RAD["d"])
    push = pull * (1 - mu) / (1 + mu)
    L.append(f"  手上 {RAD['name']} 内螺纹（铝，按抗剪 150 MPa、扣掉一个螺距）拉脱力约 {strip[0]:.0f}..{strip[1]:.0f} N。拧紧力矩定 {torque:g} N·m（沉头 K≈{kf}）→ 轴力 {pull:.0f} N，"
             f"安全系数 {strip[0] / pull:.1f}..{strip[1] / pull:.1f}；任何情况不超过 0.25 N·m；手的底座若是塑料用 0.08 N·m")
    arm_c = r0 + (rc + fit["e"] + H - ac - LIP) / 2 - rd - TUBE_D / 2  # middle of the bearing line to the spigot surface
    bear = 4 * push / (RAD["d"] * engs[1]) + 6 * push * arm_c / (RAD["d"] * engs[1] ** 2)
    L.append(f"  锥座把每颗螺钉向下推约 {push:.0f} N（摩擦 0.15）：螺钉根部弯曲应力 {push * arm_c / (np.pi * (RAD['d'] - 1.2269 * pt) ** 3 / 32):.0f} MPa（力臂 {arm_c:.2f}），"
             f"手上螺纹孔口侧压峰值约 {bear:.0f} MPa（按线性分布，偏保守）；锥座承力线长 {H - ac - LIP - rc - fit['e']:.2f} mm，其上唇厚 {LIP:g}..{H - ac - rc - fit['e']:.2f}")
    L.append(f"  销 Φ4x8：压入法兰 {8 - PIN_OUT:g}（孔深 {PIN_DEPTH:g}），进臂 {PIN_OUT:g}；臂上销孔深度未知，须 ≥ {PIN_OUT + 0.3:g}")

    # 7 minimum wall
    coarse = tess(part, 0.03)
    c, n = coarse.triangles_center, coarse.face_normals
    p2, f2 = trimesh.sample.sample_surface(coarse, 30000, seed=1)
    o, nn = np.vstack([c, p2]), np.vstack([n, n[f2]])
    loc, ray, tri = coarse.ray.intersects_location(o - nn * 1e-4, -nn, multiple_hits=False)
    t, at = np.linalg.norm(loc - o[ray], axis=1), o[ray]
    opp = (nn[ray] * n[tri]).sum(1) < -0.5  # only between faces that face each other
    t, at = t[opp], at[opp]
    zone = (np.abs(at[:, 1]) < NOTCH_R + 0.3) & (np.abs(at[:, 0]) > rb - 0.2) & (at[:, 2] > ac - NOTCH_R - 0.5)  # the two screw seats
    i, j = np.argmin(np.where(zone, t, 1e9)), np.argmin(np.where(zone, 1e9, t))
    L += ["七、最小壁厚（沿法向向内射线，只算两面大致相对的地方）",
          f"  两个径向螺钉座以外：最薄 {t[j]:.2f} mm，在 X={at[j][0]:.1f} Y={at[j][1]:.1f} Z={at[j][2]:.1f}；厚度 < 1.5 的采样点占 {100 * (t[~zone] < 1.5).mean():.1f}%",
          f"  径向螺钉座：最薄 {t[i]:.2f} mm，在 X={at[i][0]:.1f} Y={at[i][1]:.1f} Z={at[i][2]:.1f}。这里有两处是有意做薄的：锥座上方的唇 {LIP:g}（受压，夹在钉头和手底板之间）；"
          f"锥座小端到 Φ26 孔壁 {r0 - rb:.2f}（不受力，钉头的锥面从半径 {r0 + fit['e']:.2f} 才开始）",
          f"  其它关键壁（算出）：腔底 {FLOOR:g}；通道到外圆 {BODY_R - ARM_PCD / 2 - CHAN_D / 2:.2f}；臂螺钉孔到定位台边 {PILOT_R - ARM_PCD / 2 - ARM_CLEAR / 2:.2f}；"
          f"销孔到通道 {np.hypot(*(np.array(pol(ARM_PCD / 2, 30)) - pol(PIN_PCD / 2, 60))) - 2 - CHAN_D / 2:.2f}；径向过孔到顶面 {HOLE_BELOW - RAD['clear'] / 2:.2f}；"
          f"出线口上方的环高 {H - WIN_TOP:g}；舌头厚 {FLOOR - RELIEF_H:g}"]

    # 8 cable path
    path = [(0, 0, H + 1.0), (0, 0, 16.0)] + [(0, 5 - 5 * np.cos(a), 11 + 5 * (1 - np.sin(a))) for a in np.radians((15, 30, 45, 60, 75, 90))] + [(0, 19, 11.0), (0, 24.5, 7.6), (0, 33, 7.6)]
    cable = ball(path[0], 3.5)
    for a, b in zip(path[:-1], path[1:]):
        v = np.subtract(b, a)
        cable = fuse(cable, cyl(a, v, 3.5, np.linalg.norm(v)), ball(b, 3.5))
    heads = fuse(*(fast[f"arm{al}"] for al in ARM_AL))
    plug = fuse(box((-6, -7.5, 7.5), (6, 40, 15.5)), box((-6, -7.5, 7.5), (6, 7.5, H)))
    zc = np.array([p[2] for p in path])
    inner = pts[(np.hypot(pts[:, 0], pts[:, 1]) < 12.5) & (pts[:, 2] < H + 1.0)]
    L += ["八、走线（扫掠体）",
          f"  线束 Φ7（6 根线，以后加 5 根也够）：从止口中心向下、半径 5 的弯、穿出线口、落到舌头上。与法兰重叠 {common(part, cable):.3f} mm3，最近 {gap(part, cable):.2f} mm（线落在舌头面上，模型里留了 0.1）；"
          f"与臂螺钉头重叠 {common(heads, cable):.3f} mm3，最近 {gap(heads, cable):.2f} mm；离止口内壁最近 {(TUBE_ID_MODEL / 2 - 3.5):.1f} mm（模型内孔 Φ20，采样最小内半径 {np.hypot(inner[:, 0], inner[:, 1]).min():.2f}）",
          f"  插头 12×8×15：平着穿过出线口再从腔里竖直进止口，与法兰重叠 {common(part, plug):.3f} mm3，与臂螺钉头重叠 {common(heads, plug):.3f} mm3；"
          f"出线口宽 {2 * WIN_W:g}、高 {WIN_TOP - FLOOR:g}，两颗臂螺钉头以上净高 {WIN_TOP - FLOOR - ARM_SCREW['head_k']:g}（12×8 的余量：宽 {WIN_W - 6:.2f}/边，高 {(WIN_TOP - FLOOR - ARM_SCREW['head_k'] - 8) / 2:.2f}/边）",
          f"  腔：Φ{fit['bore']:.2f} × 高 {H - TUBE_LEN - FLOOR:g}（腔底到止口端面），加止口内孔；线束路径最低点 Z={zc.min() - 3.5:.1f}，在舌头面（Z={FLOOR:g}）上；捆线槽 2 处，宽 {4 + 1:g}"]

    # 9 mass
    m_al = vol * RHO["6061 铝"] / 1000
    m_fast = 4 * 0.60 + 2 * 0.25 + 2 * 0.78
    L += ["九、质量和质心（零件坐标：原点在法兰面中心，Z 朝手）",
          "  " + "；".join(f"{k} {vol * v / 1000:.1f} g" for k, v in RHO.items()) + f"；质心 X={com[0]:.2f} Y={com[1]:.2f} Z={com[2]:.2f} mm",
          f"  加螺钉和销约 {m_fast:.1f} g：铝件合计 {m_al + m_fast:.0f} g；加手 {HAND_KG * 1000:.0f} g（手册值，不含线）共 {m_al + m_fast + HAND_KG * 1000:.0f} g，臂的末端负载 1500 g"]

    # 10 compliance, by hand
    zt = TIP_Y - FACE_Y + H
    bottom = flat_region(fine, 0, -1)
    A_s, c_s, ixx_s, iyy_s, xm_s, ym_s = area_moments(contact)
    A_f, c_f, ixx_f, iyy_f, _, _ = area_moments(bottom)
    zs = np.arange(FLOOR + 0.5, H, 1.0)
    inv = np.zeros(2)
    for z in zs:  # bending of the wall between floor and seat
        _, _, ixx, iyy, _, _ = area_moments(loops_polygon([q[:, :2] for q in coarse.section(plane_origin=(0, 0, z), plane_normal=Z).discrete]), 0.1)
        inv += 1.0 / np.array([iyy, ixx])
    seat_push = (2 * push - 2 * mu * pull, 2 * push)
    arm_pre = 4 * 0.8 * 1000 / (0.2 * 3.0)
    L += ["十、指尖受 2 N 时的位移（手算估计，只算法兰和两个结合面，不含手和臂本身）",
          f"  食指尖：离法兰面 {zt:.1f}，离座面 {zt - H:.1f}，偏离腕轴 {abs(TIP_INDEX[0]):.1f}（X 向）。座面接触 {A_s:.0f} mm2，惯性矩 ∫y² {iyy_s:.0f}、∫x² {ixx_s:.0f} mm4；"
          f"法兰面接触 {A_f:.0f} mm2，∫y² {iyy_f:.0f}、∫x² {ixx_f:.0f} mm4",
          f"  座面预紧：两颗径向螺钉各 {torque:g} N·m → 压紧力 {seat_push[0]:.0f}..{seat_push[1]:.0f} N（下限扣掉孔壁摩擦）；"
          f"张口力矩 绕 X（指腹方向受力）{seat_push[0] * iyy_s / (A_s * ym_s) / 1000:.2f}..{seat_push[1] * iyy_s / (A_s * ym_s) / 1000:.2f} N·m，"
          f"绕 Y {seat_push[0] * ixx_s / (A_s * xm_s) / 1000:.2f}..{seat_push[1] * ixx_s / (A_s * xm_s) / 1000:.2f} N·m；点击力矩 {2 * (zt - H) / 1000:.2f} N·m，"
          f"手自重力矩最大约 {HAND_KG * 9.81 * 0.06:.2f} N·m（质心按离座面 60 mm 估）。超过张口力矩后仍由锥座顶住，不会松动，只是变软",
          f"  臂螺钉 4×M3 各 0.8 N·m → 压紧力 {arm_pre:.0f} N，张口力矩约 {arm_pre * iyy_f / (A_f * PILOT_R) / 1000:.0f} N·m"]
    for label, kn_seat, kn_flange in (("保守", 50.0, 300.0), ("一般", 200.0, 1000.0)):  # contact stiffness per area, N/mm3
        row = []
        for name, i_s, i_f, ib in (("沿手的 X（指腹方向）", iyy_s, iyy_f, inv[0]), ("沿手的 Z（侧向）", ixx_s, ixx_f, inv[1])):
            d_seat = 2 * (zt - H) / (kn_seat * i_s) * (zt - H)
            d_flange = 2 * zt / (kn_flange * i_f) * zt
            d_body = 2 * (zt - H / 2) * ib * 1.0 / E_MOD["6061 铝"] * (zt - H / 2)
            d_twist = 2 * abs(TIP_INDEX[0]) ** 2 / (0.5 * kn_seat * (ixx_s + iyy_s)) if "X" in name else 0.0
            row.append(f"{name} {1000 * (d_seat + d_flange + d_body + d_twist):.1f} µm（座面 {1000 * d_seat:.1f}，法兰面 {1000 * d_flange:.1f}，本体 {1000 * d_body:.2f}，扭转 {1000 * d_twist:.1f}）")
        L.append(f"  {label}（结合面刚度 座面 {kn_seat:g}、法兰面 {kn_flange:g} N/mm3）：" + "；".join(row))
    d_ax = 2 * abs(TIP_INDEX[0]) ** 2 / (50.0 * ixx_s) + 2 * abs(TIP_INDEX[0]) ** 2 / (300.0 * ixx_f)
    L.append(f"  沿手指方向（轴向）2 N：力臂只有 {abs(TIP_INDEX[0]):.0f}，指尖 {1000 * d_ax:.2f} µm（保守）。本体若是打印件（PA12 / PETG-CF），本体项乘 {E_MOD['6061 铝'] / E_MOD['PA12']:.0f} / {E_MOD['6061 铝'] / E_MOD['PETG-CF']:.0f}，"
             f"即 {1000 * 2 * (zt - H / 2) ** 2 * inv[0] / E_MOD['PA12']:.0f} / {1000 * 2 * (zt - H / 2) ** 2 * inv[0] / E_MOD['PETG-CF']:.0f} µm，另外塑料锥座会蠕变，只能试装")
    L.append(f"  几何误差比受力变形大得多：座面在窄方向（{2 * ym_s:.0f} mm）每歪 0.01 mm，指尖走 {0.01 / (2 * ym_s) * (zt - H):.3f} mm；法兰面在 Φ{2 * PILOT_R:g} 上每歪 0.01 mm，指尖走 {0.01 / (2 * PILOT_R) * zt:.3f} mm；"
             f"手绕腕轴每转 0.1°，指尖走 {np.radians(0.1) * abs(TIP_INDEX[0]):.3f} mm；不用销时臂螺钉孔间隙 ±{(ARM_CLEAR - 3) / 2:g} 可让法兰转 ±{np.degrees((ARM_CLEAR - 3) / ARM_PCD):.2f}°，指尖 ±{(ARM_CLEAR - 3) / ARM_PCD * abs(TIP_INDEX[0]):.2f} mm")
    return L, dict(hand=hand, fast=fast, contact=contact, top=top_part, plate=plate, cable=cable, coarse=coarse)


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
    rb = fit["bore"] / 2
    ah, ac, r0, rlip, rnotch = seat(fit)
    hand, fast = extra["hand"], extra["fast"]
    hand_low = hand.submesh([np.nonzero(hand.triangles[:, :, 2].min(axis=1) < H + 30)[0]], append=True)
    disc, ring = arm_model(0.0, 0.2)
    BLUE, GREY, DARK, STEEL, PIN, ORANGE = (0.36, 0.56, 0.86), (0.84, 0.84, 0.82), (0.42, 0.42, 0.45), (0.9, 0.72, 0.25), (0.85, 0.3, 0.3), (0.95, 0.5, 0.1)
    solids = [(part, BLUE), (disc, (0.6, 0.6, 0.62)), (ring, DARK), (extra["cable"], ORANGE)] + [(s, PIN if k.startswith("pin") else STEEL) for k, s in fast.items()]

    def shot(ax, items, v, centre, half, title, px=800, keep=None):
        """Orthographic picture by splatting surface samples into a depth buffer."""
        v = np.array(v, float) / np.linalg.norm(v)
        right = np.cross(Z, v) / np.linalg.norm(np.cross(Z, v))
        R = np.array([right, np.cross(v, right), v])
        l1, l2 = np.array([-0.45, 0.55, 0.7]) / np.linalg.norm([-0.45, 0.55, 0.7]), np.array([0.75, -0.1, 0.65]) / np.linalg.norm([0.75, -0.1, 0.65])
        scale = px / (2 * half)
        F, D, C, N = [], [], [], []
        for mesh, rgb, shell in items:
            p, f = trimesh.sample.sample_surface(mesh, int(mesh.area * 4 * scale**2) + 100, seed=2)
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
    fig.suptitle(f"转接法兰 PiPER → TeleHand Lite（左手）：法兰面到手底面 {H:g} mm，一件，6061 铝 {volume(part) * RHO['6061 铝'] / 1000:.0f} g", fontsize=17)

    # --- 3D: assembly, quarter cut away, the part alone from above and from below ---
    whole = meshes()
    note = shot(fig.add_subplot(gs[0, 0]), [w for w in whole if w[1] != ORANGE] + [(hand_low, GREY, True)], (0.62, 0.72, 0.2), (0, 2, 22), 50, "装配：手腕、法兰、臂末端（内盘与外圈按齐平画）",
                keep=lambda p: p[:, 2] < H + 30)
    note("手的底板", (0, 14, H + 2), (0.2, 0.36))
    note("径向螺钉 A\n（凹坑一侧，先拧）", (BODY_R, 0, ah), (0.86, 0.5))
    note("出线口 + 捆线舌头", (0, TON_TIP - 2, FLOOR), (0.2, 0.6))
    note("内盘 Φ35（转）", (0, 17.5, -1.5), (0.16, 0.74))
    note("外圈 Φ57（可能不转）\n离法兰 1.5", (10, 26, 0), (0.25, 0.9))
    quarter = box((0, 0, -60), (100, 100, 120))
    note = shot(fig.add_subplot(gs[0, 1]), meshes(quarter) + [(hand_low, GREY, True)], (0.62, 0.72, 0.36), (0, 0, 14), 37, "切掉四分之一（X>0、Y>0）：线腔、止口、径向螺钉",
                keep=lambda p: ~((p[:, 0] > 1e-6) & (p[:, 1] > 1e-6)) & (p[:, 2] < H + 30))  # the cut faces themselves stay
    note("线束 Φ7", (0, 12, 11), (0.12, 0.5))
    note("M3x8 臂螺钉\n（通道直通顶面）", (*pol(ARM_PCD / 2, 330), FLOOR + 3), (0.13, 0.7))
    note("M3x5 沉头\n拧进止口螺纹", (r0 + 0.5, 0, ah), (0.86, 0.36))
    note("止口 Φ26×8", (0, TUBE_D / 2 - 1, H - 6), (0.5, 0.2))
    note("手底面贴顶面", (16, 0, H), (0.9, 0.2))
    alone = [(tess(part, 0.03), BLUE, False)]
    note = shot(fig.add_subplot(gs[0, 2]), alone, (0.55, 0.6, 0.58), (0, 5, 12), 31, "零件：从手一侧看")
    note("顶面（手底面的座）", (-16, -6, H), (0.2, 0.12))
    note("90° 锥座 + 钉头缺口", (rnotch + 1, 0, ac), (0.86, 0.3))
    note("臂螺钉通道 Φ6.2 ×4", (*pol(ARM_PCD / 2, 210), H), (0.14, 0.3))
    note("薄壁让开 0.15", (0, -rb - 0.1, H - 1), (0.1, 0.5))
    note("出线口 14.5×12", (0, BODY_R, 10), (0.6, 0.93))
    note("凹坑 = A 侧", (BODY_R, 0, 12), (0.9, 0.62))
    note = shot(fig.add_subplot(gs[2, 2]), alone, (0.5, 0.55, -0.67), (0, 5, 12), 31, "零件：从臂一侧看")
    note("Φ34 定位台，中间让空 0.3", (0, -4, 0), (0.25, 0.9))
    note("M3 过孔 Φ3.4 ×4", (*pol(ARM_PCD / 2, 150), 0), (0.85, 0.9))
    note("销孔 Φ4 ×4", (*pol(PIN_PCD / 2, 240), 0), (0.12, 0.72))
    note("捆线舌头（扎带槽）", (TON_W - 1.5, TIE_Y, RELIEF_H), (0.82, 0.1))

    # --- 2D sections ---
    coarse = extra["coarse"]
    fm = {k: tess(s, 0.02) for k, s in fast.items()}

    def draw(ax, g, sign=1, **kw):
        for p in getattr(g, "geoms", [g]):
            if p.geom_type == "Polygon" and not p.is_empty:
                verts, codes = [], []
                for ring in (p.exterior, *p.interiors):
                    xy = np.asarray(ring.coords) * (sign, 1)
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
            ax.add_patch(plt.Rectangle((0, -9) if sign > 0 else (-DISC_D / 2, -9), DISC_D / 2, 9, fc="0.8", ec="k", lw=0.6))
            ax.add_patch(plt.Rectangle((DISC_D / 2 + 0.15, -16) if sign > 0 else (-RING_D / 2, -16), RING_D / 2 - DISC_D / 2 - 0.15, 16, fc="0.6", ec="k", lw=0.6))
            if al % 60 == 30:  # an arm thread
                ax.add_patch(plt.Rectangle((sign * ARM_PCD / 2 - 1.5, -ARM_DEPTH), 3, ARM_DEPTH, fc="white", ec="k", lw=0.5, ls="--"))
            if al % 120 == 0:  # an arm pin hole (depth not known)
                ax.add_patch(plt.Rectangle((sign * PIN_PCD / 2 - 2, -4.5), 4, 4.5, fc="white", ec="k", lw=0.5, ls="--"))
            draw(ax, half_section(coarse, al), sign, fc=BLUE, ec="k", lw=0.8, hatch="///", alpha=0.9)
            for k, m in fm.items():
                draw(ax, half_section(m, al), sign, fc=PIN if k.startswith("pin") else STEEL, ec="k", lw=0.6)
            draw(ax, half_section(hand_low, al, solid="closed"), sign, fc="0.88", ec="none")
            draw(ax, half_section(hand_low, al, solid=False), sign)
        ax.axvline(0, color="k", lw=0.5, ls="-.")
        ax.set_aspect("equal")
        ax.set_xlim(*xlim), ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=13)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25, lw=0.4)

    ax = fig.add_subplot(gs[1, 0])
    section(ax, 0, 180, "剖面 Y–Z（过出线口；右 = 手背 +Y）", (-40, 44), (-34, 40))
    dim(ax, (-33, 0), (-33, H), f"{H:g}", rot=90)
    ax.plot([-33.8, -BODY_R], [H, H], "k", lw=0.4), ax.plot([-33.8, -DISC_D / 2], [0, 0], "k", lw=0.4)
    dim(ax, (-7, 0), (-7, FLOOR), f"腔底 {FLOOR:g}", at=(-7, -2.2))
    dim(ax, (-3, FLOOR), (-3, H - TUBE_LEN), f"线腔 {H - TUBE_LEN - FLOOR:g}", rot=90)
    dim(ax, (-3, H - TUBE_LEN), (-3, H), f"止口 {TUBE_LEN:g}", rot=90)
    dim(ax, (-rb, 9), (rb, 9), f"Φ{fit['bore']:.2f}", at=(5, 9))
    dim(ax, (38, FLOOR), (38, WIN_TOP), f"出线口高 {WIN_TOP - FLOOR:g}", at=(38, WIN_TOP + 2))
    dim(ax, (28, 0), (28, RELIEF_H), "", color="r")
    ax.text(36, -2.2, f"离外圈 {RELIEF_H:g}", fontsize=10, color="r", ha="center")
    for z, r, t in ((-19, PILOT_R, f"Φ{2 * PILOT_R:g} 定位台"), (-22.5, DISC_D / 2, f"Φ{DISC_D:g} 内盘"), (-26, BODY_R, f"Φ{2 * BODY_R:g} 外圆"), (-29.5, RING_D / 2, f"Φ{RING_D:g} 外圈")):
        dim(ax, (-r, z), (r, z), t)
    lead(ax, "捆线舌头", (TON_TIP - 2, 3), (36, 8.5))
    lead(ax, f"薄壁处让开 {THIN_GAP:g}", (-rb - 0.07, H - 4), (-27, 31))
    lead(ax, "手底面 = 座面", (-16, H), (-27, 36))
    ax.text(0, 37, "手", fontsize=12, ha="center"), ax.text(-23, -12, "外圈", fontsize=10, ha="center"), ax.text(0, -6, "内盘", fontsize=10, ha="center")

    ax = fig.add_subplot(gs[1, 1])
    section(ax, 90, 270, "剖面 X–Z（过两颗径向螺钉；右 = A 侧，手的 +Z）", (-40, 40), (-12, 40))
    ax.plot([-36, 36], [ah, ah], "k", lw=0.5, ls="-.")
    dim(ax, (30, ah), (30, H), f"{HOLE_BELOW:.2f}", at=(34.5, (ah + H) / 2))
    dim(ax, (-TUBE_D / 2, 30), (TUBE_D / 2, 30), f"止口 Φ{TUBE_D:g}", at=(0, 32))
    lead(ax, f"A：M3x{RAD['length']:g} 沉头\n先拧，0.15 N·m\n2.0 扳手从侧面", (r0 + 1.8, ah), (31, 9))
    lead(ax, "B：后拧\n0.12~0.15 N·m", (-r0 - 1.8, ah), (-31, 9))
    lead(ax, "钉尖不过内壁", (TIP_NOM, ah - 1.3), (0, 14.5))
    lead(ax, "A 侧孔壁是基准", (rb, 18.6), (2, 8))
    ax.text(0, 37, "手", fontsize=12, ha="center")

    ax = fig.add_subplot(gs[1, 2])
    section(ax, 30, 240, "剖面（右过一颗臂螺钉 al=30°，左过一根销 al=240°）", (-34, 40), (-14, 34))
    x = ARM_PCD / 2
    eng = ARM_SCREW["length"] - FLOOR
    dim(ax, (x + 5.5, -eng), (x + 5.5, 0), f"旋入 {eng:g}", at=(x + 11, -eng / 2 - 0.5))
    dim(ax, (x - 4.5, -ARM_DEPTH), (x - 4.5, 0), f"螺纹深 {ARM_DEPTH:g}", at=(x - 10.5, -ARM_DEPTH / 2 - 1))
    lead(ax, f"钉尖离孔底 {ARM_DEPTH - eng:g}", (x, -ARM_DEPTH + 0.5), (x + 11, -9))
    lead(ax, f"M3x8，2.5 扳手\n从上方进通道 Φ{CHAN_D:g}", (x, H + 1), (x + 12, 30))
    ax.annotate("", xy=(x, FLOOR + 3.3), xytext=(x, H + 6), arrowprops=dict(arrowstyle="->", lw=1.2, color="r"))
    dim(ax, (x + 9, 0), (x + 9, FLOOR), f"腔底 {FLOOR:g}", at=(x + 14.5, FLOOR / 2))
    lead(ax, f"销 Φ4×8：法兰里 {8 - PIN_OUT:g}，臂里 {PIN_OUT:g}\n（臂上孔深未知）", (-PIN_PCD / 2, -1), (-20, -9))
    lead(ax, "手装上后盖住通道", (x - 1, H + 1.5), (-10, 31))

    # detail of one screw seat
    ax = fig.add_subplot(gs[2, 0])
    section(ax, 90, 270, f"放大：径向螺钉座（锥座比螺纹低 {fit['e']:g}，钉头压在锥座上侧）", (8.5, 21.5), (17, 27.3))
    ax.plot([8.5, 21.5], [ah, ah], "k", lw=0.6, ls="-.")
    ax.plot([12.6, 21.5], [ac, ac], "r", lw=0.6, ls="-.")
    ax.text(21.3, ah + 0.12, "螺纹轴线", fontsize=9, ha="right"), ax.text(21.3, ac - 0.38, "锥座轴线", fontsize=9, ha="right", color="r")
    dim(ax, (19.6, ah), (19.6, H), f"{HOLE_BELOW:.2f}", at=(20.6, (ah + H) / 2))
    dim(ax, (18.2, ac), (18.2, ah), "", color="r")
    ax.text(18.2, ac - 0.42, f"{fit['e']:g}", fontsize=10, color="r", ha="center")
    dim(ax, (rlip, H - LIP), (rlip, H), "", color="r")
    lead(ax, f"唇最薄 {LIP:g}（受压）", (rlip, H - LIP / 2), (17.6, 26.6))
    for r, t in ((TUBE_ID_MODEL / 2, "模型内壁 r10"), (TUBE_ID_MANUAL / 2, "手册 r10.5")):
        ax.plot([r, r], [18.2, 20.2], "g", lw=0.8)
        ax.text(r, 17.9 if r == 10 else 17.45, t, fontsize=9, color="g", ha="center")
    lead(ax, f"钉尖 r{TIP_NOM:g}\n（{TIP_NOM - 0.59:.2f}..{TIP_NOM + 0.55:.2f}）", (TIP_NOM, ah), (9.9, 23.6))
    lead(ax, "向下压", (r0 + 0.7, ac + 2.25), (15.6, 26.0), color="r")
    lead(ax, f"Φ{fit['bore']:.2f} 孔壁", (rb, 19.0), (14.6, 17.6))
    lead(ax, "手底板", (11.5, H + 0.6), (10.2, 26.8))
    lead(ax, "止口壁", (11.5, 19.5), (9.6, 21.0))

    # plan view
    ax = fig.add_subplot(gs[2, 1])
    draw(ax, extra["top"], fc=(0.75, 0.83, 0.95), ec="k", lw=0.8)
    draw(ax, extra["contact"], fc=(0.3, 0.75, 0.4), ec="none", alpha=0.75)
    for g in getattr(extra["plate"], "geoms", [extra["plate"]]):
        ax.plot(*np.asarray(g.exterior.coords).T, color="k", lw=1.0, ls="--")
    t = np.radians(np.arange(361))
    ax.plot(BODY_R * np.sin(t), BODY_R * np.cos(t), "k", lw=0.6)
    ax.plot(DISC_D / 2 * np.sin(t), DISC_D / 2 * np.cos(t), color="0.5", lw=0.6, ls=":")
    ax.plot(RING_D / 2 * np.sin(t), RING_D / 2 * np.cos(t), color="0.5", lw=0.6, ls=":")
    ax.add_patch(plt.Polygon([(-TON_W, 18.6), (-TON_W, TON_TIP), (TON_W, TON_TIP), (TON_W, 18.6)], fc="none", ec="k", lw=0.6))
    for al in ARM_AL:
        x, y = pol(ARM_PCD / 2, al)
        ax.add_patch(plt.Circle((x, y), ARM_SCREW["head_d"] / 2, fc=STEEL, ec="k", lw=0.5))
        ax.text(*pol(ARM_PCD / 2 + 10.5, al), f"M3 {al}°", fontsize=9, ha="center", va="center")
    for al in PIN_AL:
        x, y = pol(PIN_PCD / 2, al)
        ax.add_patch(plt.Circle((x, y), 2, fc="none", ec=PIN, lw=1.0, ls="--"))
        ax.text(*pol(PIN_PCD / 2 - 5.5, al), f"销 {al}°", fontsize=9, ha="center", va="center", color=PIN)
    for sx, name in ((1, "A"), (-1, "B")):
        ax.annotate("", xy=(sx * (rnotch - 0.5), 0), xytext=(sx * 37, 0), arrowprops=dict(arrowstyle="->", lw=1.3, color="r"))
        ax.text(sx * 36, 2.2, f"{name} 侧 2.0 扳手", fontsize=10, color="r", ha="center")
    ax.annotate("", xy=(0, 37), xytext=(0, rb), arrowprops=dict(arrowstyle="->", lw=1.3, color=ORANGE))
    ax.text(0, 38.5, "出线（al = 0，手背）", fontsize=10, color=ORANGE, ha="center")
    ax.text(0, -25, "虚线：手底板外形    绿：座面实际接触（" + f"{extra['contact'].area:.0f} mm²）", fontsize=10, ha="center")
    ax.text(0, -30, "点线：臂内盘 Φ35、外圈 Φ57", fontsize=10, ha="center", color="0.4")
    ax.set_aspect("equal")
    ax.set_xlim(-42, 42), ax.set_ylim(-33, 42)
    ax.set_xlabel("X = 手的 +Z（小指侧）→", fontsize=10), ax.set_ylabel("Y = 手的 +X（手背）→", fontsize=10)
    ax.set_title("从手一侧看顶面：座面、通道、销孔位置（al 从 +Y 向 +X 量）", fontsize=13)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, lw=0.4)
    fig.savefig(out / "render.png")
    plt.close(fig)


HEAD = """转接法兰 PiPER → TeleHand Lite（左手）检查报告
由 build_hand_flange.py 生成；尺寸 mm。零件坐标：原点在臂法兰面中心，Z 沿腕轴朝手，X = 手的 +Z，Y = 手的 +X（手背，出线口）。
角度 al 从 +Y 向 +X 量。图见 render.png。

结构（一件，6061 铝；零件上没有任何螺纹）
  法兰面到手底面 {H:g} mm。底部 Φ{pilot:g} 定位台压在臂内盘上，Φ35 以外从 {relief:g} mm 高才变粗到 Φ{body:g}，不碰外圈。
  腔底厚 {floor:g}。4 颗 M3x8 内六角圆柱头螺钉从上方穿过四条 Φ{chan:g} 通道拧进臂内盘（al = 30/150/210/330；臂上另两个 M3 孔不用，
  它们正好在径向螺钉下面）。Φ4 销孔 4 个（al = 60/120/240/300），每个安装角用其中 2 个。
  Φ{bore:.2f} 孔从顶面直通腔底：上面 {tube:g} mm 套住手的止口，下面 {cham:g} mm 是线腔，+Y 侧开 {winw:g}×{winh:g} 出线口，口外有捆线舌头。

怎么消除间隙（按重要程度）
  1 倾角和轴向：手的底面贴在法兰顶面上。两颗 {rad}x{radl:g} 内六角沉头螺钉从 ±X 侧面穿过法兰，拧进止口的两个螺纹；
    法兰上的 90° 锥座比螺纹低 {e:g}，钉头只压在锥座上侧，像楔子一样把手往下压紧在顶面上（每颗约 140 N），不是靠摩擦。
  2 沿螺钉方向（X）：两颗螺钉方向相反，单靠它们定不了位。所以规定 A 侧（外圆上有小凹坑）先拧到位，把止口壁拉到 A 侧孔壁上，
    这是基准；B 侧后拧、力矩不大于 A 侧，它只需把止口撑开一个配合间隙（孔按实测止口 +0.02 加工，直径上 0.02~0.04）。
  3 横向（Y）和绕腕轴转角：钉头在锥座里自己对中（偏开就得把手往下压，压不动），再加座面摩擦锁住。
  4 止口 ±Y 两处薄壁（只剩 1 mm）：孔在这里让开 {thin:g}，薄壁不受力；螺钉的力都落在 ±X 的厚壁上。

螺钉清单
  4 × M3x8 内六角圆柱头（GB/T 70.1，12.9 或 A2-70），2.5 扳手，0.8 N·m
  2 × {rad}x{radl:g} 内六角沉头（GB/T 70.3 / DIN 7991，10.9），{key:g} 扳手，0.15 N·m（不得超过 0.25），头径 ≤ {dk:g}
  2 × 圆柱销 Φ4 m6 × 8（GB/T 119.2），可选：法兰要反复拆装又不想重新标定时用

装配顺序
  1 选好手的安装角（60° 一档，见第四节的表）。把 2 根销从底面压进对应的两个销孔，露出 {pin_out:g}。
  2 法兰放到臂内盘上，4 颗 M3x8 从上方通道放入，对角分两次拧到 0.8 N·m。此时手还没装，2.5 扳手直上直下。
  3 线先插到手上，线的另一头从孔里向下、再从出线口穿出（或把插头从出线口穿进来再插到手上）。
  4 手的止口对准孔放下，手背朝出线口（也可以转 180°），边放边把多余的线从出线口拉出，手底面压实在顶面上。
  5 两颗沉头螺钉从 ±X 侧面的缺口带上。按住手，先把 A 侧拧到 0.15 N·m，再拧 B 侧到 0.12~0.15 N·m。
    拧完用 0.02 塞尺沿手底面四周查，塞不进才对；塞得进说明螺纹孔比图纸低，停下来量（见“加工前必须量”第 3 条）。
  6 线在舌头上用扎带捆住，留出 J6 转动的余量。
  拆：反过来。臂螺钉只有把手拿掉才够得着；销从腔里用 Φ2 冲子向下顶出（销顶有一半露在腔底）。
"""

TODO = """加工前必须在实物上量的（括号里是现在用的值，都是脚本开头的参数）
  1 止口外径（Φ{tube_d:g}）和圆度：孔按实测 +0.02~+0.04 加工。
  2 径向螺纹是不是 {rad}×{pitch:g}：拿一颗真螺钉试；能拧进几圈（壁厚 2.5 还是 3.0：手册内孔 Φ21，模型 Φ20）。螺纹规格是参数 RAD。
  3 螺纹孔中心到手底面的距离（{below:.2f}，模型量得；手册只标了到止口端面 4.90，止口长 {tube:g}）。锥座比它低 {e:g}：
    实测若差 0.15 以上，改参数 HOLE_BELOW 重出图；若螺纹孔实际比锥座还低，螺钉会把手顶离座面，绝对不行。
  4 手底面平不平，6 颗固定螺钉的头是否都不高出底面（模型里内六花头低 0.2）；止口根部有没有圆角（模型没有，法兰孔口倒角 0.5）。
  5 止口两侧薄壁的槽宽（手册 7，模型 9.6；法兰让开的宽度 {thin_w:g}）；插座个数（手册 4，模型 3）；插头实际尺寸，
    能否过 {winw:g}×{winh:g} 的出线口（两颗臂螺钉头以上净高 {clear:g}）。过不去：把出线口两侧的臂螺钉换成薄头（DIN 7984，多 1 mm），或先穿线后压端子。
  6 手的底座材料（铝还是塑料）：决定径向螺钉力矩 0.15 还是 0.08 N·m。手的重量（手册 408 g，模型 395 g）。
  7 臂：内盘与外圈齐平还是凸出、外圈转不转；M3 螺纹实际深度（图上 5）；Φ4 销孔深度（图上没标，要 ≥ {pin_need:g}）和实际孔径；
    三处异形凹口的范围；末端插座在哪一侧（决定出线口朝哪个安装角）。
  8 买到的沉头螺钉：头径 ≤ {dk:g}、长 {radl:g}±{tol:g}。拧紧后从止口里面看，钉尖不应伸出内壁超过 0.5。

加工要求（给加工方）
  顶面与底面平行 0.02，Φ{tube_d:g} 孔对顶面垂直 0.02；两个 90° 锥座的轴线离顶面 {cone:.2f} ±0.03、与径向过孔同在 X 轴上，锥面 Ra1.6；销孔 Φ4 H7。
  锥座上方的唇最薄 {lip:g}，去毛刺时不要倒大角。内角：出线口上角 R2，舌头根部 R2；没有倒扣，三轴加四个侧面装夹能做完。
  阳极氧化的话，Φ{tube_d:g} 孔、销孔和锥座按氧化后的尺寸控制；顶面和底面不喷漆。
  打印件（hand_flange_print.*）：孔 Φ{pbore:.2f}，锥座低 {pe:g}，销孔 Φ{ppin:g} 留铰；法兰面朝下打印，定位台台阶和舌头下面加支撑；
  只用于试装和走线检查，径向螺钉 ≤ 0.08 N·m，不能用来验证精度。
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
    lines += ["十一、打印件", f"  B-rep 有效 {BRepCheck_Analyzer(parts['print']).IsValid()}，STL 封闭 {pm.is_watertight}，连通体 {pm.body_count}，体积 {pm.volume / 1000:.3f} cm3"]
    head = HEAD.format(H=H, pilot=2 * PILOT_R, relief=RELIEF_H, body=2 * BODY_R, floor=FLOOR, chan=CHAN_D, bore=fit["bore"], tube=TUBE_LEN, cham=H - TUBE_LEN - FLOOR,
                       winw=2 * WIN_W, winh=WIN_TOP - FLOOR, e=fit["e"], thin=THIN_GAP, rad=RAD["name"], radl=RAD["length"], key=RAD["key"], dk=RAD["dk"], pin_out=PIN_OUT)
    todo = TODO.format(tube_d=TUBE_D, rad=RAD["name"], pitch=RAD["pitch"], below=HOLE_BELOW, tube=TUBE_LEN, e=fit["e"], thin_w=2 * THIN_X, winw=2 * WIN_W, winh=WIN_TOP - FLOOR,
                       clear=WIN_TOP - FLOOR - ARM_SCREW["head_k"], pin_need=PIN_OUT + 0.3, dk=RAD["dk"], radl=RAD["length"], tol=RAD["tol"], cone=HOLE_BELOW + fit["e"], lip=LIP,
                       pbore=FITS["print"]["bore"], pe=FITS["print"]["e"], ppin=FITS["print"]["pin"])
    text = head + "\n检查结果\n" + "\n".join(lines) + "\n\n" + todo
    (out / "check_report.txt").write_text(text)
    print(text)
    render(parts["cnc"], fit, extra, out)


if __name__ == "__main__":
    main()
