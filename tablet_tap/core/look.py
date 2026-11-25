"""Where to hold the arm so that the camera on its wrist sees the whole screen: the look pose.

``WristCamera`` is the camera on the flange; ``LookPlanner`` searches joints from which the screen is large, upright
enough, not behind the pen, and which the arm reaches without coming near the table, the post or its own links. The
arm's shape is points sampled from the collision meshes; pen, holder and camera are boxes.
"""

from __future__ import annotations

import dataclasses
import functools
import pathlib
from typing import Any

import cv2
import numpy as np
import pinocchio as pin
from scipy.spatial import cKDTree

from tablet_tap.core.kinematics import FINGERTIP_IN_FLANGE
from tablet_tap.core.kinematics import URDF_PATH
from tablet_tap.core.kinematics import PenArm
from tablet_tap.core.kinematics import pose
from tablet_tap.core.pinhole import CameraModel
from tablet_tap.core.screen_map import ScreenMap
from tablet_tap.errors import Unreachable

COLLISION_DIR = URDF_PATH.parent / "Collision"
LINKS = ("link1", "link2", "link3", "link4", "link5", "link6")
POINTS_PER_LINK = 500
# (low, high) corners in the flange (link6) frame: z out of the gripper, -x toward the camera
HOLDER_BOX = ((-0.02, -0.038, 0.10), (0.04, 0.038, 0.165))
PEN_HALF_M = 0.006
MOUNT_BOX = ((-0.10, -0.03, 0.0), (-0.03, 0.03, 0.06))  # the camera's column
# the D435 around its colour lens, in the camera frame (x right, y down, z forward)
CAMERA_BOX = ((-0.0125, -0.0125, -0.021), (0.0775, 0.0125, 0.004))
# (low, high) corners in the arm's base frame: the post behind the left arm and the other arm's side of the bench
POST_BOX = ((-0.085, -0.315, 0.0), (-0.055, -0.285, 0.9))
OTHER_ARM_Y = -0.42  # nothing of this arm goes beyond this line

# The wrist camera as it is mounted: the colour lens above the gripper axis (toward -x of the flange), beside it (+y),
# ahead of the flange, and how far the optical axis points down toward the pen.
KIT_MOUNT = (0.0717, 0.0025, 0.0409, 20.0)  # with it the pen tip covers the middle of the picture
TAP_MOUNT = (0.100, 0.0025, 0.050, 15.0)  # cad/wrist_camera/out_one_piece
SIM_MOUNT = (0.120, 0.010, 0.030, 32.05)  # the printed mount of the simulator's wrist view, on the left arm from 2026-11-23 21:40
MOUNTS = {"tap": TAP_MOUNT, "kit": KIT_MOUNT, "sim": SIM_MOUNT}
# What pen, holder and fingers cover in the 1920 x 1080 picture of a mount, outlined in pictures of 2026-11-23 with a
# margin of about 20 px. They hold for that pen in that holder: look again when the pen sits differently.
COVERED_PX = {
    "sim": [[(990, 555), (1070, 555), (1080, 700), (990, 700)], [(930, 665), (1090, 665), (1250, 1080), (920, 1080)],
            [(890, 850), (1240, 850), (1390, 1080), (815, 1080)]],
}  # fmt: skip


def camera_in_flange(up_m: float, side_m: float, ahead_m: float, tilt_deg: float) -> np.ndarray:
    """(4, 4) the camera (x right, y down, z forward) in the flange frame, for a mount without roll or yaw."""
    t = np.radians(tilt_deg)
    forward = np.array([np.sin(t), 0.0, np.cos(t)])
    down = np.array([np.cos(t), 0.0, -np.sin(t)])
    return pose(np.column_stack([np.cross(down, forward), down, forward]), np.array([-up_m, side_m, ahead_m]))


@dataclasses.dataclass(frozen=True)
class WristCamera:
    model: CameraModel
    in_flange: np.ndarray  # (4, 4)

    def pixels(self, points_base: np.ndarray, camera_in_base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(..., 2) pixels and (...,) depths of points of the arm's base frame."""
        rot, eye = camera_in_base[:3, :3], camera_in_base[:3, 3]
        seen = (np.asarray(points_base, np.float64) - eye) @ rot
        return self.model.project(seen), seen[..., 2]


def tip_in_flange(pen: PenArm) -> np.ndarray:
    return (FINGERTIP_IN_FLANGE * pin.SE3(np.eye(3), np.array([0.0, 0.0, float(pen.tip_len)]))).homogeneous


def flange_pose(pen: PenArm, q: np.ndarray) -> np.ndarray:
    """(..., 4, 4) the flange (link6) in the base frame."""
    return pen.fk(q) @ np.linalg.inv(tip_in_flange(pen))


def _box_points(low: tuple, high: tuple, n: int = 4) -> np.ndarray:
    axes = [np.linspace(a, b, n) for a, b in zip(low, high)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    edge = np.any((grid == np.array(low)) | (grid == np.array(high)), axis=1)
    return grid[edge]


def _mesh_points(path: pathlib.Path, n: int, seed: int) -> np.ndarray:
    """``n`` points spread over the surface of an .obj mesh."""
    vertices, faces = [], []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            vertices.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            corner = [int(x.split("/")[0]) - 1 for x in line.split()[1:]]
            faces += [(corner[0], corner[k], corner[k + 1]) for k in range(1, len(corner) - 1)]
    v, f = np.array(vertices), np.array(faces)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    rng = np.random.default_rng(seed)
    pick, st = rng.choice(len(f), n, p=area / area.sum()), rng.random((n, 2))
    st = np.where(st.sum(1, keepdims=True) > 1, 1 - st, st)
    return a[pick] + st[:, :1] * (b - a)[pick] + st[:, 1:] * (c - a)[pick]


@functools.lru_cache(maxsize=2)
def _link_points(n: int = POINTS_PER_LINK) -> dict[str, np.ndarray]:
    points = {name: _mesh_points(COLLISION_DIR / f"{name}_acd.obj", n, k) for k, name in enumerate(LINKS)}
    points["gripper"] = _mesh_points(COLLISION_DIR / "gripper_base_v100_acd.obj", n, 17)
    return points


@functools.lru_cache(maxsize=2)
def _arm_model(path: str) -> pin.Model:
    return pin.buildModelFromUrdf(path)


def _box_gap(points: np.ndarray, low: tuple, high: tuple) -> float:
    """The smallest distance (m) from any of the points to a box; 0 inside."""
    outside = np.maximum(np.maximum(np.array(low) - points, points - np.array(high)), 0.0)
    return float(np.linalg.norm(outside, axis=1).min())


class ArmShape:
    """The arm with pen, holder and wrist camera, as points: what could hit something at given joints."""

    def __init__(self, pen: PenArm, in_flange: np.ndarray) -> None:
        self.pen, self.in_flange = pen, np.asarray(in_flange, np.float64)
        self.model = _arm_model(pen.urdf_path)
        self.data = self.model.createData()
        tip = tip_in_flange(pen)[2, 3]
        half = PEN_HALF_M
        holder = [_box_points(*HOLDER_BOX), _box_points((-half, -half, 0.16), (half, half, float(tip)))]
        camera = _box_points(*CAMERA_BOX) @ self.in_flange[:3, :3].T + self.in_flange[:3, 3]
        self.tool = np.concatenate([_link_points()["gripper"], *holder])
        self.camera = np.concatenate([camera, _box_points(*MOUNT_BOX)])
        self.camera_body = camera

    def points(self, q: np.ndarray) -> dict[str, np.ndarray]:
        pin.forwardKinematics(self.model, self.data, np.asarray(q, np.float64))
        pin.updateFramePlacements(self.model, self.data)
        out = {}
        for name in LINKS:
            placed = self.data.oMf[self.model.getFrameId(name)]
            local = _link_points()[name]
            out[name] = local @ placed.rotation.T + placed.translation
            if name == "link6":
                for part, body in (("tool", self.tool), ("camera", self.camera), ("camera_body", self.camera_body)):
                    out[part] = body @ placed.rotation.T + placed.translation
        return out

    def clearance(self, q: np.ndarray) -> dict[str, float]:
        """The smallest gaps (m): to the table (z = 0 of the arm's base), the post, the other arm's side, and between
        the wrist camera and the arm's own links."""
        p = self.points(q)
        moving = np.concatenate([p[k] for k in ("link3", "link4", "link5", "link6", "tool", "camera")])
        whole = np.concatenate([moving, p["link2"]])
        links = np.concatenate([p[k] for k in ("link1", "link2", "link3", "link4")])
        return {
            "table": float(moving[:, 2].min()),
            "post": _box_gap(whole, *POST_BOX),
            "other_arm": float(whole[:, 1].min() - OTHER_ARM_Y),
            "camera": float(cKDTree(links).query(p["camera_body"])[0].min()),
        }


@dataclasses.dataclass(frozen=True)
class LookRules:
    margin_px: float = 40.0  # the screen stays this far inside the picture
    pen_px: float = 25.0  # and this far from the pen and its holder
    off_normal_deg: float = 25.0
    distance_m: tuple[float, float] = (0.20, 0.50)
    tip_m: float = 0.04  # the pen stays this high above the glass
    limit_deg: float = 4.0  # distance from any joint limit, beyond the planner's own margin
    table_m: float = 0.015
    post_m: float = 0.04
    camera_m: float = 0.03
    travel_deg: float = 120.0  # no joint goes further than this from the park pose (no flipped wrist)


# for calibration views: the screen may be partly outside the picture and is seen at a slant
CALIBRATION_RULES = LookRules(margin_px=8.0, pen_px=10.0, off_normal_deg=42.0, distance_m=(0.20, 0.55), limit_deg=2.0)


@dataclasses.dataclass(frozen=True)
class Look:
    q: np.ndarray
    camera_in_base: np.ndarray
    width_px: float  # the screen's shorter horizontal edge in the picture
    height_px: float
    distance_m: float
    off_normal_deg: float
    tip_height_m: float
    limit_deg: float  # distance from the nearest joint limit
    clearance: dict[str, float]

    @property
    def detail_px(self) -> float:
        """The width of a 16:10 screen seen with the detail of the worse direction."""
        return min(self.width_px, 1.6 * self.height_px)

    def describe(self) -> str:
        gaps = "、".join(f"{name} {gap * 1e3:.0f}" for name, gap in zip(
            ("离桌面", "离立柱", "离右臂一侧", "相机离手臂连杆"), self.clearance.values()))  # fmt: skip
        return (
            f"屏幕在画面里约 {self.width_px:.0f} × {self.height_px:.0f} px，镜头离屏幕中心 {self.distance_m * 1e2:.0f} cm、"
            f"偏离法线 {self.off_normal_deg:.0f}°，笔尖在屏幕上方 {self.tip_height_m * 1e3:.0f} mm，"
            f"离最近的关节限位 {self.limit_deg:.0f}°；{gaps} mm"
        )

    def to_json(self) -> dict[str, Any]:
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in dataclasses.asdict(self).items()}


class LookPlanner:
    def __init__(
        self,
        pen: PenArm,
        camera: WristCamera,
        rules: LookRules | None = None,
        seed: int = 0,
        covered: list[np.ndarray] | None = None,
    ) -> None:
        """``covered``: what pen and holder were measured to cover in the picture (convex outlines, pixels); None: boxes
        of their size, which hide more than the real ones do."""
        self.pen, self.camera, self.rules = pen, camera, rules or LookRules()
        self.shape = ArmShape(pen, camera.in_flange)
        self.rng = np.random.default_rng(seed)
        self._to_tip = np.linalg.inv(camera.in_flange) @ tip_in_flange(pen)
        self._hidden = self._hidden_hulls() if covered is None else [cv2.convexHull(np.float32(h)) for h in covered]

    # ---- what the pen hides ------------------------------------------------------------------------------------------
    def _hidden_hulls(self) -> list[np.ndarray]:
        """Outlines (convex, pixels) of pen and holder in the picture: the same at every arm pose."""
        tip = float(tip_in_flange(self.pen)[2, 3])
        half = PEN_HALF_M
        hulls = []
        for low, high in (HOLDER_BOX, ((-half, -half, 0.16), (half, half, tip + 0.002))):
            points = _box_points(low, high, 12)
            seen = (points - self.camera.in_flange[:3, 3]) @ self.camera.in_flange[:3, :3]
            seen = seen[seen[:, 2] > 0.02]
            if len(seen) >= 3:
                hulls.append(cv2.convexHull(self.camera.model.project(seen).astype(np.float32)))
        return hulls

    def hides(self, quad: np.ndarray, pad: float) -> bool:
        """Whether pen or holder come within ``pad`` pixels of the screen's outline (4, 2) in the picture."""
        centre = quad.mean(0)
        outward = quad - centre
        grown = (quad + pad * outward / np.linalg.norm(outward, axis=1, keepdims=True) * np.sqrt(2)).astype(np.float32)
        grown = cv2.convexHull(grown)
        return any(cv2.intersectConvexConvex(grown, hull)[0] > 0 for hull in self._hidden)

    # ---- candidates --------------------------------------------------------------------------------------------------
    def _views(self, screen: ScreenMap, n: int) -> tuple[np.ndarray, ...]:
        """Random camera poses looking at the screen, at any turn, that keep it in the frame: poses (k, 4, 4), outlines (k, 4, 2), distances, angles off the normal."""
        rng, rules = self.rng, self.rules
        corners = np.array([screen.point(u, v) for u, v in ((0, 0), (1, 0), (1, 1), (0, 1))])
        centre, normal = screen.point(0.5, 0.5), screen.normal
        ex = screen.right / np.linalg.norm(screen.right)
        ey = np.cross(normal, ex)
        distance = rng.uniform(*rules.distance_m, n)
        lean = np.arccos(rng.uniform(np.cos(np.radians(rules.off_normal_deg)), 1.0, n))
        azimuth, roll = rng.uniform(-np.pi, np.pi, n), rng.uniform(-np.pi, np.pi, n)
        away = (np.sin(lean) * np.cos(azimuth))[:, None] * ex + (np.sin(lean) * np.sin(azimuth))[:, None] * ey
        away += np.cos(lean)[:, None] * normal
        eye = centre + distance[:, None] * away
        width, height = screen.size_m
        aim = centre + rng.uniform(-0.45, 0.45, n)[:, None] * width * ex + rng.uniform(-0.6, 0.6, n)[:, None] * height * ey
        forward = aim - eye
        forward /= np.linalg.norm(forward, axis=1, keepdims=True)
        hint = np.cos(roll)[:, None] * ex + np.sin(roll)[:, None] * ey
        down = hint - np.sum(hint * forward, 1, keepdims=True) * forward
        down /= np.linalg.norm(down, axis=1, keepdims=True)
        rot = np.stack([np.cross(down, forward), down, forward], axis=2)
        seen = (corners[None] - eye[:, None]) @ rot
        ok = np.all(seen[..., 2] > 0.05, axis=1)
        pixel = self.camera.model.project(np.where(seen[..., 2:3] > 0.05, seen, 1.0))
        size = np.array([self.camera.model.width, self.camera.model.height])
        m = rules.margin_px
        ok &= np.all((pixel > m) & (pixel < size - 1 - m), axis=(1, 2))
        poses = np.tile(np.eye(4), (n, 1, 1))
        poses[:, :3, :3], poses[:, :3, 3] = rot, eye
        return poses[ok], pixel[ok], distance[ok], np.degrees(lean[ok])

    def feasible(self, screen: ScreenMap, camera_pose: np.ndarray, seeds: list[np.ndarray]) -> Look | None:
        """The look from ``camera_pose``, when the arm can hold it and everything keeps its distance."""
        rules = self.rules
        tip = camera_pose @ self._to_tip
        tip_height = float((tip[:3, 3] - screen.origin) @ screen.normal)
        if tip_height < rules.tip_m:
            return None
        for start in seeds:
            q, miss, turn = self.pen.ik(tip, start)
            if miss < 1e-3 and turn < 5e-3:
                break
        else:
            return None
        limit = float(np.degrees(np.min(np.minimum(q - self.pen.lower, self.pen.upper - q))))
        if limit < rules.limit_deg:
            return None
        gaps = self.shape.clearance(q)
        if (
            gaps["table"] < rules.table_m
            or gaps["post"] < rules.post_m
            or gaps["camera"] < rules.camera_m
            or gaps["other_arm"] < 0
        ):
            return None
        return self._look(screen, q, gaps, tip_height, limit)

    def _look(self, screen: ScreenMap, q: np.ndarray, gaps: dict, tip_height: float, limit: float) -> Look:
        camera_pose = flange_pose(self.pen, q) @ self.camera.in_flange
        corners = np.array([screen.point(u, v) for u, v in ((0, 0), (1, 0), (1, 1), (0, 1))])
        px, _ = self.camera.pixels(corners, camera_pose)
        edge = np.linalg.norm(np.roll(px, -1, axis=0) - px, axis=1)  # top, right, bottom, left
        to_eye = camera_pose[:3, 3] - screen.point(0.5, 0.5)
        distance = float(np.linalg.norm(to_eye))
        off = float(np.degrees(np.arccos(np.clip(to_eye @ screen.normal / distance, -1, 1))))
        return Look(q, camera_pose, float(min(edge[0], edge[2])), float(min(edge[1], edge[3])), distance, off, tip_height, limit, gaps)

    def _seeds(self, screen: ScreenMap, seed_q: np.ndarray, park_q: np.ndarray) -> list[np.ndarray]:
        park = np.asarray(park_q, np.float64)
        turned = np.asarray(seed_q, np.float64).copy()
        centre = screen.point(0.5, 0.5)
        turned[0] = np.arctan2(centre[1], centre[0])
        return [turned, np.asarray(seed_q, np.float64), park, park + [0.0, 0.5, -0.4, 0.0, 0.3, 0.0]]

    def _from_postures(self, screen: ScreenMap, seed_q: np.ndarray, park_q: np.ndarray, *, postures: int = 20000) -> list[Look]:
        """Looks found the other way round. Most camera poses over the screen are beyond the wrist's pitch, and an
        inverse solve started at the touch or the park pose does not arrive at the few that are within it. So: camera
        positions on a cap over the screen, the picture level with the screen and the screen moved up in it, away from
        the pen; each solved from the nearest of many random postures of the arm, which is already close to it."""
        rules, model, pen = self.rules, self.camera.model, self.pen
        margin = np.radians(rules.limit_deg)
        spread = np.array([0.6, 0.9, 0.9, 1.2, 0.8, 1.8])
        cloud = np.clip(np.asarray(seed_q, np.float64) + self.rng.uniform(-1, 1, (postures, 6)) * spread, pen.lower + margin, pen.upper - margin)
        held = flange_pose(pen, cloud) @ self.camera.in_flange
        nearest = cKDTree(np.column_stack([held[:, :3, 3], 0.15 * held[:, :3, 2], 0.05 * held[:, :3, 1]]))
        corners = np.array([screen.point(u, v) for u, v in ((0, 0), (1, 0), (1, 1), (0, 1))])
        centre, normal, height = screen.point(0.5, 0.5), screen.normal, screen.size_m[1]
        ex, ey = screen.right / np.linalg.norm(screen.right), screen.down / np.linalg.norm(screen.down)
        size, found = np.array([model.width, model.height]), []
        # the pen is at the picture's lower edge (at its upper edge with the camera mounted the other way up): the
        # screen is moved away from it
        below = np.mean([hull.reshape(-1, 2)[:, 1].mean() for hull in self._hidden]) >= model.cy if self._hidden else True
        shifts = [side * (1.0 if below else -1.0) for side in (0.1, 0.25, 0.0)]
        for distance in np.linspace(rules.distance_m[0] + 0.02, min(rules.distance_m[1], 0.42), 9):
            for lean in np.radians(np.arange(0.0, rules.off_normal_deg + 0.1, 8.0)):
                for azimuth in [0.0] if lean == 0 else np.linspace(0, 2 * np.pi, 12, endpoint=False):
                    away = np.sin(lean) * (np.cos(azimuth) * ex + np.sin(azimuth) * ey) + np.cos(lean) * normal
                    eye = centre + distance * away
                    for hint in (ey, -ey):  # the picture's lower edge toward the screen's, or turned half round
                        for shift in shifts:
                            forward = centre + shift * height * hint - eye
                            forward /= np.linalg.norm(forward)
                            down = hint - (hint @ forward) * forward
                            down /= np.linalg.norm(down)
                            rot = np.column_stack([np.cross(down, forward), down, forward])
                            seen = (corners - eye) @ rot
                            pixel = model.project(seen)
                            if np.any(seen[:, 2] < 0.05) or np.any((pixel < rules.margin_px) | (pixel > size - 1 - rules.margin_px)):
                                continue
                            if self.hides(pixel, rules.pen_px):
                                continue
                            camera_pose = pose(rot, eye)
                            tip = camera_pose @ self._to_tip
                            starts = []
                            for k in nearest.query(np.concatenate([eye, 0.15 * forward, 0.05 * down]), k=3)[1]:
                                q, miss, turn = pen.ik(tip, cloud[k])
                                if miss < 1e-3 and turn < 5e-3:
                                    starts.append(q)
                                    break
                            look = self.feasible(screen, camera_pose, starts) if starts else None
                            if look is None or np.max(np.abs(look.q - np.asarray(park_q))) > np.radians(rules.travel_deg):
                                continue
                            if self.path_problem(screen, park_q, look.q) is None:
                                found.append(look)
        return found

    def path_problem(self, screen: ScreenMap, q_start: np.ndarray, q_goal: np.ndarray, *, tip_m: float = 0.0135) -> str | None:
        """What a straight joint move between the two would come too near to; None when the way is free."""
        rules = self.rules
        q_start, q_goal = np.asarray(q_start, np.float64), np.asarray(q_goal, np.float64)
        n = max(int(np.ceil(np.max(np.abs(q_goal - q_start)) / 0.05)), 1) + 1
        for q in q_start + np.linspace(0, 1, n)[:, None] * (q_goal - q_start):
            gaps = self.shape.clearance(q)
            for name, needed in (("table", 0.5 * rules.table_m), ("post", 0.5 * rules.post_m), ("camera", 0.5 * rules.camera_m),
                                 ("other_arm", 0.0)):  # fmt: skip
                if gaps[name] < needed:
                    return f"{ {'table': '桌面', 'post': '立柱', 'camera': '相机和手臂连杆', 'other_arm': '右臂一侧'}[name] }只差 {gaps[name] * 1e3:.0f} mm"
            p = self.shape.points(q)
            low = np.concatenate([p["tool"], p["camera"]])
            basis = np.stack([screen.right, screen.down, screen.normal], axis=1)
            uvh = np.linalg.solve(basis, (low - screen.origin).T).T
            over = (np.abs(uvh[:, 0] - 0.5) < 0.6) & (np.abs(uvh[:, 1] - 0.5) < 0.6)
            if over.any() and uvh[over, 2].min() < tip_m:
                return f"平板只差 {uvh[over, 2].min() * 1e3:.0f} mm"
        return None

    # ---- the two searches --------------------------------------------------------------------------------------------
    def best(
        self,
        screen: ScreenMap,
        seed_q: np.ndarray,
        park_q: np.ndarray,
        *,
        samples: int = 60000,
        tries: int = 250,
        keep: int = 12,
    ) -> Look:
        """The look pose: the screen as large as it gets, within reason of reach, limits and travel. ``Unreachable``
        when no pose satisfies the rules."""
        found = self._from_postures(screen, seed_q, park_q)
        poses, pixels, _, off = self._views(screen, samples if len(found) < 3 else 0)
        edge = np.linalg.norm(np.roll(pixels, -1, axis=1) - pixels, axis=2)
        detail = np.minimum(np.minimum(edge[:, 0], edge[:, 2]), 1.6 * np.minimum(edge[:, 1], edge[:, 3]))
        order = np.argsort(-detail)
        seeds, tried = self._seeds(screen, seed_q, park_q), 0
        for k in order:
            if tried >= tries and found or tried >= 6 * tries or len(found) >= keep:
                break
            if self.hides(pixels[k], self.rules.pen_px):
                continue
            tried += 1
            look = self.feasible(screen, poses[k], seeds)
            if look is None or np.max(np.abs(look.q - np.asarray(park_q))) > np.radians(self.rules.travel_deg):
                continue
            if self.path_problem(screen, park_q, look.q) is None:
                found.append(look)
        if not found:
            ahead = float(np.hypot(*screen.point(0.5, 0.5)[:2])) * 1e2
            raise Unreachable(
                f"找不到能看全屏幕的看屏位（被笔挡住、够不到、离关节限位太近，或者途中会碰到东西）。平板中心现在离底座 {ahead:.0f} cm："
                + (
                    "腕部相机要退到屏幕上方 25 cm 以外才看得全，太近了手腕转不过来，放到 31–33 cm 再试。"
                    if ahead < 29.5
                    else "距离是够的，多半是笔挡住了画面（原装支架 kit 就是这样，要用新支架 tap），或者平板放得太偏、太斜。"
                )
            )

        def score(look: Look) -> float:
            travel = np.degrees(np.max(np.abs(look.q - np.asarray(park_q))))
            return look.detail_px + 8 * min(look.limit_deg, 15.0) - 1.0 * travel - 3 * look.off_normal_deg

        return max(found, key=score)

    def calibration_views(
        self,
        screen: ScreenMap,
        around: Look,
        park_q: np.ndarray,
        boxes: list[tuple[float, float, float, float]],
        n: int = 14,
        *,
        spread_deg: float = 28.0,
        min_boxes: int = 4,
        samples: int = 6000,
        tries: int = 160,
    ) -> list[Look]:
        """Looks near ``around`` that differ in direction, turn and distance and see at least ``min_boxes`` of the
        markers (u0, v0, u1, v1), in an order the arm can go through with straight joint moves, beginning at the park
        pose. A leg that has to go by the park pose is marked in its clearance."""
        rules, size = self.rules, np.array([self.camera.model.width, self.camera.model.height])
        corners = np.array([[screen.point(u, v) for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))] for u0, v0, u1, v1 in boxes])
        pool: list[Look] = []
        scale = np.radians(spread_deg) * np.array([0.5, 0.6, 0.6, 1.0, 0.8, 1.6])
        for step in self.rng.uniform(-1, 1, (samples, 6)) * scale:
            if len(pool) >= tries:
                break
            q = np.clip(around.q + step, self.pen.lower, self.pen.upper)
            camera_pose = flange_pose(self.pen, q) @ self.camera.in_flange
            px, depth = self.camera.pixels(corners, camera_pose)
            inside = np.all((depth > 0.1)[..., None] & (px > rules.margin_px) & (px < size - 1 - rules.margin_px), axis=(1, 2))
            seen = [k for k in np.flatnonzero(inside) if not self.hides(px[k], rules.pen_px)]
            if len(seen) < min_boxes:
                continue
            tip = self.pen.fk(q)[:3, 3]
            tip_height = float((tip - screen.origin) @ screen.normal)
            if tip_height < rules.tip_m:
                continue
            limit = float(np.degrees(np.min(np.minimum(q - self.pen.lower, self.pen.upper - q))))
            gaps = self.shape.clearance(q)
            if limit < rules.limit_deg or gaps["table"] < rules.table_m or gaps["post"] < rules.post_m or gaps["camera"] < rules.camera_m or gaps["other_arm"] < 0:  # fmt: skip
                continue
            look = self._look(screen, q, gaps, tip_height, limit)
            if look.off_normal_deg > rules.off_normal_deg or not rules.distance_m[0] < look.distance_m < rules.distance_m[1]:
                continue
            if self.path_problem(screen, park_q, q) is None:
                pool.append(look)
        if len(pool) < n:
            raise Unreachable(f"只找到 {len(pool)} 个能看到标定页的位姿，标定至少要 {n} 个；换一下平板的位置再试。")
        rot0 = around.camera_in_base[:3, :3].T
        feature = np.array([np.concatenate([pin.log3(look.camera_in_base[:3, :3] @ rot0), 3.0 * look.camera_in_base[:3, 3]]) for look in pool])
        chosen = [int(np.argmax(np.linalg.norm(feature - feature.mean(0), axis=1)))]
        while len(chosen) < n:  # farthest point sampling: each next view is the most different from those taken
            gap = np.min(np.linalg.norm(feature[:, None] - feature[chosen], axis=2), axis=1)
            chosen.append(int(np.argmax(gap)))
        rest, route, here = [pool[k] for k in chosen], [], np.asarray(park_q, np.float64)
        while rest:
            rest.sort(key=lambda look: float(np.max(np.abs(look.q - here))))  # noqa: B023
            step = next((look for look in rest if self.path_problem(screen, here, look.q) is None), None)
            if step is None:  # go by the park pose, from which every view was checked
                step = rest[0]
                route.append(dataclasses.replace(step, clearance=step.clearance | {"via_park": 1.0}))
            else:
                route.append(step)
            rest.remove(step)
            here = step.q
        return route
