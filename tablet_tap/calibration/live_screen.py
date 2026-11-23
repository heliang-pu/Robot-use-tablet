"""Where the tablet is right now, from whatever calibration markers the camera sees, and the pen contacts that correct it.

The tablet slides on one plane. A look with at least four markers (``hand_eye.screen_from_camera``) gives that plane, at
whatever slant it is; afterwards two markers are enough to find the tablet on it, so the arm may cover the rest. Pen
contacts are kept in the tablet's own frame: they correct what the camera gets wrong, and stay valid when the tablet is
moved. Printed tags fixed beside the screen (``TAG_IDS``) are learned while the calibration page shows its markers and
stand in for them when it does not, as long as tablet and tags move together.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from tablet_tap.core.kinematics import pose
from tablet_tap.core.pinhole import CameraModel
from tablet_tap.core.screen_map import ScreenMap
from tablet_tap.vision.layout import ScreenLayout

MIN_TRACK_MARKERS = 2
TAG_IDS = range(40, 48)  # of the page's dictionary, which uses 0-5 itself
TAG_LOOKS = 30  # looks averaged into where a tag is
TRACK_RMS_M = 0.004  # markers that fit the plane worse than this are not the tablet lying on it
AFFINE_CONTACTS = 4
AFFINE_SPREAD = 0.15  # standard deviation of u and of v that the contacts need before a tilt or scale is fitted
MAX_CONTACTS = 40
OUTLIER_M = 0.006
CONTACT_SIGMA_M = 0.001  # scatter of one contact
PRIOR_SLOPE = np.tan(np.radians(0.7))  # how far the camera's plane may tilt, turn or stretch


@dataclasses.dataclass(frozen=True)
class Contact:
    """One touch: what the screen reported, and the pen tip in the tablet's frame (x right, y down, z into the glass)."""

    u: float
    v: float
    p: tuple[float, float, float]


@dataclasses.dataclass
class LiveScreen:
    pose: np.ndarray  # (4, 4) the tablet's frame in the arm's base: origin at the screen's top-left corner
    size_m: tuple[float, float]
    tilt_deg: float = 0.0
    contacts: list[Contact] = dataclasses.field(default_factory=list)
    tags: dict[int, tuple[np.ndarray, int]] = dataclasses.field(default_factory=dict)  # id -> (4, 2) uv, looks

    @classmethod
    def from_map(cls, screen: ScreenMap) -> LiveScreen:
        x = screen.right / np.linalg.norm(screen.right)
        z = -screen.normal
        return cls(pose(np.stack([x, np.cross(z, x), z], axis=1), screen.origin), screen.size_m, screen.tilt_deg)

    # ---- the camera ------------------------------------------------------------------------------------------------
    def track(
        self, markers: dict[int, np.ndarray], layout: ScreenLayout, model: CameraModel, camera_in_base: np.ndarray
    ) -> np.ndarray | None:
        """The tablet's pose on its plane as the markers of one frame show it; None when they do not tell."""
        known = sorted(set(markers) & set(layout.markers))
        seen, drawn = [], []
        for marker_id in known:
            u0, v0, u1, v1 = layout.markers[marker_id]
            drawn += [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
            seen += np.asarray(markers[marker_id], np.float64).reshape(4, 2).tolist()
        if len(known) < MIN_TRACK_MARKERS:
            for marker_id in sorted(set(markers) & set(self.tags)):
                drawn += self.tags[marker_id][0].tolist()
                seen += np.asarray(markers[marker_id], np.float64).reshape(4, 2).tolist()
        if len(seen) < 4 * MIN_TRACK_MARKERS:
            return None
        flat = np.array(drawn) * self.size_m
        rays = np.array([((x - model.cx) / model.fx, (y - model.cy) / model.fy, 1.0) for x, y in seen])
        rays = rays @ np.asarray(camera_in_base)[:3, :3].T
        eye, rot, origin = np.asarray(camera_in_base)[:3, 3], self.pose[:3, :3], self.pose[:3, 3]
        along = ((origin - eye) @ rot[:, 2]) / (rays @ rot[:, 2])
        if np.any(along <= 0):
            return None
        on_plane = (eye + along[:, None] * rays - origin) @ rot[:, :2]
        a, b = flat - flat.mean(0), on_plane - on_plane.mean(0)
        turn = np.arctan2(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]), np.sum(a * b))
        r2 = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
        shift = on_plane.mean(0) - r2 @ flat.mean(0)
        if np.sqrt(np.mean(np.sum((flat @ r2.T + shift - on_plane) ** 2, axis=1))) > TRACK_RMS_M:
            return None
        move = np.eye(4)
        move[:2, :2], move[:2, 3] = r2, shift
        return self.pose @ move

    def learn_tags(self, markers: dict[int, np.ndarray], cam_to_norm: np.ndarray) -> None:
        """Remember where the printed tags are beside the screen, from a look that also shows the screen's markers."""
        for tag in set(markers) & set(TAG_IDS):
            quad = np.column_stack([np.asarray(markers[tag], np.float64).reshape(4, 2), np.ones(4)]) @ cam_to_norm.T
            quad = quad[:, :2] / quad[:, 2:]
            mean, looks = self.tags.get(tag, (quad, 0))
            if looks and np.abs(quad - mean).max() > 0.03:  # it was moved: start over
                mean, looks = quad, 0
            looks = min(looks + 1, TAG_LOOKS)
            self.tags[tag] = (mean + (quad - mean) / looks, looks)

    def off_plane(self, other: np.ndarray) -> tuple[float, float]:
        """How far ``other`` lies from this plane: along the normal at its centre (m), and the angle (deg)."""
        centre = other @ np.array([0.5 * self.size_m[0], 0.5 * self.size_m[1], 0.0, 1.0])
        normal = self.pose[:3, 2]
        cos = np.clip(normal @ other[:3, 2], -1.0, 1.0)
        return float(abs((centre[:3] - self.pose[:3, 3]) @ normal)), float(np.degrees(np.arccos(cos)))

    def moved(self, other: np.ndarray) -> tuple[float, float]:
        """How far the screen's centre is from where ``other`` has it (m), and the turn between the two (deg)."""
        centre = np.array([0.5 * self.size_m[0], 0.5 * self.size_m[1], 0.0, 1.0])
        cos = np.clip(self.pose[:3, 0] @ other[:3, 0], -1.0, 1.0)
        return float(np.linalg.norm((self.pose @ centre - other @ centre)[:3])), float(np.degrees(np.arccos(cos)))

    def corners(self) -> np.ndarray:
        """(4, 3) the screen's corners as the camera places them: top-left, top-right, bottom-right, bottom-left."""
        w, h = self.size_m
        return np.array([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]]) @ self.pose[:3, :3].T + self.pose[:3, 3]

    # ---- the contacts ----------------------------------------------------------------------------------------------
    def add(self, uv: tuple[float, float], tip: np.ndarray) -> None:
        p = (np.asarray(tip, np.float64) - self.pose[:3, 3]) @ self.pose[:3, :3]
        self.contacts = [*self.contacts, Contact(float(uv[0]), float(uv[1]), tuple(float(x) for x in p))][
            -MAX_CONTACTS:
        ]

    def trusted(self) -> list[Contact]:
        """The contacts without those far off the others: a touch the screen reported late has been pressed deep and
        has slid."""
        if len(self.contacts) < 3:
            return self.contacts
        offsets = np.array([np.array(c.p) - (c.u * self.size_m[0], c.v * self.size_m[1], 0.0) for c in self.contacts])
        gap = np.linalg.norm(offsets - np.median(offsets, axis=0), axis=1)
        return [c for c, g in zip(self.contacts, gap) if g < OUTLIER_M]

    @property
    def affine(self) -> bool:
        """The contacts are enough to fix the whole plane, not only to shift it."""
        kept = self.trusted()
        if len(kept) < AFFINE_CONTACTS:
            return False
        uv = np.array([(c.u, c.v) for c in kept])
        return bool(np.linalg.svd(uv - uv.mean(0), compute_uv=False)[-1] / np.sqrt(len(uv)) > AFFINE_SPREAD)

    def screen_map(self) -> ScreenMap:
        """Where to send the pen: the camera's placement, corrected by the contacts."""
        w, h = self.size_m
        local = np.array([[0.0, 0.0, 0.0], [w, 0.0, 0.0], [0.0, h, 0.0]])  # origin, right, down in the tablet's frame
        residual = float("nan")
        kept = self.trusted()
        if kept:
            uv, p = np.array([(c.u, c.v) for c in kept]), np.array([c.p for c in kept])
            local[0] += np.median(p - np.column_stack([np.ones(len(uv)), uv]) @ local, axis=0)
            centred = np.column_stack([np.ones(len(uv)), uv - 0.5])
            if self.affine:  # ridge: the camera's plane and scale hold unless the contacts clearly say otherwise
                prior = np.diag([0.0, *(CONTACT_SIGMA_M / (PRIOR_SLOPE * np.array([w, h]))) ** 2])
                error = p - np.column_stack([np.ones(len(uv)), uv]) @ local
                change = np.linalg.solve(centred.T @ centred + prior, centred.T @ error)
                local += np.array([change[0] - 0.5 * (change[1] + change[2]), change[1], change[2]])
            residual = float(
                np.sqrt(np.mean(np.sum((p - np.column_stack([np.ones(len(uv)), uv]) @ local) ** 2, axis=1))) * 1e3
            )
        rot, origin = self.pose[:3, :3], self.pose[:3, 3]
        corner, right, down = origin + rot @ local[0], rot @ local[1], rot @ local[2]
        normal = np.cross(down, right) / np.linalg.norm(np.cross(down, right))
        centre = corner + 0.5 * right + 0.5 * down
        outward = np.array([centre[0], centre[1], 0.0])
        tool_x = outward - (outward @ normal) * normal
        tool_x /= np.linalg.norm(tool_x)
        return ScreenMap(corner, right, down, tool_x, residual, len(self.contacts), self.tilt_deg)

    # ---- on disk ---------------------------------------------------------------------------------------------------
    def to_json(self) -> dict:
        return {
            "pose": self.pose.tolist(),
            "size_m": list(self.size_m),
            "tilt_deg": self.tilt_deg,
            "contacts": [dataclasses.asdict(c) for c in self.contacts],
            "tags": {str(k): [quad.tolist(), looks] for k, (quad, looks) in self.tags.items()},
        }

    @classmethod
    def from_json(cls, data: dict) -> LiveScreen:
        contacts = [Contact(c["u"], c["v"], tuple(c["p"])) for c in data["contacts"]]
        tags = {int(k): (np.array(quad), int(looks)) for k, (quad, looks) in data.get("tags", {}).items()}
        return cls(np.array(data["pose"]), tuple(data["size_m"]), float(data["tilt_deg"]), contacts, tags)


def mean_pose(poses: list[np.ndarray]) -> np.ndarray:
    """The average of poses that differ by little."""
    u, _, vt = np.linalg.svd(np.mean([p[:3, :3] for p in poses], axis=0))
    return pose(u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt, np.mean([p[:3, 3] for p in poses], axis=0))
