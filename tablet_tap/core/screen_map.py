"""Where the screen is, in the arm's base frame.

``ScreenMap`` is the calibration: where the pen tip is when it touches the screen point ``(u, v)`` (normalized: u left
to right, v top to bottom over the whole display). It is an affine map fitted to touches, so the tablet may lie
anywhere and at any tilt. The pen is always held along the screen's normal with the yaw of the calibration touches,
which makes the result independent of how exactly the pen sits in the gripper.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np

from tablet_tap.errors import CalibrationError


@dataclasses.dataclass(frozen=True)
class ScreenMap:
    """Pen-tip position of a touch at (u, v): ``origin + u * right + v * down`` (metres, arm base frame)."""

    origin: np.ndarray  # (3,) the screen's top-left corner
    right: np.ndarray  # (3,) top-left -> top-right: its length is the screen width
    down: np.ndarray  # (3,) top-left -> bottom-left: its length is the screen height
    tool_x: np.ndarray  # (3,) the pen-tip frame's x axis while tapping (unit, in the screen plane)
    residual_mm: float = float("nan")  # RMS distance of the calibration touches from the fit
    n_points: int = 0
    tilt_deg: float = 0.0  # the pen leans by this from the normal, its tip toward tool_x (gives the wrist more reach)

    @property
    def normal(self) -> np.ndarray:
        """Unit normal out of the screen, toward the pen."""
        n = np.cross(self.down, self.right)
        return n / np.linalg.norm(n)

    @property
    def size_m(self) -> tuple[float, float]:
        return float(np.linalg.norm(self.right)), float(np.linalg.norm(self.down))

    @property
    def skew_deg(self) -> float:
        """How far ``right`` and ``down`` are from perpendicular."""
        cos = float(self.right @ self.down / (np.linalg.norm(self.right) * np.linalg.norm(self.down)))
        return float(np.degrees(np.arcsin(np.clip(cos, -1.0, 1.0))))

    def point(self, u: float, v: float, height: float = 0.0) -> np.ndarray:
        """Pen-tip position ``height`` metres above the screen at (u, v); negative presses into it."""
        return self.origin + u * self.right + v * self.down + height * self.normal

    def tool_rotation(self) -> np.ndarray:
        """Pen-tip orientation for tapping: z into the screen (leaning by ``tilt_deg`` toward x), x as calibrated."""
        z = -self.normal
        x = self.tool_x - (self.tool_x @ z) * z
        x = x / np.linalg.norm(x)
        lean = np.radians(self.tilt_deg)
        x, z = np.cos(lean) * x - np.sin(lean) * z, np.cos(lean) * z + np.sin(lean) * x
        return np.stack([x, np.cross(z, x), z], axis=1)

    def uv_of(self, position: np.ndarray) -> tuple[float, float, float]:
        """(u, v, height) of a pen-tip position: the inverse of ``point``."""
        basis = np.stack([self.right, self.down, self.normal], axis=1)
        u, v, h = np.linalg.solve(basis, np.asarray(position, np.float64) - self.origin)
        return float(u), float(v), float(h)

    def error_mm(self, uv_wanted: tuple[float, float], uv_touched: tuple[float, float]) -> float:
        """Distance on the glass between two screen points."""
        du, dv = uv_touched[0] - uv_wanted[0], uv_touched[1] - uv_wanted[1]
        return float(np.linalg.norm(du * self.right + dv * self.down) * 1e3)

    def to_json(self) -> dict:
        return {
            "origin": self.origin.tolist(),
            "right": self.right.tolist(),
            "down": self.down.tolist(),
            "tool_x": self.tool_x.tolist(),
            "residual_mm": self.residual_mm,
            "n_points": self.n_points,
            "tilt_deg": self.tilt_deg,
        }

    @classmethod
    def from_json(cls, data: dict) -> ScreenMap:
        return cls(
            origin=np.asarray(data["origin"], np.float64),
            right=np.asarray(data["right"], np.float64),
            down=np.asarray(data["down"], np.float64),
            tool_x=np.asarray(data["tool_x"], np.float64),
            residual_mm=float(data.get("residual_mm", float("nan"))),
            n_points=int(data.get("n_points", 0)),
            tilt_deg=float(data.get("tilt_deg", 0.0)),
        )

    def save(self, path: str | pathlib.Path) -> None:
        pathlib.Path(path).write_text(json.dumps(self.to_json(), indent=2))

    @classmethod
    def load(cls, path: str | pathlib.Path) -> ScreenMap:
        return cls.from_json(json.loads(pathlib.Path(path).read_text()))


def fit_screen_map(uv: np.ndarray, tip_poses: np.ndarray, *, aspect: float | None = None) -> ScreenMap:
    """Fit a ``ScreenMap`` to touches: ``uv`` (N, 2) reported by the screen, ``tip_poses`` (N, 4, 4) of the pen tip.

    Needs at least 3 touches that are not on one line (4 or more give a residual). ``aspect`` (screen width / height),
    when known, is checked against the fit. Raises ``CalibrationError`` when the touches are degenerate or the fit says
    the screen is seen mirrored, which means the (u, v) and the poses do not belong together.
    """
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    tip_poses = np.asarray(tip_poses, np.float64).reshape(-1, 4, 4)
    if len(uv) != len(tip_poses) or len(uv) < 3:
        raise CalibrationError(f"需要至少 3 个触点，现在有 {len(uv)} 个。")
    design = np.concatenate([np.ones((len(uv), 1)), uv], axis=1)  # (N, 3): 1, u, v
    if np.linalg.matrix_rank(design, tol=1e-3) < 3:
        raise CalibrationError("触点几乎在一条直线上，无法确定屏幕平面；请在屏幕的不同角落各点一次。")
    coeffs, *_ = np.linalg.lstsq(design, tip_poses[:, :3, 3], rcond=None)  # rows: origin, right, down
    origin, right, down = coeffs
    residual = tip_poses[:, :3, 3] - design @ coeffs
    rms_mm = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))) * 1e3)

    mean_z = tip_poses[:, :3, 2].mean(axis=0)  # the pen pointed at the screen while touching
    into = np.cross(right, down)
    into = into / np.linalg.norm(into)
    if into @ mean_z < 0.5 * np.linalg.norm(mean_z):
        raise CalibrationError(
            "拟合出的屏幕法线和笔的方向对不上（屏幕像是被镜像了，或者笔和屏幕的夹角超过 60°）。"
            "请检查触点的 (u, v) 是否和记录的位姿一一对应，并让笔大致垂直于屏幕。"
        )
    mean_x = tip_poses[:, :3, 0].mean(axis=0)
    tool_x = mean_x - (mean_x @ into) * into
    if np.linalg.norm(tool_x) < 0.2:
        raise CalibrationError("各触点的笔朝向相差太大，无法确定点按时的手腕转角。")
    fitted = ScreenMap(origin, right, down, tool_x / np.linalg.norm(tool_x), residual_mm=rms_mm, n_points=len(uv))
    width, height = fitted.size_m
    if not (0.05 < width < 0.5 and 0.03 < height < 0.4):
        raise CalibrationError(f"拟合出的屏幕尺寸 {width * 1e3:.0f} × {height * 1e3:.0f} mm 不合理。")
    if abs(fitted.skew_deg) > 5.0:
        raise CalibrationError(f"拟合出的屏幕两条边不垂直（偏 {fitted.skew_deg:.1f}°）；触点数据有误。")
    if aspect is not None and abs(width / height / aspect - 1.0) > 0.06:
        raise CalibrationError(
            f"拟合出的宽高比 {width / height:.3f} 和屏幕的 {aspect:.3f} 相差超过 6 %；触点数据有误。"
        )
    return fitted
