"""The automatic calibration steps: probing a grid of screen points, the tap-accuracy test, and the fit's report.

They take the robot and the calibration page as objects and return plain results; printing and saving is the command
line's business (cli/commands).
"""

from __future__ import annotations

from collections.abc import Callable
import dataclasses

import numpy as np

from tablet_tap.calibration.page_client import TouchPage
from tablet_tap.core.kinematics import PenArm
from tablet_tap.core.motion import MotionLimits
from tablet_tap.core.motion import TapPlanner
from tablet_tap.core.screen_map import ScreenMap
from tablet_tap.core.screen_map import fit_screen_map
from tablet_tap.errors import CalibrationError
from tablet_tap.errors import Unreachable
from tablet_tap.robot.tablet_robot import TabletRobot

MIN_PROBE_POINTS = 4


def grid_points(nu: int, nv: int, margin: float) -> list[tuple[float, float]]:
    """An nu x nv grid of screen points, ``margin`` away from the edges, row by row in a serpentine."""
    us, vs = np.linspace(margin, 1 - margin, nu), np.linspace(margin, 1 - margin, nv)
    rows = [[(float(u), float(v)) for u in (us if j % 2 == 0 else us[::-1])] for j, v in enumerate(vs)]
    return [point for row in rows for point in row]


@dataclasses.dataclass(frozen=True)
class ProbePoint:
    """One probed point: where the pen was sent, where the screen saw it, and the pen-tip pose at the contact."""

    u_cmd: float
    v_cmd: float
    u: float
    v: float
    pose: list[list[float]]  # (4, 4)
    error_mm: float  # distance on the glass between the commanded and the touched point
    height_mm: float  # contact above (+) or below (-) the plane of the calibration in use


def probe_grid(
    robot: TabletRobot,
    page: TouchPage,
    points: list[tuple[float, float]],
    depth_m: float,
    on_point: Callable[[int, tuple[float, float], ProbePoint | Unreachable | None], None] | None = None,
) -> list[ProbePoint]:
    """Go down slowly at every point until the page reports the touch. Points the arm cannot reach and points the
    screen never reported are left out; ``on_point`` hears about them with the ``Unreachable`` and with None. The
    robot ends at the park pose."""
    page.set_mode("blank")
    found = []
    for index, (u, v) in enumerate(points, 1):
        seq = page.last_down_seq()
        try:  # the whole move is planned before any of it is sent: a point out of reach leaves the arm where it hovers
            contact = robot.probe(u, v, touched=lambda seq=seq: page.last_down_seq() > seq, depth_m=depth_m)
        except Unreachable as exc:
            if on_point is not None:
                on_point(index, (u, v), exc)
            continue
        touched = page.wait_down(seq, 0.5) if contact is not None else None
        if touched is None and not found:  # pressing on at every other point could only hurt the screen
            if on_point is not None:
                on_point(index, (u, v), None)
            robot.to_park()
            raise CalibrationError("第一个点压到最深屏幕也没有报告触摸，已停止：检查笔是否导电、屏幕平面估计是否准。")
        point = None
        if touched is not None:
            point = ProbePoint(
                u_cmd=u,
                v_cmd=v,
                u=touched[0],
                v=touched[1],
                pose=contact.tolist(),
                error_mm=robot.screen.error_mm((u, v), touched),
                height_mm=robot.screen.uv_of(contact[:3, 3])[2] * 1e3,
            )
            found.append(point)
        if on_point is not None:
            on_point(index, (u, v), point)
    robot.to_park()
    return found


def refit(found: list[ProbePoint], aspect: float | None, tilt_deg: float = 0.0) -> ScreenMap:
    """The screen fitted to the probed contacts; ``tilt_deg`` is the pen's lean they were made with, which is kept."""
    if len(found) < MIN_PROBE_POINTS:
        raise CalibrationError(f"只有 {len(found)} 个点触发，不足以重新拟合；检查笔是否导电、下压深度是否够。")
    uv, poses = np.array([[p.u, p.v] for p in found]), np.array([p.pose for p in found])
    return dataclasses.replace(fit_screen_map(uv, poses, aspect=aspect), tilt_deg=tilt_deg)


@dataclasses.dataclass(frozen=True)
class TapOutcome:
    u: float
    v: float
    touched: tuple[float, float] | None  # where the screen saw the tap; None: it saw none
    error_mm: float | None
    unreachable: bool = False  # the arm cannot get there: the point was not tapped


def tap_test(
    robot: TabletRobot,
    page: TouchPage,
    points: list[tuple[float, float]],
    on_point: Callable[[int, TapOutcome], None] | None = None,
) -> list[TapOutcome]:
    """Tap every point (the page shows a target there) and compare with where the screen saw the tap. The robot ends
    at the park pose and the page back on its markers."""
    outcomes = []
    for index, (u, v) in enumerate(points, 1):
        page.set_mode("target", (u, v))
        seq = page.last_down_seq()
        try:
            robot.tap(u, v)
        except Unreachable:
            outcome = TapOutcome(u, v, None, None, unreachable=True)
        else:
            touched = page.wait_down(seq, 1.5)
            outcome = TapOutcome(u, v, touched, None if touched is None else robot.screen.error_mm((u, v), touched))
        outcomes.append(outcome)
        if on_point is not None:
            on_point(index, outcome)
    robot.to_park()
    page.set_mode("markers")
    return outcomes


def summarize_taps(outcomes: list[TapOutcome]) -> dict[str, float | int]:
    errors = np.array([o.error_mm for o in outcomes if o.error_mm is not None])
    tapped = [o for o in outcomes if not o.unreachable]
    summary: dict[str, float | int] = {"n": len(tapped), "registered": int(len(errors))}
    if len(tapped) < len(outcomes):
        summary["unreachable"] = len(outcomes) - len(tapped)
    if len(errors):
        summary.update(
            mean_mm=float(errors.mean()),
            median_mm=float(np.median(errors)),
            p95_mm=float(np.percentile(errors, 95)),
            max_mm=float(errors.max()),
            within_2mm=int(np.sum(errors <= 2.0)),
        )
    return summary


def fit_report(fitted: ScreenMap, pen: PenArm, limits: MotionLimits, seed_q: list[float]) -> list[str]:
    """What the operator should know about a fitted screen: its size and tilt, the residual, and what the arm reaches."""
    width, height = fitted.size_m
    tilt = np.degrees(np.arccos(np.clip(fitted.normal[2], -1, 1)))
    lines = [
        f"屏幕平面：{width * 1e3:.1f} × {height * 1e3:.1f} mm，倾角 {tilt:.1f}°，两边垂直度偏差 {fitted.skew_deg:+.2f}°，"
        f"{fitted.n_points} 个触点的拟合残差 {fitted.residual_mm:.2f} mm（均方根）"
    ]
    planner = TapPlanner(pen, fitted, limits)
    try:
        reach = planner.reach_report(planner.hover_q(0.5, 0.5, np.asarray(seed_q)))
    except Unreachable as exc:
        return [*lines, f"⚠ 屏幕中心的悬停位姿够不到：{exc}"]
    if reach["unreachable"]:
        lines.append(
            f"⚠ {reach['n']} 个检查点里有 {len(reach['unreachable'])} 个够不到（u, v）：{reach['unreachable']}\n"
            "  把平板往机械臂基座挪近一些（屏幕中心在基座前方 24–36 cm 最合适）再重新标定。"
        )
    else:
        lines.append(f"可达性：{reach['n']} / {reach['n']} 个检查点（含屏幕边缘外 2 %）都够得到。")
    return lines
