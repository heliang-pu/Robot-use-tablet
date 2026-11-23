"""What is left of the tapping error after the screen is calibrated, learned from the taps themselves.

Every tap that the screen reports says by how much the pen missed the point it was aimed at. That miss is a smooth
function of where on the screen the tap is (the arm's geometry is not exactly its model's), so a low-order polynomial is
fitted to it and taken off the aim. Which order the taps can support is decided by leave-one-out error, never by how well
the fit describes the taps it was made from.
"""

from __future__ import annotations

import dataclasses

import numpy as np

MAX_SAMPLES = 120
MAX_SHIFT_MM = 12.0  # a correction larger than this means something else is wrong
OUTLIER_MM = 4.0
NEEDED = {0: 2, 1: 6, 2: 12}  # order -> taps it needs
RIDGE = 0.05


def _terms(uv: np.ndarray, order: int) -> np.ndarray:
    x, y = 2 * uv[:, 0] - 1, 2 * uv[:, 1] - 1
    columns = [np.ones(len(uv))]
    if order >= 1:
        columns += [x, y]
    if order >= 2:
        columns += [x * y, x * x - 1 / 3, y * y - 1 / 3]
    return np.stack(columns, axis=1)


def _fit(uv: np.ndarray, miss: np.ndarray, order: int) -> np.ndarray:
    design = _terms(uv, order)
    ridge = RIDGE * np.eye(design.shape[1])
    ridge[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + ridge, design.T @ miss)


@dataclasses.dataclass
class TapCorrection:
    size_m: tuple[float, float]
    samples: list[tuple[float, float, float, float]] = dataclasses.field(default_factory=list)  # aimed u, v; miss mm

    def add(self, aimed: tuple[float, float], touched: tuple[float, float]) -> None:
        miss = [(t - a) * s * 1e3 for a, t, s in zip(aimed, touched, self.size_m)]
        self.samples = [*self.samples, (float(aimed[0]), float(aimed[1]), miss[0], miss[1])][-MAX_SAMPLES:]

    def _clean(self) -> tuple[np.ndarray, np.ndarray]:
        data = np.array(self.samples, np.float64).reshape(-1, 4)
        if len(data) >= 4:
            far = np.linalg.norm(data[:, 2:] - np.median(data[:, 2:], axis=0), axis=1)
            data = data[far < max(OUTLIER_MM, 3 * np.median(far))]
        return data[:, :2], data[:, 2:]

    def model(self) -> tuple[int, np.ndarray, float] | None:
        """(order, coefficients, leave-one-out RMS in mm) of the order that predicts best; None with too few taps."""
        uv, miss = self._clean()
        best: tuple[int, np.ndarray, float] | None = None
        for order, needed in NEEDED.items():
            if len(uv) < needed:
                continue
            held_out = []
            for k in range(len(uv)):
                rest = np.arange(len(uv)) != k
                held_out.append(miss[k] - (_terms(uv[k : k + 1], order) @ _fit(uv[rest], miss[rest], order))[0])
            error = float(np.sqrt(np.mean(np.sum(np.square(held_out), axis=1))))
            if best is None or error < 0.95 * best[2]:  # a higher order has to earn it
                best = (order, _fit(uv, miss, order), error)
        return best

    def aim(self, u: float, v: float) -> tuple[float, float]:
        """Where to send the pen so that the screen is touched at (u, v)."""
        found = self.model()
        if found is None:
            return u, v
        order, coeffs, _ = found
        size_mm = np.array(self.size_m) * 1e3
        want = np.array([u, v], np.float64)
        at = want.copy()
        for _ in range(4):
            shift = (_terms(at[None], order) @ coeffs)[0]
            norm = float(np.linalg.norm(shift))
            if norm > MAX_SHIFT_MM:
                shift *= MAX_SHIFT_MM / norm
            at = want - shift / size_mm
        return float(at[0]), float(at[1])

    def describe(self) -> str:
        found = self.model()
        if found is None:
            return f"点按修正：{len(self.samples)} 次，还不够"
        return f"点按修正：{len(self.samples)} 次，{('平移', '线性', '二次')[found[0]]}，留一误差 {found[2]:.2f} mm"
