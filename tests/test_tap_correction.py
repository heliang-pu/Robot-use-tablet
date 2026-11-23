"""Learning the remaining tapping error from the taps."""

import numpy as np

from tablet_tap.calibration.tap_correction import TapCorrection

SIZE = (0.2369, 0.1481)


def miss_of(u, v):
    """A smooth 2-4 mm miss, as of an arm whose geometry is slightly off (mm)."""
    x, y = 2 * u - 1, 2 * v - 1
    return np.array([3.0 + 1.2 * x - 0.6 * y + 0.8 * x * y, -1.0 + 0.5 * x + 1.0 * y - 0.7 * x * x])


def tap(aim, rng, noise=0.3):
    miss = miss_of(*aim) + rng.normal(0, noise, 2)
    return aim[0] + miss[0] / SIZE[0] / 1e3, aim[1] + miss[1] / SIZE[1] / 1e3


def errors(correction, rng, n=60):
    out = []
    for u, v in rng.uniform(0.06, 0.94, (n, 2)):
        t = tap(correction.aim(u, v), rng)
        out.append(np.hypot((t[0] - u) * SIZE[0], (t[1] - v) * SIZE[1]) * 1e3)
    return np.array(out)


def test_a_grid_of_taps_brings_the_error_down_to_the_noise():
    rng = np.random.default_rng(0)
    correction = TapCorrection(SIZE)
    before = errors(correction, rng)
    for v in np.linspace(0.08, 0.92, 4):
        for u in np.linspace(0.08, 0.92, 5):
            correction.add((u, v), tap((u, v), rng))
    after = errors(correction, rng)
    assert before.mean() > 2.5 and after.mean() < 0.6 and np.percentile(after, 95) < 1.0


def test_no_taps_no_change_and_few_taps_only_shift():
    correction = TapCorrection(SIZE)
    assert correction.aim(0.3, 0.7) == (0.3, 0.7)
    rng = np.random.default_rng(1)
    for u, v in ((0.2, 0.2), (0.8, 0.3), (0.5, 0.8)):
        correction.add((u, v), tap((u, v), rng))
    assert correction.model()[0] == 0


def test_noise_is_not_fitted():
    rng = np.random.default_rng(2)
    correction = TapCorrection(SIZE)
    for _ in range(14):
        u, v = rng.uniform(0.1, 0.9, 2)
        t = np.array([u, v]) + rng.normal(0, 0.5, 2) / np.array(SIZE) / 1e3
        correction.add((u, v), tuple(t))
    au, av = correction.aim(0.5, 0.5)
    assert np.hypot((au - 0.5) * SIZE[0], (av - 0.5) * SIZE[1]) * 1e3 < 0.6


def test_one_wild_tap_is_left_out():
    rng = np.random.default_rng(3)
    correction = TapCorrection(SIZE)
    for u in (0.1, 0.5, 0.9):
        for v in (0.1, 0.5, 0.9):
            correction.add((u, v), tap((u, v), rng))
    correction.add((0.5, 0.2), (0.6, 0.2))  # a finger, 24 mm away
    assert errors(correction, rng).mean() < 0.9
