"""Contracts for geometry.py's pure-geometry sanity checks on a detected
card quad, independent of SigLIP similarity.

estimate_aspect_ratio() is validated against synthetic rectangles at
random 3D poses (known ground truth), not just a couple of handpicked
cases -- getting projective geometry subtly wrong is exactly the kind of
bug that wouldn't show up on a quick manual check but would then quietly
reject (or fail to reject) real scans in production."""
import numpy as np
import pytest

from geometry import estimate_aspect_ratio, aspect_ratio_matches, quad_visible_fraction


def _rotation_matrix(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _random_projected_quad(rng, true_w, true_h, f_true=800, img_size=(600, 800)):
    """Places a WxH rectangle at a random 3D rotation/distance in front of
    a pinhole camera and returns its projected 4 corners (cyclic TL, TR,
    BR, BL, matching Scanner.order_points()' convention) plus the
    principal point -- or None if the pose is invalid (behind camera)."""
    corners_local = np.array([
        [-true_w / 2, -true_h / 2, 0],
        [true_w / 2, -true_h / 2, 0],
        [true_w / 2, true_h / 2, 0],
        [-true_w / 2, true_h / 2, 0],
    ])
    R = _rotation_matrix(rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), rng.uniform(0, 2 * np.pi))
    distance = rng.uniform(3, 8) * max(true_w, true_h)
    translation = np.array([rng.uniform(-1, 1) * true_w, rng.uniform(-1, 1) * true_h, distance])
    world = corners_local @ R.T + translation
    if np.any(world[:, 2] <= 0.01):
        return None
    cx, cy = img_size[0] / 2, img_size[1] / 2
    quad = [(f_true * X / Z + cx, f_true * Y / Z + cy) for X, Y, Z in world]
    return quad, (cx, cy)


@pytest.mark.parametrize("true_w,true_h", [(63.0, 88.0), (100.0, 100.0), (88.0, 63.0)])
def test_estimate_aspect_ratio_recovers_ground_truth_across_random_poses(true_w, true_h):
    rng = np.random.default_rng(42)
    true_ratio = true_w / true_h
    n_checked = 0
    n_correct = 0
    for _ in range(500):
        result = _random_projected_quad(rng, true_w, true_h)
        if result is None:
            continue
        quad, principal_point = result
        estimated = estimate_aspect_ratio(quad, principal_point)
        n_checked += 1
        if aspect_ratio_matches(estimated, true_ratio, tolerance=0.05):
            n_correct += 1

    assert n_checked > 300  # sanity: most random poses should be valid
    assert n_correct / n_checked > 0.99


def test_aspect_ratio_matches_rejects_a_genuinely_different_ratio():
    rng = np.random.default_rng(1)
    card_ratio = 63.0 / 88.0
    n_checked = 0
    n_rejected = 0
    for _ in range(300):
        result = _random_projected_quad(rng, true_w=63.0, true_h=88.0)
        if result is None:
            continue
        quad, principal_point = result
        estimated = estimate_aspect_ratio(quad, principal_point)
        n_checked += 1
        # A square (ratio 1.0) is a very different shape from a card (~0.72)
        if not aspect_ratio_matches(estimated, expected=1.0, tolerance=0.05):
            n_rejected += 1

    assert n_rejected / n_checked > 0.99


def test_aspect_ratio_matches_treats_unknown_estimate_or_expected_as_inconclusive():
    # A None on either side must never cause a rejection -- an
    # inconclusive geometry check should never reject a genuine match.
    assert aspect_ratio_matches(None, 0.7, tolerance=0.1) is True
    assert aspect_ratio_matches(0.7, None, tolerance=0.1) is True
    assert aspect_ratio_matches(None, None, tolerance=0.1) is True


def test_quad_visible_fraction_fully_on_screen():
    quad = [(100, 100), (200, 100), (200, 200), (100, 200)]
    assert quad_visible_fraction(quad, 400, 400) == pytest.approx(1.0)


def test_quad_visible_fraction_half_off_screen():
    quad = [(300, 100), (500, 100), (500, 200), (300, 200)]
    assert quad_visible_fraction(quad, 400, 400) == pytest.approx(0.5, abs=0.01)


def test_quad_visible_fraction_fully_off_screen():
    quad = [(500, 500), (600, 500), (600, 600), (500, 600)]
    assert quad_visible_fraction(quad, 400, 400) == pytest.approx(0.0)


def test_quad_visible_fraction_partial():
    # 30% hanging off the left edge -> 70% visible
    quad = [(-30, 100), (70, 100), (70, 200), (-30, 200)]
    assert quad_visible_fraction(quad, 400, 400) == pytest.approx(0.7, abs=0.01)
