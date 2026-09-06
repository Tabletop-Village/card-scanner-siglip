"""Regression tests for Scanner.order_points(): a card rotated close to
45 degrees must still produce a valid, non-degenerate cyclic ordering.

A prior sum/diff-based heuristic had a real degeneracy exactly at 45
degrees -- two corners tie for the sum (or diff) classification, so two
of the four "ordered" points collapse to the same source point, handing
cv2.getPerspectiveTransform a degenerate triangle instead of a
quadrilateral. warpPerspective then produces a garbage collapsed crop
(confirmed manually: a solid-color blob with no real card content) rather
than merely a lower-quality one -- this is the real bug behind cards at a
45-degree angle being "extremely difficult" to match. With realistic
keypoint jitter the practical failure zone was empirically ~42-47
degrees, peaking near 87% failures right at 45 degrees."""
import numpy as np
import pytest

from scanner import Scanner


def _rotated_card_quad(angle_deg, w=63.0, h=88.0, center=(300.0, 300.0)):
    """4 corners (TL, TR, BR, BL) of a WxH rectangle rotated angle_deg
    degrees about its center, in image (y-down) pixel coordinates."""
    theta = np.radians(angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    local = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    return (local @ R.T) + np.array(center)


def _quad_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _is_cyclic_rotation(rect, quad, atol=1e-2):
    for k in range(4):
        if np.allclose(rect, np.roll(quad, -k, axis=0), atol=atol):
            return True
    return False


@pytest.fixture
def scanner():
    return object.__new__(Scanner)  # order_points() needs no loaded models


@pytest.mark.parametrize("angle", [0, 15, 30, 44, 45, 45.5, 46, 60, 90, 135, 180, 270])
def test_order_points_is_a_valid_cyclic_rotation_at_every_angle(scanner, angle):
    quad = _rotated_card_quad(angle)
    rect = scanner.order_points(quad.astype(np.float32))

    assert _is_cyclic_rotation(rect, quad), f"order_points scrambled the quad at {angle} degrees"


@pytest.mark.parametrize("angle", [44.9, 45.0, 45.1, 135.0])
def test_order_points_never_collapses_the_quad_area(scanner, angle):
    """The old sum/diff heuristic collapsed the quad to ~50% of its true
    area exactly at 45/135 degrees (two points made identical). No area
    loss should occur at any angle."""
    quad = _rotated_card_quad(angle)
    true_area = _quad_area(quad)

    rect = scanner.order_points(quad.astype(np.float32))

    assert _quad_area(rect) == pytest.approx(true_area, rel=1e-3)


def test_order_points_robust_to_realistic_keypoint_jitter_near_45_degrees():
    """The practical failure mode: with a few pixels of realistic keypoint
    detection noise, angles near 45 degrees must not collapse the quad
    area on a meaningful fraction of trials."""
    scanner = object.__new__(Scanner)
    rng = np.random.default_rng(0)
    true_area = _quad_area(_rotated_card_quad(0, w=252, h=352))  # ~4x scale, realistic on-screen size
    bad = 0
    n_trials = 300
    for _ in range(n_trials):
        quad = _rotated_card_quad(45.0, w=252, h=352) + rng.normal(0, 3.0, size=(4, 2))
        rect = scanner.order_points(quad.astype(np.float32))
        if _quad_area(rect) / true_area < 0.9:
            bad += 1

    assert bad == 0
