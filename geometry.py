"""
Pure-geometry sanity checks on a detected card quadrilateral, independent
of (and complementary to) SigLIP similarity:

  - estimate_aspect_ratio(): recovers the true width/height of the
    physical rectangle from its perspective-projected quad, via single-
    view metrology (the two edge-direction vanishing points must be
    orthogonal, since the rectangle's edges are perpendicular in 3D --
    that alone solves for the unknown focal length, and the corner
    positions then give the edge-length ratio). If the recovered ratio
    doesn't match the ratio of whichever card the quad's crop got
    SigLIP-matched to, the match is geometrically implausible regardless
    of how visually similar the crop looked.

  - quad_visible_fraction(): how much of the quad's area actually falls
    within the image frame. A card mostly off-screen shouldn't be
    processed at all -- its crop is mostly extrapolated/fabricated by the
    perspective warp, not real image content.

NOTE ambiguity: from 4 points alone, there is no way to tell which edge
pair is "width" vs "height" -- a rectangle photographed rotated 90 degrees
in-plane looks geometrically identical to the same rectangle's edges
swapped. estimate_aspect_ratio() returns one consistent convention (see
its docstring); callers must compare against BOTH the expected ratio and
its reciprocal.
"""
import numpy as np


def estimate_aspect_ratio(quad_cyclic, principal_point):
    """quad_cyclic: 4 (x,y) points in cyclic order [TL, TR, BR, BL] (the
    same convention Scanner.order_points() already produces -- geometric
    corners derived from pixel positions, not true semantic identity).
    principal_point: (cx, cy), normally the image center.

    Returns the estimated width/height ratio in the TL->TR vs TL->BL edge
    convention, or None if the configuration is degenerate (can't solve
    for focal length, or the result isn't numerically reliable).
    Ambiguous with its reciprocal -- see module docstring.
    """
    tl, tr, br, bl = (np.asarray(p, dtype=np.float64) for p in quad_cyclic)
    cx, cy = principal_point
    m1 = np.array([tl[0] - cx, tl[1] - cy, 1.0])  # TL
    m2 = np.array([tr[0] - cx, tr[1] - cy, 1.0])  # TR
    m3 = np.array([bl[0] - cx, bl[1] - cy, 1.0])  # BL
    m4 = np.array([br[0] - cx, br[1] - cy, 1.0])  # BR

    denom2 = np.dot(np.cross(m2, m4), m3)
    denom3 = np.dot(np.cross(m3, m4), m2)
    if abs(denom2) < 1e-9 or abs(denom3) < 1e-9:
        return None
    k2 = np.dot(np.cross(m1, m4), m3) / denom2
    k3 = np.dot(np.cross(m1, m4), m2) / denom3

    n2 = k2 * m2 - m1
    n3 = k3 * m3 - m1

    if abs(n2[2]) < 1e-6 or abs(n3[2]) < 1e-6:
        # Near-fronto-parallel: edges already ~parallel in image, can't
        # solve for focal length -- fall back to the raw pixel-space
        # edge-length ratio (no perspective distortion left to correct).
        w = np.linalg.norm(m2[:2] - m1[:2])
        h = np.linalg.norm(m3[:2] - m1[:2])
        return w / h if h > 1e-9 else None

    f2 = -(n2[0] * n3[0] + n2[1] * n3[1]) / (n2[2] * n3[2])
    if f2 <= 0:
        return None
    f = np.sqrt(f2)

    a_vec = np.array([n2[0], n2[1], n2[2] * f])
    b_vec = np.array([n3[0], n3[1], n3[2] * f])
    denom = float(np.dot(b_vec, b_vec))
    if denom < 1e-9 or not np.isfinite(denom):
        return None
    ratio_sq = float(np.dot(a_vec, a_vec)) / denom
    if not np.isfinite(ratio_sq) or ratio_sq <= 1e-9:
        return None
    return float(np.sqrt(ratio_sq))


def aspect_ratio_matches(estimated, expected, tolerance):
    """True if `estimated` is within `tolerance` (relative) of `expected`
    OR of 1/expected -- see module docstring on the width/height
    ambiguity. `estimated` may be None (estimation failed/degenerate),
    in which case this returns True: an inconclusive geometry check
    should never be the reason a genuine match gets rejected."""
    if estimated is None or expected is None or expected <= 0:
        return True
    direct = abs(estimated - expected) / expected
    reciprocal = abs(estimated - 1.0 / expected) / (1.0 / expected)
    return min(direct, reciprocal) <= tolerance


def quad_visible_fraction(quad, image_width, image_height):
    """Fraction of the quad's area that falls within [0,image_width] x
    [0,image_height]. Uses OpenCV's convex-polygon intersection since a
    detected card quad and the image frame are both convex."""
    import cv2
    quad = np.asarray(quad, dtype=np.float32).reshape(-1, 1, 2)
    frame = np.array([[0, 0], [image_width, 0], [image_width, image_height], [0, image_height]],
                      dtype=np.float32).reshape(-1, 1, 2)
    quad_area = cv2.contourArea(quad)
    if quad_area <= 1e-6:
        return 0.0
    intersection_area, _ = cv2.intersectConvexConvex(quad, frame)
    return float(intersection_area) / float(quad_area)
