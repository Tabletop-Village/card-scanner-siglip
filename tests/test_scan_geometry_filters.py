"""Integration contracts for Scanner.scan()'s two geometry-only filters
(see geometry.py): off-screen detections are skipped before matching at
all, and matches whose expected aspect ratio doesn't match the detected
quad's own recovered shape are dropped after matching."""
import numpy as np
import pytest

from scanner import Scanner


class _FakeTensor:
    def __init__(self, arr):
        self._arr = np.asarray(arr, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _FakeKeypoint:
    def __init__(self, pts):
        self.xy = [_FakeTensor(pts)]


class _FakeBox:
    def __init__(self, xyxy_row):
        self.xyxy = [np.array(xyxy_row, dtype=np.float32)]


class _FakeResult:
    def __init__(self, box_xyxy, keypoints):
        self.boxes = [_FakeBox(box_xyxy)]
        self.keypoints = [_FakeKeypoint(keypoints)]


class _FakeMatcher:
    def __init__(self, expected_ratio):
        self._expected_ratio = expected_ratio
        self.search_calls = 0

    def search(self, cropped_image, top_k=1, margin_pct=None, min_similarity=None):
        self.search_calls += 1
        return [("1", 0.95)]

    def search_verified(self, *args, **kwargs):
        card_id, sim = self.search(*args, **kwargs)[0]
        return [(card_id, sim, 0)]

    def get_expected_aspect_ratio(self, product_id):
        return self._expected_ratio


def _make_scanner(quad, box_xyxy, expected_ratio, img_shape=(600, 800, 3)):
    scanner = object.__new__(Scanner)
    scanner.model = lambda image, device, verbose: [_FakeResult(box_xyxy, quad)]
    scanner.device = "cpu"
    scanner.matcher = _FakeMatcher(expected_ratio)
    image = np.zeros(img_shape, dtype=np.uint8)
    return scanner, image


def test_scan_skips_detection_more_than_max_offscreen_fraction_off_frame():
    # Card mostly hanging off the right edge of an 800-wide frame: only
    # ~30% of its width is on-screen, well past the default 40% cutoff.
    quad = [(700, 100), (1100, 100), (1100, 300), (700, 300)]
    scanner, image = _make_scanner(quad, box_xyxy=[700, 100, 800, 300], expected_ratio=0.716)

    cards = scanner.scan(image, k=1)

    assert cards == []
    assert scanner.matcher.search_calls == 0  # skipped before ever matching


def test_scan_rejects_match_whose_aspect_ratio_does_not_fit():
    # A fronto-parallel SQUARE (ratio ~1.0) matched against a product
    # whose real card image is the standard ~63:88 portrait ratio --
    # neither the ratio nor its reciprocal is anywhere close to 1.0.
    quad = [(300, 200), (500, 200), (500, 400), (300, 400)]
    scanner, image = _make_scanner(quad, box_xyxy=[300, 200, 500, 400], expected_ratio=63.0 / 88.0)

    cards = scanner.scan(image, k=1)

    assert cards == []
    assert scanner.matcher.search_calls == 1  # matching did happen; the match was then dropped


def test_scan_keeps_match_with_a_plausible_aspect_ratio():
    # A fronto-parallel rectangle at the real card ratio, matched against
    # a product with that same expected ratio -- should pass through.
    card_ratio = 63.0 / 88.0
    w, h = 200, 200 / card_ratio
    quad = [(300, 100), (300 + w, 100), (300 + w, 100 + h), (300, 100 + h)]
    scanner, image = _make_scanner(quad, box_xyxy=[300, 100, 300 + w, 100 + h], expected_ratio=card_ratio)

    cards = scanner.scan(image, k=1)

    assert len(cards) == 1
    assert cards[0]["matches"][0]["card_id"] == "1"


def test_scan_keeps_match_when_expected_ratio_is_unknown():
    # get_expected_aspect_ratio() returning None (e.g. an older gallery
    # without per-product ratios) must never cause a rejection.
    quad = [(300, 200), (500, 200), (500, 400), (300, 400)]  # a square -- shape is irrelevant here
    scanner, image = _make_scanner(quad, box_xyxy=[300, 200, 500, 400], expected_ratio=None)

    cards = scanner.scan(image, k=1)

    assert len(cards) == 1
