"""Integration contracts for Scanner.scan()'s three geometry-only filters
(see geometry.py): a collapsed keypoint quad is discarded in favour of the
plain box crop before anything else, off-screen detections are skipped
before matching at all, and matches whose expected aspect ratio doesn't
match the detected quad's own recovered shape are dropped after
matching."""
import numpy as np
import pytest

from config import settings
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
    scanner.model = lambda image, device, verbose, conf, imgsz: [_FakeResult(box_xyxy, quad)]
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


def test_segment_passes_the_configured_confidence_threshold():
    """Regression test: segment() must actually apply
    config.yolo_confidence_threshold, not rely on ultralytics' own
    permissive 0.25 default -- see config.py for why (real cards scored
    0.94-0.98 in testing; false detections on non-card objects like a
    laptop screen or water bottle need a higher bar to get filtered)."""
    quad = [(300, 200), (500, 200), (500, 400), (300, 400)]
    scanner, image = _make_scanner(quad, box_xyxy=[300, 200, 500, 400], expected_ratio=None)

    captured = {}

    def fake_model(image, device, verbose, conf, imgsz):
        captured["conf"] = conf
        return [_FakeResult([300, 200, 500, 400], quad)]

    scanner.model = fake_model
    scanner.scan(image, k=1)

    assert captured["conf"] == settings.yolo_confidence_threshold




@pytest.mark.parametrize("largest_index", [0, 1])
def test_largest_only_matches_biggest_detection_before_retrieval(largest_index):
    scanner = object.__new__(Scanner)
    boxes = [_FakeBox([0, 0, 40, 60]), _FakeBox([100, 100, 250, 300])]
    if largest_index == 0:
        boxes.reverse()
    result = type("Result", (), {"boxes": boxes, "keypoints": None})()
    scanner.segment = lambda image: [result]
    cropped = []
    scanner.crop = lambda image, box, keypoints: cropped.append(box) or image
    scanner.match = lambda *args, **kwargs: [("1", .9)]
    cards = scanner.scan(np.zeros((400, 400, 3), dtype=np.uint8), largest_only=True)
    assert cropped == [boxes[largest_index]]
    assert len(cards) == 1
    assert cards[0]["box"] == [100, 100, 250, 300]


def test_largest_card_without_matches_does_not_substitute_smaller_card():
    scanner = object.__new__(Scanner)
    boxes = [_FakeBox([0, 0, 40, 60]), _FakeBox([100, 100, 250, 300])]
    result = type("Result", (), {"boxes": boxes, "keypoints": None})()
    scanner.segment = lambda image: [result]
    scanner.crop = lambda image, box, keypoints: box
    scanner.match = lambda box, **kwargs: [] if box is boxes[1] else [("1", .9)]
    assert scanner.scan(np.zeros((400, 400, 3), dtype=np.uint8), largest_only=True) == []


def test_largest_only_handles_no_detections():
    scanner = object.__new__(Scanner)
    scanner.segment = lambda image: []
    assert scanner.scan(np.zeros((400, 400, 3), dtype=np.uint8), largest_only=True) == []




def test_segment_passes_the_configured_inference_size():
    """Regression test: segment() must apply config.yolo_imgsz, not
    ultralytics' 640 default -- this model's keypoint head collapses on
    cards that are large in the network input, and 640 is right on that
    edge for a card filling a fixed-camera frame (see config.py)."""
    quad = [(300, 200), (500, 200), (500, 400), (300, 400)]
    scanner, image = _make_scanner(quad, box_xyxy=[300, 200, 500, 400], expected_ratio=None)

    captured = {}

    def fake_model(image, device, verbose, conf, imgsz):
        captured["imgsz"] = imgsz
        return [_FakeResult([300, 200, 500, 400], quad)]

    scanner.model = fake_model
    scanner.scan(image, k=1)

    assert captured["imgsz"] == settings.yolo_imgsz


def test_scan_falls_back_to_the_box_crop_when_keypoints_collapse():
    """The failure this filter exists for: box and confidence correct,
    keypoints bunched near the card's centre. The detection must still be
    matched -- from the box crop -- not warped from the tiny quad and not
    dropped."""
    box = [300, 100, 500, 379]  # a real ~63:88 card box
    quad = [(395, 230), (405, 228), (407, 245), (393, 247)]  # ~0.3% of the box

    scanner, image = _make_scanner(quad, box_xyxy=box, expected_ratio=63.0 / 88.0)
    captured = {}
    original_crop = scanner.crop

    def spy_crop(img, b, keypoints=None):
        captured["keypoints"] = keypoints
        return original_crop(img, b, keypoints)

    scanner.crop = spy_crop
    cards = scanner.scan(image, k=1)

    assert captured["keypoints"] is None  # collapsed quad discarded before cropping
    assert len(cards) == 1                # and the detection still got matched
    assert cards[0]["matches"][0]["card_id"] == "1"


def test_scan_does_not_run_the_aspect_ratio_check_on_a_collapsed_quad():
    """A collapsed quad's recovered aspect ratio is meaningless. It must
    not be used to reject the match -- that would turn a wrong answer
    into no answer instead of into the right one."""
    box = [300, 100, 500, 379]
    # A tiny quad whose own shape is nothing like a card's (a wide
    # slither): if this ratio were checked against the standard 63:88
    # expectation, the match would be dropped.
    quad = [(380, 235), (420, 234), (420, 241), (380, 242)]

    scanner, image = _make_scanner(quad, box_xyxy=box, expected_ratio=63.0 / 88.0)
    cards = scanner.scan(image, k=1)

    assert len(cards) == 1


def test_scan_keeps_keypoints_for_a_steeply_rotated_card():
    """The guard must not fire on a genuine pose. A card rotated 45
    degrees in-plane fills the least of its own bounding box that a real
    card ever can (2wh/(w+h)^2 = 0.486 for 63:88) -- the worst honest
    case, and it has to survive."""
    w, h = 63.0 * 3, 88.0 * 3
    centre = np.array([400.0, 300.0])
    corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    theta = np.pi / 4
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    quad = [tuple(p) for p in (corners @ rotation.T + centre)]
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]

    scanner, image = _make_scanner(quad, box_xyxy=[min(xs), min(ys), max(xs), max(ys)],
                                    expected_ratio=63.0 / 88.0)
    captured = {}
    original_crop = scanner.crop

    def spy_crop(img, b, keypoints=None):
        captured["keypoints"] = keypoints
        return original_crop(img, b, keypoints)

    scanner.crop = spy_crop
    cards = scanner.scan(image, k=1)

    assert captured["keypoints"] is not None  # a real 45-degree pose is not "collapsed"
    assert len(cards) == 1
