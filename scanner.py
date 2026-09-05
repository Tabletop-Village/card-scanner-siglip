"""
Handles the scanner for the card scanner.
Uses an ultralytics YOLO pose model to detect each card's 4 corners
directly (see MODEL_HISTORY.md / commit history for the earlier
segmentation-mask + approxPolyDP approach this replaced -- the pose
model's direct keypoint regression stays accurate at steep rotation
angles where mask-polygon approximation degraded badly).
Uses OpenCV to perspective-warp the card to a flat, upright crop from
those 4 corners.
Checks the card against the SigLIP2 LoRA embedding matcher (siglip_matcher.py)
to get the product ID. Returns the product ID.
"""

from pathlib import Path

from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml
from ultralytics.trackers import BOTSORT, BYTETracker
import cv2
import numpy as np
import geometry
import siglip_matcher
from config import settings

_TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}

class Scanner:
    def __init__(self, model_path=None, vectors_path=None, lora_path=None):
        """
        Initialize the Scanner with the YOLO pose model and SigLIP2 LoRA
        matcher. (Was CudaSift RootSIFT+VLAD via vlad_matcher.VLADCardSearch
        -- see siglip_matcher.py's docstring for why/how that was replaced.
        That module is left in place, unimported, in case of rollback.)

        model_path: explicit local .pt path, overriding the usual
        local-file-then-HF-Hub resolution (see _resolve_model_path()).
        """
        self.model = YOLO(model_path or self._resolve_model_path())
        self.device = settings.yolo_device
        self.model.to(self.device)
        self.matcher = siglip_matcher.SigLIPCardSearch(vectors_path=vectors_path, lora_path=lora_path)

    @staticmethod
    def _resolve_model_path():
        """A local models/ file, if present (offline dev / not-yet-uploaded
        checkpoint), takes priority over the HF Hub -- same pattern as
        siglip_matcher.py's adapter/embeddings resolution."""
        local_file = Path(settings.yolo_model_path)
        if local_file.exists():
            return str(local_file)
        return hf_hub_download(settings.yolo_hf_repo_id, settings.yolo_hf_filename)

    @staticmethod
    def new_tracker(tracker_yaml='bytetrack.yaml'):
        """
        Build a fresh, isolated tracker instance for one video/websocket
        stream. Deliberately NOT `model.track(..., persist=True)`: that
        stores tracker state as an attribute of the shared YOLO model
        object (`model.predictor.trackers`), which would mix up track IDs
        across concurrent connections if multiple clients share one
        Scanner. A standalone BYTETracker/BOTSORT instance, driven by
        hand via `.update()`, keeps each connection's tracks isolated
        while still sharing one loaded model for the actual detection.

        ultralytics>=8.4 dropped BYTETracker/BOTSORT's `frame_rate`
        constructor argument (max-lost-frames now comes straight from
        args.track_buffer, with no frame_rate scaling) -- passing it
        raises TypeError and crashes every /live-recognize connection at
        accept time, right after `websocket.accept()`, before any frame
        is even read. If a client only notices when its first `send()`/
        `recv()` fails against the now-dead connection, it looks like
        "the socket drops on the first frame" even though the server
        actually crashed during setup.
        """
        cfg = IterableSimpleNamespace(**YAML.load(check_yaml(tracker_yaml)))
        return _TRACKER_MAP[cfg.tracker_type](args=cfg)

    def start_scheduled_updates(self):
        """Start the scheduled update background task for the VLAD matcher."""
        self.matcher.start_scheduled_updates()

    def segment(self, image):
        """
        Run the YOLO pose model over the image.
        Returns a list of results containing boxes/keypoints (4 corners
        per detected card).
        """
        results = self.model(image, device=self.device, verbose=False)
        return results

    def order_points(self, pts):
        """
        Orders points in order: top-left, top-right, bottom-right, bottom-left.
        """
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def crop(self, image, box, keypoints=None):
        """
        Crop and dewarp the image based on the bounding box and the pose
        model's 4 corner keypoints, if provided.
        """
        if keypoints is not None:
            pts = np.asarray(keypoints, dtype=np.float32)
            if pts.shape == (4, 2):
                rect = self.order_points(pts)
                (tl, tr, br, bl) = rect

                # Define standard card dimensions (63mm x 88mm ratio)
                # We'll use a fixed width of 400px for consistency
                card_width = 400
                card_height = int(card_width * (88 / 63))

                # Check if the detected card is horizontal or vertical
                widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

                orig_width = max(int(widthA), int(widthB))
                orig_height = max(int(heightA), int(heightB))

                if orig_width > orig_height:
                    # Detected horizontal, rotate mapping to make it vertical
                    # Map: bl->tl, tl->tr, tr->br, br->bl
                    rect = np.array([rect[3], rect[0], rect[1], rect[2]], dtype="float32")

                dst = np.array([
                    [0, 0],
                    [card_width - 1, 0],
                    [card_width - 1, card_height - 1],
                    [0, card_height - 1]], dtype="float32")

                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(image, M, (card_width, card_height))
                return warped

        # Fallback to simple crop (e.g. keypoints unexpectedly missing)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = image[y1:y2, x1:x2]
        return cropped

    def match(self, cropped_image, top_k=1, verify=False, margin_pct=None, min_similarity=None):
        """
        Match the cropped card image using the SigLIP2 LoRA embedding matcher.
        verify=True has no geometric-verification equivalent for a global
        embedding (see siglip_matcher.py) -- inliers is always 0, kept for
        response-shape compatibility.
        top_k=None switches to margin mode -- see siglip_matcher.search().
        min_similarity overrides config.min_match_similarity -- matches
        below it are dropped entirely (see siglip_matcher.search()).
        Returns a list of (card_id, similarity) or (card_id, similarity, inliers) tuples.
        """
        if verify:
            return self.matcher.search_verified(cropped_image, top_k=top_k, margin_pct=margin_pct, min_similarity=min_similarity)
        results = self.matcher.search(cropped_image, top_k=top_k, margin_pct=margin_pct, min_similarity=min_similarity)
        return results

    @staticmethod
    def _match_dicts(matches):
        """Convert match tuples (with or without inlier counts) to response dicts."""
        return [
            {'card_id': m[0], 'similarity': float(m[1])}
            | ({'inliers': int(m[2])} if len(m) > 2 else {})
            for m in matches
        ]

    def identify_card(self, image, k=1, verify=False, margin_pct=None, min_similarity=None):
        """
        Identify a single pre-cropped card image.
        Resizes the image to the standard card size used in the pipeline before matching.
        k=None switches to margin mode -- see siglip_matcher.search().
        Returns a dictionary with matches and the bounding box (full image).
        """
        # Resize to standard dimensions used in crop()
        card_width = 400
        card_height = int(card_width * (88 / 63))

        resized = cv2.resize(image, (card_width, card_height))
        matches = self.match(resized, top_k=k, verify=verify, margin_pct=margin_pct, min_similarity=min_similarity)

        return {
            'matches': self._match_dicts(matches),
            'box': [0, 0, image.shape[1], image.shape[0]] # Full image box
        }

    def scan(self, image, k=1, verify=False, tracker=None, margin_pct=None, min_similarity=None):
        """
        Full scan pipeline: segment, crop (with dewarp), and match.
        Returns a list of dictionaries, each containing the bounding box and
        a list of matches. If `tracker` is given (see `new_tracker()`), each
        returned dict also gets a `track_id` that stays stable for the same
        physical card across calls on that tracker -- callers own the
        tracker's lifetime (one per video/websocket stream) and pass the
        same instance on every frame. Without a tracker (the stateless
        REST /scan and /identify path), no `track_id` key is present at all.
        k=None switches to margin mode -- see siglip_matcher.search().

        Two geometry-only sanity checks, independent of SigLIP similarity
        (see geometry.py for the full reasoning):
        - A detection more than settings.max_offscreen_fraction off-frame
          is skipped entirely, before even cropping/matching it -- its
          crop would be mostly extrapolated by the perspective warp, not
          real image content.
        - Each candidate match is checked against the detected quad's own
          recovered 3D aspect ratio (single-view metrology from its 4
          corners) vs. that specific matched product's real catalog image
          aspect ratio (most cards are the standard ~63:88 portrait ratio,
          but a few legitimate formats aren't -- e.g. Magic art cards,
          Pokemon BREAK secondary photos -- so this is checked per-match,
          not against one fixed constant). A candidate whose shape
          doesn't match is dropped, regardless of how visually similar
          the crop looked.
        """
        results = self.segment(image)
        scanned_cards = []
        img_h, img_w = image.shape[:2]
        principal_point = (img_w / 2, img_h / 2)

        for result in results:
            if not result.boxes:
                continue

            if tracker is not None:
                tracks = tracker.update(result.boxes.cpu().numpy(), image)
                if len(tracks) == 0:
                    continue
                track_ids = tracks[:, 4]
                det_indices = tracks[:, -1].astype(int)
            else:
                track_ids = [None] * len(result.boxes)
                det_indices = range(len(result.boxes))

            for track_id, i in zip(track_ids, det_indices):
                box = result.boxes[i]
                keypoints = result.keypoints[i].xy[0].cpu().numpy() if result.keypoints is not None else None

                if keypoints is not None:
                    visible_fraction = geometry.quad_visible_fraction(keypoints, img_w, img_h)
                    if visible_fraction < 1.0 - settings.max_offscreen_fraction:
                        continue

                cropped = self.crop(image, box, keypoints)
                matches = self.match(cropped, top_k=k, verify=verify, margin_pct=margin_pct, min_similarity=min_similarity)

                if keypoints is not None and matches:
                    ordered_quad = self.order_points(np.asarray(keypoints, dtype=np.float32))
                    estimated_ratio = geometry.estimate_aspect_ratio(ordered_quad, principal_point)
                    matches = [
                        m for m in matches
                        if geometry.aspect_ratio_matches(
                            estimated_ratio,
                            self.matcher.get_expected_aspect_ratio(m[0]),
                            settings.aspect_ratio_tolerance,
                        )
                    ]

                if matches:
                    card = {
                        'matches': self._match_dicts(matches),
                        'box': box.xyxy[0].tolist()
                    }
                    if track_id is not None:
                        card['track_id'] = int(track_id)
                    scanned_cards.append(card)

        return scanned_cards

if __name__ == "__main__":
    # Test script for scanner
    scanner = Scanner()
    test_images = ['test_images/257279.png', 'test_images/276982.png']
    
    for test_img_path in test_images:
        print(f"\nScanning {test_img_path}...")
        img = cv2.imread(test_img_path)
        if img is not None:
            cards = scanner.scan(img, k=3)
            print(f"Detected {len(cards)} card segments:")
            for i, card in enumerate(cards):
                print(f"  Segment {i+1} matches:")
                for match in card['matches']:
                    print(f"    ID: {match['card_id']}, Similarity: {match['similarity']:.4f}")
        else:
            print(f"Could not load image {test_img_path}")