"""
Handles the scanner for the card scanner.
Uses ultralytics YOLO to segment the card from the image.
Uses OpenCV to crop the image using the segmentation mask.
Checks the card against the SigLIP2 LoRA embedding matcher (siglip_matcher.py)
to get the product ID. Returns the product ID.
"""

from ultralytics import YOLO
import cv2
import numpy as np
import siglip_matcher
from config import settings

class Scanner:
    def __init__(self, model_path='models/best(2).pt', vectors_path=None, lora_path=None):
        """
        Initialize the Scanner with YOLO model and SigLIP2 LoRA matcher.
        (Was CudaSift RootSIFT+VLAD via vlad_matcher.VLADCardSearch -- see
        siglip_matcher.py's docstring for why/how this was replaced. That
        module is left in place, unimported, in case of rollback.)
        """
        self.model = YOLO(model_path)
        self.device = settings.yolo_device
        self.model.to(self.device)
        self.matcher = siglip_matcher.SigLIPCardSearch(vectors_path=vectors_path, lora_path=lora_path)

    def start_scheduled_updates(self):
        """Start the scheduled update background task for the VLAD matcher."""
        self.matcher.start_scheduled_updates()

    def segment(self, image):
        """
        Use YOLO to segment cards from the image.
        Returns a list of results containing boxes/masks.
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

    def crop(self, image, box, mask=None):
        """
        Crop and dewarp the image based on the bounding box and optional mask.
        If mask is provided, performs perspective transformation.
        """
        if mask is not None:
            # Extract polygon from mask and ensure contiguous float32 array for OpenCV
            polygon = np.ascontiguousarray(mask.xy[0], dtype=np.float32)
            # Approximate the polygon to 4 points
            peri = cv2.arcLength(polygon, True)
            approx = cv2.approxPolyDP(polygon, 0.02 * peri, True)
            
            if len(approx) == 4:
                pts = approx.reshape(4, 2)
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

        # Fallback to simple crop
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = image[y1:y2, x1:x2]
        return cropped

    def match(self, cropped_image, top_k=1, verify=False):
        """
        Match the cropped card image using the SigLIP2 LoRA embedding matcher.
        verify=True has no geometric-verification equivalent for a global
        embedding (see siglip_matcher.py) -- inliers is always 0, kept for
        response-shape compatibility.
        Returns a list of (card_id, similarity) or (card_id, similarity, inliers) tuples.
        """
        if verify:
            return self.matcher.search_verified(cropped_image, top_k=top_k)
        results = self.matcher.search(cropped_image, top_k=top_k)
        return results

    @staticmethod
    def _match_dicts(matches):
        """Convert match tuples (with or without inlier counts) to response dicts."""
        return [
            {'card_id': m[0], 'similarity': float(m[1])}
            | ({'inliers': int(m[2])} if len(m) > 2 else {})
            for m in matches
        ]

    def identify_card(self, image, k=1, verify=False):
        """
        Identify a single pre-cropped card image.
        Resizes the image to the standard card size used in the pipeline before matching.
        Returns a dictionary with matches and the bounding box (full image).
        """
        # Resize to standard dimensions used in crop()
        card_width = 400
        card_height = int(card_width * (88 / 63))
        
        resized = cv2.resize(image, (card_width, card_height))
        matches = self.match(resized, top_k=k, verify=verify)
        
        return {
            'matches': self._match_dicts(matches),
            'box': [0, 0, image.shape[1], image.shape[0]] # Full image box
        }

    def scan(self, image, k=1, verify=False):
        """
        Full scan pipeline: segment, crop (with dewarp), and match.
        Returns a list of dictionaries, each containing the bounding box and a list of matches.
        """
        results = self.segment(image)
        scanned_cards = []

        for result in results:
            if result.boxes:
                for i in range(len(result.boxes)):
                    box = result.boxes[i]
                    mask = result.masks[i] if result.masks is not None else None
                    cropped = self.crop(image, box, mask)
                    matches = self.match(cropped, top_k=k, verify=verify)
                    
                    if matches:
                        scanned_cards.append({
                            'matches': self._match_dicts(matches),
                            'box': box.xyxy[0].tolist()
                        })
        
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