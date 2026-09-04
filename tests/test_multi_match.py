import cv2
import numpy as np
from scanner import Scanner
import os

def test_scanner_multi_match():
    scanner = Scanner()
    # Use one of the test images
    test_img_path = 'test_images/257279.png'
    if not os.path.exists(test_img_path):
        print(f"Test image {test_img_path} not found, skipping...")
        return

    img = cv2.imread(test_img_path)
    if img is None:
        print(f"Could not load image {test_img_path}, skipping...")
        return

    k = 3
    results = scanner.scan(img, k=k)
    
    print(f"Detected {len(results)} card segments.")
    for i, res in enumerate(results):
        print(f"Segment {i+1}:")
        assert 'matches' in res
        assert 'box' in res
        assert len(res['matches']) <= k
        print(f"  Got {len(res['matches'])} matches.")
        for match in res['matches']:
            assert 'card_id' in match
            assert 'similarity' in match
            print(f"    ID: {match['card_id']}, Sim: {match['similarity']:.4f}")

if __name__ == "__main__":
    test_scanner_multi_match()
