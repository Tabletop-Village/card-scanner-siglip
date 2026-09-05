---
license: agpl-3.0
tags:
  - ultralytics
  - yolo
  - pose-estimation
  - object-detection
  - trading-cards
---

# Card Scanner YOLO Pose

A `yolo26n-pose` (Ultralytics) model fine-tuned to detect trading cards and
regress their 4 corner keypoints directly, for the
[card-scanner-siglip](https://github.com/Tabletop-Village/card-scanner-siglip)
pipeline's perspective-dewarp step.

## Why pose instead of segmentation

The original pipeline used a YOLO segmentation model: detect a card's
mask, then approximate its 4 corners from the mask polygon
(`cv2.approxPolyDP`). That approach degrades on steeply rotated cards --
the polygon approximation gets noisy exactly when you need it most.
Predicting the 4 corners directly as pose keypoints stays accurate
regardless of rotation angle, since it's a direct regression rather than
a derived approximation.

Live detections on a real, messy multi-card photo (green quad = the
model's 4 predicted corners, gray box = its bounding box) -- all 7 cards
found and tightly outlined despite heavy rotation and overlap. Notably,
one of them (the holographic "Krillin, Surprise Move") is a Dragon Ball
Super card -- a game not represented anywhere in the training corpus below
-- and it's still detected and cleanly outlined just as well as the rest,
since the model only ever learned "a rectangular card, at any rotation,"
not per-game visual conventions:

![Sample detections on a real photo](docs/sample_detection.jpg)

## Training data

Synthetic composites: real trading-card catalog photos (perspective-warped,
rotated, randomly placed, 1-8 per image) over COCO train2017/val2017
backgrounds. The card pool spans 9 TCGs (Pokemon, Pokemon Japan, One
Piece, Magic: The Gathering, YuGiOh, Digimon, Lorcana, Flesh & Blood, Star
Wars Unlimited), each capped to 10k images so no single game dominates --
the model only needs to learn "a rectangular card, at any rotation/scale,
photographed against a cluttered background," not per-game visual
conventions, so per-game diversity matters more than any one game's full
catalog coverage.

A sample of labeled training examples (ground-truth keypoints drawn in
green), across several of COCO's very different background scenes:

![Sample training examples](docs/training_samples.jpg)

Full training pipeline + results:
[Tabletop-Village/yolo-pose-training](https://github.com/Tabletop-Village/yolo-pose-training).

## Usage

```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model(image)
for result in results:
    for i, box in enumerate(result.boxes):
        corners = result.keypoints[i].xy[0]  # (4, 2) pixel coords
        # perspective-warp using these 4 points -- see
        # card-scanner-siglip's scanner.py Scanner.crop()/order_points()
```

`kpt_shape: [4, 3]` (x, y, visibility per corner). Corner order is a
stable-but-not-semantically-named convention from the training data
generator; re-derive true top-left/top-right/bottom-right/bottom-left
from pixel coordinates downstream (`order_points()`) rather than relying
on keypoint index order.
