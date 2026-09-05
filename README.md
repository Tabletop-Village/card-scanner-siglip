# Card Scanner (SigLIP)

>[!WARNING]
>This project was created using AI tools. The tools were guided by me, but much of the implementation was left to the tools.

A trading card scanning and identification system using a YOLO pose model
for corner detection and a LoRA-fine-tuned SigLIP2 vision encoder for
identification.

> [!IMPORTANT]
> **This is a break-out of [`card-scanner`](https://github.com/Tabletop-Village/card-scanner)'s CUDA branch**,
> replacing its CudaSift RootSIFT+VLAD matcher with global-embedding cosine
> similarity search. All VLAD/CudaSift code, the CMake build, and the
> RANSAC geometric-verification path were removed here; that history lives
> in the original repo if you need it.

## Why

The VLAD baseline scored 94.2% top-1 / 97.7% top-3 on a 514-scan real-camera
benchmark, at ~224ms per scan. A LoRA fine-tune of SigLIP2's vision tower
(gradients from self-supervised augmentation of the catalog gallery only —
the real scans are never trained on) scores **98.8% top-1 / 100.0% top-3** on
the same benchmark, at **~17ms per scan**. See the `siglip-scanner` project
this model came from for the full fine-tuning writeup and methodology.

## System Architecture

- **Detection**: A `yolo26n-pose` model (`scanner.py`) detects each card and regresses its 4 corners directly, then perspective-warps it to a flat, upright crop. This replaced an earlier YOLOv11 segmentation model that approximated corners from a mask polygon (`cv2.approxPolyDP`) -- that approach degraded badly on steeply rotated cards, where direct keypoint regression stays accurate. See "The pose model" below for hosting/training details.
- **Identification**: SigLIP2 (`google/siglip2-so400m-patch14-384`) vision tower + a merged LoRA
  adapter encodes the cropped card to a single embedding, matched by cosine
  similarity against a precomputed gallery index (`siglip_matcher.py`).
- **Database**: Asynchronous SQLite database stores product metadata and real-time market prices (unchanged).

## The pose model

The corner-detection model is hosted on the HuggingFace Hub:
[jackttv/card-scanner-yolo-pose](https://huggingface.co/jackttv/card-scanner-yolo-pose),
fetched via `huggingface_hub.hf_hub_download` and cached locally after the
first startup -- same pattern as the vector index below. A local
`models/pose_best.pt`, if present, overrides the Hub fetch (offline dev /
testing a not-yet-uploaded checkpoint).

Trained on synthetic composites (real catalog card photos, perspective-warped
and randomly placed over COCO backgrounds) spanning 9 TCGs, each capped to
10k images so no single game's art style dominates -- the model only needs
to learn "a rectangular card at any rotation," not per-game visual
conventions. See the `yolo-pose-training` project for the full training
writeup.

## The vector index

The LoRA adapter and gallery embeddings are hosted on the HuggingFace Hub:
[jackttv/card-scanner-siglip-lora](https://huggingface.co/jackttv/card-scanner-siglip-lora).

- `embeddings.pt` — `{ids, embeds}`, one L2-normalized fp16 embedding per
  catalog product, built from the fine-tuning project's own cached gallery
  embeddings (no need to re-encode the catalog from scratch). Fetched with
  `huggingface_hub.hf_hub_download` and cached at `~/.cache/huggingface`.
- the LoRA adapter — merged onto the base SigLIP2 weights at load time via
  `PeftModel.from_pretrained("jackttv/card-scanner-siglip-lora")`, which
  fetches/caches it the same way.

Only the first startup on a machine needs network access; both are cached
locally after that. A local `siglip_vectors/` directory (same
`embeddings.pt` + `lora_best/` layout), if present, takes priority over the
Hub — useful for offline dev or testing a not-yet-uploaded adapter.

To add newly released cards: encode their catalog photos with the same
model/LoRA adapter, append to `embeddings.pt`, and re-upload to the HF
repo — no retraining needed, identical in spirit to how the VLAD version
added cards to its vector database.

## Geometric verification is gone

`verify=true` on `/scan` and `/identify` is still accepted for API
compatibility but is a no-op (`inliers` is always `0`) — a global embedding
has no local keypoints for RANSAC to re-rank. If you want a real
quality-boost for that flag, a ColBERT-style per-patch-token late-interaction
reranker over the top-K global-embedding candidates is the natural fit (see
the `siglip-scanner` project's `rerank_colbert.py` for a prototype); it
wasn't carried over here since it needs further tuning before it's a clear
win.

## Installation

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# Static UI/liveness server (does not require the model or vector index)
.venv/bin/python frontend_server.py --host 127.0.0.1 --port 8080

# Full API server
.venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000
```

No CUDA build step is required (no CMake/CudaSift bridge) — just PyTorch +
transformers + peft, same as the rest of the stack.

## Web Frontend

Navigate to `http://localhost:8000` to use the built-in scanner UI. It provides a live webcam viewfinder — capture a photo to detect and identify cards. Results show the top 3 matches with thumbnails, pricing, and similarity scores. Click any result to see full card details.

## API Usage

Base URL: `http://localhost:8000`. No authentication is required by default
-- see [Configuration](#configuration) below to require an API key.

### Scanning & Identification

#### POST `/scan`
Upload an image containing one or more cards; YOLO detects and perspective-corrects each card, then SigLIP2 LoRA embedding matching identifies it.

| Parameter    | Type         | Required | Description                              |
|--------------|--------------|----------|------------------------------------------|
| `image`      | File (image) | Yes      | The image file to scan                   |
| `top_n`      | Integer      | No       | Number of top matches per card. If omitted, returns every match within `margin_pct` percentage points of the best match instead of a fixed count (see below) |
| `margin_pct` | Float        | No       | Margin used when `top_n` is omitted (default: 2.0). Ignored if `top_n` is given |
| `verify`     | Boolean      | No       | Accepted for compatibility with the old RANSAC re-rank flag; always a no-op now (`inliers` is always `0`) -- see [Geometric verification is gone](#geometric-verification-is-gone) |

```bash
curl -X POST "http://localhost:8000/scan" \
     -F "image=@your_card_photo.jpg" \
     -F "top_n=5"
```

**Margin mode (dynamic count):** Omit `top_n` entirely to get every gallery
match within `margin_pct` points of the top similarity, instead of a fixed
number of results per card. This is meant for reprints/near-duplicates
that shouldn't be arbitrarily narrowed down to a single "winner" -- e.g. a
card with two visually-near-identical reprints might return both at
98.2% and 97.6% rather than picking one:

```bash
curl -X POST "http://localhost:8000/scan" -F "image=@your_card_photo.jpg"
curl -X POST "http://localhost:8000/scan" -F "image=@your_card_photo.jpg" -F "margin_pct=5"
```

**Response:**
```json
[
  {
    "card_id": 123456,
    "similarity": 0.9523,
    "box": [100.5, 200.3, 450.2, 800.7],
    "details": {
      "product_id": 123456,
      "name": "Pikachu VMAX",
      "clean_name": "Pikachu VMAX",
      "image_url": "https://example.com/image.jpg",
      "category_id": 3,
      "group_id": 2831,
      "url": "https://www.tcgplayer.com/product/123456",
      "modified_on": "2024-01-15T10:30:00",
      "image_count": 2,
      "low_price": 15.99,
      "mid_price": 22.50,
      "high_price": 35.00,
      "market_price": 20.75,
      "direct_low_price": 18.00,
      "sub_type_name": "Normal",
      "ext_data": {"extRarity": "Ultra Rare", "extNumber": "044/185"},
      "last_updated": "2024-01-20T03:00:00"
    },
    "variants": [
      {"sub_type_name": "Normal", "low_price": 15.99, "mid_price": 22.50, "high_price": 35.00, "market_price": 20.75, "direct_low_price": 18.00},
      {"sub_type_name": "Reverse Holofoil", "low_price": 24.99, "mid_price": 32.10, "high_price": 55.00, "market_price": 29.40, "direct_low_price": 26.00}
    ]
  }
]
```

| Field        | Type   | Description                                       |
|--------------|--------|---------------------------------------------------|
| `card_id`    | Integer| The product ID of the matched card                |
| `similarity` | Float  | Match confidence score (0-1, higher is better)    |
| `box`        | Array  | Bounding box coordinates [x1, y1, x2, y2]         |
| `details`    | Object | Full product row -- columns match `/columns` below |
| `variants`   | Array  | Every priced finish of this card (Normal, Reverse Holofoil, 1st Edition, etc.) -- see below |

**Card finishes / variants:** TCGCSV assigns the *same* `product_id` and
catalog image to every finish of a card (Normal, Reverse Holofoil, 1st
Edition, ...) -- a match can't tell which one is physically in hand,
since they're visually identical in the catalog. `details`'s flat price
fields (`low_price`, `market_price`, etc.) are just one canonical finish
(preferring "Normal" when it exists); `variants` lists every finish's
price so a client can show all of them or let the user pick.

#### POST `/identify`
Identify a pre-cropped card image directly (skips YOLO detection) for faster processing when the card is already isolated.

| Parameter    | Type         | Required | Description                              |
|--------------|--------------|----------|------------------------------------------|
| `image`      | File (image) | Yes      | The pre-cropped card image               |
| `top_n`      | Integer      | No       | Number of top matches to return. If omitted, returns every match within `margin_pct` percentage points of the best match instead of a fixed count -- see `/scan`'s margin mode above |
| `margin_pct` | Float        | No       | Margin used when `top_n` is omitted (default: 2.0). Ignored if `top_n` is given |
| `verify`     | Boolean      | No       | Same no-op compatibility flag as `/scan`  |

```bash
curl -X POST "http://localhost:8000/identify" \
     -F "image=@cropped_card.jpg" \
     -F "top_n=3"
```

**Response:** Same format as `/scan`.

#### WS `/live-recognize`
Streams JPEG camera frames for continuous recognition. Each connection gets
its own isolated YOLO tracker (ByteTrack), so cards keep a stable `track_id`
across frames as long as they stay in view -- there's no server-side score
smoothing or aggregation. If you want a "hold steady for N frames before
committing" UX, implement it client-side by grouping results on `track_id`.

Same `top_n`/`margin_pct` as `/scan`, but set once as query params on the
connection URL (fixed for the connection's lifetime, alongside its
tracker) rather than per-request:

```
wss://host/live-recognize?top_n=5
wss://host/live-recognize?margin_pct=3
```

Send binary JPEG frames (max 20 FPS; faster sends get an `error` reply, not
a queued frame). The server replies with one JSON message per frame:

```json
{
  "type": "result",
  "results": [
    {
      "card_id": 123456,
      "track_id": 1,
      "similarity": 0.9523,
      "box": [100.5, 200.3, 450.2, 800.7],
      "details": { "...": "same shape as /scan's details" },
      "variants": [ { "...": "same shape as /scan's variants" } ]
    }
  ]
}
```

An empty `results` array means no cards were detected in that frame.

| `error`               | Cause                                    |
|-----------------------|-------------------------------------------|
| `frame_rate_exceeded`  | More than 20 frames/sec sent on this connection |
| `invalid_jpeg`         | Binary payload didn't decode as a JPEG    |

| Field        | Type    | Description                                                        |
|--------------|---------|----------------------------------------------------------------------|
| `card_id`    | Integer | The product ID of the matched card                                  |
| `track_id`   | Integer | Stable per-connection ID for this physical card while it stays in view (ByteTrack, globally unique across connections -- not reset per socket) |
| `similarity` | Float   | Raw match confidence for this frame (0-1, higher is better) -- no smoothing/averaging is applied |
| `box`        | Array   | Bounding box coordinates [x1, y1, x2, y2]                            |
| `details`    | Object  | Full product details, or `null` if not found                        |
| `variants`   | Array   | Every priced finish of this card -- see `/scan`'s "Card finishes / variants" above |

### Pricing

#### GET `/price`
Get pricing information for a specific card by product ID.

| Parameter    | Type    | Required | Description           |
|--------------|---------|----------|-----------------------|
| `product_id` | Integer | Yes      | The product ID        |

```bash
curl "http://localhost:8000/price?product_id=123456"
```

**Response:**
```json
{
  "low_price": 15.99,
  "mid_price": 22.50,
  "high_price": 35.00,
  "market_price": 20.75,
  "direct_low_price": 18.00
}
```

#### POST `/prices`
Get pricing information for multiple cards in a single request.

| Parameter     | Type           | Required | Description                    |
|---------------|----------------|----------|--------------------------------|
| `product_ids` | Array[Integer] | Yes      | List of product IDs to query   |

```bash
curl -X POST "http://localhost:8000/prices" \
     -H "Content-Type: application/json" \
     -d '{"product_ids": [123456, 789012]}'
```

**Response:**
```json
{
  "prices": {
    "123456": {"low_price": 15.99, "mid_price": 22.50, "high_price": 35.00, "market_price": 20.75, "direct_low_price": 18.00},
    "789012": {"low_price": 5.00, "mid_price": 7.50, "high_price": 12.00, "market_price": 6.25, "direct_low_price": 5.50}
  }
}
```

> [!NOTE]
> Products that don't exist in the database are omitted from the response rather than returning an error.

### Metadata

#### GET `/categories`
Get all available categories stored in the database.

```bash
curl "http://localhost:8000/categories"
```
```json
{"categories": [{"category_id": 3, "category_name": "Pokemon"}]}
```

#### GET `/groups`
Get all groups (card sets) for a specific category.

| Parameter     | Type    | Required | Description                        |
|---------------|---------|----------|-------------------------------------|
| `category_id` | Integer | No       | Category ID (default: 3 - Pokemon)  |

```bash
curl "http://localhost:8000/groups?category_id=3"
```
```json
{
  "groups": [
    {"group_id": 604, "category_id": 3, "group_name": "Base Set"},
    {"group_id": 635, "category_id": 3, "group_name": "Jungle"}
  ]
}
```

#### GET `/columns`
Get the list of column names available in the products table (i.e. the keys in `/scan`'s `details`).

```bash
curl "http://localhost:8000/columns"
```
```json
{
  "columns": [
    "product_id", "name", "clean_name", "image_url", "category_id",
    "group_id", "url", "modified_on", "image_count", "low_price",
    "mid_price", "high_price", "market_price", "direct_low_price",
    "sub_type_name", "ext_data", "last_updated"
  ]
}
```

#### GET `/ext-data`
Get the extended data column names available for a specific category (the keys inside `details.ext_data`).

| Parameter     | Type    | Required | Description                        |
|---------------|---------|----------|-------------------------------------|
| `category_id` | Integer | No       | Category ID (default: 3 - Pokemon)  |

```bash
curl "http://localhost:8000/ext-data?category_id=3"
```
```json
{"ext_data_columns": ["extRarity", "extNumber", "extColor", "extDescription", "extFlavorText"]}
```

### Admin

#### POST `/update`
Trigger a manual database update. Downloads the latest product data from TCGPlayer/TCGCSV and refreshes `product_variants`/`products` in the background.

```bash
curl -X POST "http://localhost:8000/update"
```
```json
{"status": "Update started"}
```

### System Health

- **`GET /health`**: Liveness probe (always returns 200 if the service is running).
- **`GET /ready`**: Readiness probe (checks database and scanner initialization; 503 if not ready).
- **`GET /metrics`**: Prometheus metrics.

## Configuration

The application is configured via environment variables. Copy `.env.example` to `.env` to customize.

### Key Settings

- **Authentication**: Set `CARD_SCANNER_API_KEYS` to a comma-separated list of keys to enable auth.
- **CORS**: Set `CARD_SCANNER_CORS_ORIGINS` to allow specific domains (default is `*`).
- **Rate Limits**: Adjust `CARD_SCANNER_RATE_LIMIT_*` variables to control request throttling.
- **Vector index / adapter**: `CARD_SCANNER_SIGLIP_HF_REPO_ID` (default `jackttv/card-scanner-siglip-lora`); `CARD_SCANNER_SIGLIP_VECTORS_PATH` (default `siglip_vectors`) overrides with a local directory if present.
- **Pose detector**: `CARD_SCANNER_YOLO_HF_REPO_ID` (default `jackttv/card-scanner-yolo-pose`) / `CARD_SCANNER_YOLO_HF_FILENAME` (default `best.pt`); `CARD_SCANNER_YOLO_MODEL_PATH` (default `models/pose_best.pt`) overrides with a local file if present.
- **Margin-match mode**: `CARD_SCANNER_MATCH_MARGIN_PCT` (default `2.0`) and `CARD_SCANNER_MATCH_MARGIN_POOL_SIZE` (default `30`) -- see `/scan`'s margin mode above.
- **Schedule**:
    - **Vector index reload**: checked once every 24 hours (default 4:00 AM) — reloads `embeddings.pt` from disk in place if it's been refreshed.
    - **Database Update**: Product metadata updates daily (default 3:00 AM).

## License

AGPLv3
