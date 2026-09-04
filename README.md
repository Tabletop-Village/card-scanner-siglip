# Card Scanner (SigLIP)

>[!WARNING]
>This project was created using AI tools. The tools were guided by me, but much of the implementation was left to the tools.

A trading card scanning and identification system using YOLOv11 for
segmentation and a LoRA-fine-tuned SigLIP2 vision encoder for identification.

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

- **Segmentation**: YOLOv11 handles card detection and perspective correction (unchanged from the VLAD version).
- **Identification**: SigLIP2 (`google/siglip2-so400m-patch14-384`) vision tower + a merged LoRA
  adapter encodes the cropped card to a single embedding, matched by cosine
  similarity against a precomputed gallery index (`siglip_matcher.py`).
- **Database**: Asynchronous SQLite database stores product metadata and real-time market prices (unchanged).

## The vector index

`siglip_vectors/` holds:
- `embeddings.pt` — `{ids, embeds}`, one L2-normalized fp16 embedding per
  catalog product, built from the fine-tuning project's own cached gallery
  embeddings (no need to re-encode the catalog from scratch).
- `lora_best/` — the merged-at-load-time LoRA adapter (base SigLIP2 weights
  are not modified; the adapter loads on top of the stock checkpoint from
  HuggingFace at startup).

To add newly released cards: encode their catalog photos with the same
model/LoRA adapter and append to `embeddings.pt` — no retraining needed,
identical in spirit to how the VLAD version added cards to its vector
database.

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

### Scanning & Identification

#### Scan an Image
Upload an image containing one or more cards to detect and identify them.

```bash
curl -X POST "http://localhost:8000/scan" \
  -F "image=@your_card_photo.jpg"
```

#### Identify a Cropped Card
Identify a pre-cropped card image for maximum accuracy.

```bash
curl -X POST "http://localhost:8000/identify" \
  -F "image=@cropped_card.jpg"
```

#### Live recognition (WebSocket)
`/live-recognize` accepts a stream of JPEG camera frames (binary websocket
messages, max 20 FPS) and returns per-frame + rolling-stable-window match
scores. See `live_recognition.py` for the aggregation logic.

### Pricing & Data

#### Get Card Price
Get market pricing for a specific card.

```bash
curl "http://localhost:8000/price?product_id=123"
```

#### Batch Prices
Get prices for multiple cards in one request.

```bash
curl -X POST "http://localhost:8000/prices" \
  -H "Content-Type: application/json" \
  -d '{"product_ids": [123, 456, 789]}'
```

### Metadata

#### List Categories
Get all supported card categories.

```bash
curl "http://localhost:8000/categories"
```

#### List Groups
Get sets/groups for a category.

```bash
curl "http://localhost:8000/groups?category_id=3"
```

### System Health

- **/health**: Liveness probe (returns 200 OK if service is running).
- **/ready**: Readiness probe (checks database and scanner initialization).

## Configuration

The application is configured via environment variables. Copy `.env.example` to `.env` to customize.

### Key Settings

- **Authentication**: Set `CARD_SCANNER_API_KEYS` to a comma-separated list of keys to enable auth.
- **CORS**: Set `CARD_SCANNER_CORS_ORIGINS` to allow specific domains (default is `*`).
- **Rate Limits**: Adjust `CARD_SCANNER_RATE_LIMIT_*` variables to control request throttling.
- **Vector index path**: `CARD_SCANNER_SIGLIP_VECTORS_PATH` (default `siglip_vectors`).
- **Schedule**:
    - **Vector index reload**: checked once every 24 hours (default 4:00 AM) — reloads `embeddings.pt` from disk in place if it's been refreshed.
    - **Database Update**: Product metadata updates daily (default 3:00 AM).

## License

AGPLv3
