# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Trading card scanner using a YOLO pose model for corner detection and a
LoRA-fine-tuned SigLIP2 vision encoder for identification (cosine-similarity
search over a precomputed gallery embedding index). FastAPI REST + WebSocket
API with async SQLite database for product metadata and pricing from
TCGPlayer.

**Broken out from [`card-scanner`](https://github.com/Tabletop-Village/card-scanner)'s
CUDA branch**, which used CudaSift RootSIFT+VLAD instead. That matcher (and
its CMake build, RANSAC re-rank path, and metalSIFT/Apple-Silicon variant)
was removed here in favor of the SigLIP2 embedding matcher -- see the
original repo if you need the VLAD history/code.

## Commands

```bash
# Install
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# Run server
uvicorn api:app --host 0.0.0.0 --port 8000

# Run tests
pytest tests/
```

## Architecture

**Data Flow:**
1. Image upload -> YOLO pose model detects each card + its 4 corner keypoints -> perspective-warp to a flat, upright crop
2. SigLIP2 (base model + merged LoRA adapter) encodes the crop to one embedding -> cosine similarity search against the gallery index (GPU-resident fp16 tensor)
3. Database lookup for product details/pricing -> return matches

**Key Modules:**
- `config.py` - Centralized settings with environment variable support (pydantic-settings)
- `logging_config.py` - Structured JSON logging with correlation ID support
- `api.py` - FastAPI app with endpoints, middleware, rate limiting, auth, the `/live-recognize` WebSocket
- `scanner.py` - YOLO pose detection + perspective correction + matcher orchestration. Loads the pose model from the HF Hub (`jackttv/card-scanner-yolo-pose`, or a local `models/pose_best.pt` override) via `_resolve_model_path()`. `Scanner.crop()` perspective-warps using the 4 keypoints directly (`order_points()` re-derives true TL/TR/BR/BL from pixel coordinates, since the model's raw keypoint index order isn't semantically meaningful) -- this replaced an earlier YOLOv11 segmentation model that approximated corners from a mask polygon (`cv2.approxPolyDP`), which degraded on steeply rotated cards. `Scanner.new_tracker()` builds an isolated ByteTrack/BOTSORT instance per live-recognize connection (deliberately not `model.track(persist=True)`, which stores tracker state on the shared model and would corrupt track IDs across concurrent clients); `scan(..., tracker=...)` drives it by hand and stamps a `track_id` onto each detection. `scan()` also runs two geometry-only sanity checks per detected quad (see `geometry.py`) independent of SigLIP similarity: a detection more than `config.max_offscreen_fraction` off-frame is skipped before it's even matched, and each candidate match is checked against the quad's own recovered 3D aspect ratio (single-view metrology) vs. that specific product's real catalog image ratio (`config.aspect_ratio_tolerance`) -- most cards are ~63:88 portrait, but a few real formats aren't (e.g. Pokemon BREAK secondary photos), so this is a per-match lookup, not one fixed constant.
- `geometry.py` - Pure-geometry helpers used by `scanner.py`'s two checks above: `estimate_aspect_ratio()` recovers a rectangle's true width/height from its perspective-projected quad (the two vanishing points of its edge-direction pairs must be orthogonal, since the rectangle's edges are perpendicular in 3D -- that alone solves for the unknown camera focal length); `quad_visible_fraction()` is the on-frame area fraction via `cv2.intersectConvexConvex`. Validated against 2000+ synthetic rectangles at random 3D poses (>99% within 5% of ground truth) before being wired into `scanner.py` -- see `tests/test_geometry.py`. Note the inherent width/height ambiguity: 4 points alone can't distinguish a rectangle from the same rectangle rotated 90 degrees, so callers must check the estimated ratio against both the expected ratio and its reciprocal (`aspect_ratio_matches()` does this).
- `siglip_matcher.py` - Loads the gallery embeddings + LoRA adapter from the HF Hub (`jackttv/card-scanner-siglip-lora`, or a local `siglip_vectors/` override), GPU-resident cosine-similarity search. Mirrors the old `vlad_matcher.VLADCardSearch` interface (`search`/`search_verified`/`database`/`update_task`) so `scanner.py`/`api.py` needed no other changes.
- `database.py` - Async SQLite operations, CSV sync from tcgcsv.com with retry logic. TCGCSV lists one row per (productId, subTypeName) -- the same card commonly appears as both "Normal" and "Reverse Holofoil" sharing one productId/image but different prices. `load_csv_to_db` preserves every subTypeName's price in `product_variants` (keyed by product_id + sub_type_name) rather than letting later rows silently clobber earlier ones in `products`, which picks one canonical row per productId via `_pick_canonical_variant` (prefers "Normal"/plain finishes). `query_variants_by_id()` returns all of a product's finishes.

**Endpoints:**
- `GET /health` - Liveness probe (always returns 200)
- `GET /ready` - Readiness probe (checks DB and scanner)
- `GET /metrics` - Prometheus metrics
- `POST /scan` - Full pipeline: detect and identify cards in image. `top_n` fixes the match count per card; omitting it switches to margin mode (`config.match_margin_pct`, default 2.0 points) -- every gallery match within that many percentage points of the best similarity is returned instead, for reprints/near-duplicates that shouldn't be arbitrarily narrowed to one. Regardless of mode, `min_similarity` (`config.min_match_similarity`, default 0.3) drops any match below that cosine similarity entirely -- fixed a real incident where a few gallery images were a generic TCGplayer "Image Coming Soon" placeholder (served with a 200 OK instead of a 403) that acted as a false attractor for completely unrelated photos at 50-65% similarity. See `siglip_matcher.SigLIPCardSearch.search()` (`top_k=None`, `min_similarity`). Each matched card's response also includes `variants` (every priced finish -- see `database.py` above), since matching can't distinguish finishes that share one catalog image.
- `POST /identify` - Fast path: identify pre-cropped card (skips YOLO). Same `top_n`/margin-mode/`variants` behavior as `/scan`.
- `WS /live-recognize` - Streams JPEG camera frames (max 20 FPS); each connection gets its own tracker, so responses carry a raw per-frame `similarity` plus a stable per-connection `track_id` -- no server-side smoothing/aggregation (frontends implement their own if they want it). Takes the same `top_n`/`margin_pct`/`min_similarity` as `/scan`, but as query params on the connection URL (fixed for the connection's lifetime, alongside its tracker) rather than per-request.
- Both `/scan` and `/identify` accept `verify=true` for API compatibility with the old RANSAC re-rank flag, but it's a no-op now (`inliers` always `0`) -- a global embedding has no keypoints to verify geometrically. See README's "Geometric verification is gone" section before re-adding real reranking.
- `GET /price`, `POST /prices` - Pricing lookups
- `POST /update` - Trigger database update (runs in background)

**External Dependencies:**
- Product data from `tcgcsv.com/tcgplayer/{categoryid}/{groupid}/ProductsAndPrices.csv`
- SigLIP2 base weights (`google/siglip2-so400m-patch14-384`) and the LoRA adapter + gallery embeddings (`jackttv/card-scanner-siglip-lora`) from the HF Hub at startup, both cached locally after first fetch
- YOLO pose detector (`jackttv/card-scanner-yolo-pose`) from the HF Hub at startup, cached locally after first fetch

**Scheduled Tasks:**
- Database update: 3:00 AM daily (configurable)
- Vector index reload: 4:00 AM daily (configurable) -- reloads `embeddings.pt` from disk in place; there's no external repo to sync from

## Configuration

All settings are configurable via environment variables with `CARD_SCANNER_` prefix.
See `.env.example` for all available options.

**Key Settings:**
- `CARD_SCANNER_API_KEYS` - Comma-separated API keys (empty = auth disabled)
- `CARD_SCANNER_CORS_ORIGINS` - Comma-separated allowed origins
- `CARD_SCANNER_MAX_FILE_SIZE` - Max upload size in bytes (default: 10MB)
- `CARD_SCANNER_LOG_JSON` - Enable JSON logging (default: true)
- `CARD_SCANNER_SIGLIP_HF_REPO_ID` - HF Hub repo for the LoRA adapter + gallery embeddings (default: `jackttv/card-scanner-siglip-lora`)
- `CARD_SCANNER_SIGLIP_VECTORS_PATH` - Local directory that overrides the HF Hub if present (default: `siglip_vectors`)
- `CARD_SCANNER_YOLO_HF_REPO_ID` / `CARD_SCANNER_YOLO_HF_FILENAME` - HF Hub repo/file for the pose detector (default: `jackttv/card-scanner-yolo-pose`, `best.pt`)
- `CARD_SCANNER_YOLO_MODEL_PATH` - Local file that overrides the HF Hub if present (default: `models/pose_best.pt`)
- `CARD_SCANNER_YOLO_CONFIDENCE_THRESHOLD` - Minimum pose-detection confidence (default: 0.6, raised from ultralytics' permissive 0.25 default -- genuine cards scored 0.94-0.98 in testing, so this cuts down false detections on non-card objects without losing real ones)
- `CARD_SCANNER_MAX_OFFSCREEN_FRACTION` - Skip a detection more than this fraction off-frame, before matching it at all (default: 0.2)
- `CARD_SCANNER_ASPECT_RATIO_TOLERANCE` - Relative tolerance when checking a match's expected image aspect ratio against the detected quad's recovered 3D shape (default: 0.15). Not exposed as a per-request API param like `margin_pct`/`min_similarity` -- these are pipeline correctness gates, not per-request tuning knobs

## Key Files

- `config.py` - All configurable settings with defaults
- `models/pose_best.pt` (HF Hub `jackttv/card-scanner-yolo-pose`, or local override) - YOLO26n-pose corner-keypoint model
- `database.db` - SQLite database with products, groups, categories
- `embeddings.pt` (HF Hub, or local `siglip_vectors/`) - `{ids, embeds, aspect_ratios}`: one L2-normalized fp16 embedding + real catalog image width/height ratio per product (the latter used by `scanner.py`'s aspect-ratio geometry check; optional field, gracefully absent on older files)
- `geometry.py` - Pure-geometry card-quad sanity checks (aspect ratio recovery, off-frame fraction) -- see Architecture above
- LoRA adapter (HF Hub `jackttv/card-scanner-siglip-lora`, or local `siglip_vectors/lora_best/`) - Merged-at-load, PEFT format
- `.env.example` - Example environment configuration

## Production Features

- **Persistence**: `deploy/card-scanner-siglip.service` -- systemd user unit (`Restart=on-failure`), see `deploy/README.md` for install steps (needs `loginctl enable-linger` for true reboot survival, not just logout/login)
- **Authentication**: Optional API key auth via `X-API-Key` header
- **Rate Limiting**: Configurable per-endpoint limits (slowapi)
- **Metrics**: Prometheus metrics at `/metrics`
- **Health Checks**: `/health` (liveness) and `/ready` (readiness)
- **Correlation IDs**: Request tracing via `X-Correlation-ID` header
- **Structured Logging**: JSON format with correlation ID support
- **Security Headers**: X-Content-Type-Options, X-Frame-Options, HSTS
- **Retry Logic**: Automatic retries for HTTP operations (tenacity)
- **Graceful Shutdown**: Proper signal handling for SIGTERM/SIGINT
- **Error Sanitization**: Internal errors logged, sanitized responses to clients

## Notes

- All I/O is async (`aiosqlite`, `httpx`, `asyncio`); model inference runs in a thread pool via `asyncio.to_thread`
- Default category is 3 (Pokemon), but the matcher and database are catalog-agnostic -- a card from any TCGCSV category works as long as its embedding is in the index
- Authentication is disabled by default for backwards compatibility
- CORS allows all origins by default (configure `CARD_SCANNER_CORS_ORIGINS` for production)
