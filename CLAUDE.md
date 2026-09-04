# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Trading card scanner using YOLOv11 for detection/segmentation and a
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
1. Image upload -> YOLO detects card bounding boxes/masks -> perspective correction
2. SigLIP2 (base model + merged LoRA adapter) encodes the crop to one embedding -> cosine similarity search against the gallery index (GPU-resident fp16 tensor)
3. Database lookup for product details/pricing -> return matches

**Key Modules:**
- `config.py` - Centralized settings with environment variable support (pydantic-settings)
- `logging_config.py` - Structured JSON logging with correlation ID support
- `api.py` - FastAPI app with endpoints, middleware, rate limiting, auth, the `/live-recognize` WebSocket
- `scanner.py` - YOLO detection + perspective correction + matcher orchestration. `Scanner.new_tracker()` builds an isolated ByteTrack/BOTSORT instance per live-recognize connection (deliberately not `model.track(persist=True)`, which stores tracker state on the shared model and would corrupt track IDs across concurrent clients); `scan(..., tracker=...)` drives it by hand and stamps a `track_id` onto each detection.
- `siglip_matcher.py` - Loads the gallery embeddings + LoRA adapter from the HF Hub (`jackttv/card-scanner-siglip-lora`, or a local `siglip_vectors/` override), GPU-resident cosine-similarity search. Mirrors the old `vlad_matcher.VLADCardSearch` interface (`search`/`search_verified`/`database`/`update_task`) so `scanner.py`/`api.py` needed no other changes.
- `database.py` - Async SQLite operations, CSV sync from tcgcsv.com with retry logic

**Endpoints:**
- `GET /health` - Liveness probe (always returns 200)
- `GET /ready` - Readiness probe (checks DB and scanner)
- `GET /metrics` - Prometheus metrics
- `POST /scan` - Full pipeline: detect and identify cards in image
- `POST /identify` - Fast path: identify pre-cropped card (skips YOLO)
- `WS /live-recognize` - Streams JPEG camera frames (max 20 FPS); each connection gets its own tracker, so responses carry a raw per-frame `similarity` plus a stable per-connection `track_id` -- no server-side smoothing/aggregation (frontends implement their own if they want it)
- Both `/scan` and `/identify` accept `verify=true` for API compatibility with the old RANSAC re-rank flag, but it's a no-op now (`inliers` always `0`) -- a global embedding has no keypoints to verify geometrically. See README's "Geometric verification is gone" section before re-adding real reranking.
- `GET /price`, `POST /prices` - Pricing lookups
- `POST /update` - Trigger database update (runs in background)

**External Dependencies:**
- Product data from `tcgcsv.com/tcgplayer/{categoryid}/{groupid}/ProductsAndPrices.csv`
- SigLIP2 base weights (`google/siglip2-so400m-patch14-384`) and the LoRA adapter + gallery embeddings (`jackttv/card-scanner-siglip-lora`) from the HF Hub at startup, both cached locally after first fetch

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

## Key Files

- `config.py` - All configurable settings with defaults
- `models/best(2).pt` - YOLOv11 trained model (40.8 MB)
- `database.db` - SQLite database with products, groups, categories
- `embeddings.pt` (HF Hub, or local `siglip_vectors/`) - `{ids, embeds}`: one L2-normalized fp16 embedding per catalog product
- LoRA adapter (HF Hub `jackttv/card-scanner-siglip-lora`, or local `siglip_vectors/lora_best/`) - Merged-at-load, PEFT format
- `.env.example` - Example environment configuration

## Production Features

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
