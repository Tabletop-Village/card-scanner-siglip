"""
Handles the API for the card scanner.
Uses FastAPI to create a REST API.
/scan - scans a card from file upload
/identify - identifies a pre-cropped card
/price - gets the price of a card
/prices - gets prices for multiple cards
/health - liveness probe
/ready - readiness probe
/update - updates the database
"""

import asyncio
import csv
import json
import os
import signal
import time
import uuid
import pickle
from typing import List, Optional

import fastapi
from fastapi import UploadFile, Request, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn
import cv2
import numpy as np
from pydantic import BaseModel
from contextlib import asynccontextmanager
from pathlib import Path
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from prometheus_fastapi_instrumentator import Instrumentator

import database
from scanner import Scanner
from config import settings
from logging_config import get_logger, set_correlation_id

logger = get_logger(__name__)


# =============================================================================
# Error Handling
# =============================================================================

class APIError(Exception):
    """Custom API error with error tracking ID."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_id: Optional[str] = None,
        internal_message: Optional[str] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.error_id = error_id or str(uuid.uuid4())[:8]
        self.internal_message = internal_message or message
        super().__init__(self.message)


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """Handle APIError exceptions."""
    logger.error(
        f"API Error [{exc.error_id}]: {exc.internal_message}",
        extra={"error_id": exc.error_id, "status_code": exc.status_code},
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message, "error_id": exc.error_id},
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions with sanitized response."""
    error_id = str(uuid.uuid4())[:8]
    logger.exception(f"Unhandled exception [{error_id}]: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "An internal error occurred",
            "error_id": error_id,
        },
    )


# =============================================================================
# Rate Limiting
# =============================================================================

limiter = Limiter(key_func=get_remote_address)


# =============================================================================
# Middleware
# =============================================================================

class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Middleware to extract or generate correlation IDs for request tracing."""

    async def dispatch(self, request: Request, call_next):
        # Extract or generate correlation ID
        correlation_id = request.headers.get("X-Correlation-ID")
        if not correlation_id:
            correlation_id = str(uuid.uuid4())[:8]

        # Set in context for logging
        set_correlation_id(correlation_id)

        # Process request
        response = await call_next(request)

        # Add to response headers
        response.headers["X-Correlation-ID"] = correlation_id

        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # HSTS for HTTPS deployments
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )

        return response


# =============================================================================
# Authentication
# =============================================================================

async def verify_api_key(request: Request) -> Optional[str]:
    """
    Verify API key if authentication is enabled.
    Returns the API key if valid, None if auth is disabled.
    Raises HTTPException if auth is enabled but key is invalid.
    """
    if not settings.auth_enabled:
        return None

    api_key = request.headers.get("X-API-Key")
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide X-API-Key header.",
        )

    if api_key not in settings.api_keys:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
        )

    return api_key


# =============================================================================
# File Validation
# =============================================================================

async def validate_file_size(image: UploadFile) -> bytes:
    """
    Read and validate uploaded file size.
    Returns file contents if valid.
    Raises HTTPException if file is too large.
    """
    contents = await image.read()

    if len(contents) > settings.max_file_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.max_file_size // (1024 * 1024)}MB.",
        )

    return contents


def archive_scan_image(contents: bytes, upload_name: Optional[str]) -> Path:
    """Persist a validated /scan upload with private file permissions."""
    archive_dir = Path(settings.scan_archive_dir)
    archive_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    archive_dir.chmod(0o700)

    suffix = Path(upload_name or "").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
        suffix = ".img"
    saved = archive_dir / f"scan-{time.time_ns()}-{uuid.uuid4().hex}{suffix}"
    with saved.open("xb") as handle:
        handle.write(contents)
    saved.chmod(0o600)
    return saved


# =============================================================================
# Application Lifecycle
# =============================================================================

# Track background tasks for graceful shutdown
background_tasks: List[asyncio.Task] = []
shutdown_event: Optional[asyncio.Event] = None
shutdown_initiated = False


async def graceful_shutdown():
    """Handle graceful shutdown on SIGTERM/SIGINT."""
    global shutdown_initiated, shutdown_event
    if shutdown_initiated:
        return
    shutdown_initiated = True
    logger.info("Received shutdown signal, initiating graceful shutdown...")
    if shutdown_event is not None:
        shutdown_event.set()

    # Cancel background tasks
    for task in background_tasks:
        if not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

    logger.info("Background tasks cancelled")


def setup_signal_handlers(loop: asyncio.AbstractEventLoop):
    """Setup signal handlers for graceful shutdown."""
    import threading

    # Signal handlers can only be set in the main thread
    if threading.current_thread() is not threading.main_thread():
        logger.debug("Skipping signal handler setup (not main thread)")
        return

    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig,
                lambda: asyncio.create_task(graceful_shutdown()),
            )
    except ValueError as e:
        # This can happen if we're not in the main thread of the main interpreter
        logger.debug(f"Could not set signal handlers: {e}")


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    """Application lifespan manager."""
    global shutdown_event

    # Create shutdown event inside async context (required for Python 3.10+)
    shutdown_event = asyncio.Event()

    # Setup signal handlers
    loop = asyncio.get_running_loop()
    try:
        setup_signal_handlers(loop)
    except NotImplementedError:
        # Signal handlers not supported on Windows
        logger.warning("Signal handlers not supported on this platform")

    # Startup
    scanner = Scanner()
    db = database.Database(categories=settings.default_categories)
    db.initialize()

    # Check if database exists before opening (open() creates it)
    db_exists = Path(settings.database_path).exists()

    try:
        await db.open()

        # Only run initial update if database didn't exist
        if not db_exists:
            logger.info("Database not found, running initial update...")
            await db.update()
        else:
            logger.info("Database exists, skipping initial update (will update on schedule)")

        # Start background tasks and track them
        db.start_scheduled_updates()
        scanner.start_scheduled_updates()

        if db.update_task:
            background_tasks.append(db.update_task)
        if scanner.matcher.update_task:
            background_tasks.append(scanner.matcher.update_task)

        app.state.db = db
        app.state.scanner = scanner

        logger.info("Application startup complete")
        yield

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    finally:
        # Shutdown - always runs
        logger.info("Shutting down...")
        await db.close()
        logger.info("Shutdown complete")


# =============================================================================
# Application Setup
# =============================================================================

app = fastapi.FastAPI(
    lifespan=lifespan,
    title="Card Scanner API",
    description="Trading card scanner using a YOLO pose detector + SigLIP2 LoRA embedding search",
    version="1.0.0",
)

# Add exception handlers
app.add_exception_handler(APIError, api_error_handler)
app.add_exception_handler(Exception, generic_exception_handler)
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add rate limiter state
app.state.limiter = limiter

# Add middleware (order matters - first added = outermost)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(CorrelationIdMiddleware)

# CORS configuration
# Fix: allow_credentials must be False when using wildcard origins
if settings.cors_origins == ["*"]:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,  # Must be False with wildcard
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

# Serve frontend
app.mount("/static", StaticFiles(directory="static", html=True), name="static")


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to the frontend."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html")


# =============================================================================
# Label-review endpoints
# =============================================================================

LABEL_REVIEW_FIELDS = (
    "filename", "product_id", "candidate_name", "confidence",
    "physical_finish_if_visually_verified", "visual_evidence", "catalog_image_url",
)


class LabelReviewUpdate(BaseModel):
    product_id: int
    review_note: str = ""


def _label_csv_path() -> Path:
    return Path(settings.label_review_dir) / "visual-only-agent.csv"


def _label_overrides_path() -> Path:
    return Path(settings.label_review_dir) / "visual-only-review-overrides.json"


def _read_label_overrides() -> dict:
    path = _label_overrides_path()
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        logger.warning("Unable to read label-review overrides")
        return {}


def _read_label_rows() -> list[dict]:
    path = _label_csv_path()
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as source:
        rows = [dict(row) for row in csv.DictReader(source) if row.get("filename")]
    overrides = _read_label_overrides()
    for row in rows:
        original_id = row.get("product_id", "")
        row["original_product_id"] = original_id
        row["original_candidate_name"] = row.get("candidate_name", "")
        override = overrides.get(row["filename"])
        if override:
            row.update(override)
            row["review_status"] = "changed"
        else:
            row["review_status"] = "unreviewed"
    return rows


def _product_for_review(product_id: int) -> dict | None:
    # The catalog is intentionally read-only here. User corrections are stored
    # separately from the TCGCSV ingestion database.
    import sqlite3
    db_path = Path(settings.database_path).resolve()
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as connection:
        row = connection.execute(
            "SELECT product_id, name, image_url, sub_type_name FROM products WHERE product_id = ?",
            (product_id,),
        ).fetchone()
    if not row:
        return None
    return {"product_id": row[0], "name": row[1], "image_url": row[2] or "", "sub_type_name": row[3] or ""}


@app.get("/label-review/labels", tags=["Label review"])
async def label_review_labels():
    """Return agent choices merged with user review overrides."""
    return {"labels": _read_label_rows()}


@app.get("/label-review/images/{filename}", tags=["Label review"])
async def label_review_image(filename: str):
    safe_name = Path(filename).name
    if safe_name != filename or Path(safe_name).suffix.lower() not in {".jpg", ".jpeg"}:
        raise HTTPException(status_code=400, detail="Invalid image filename")
    image_path = Path(settings.label_review_images_dir) / safe_name
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Review image not found")
    return FileResponse(image_path, media_type="image/jpeg")


@app.get("/label-review/products", tags=["Label review"])
async def label_review_products(q: str = ""):
    query = q.strip()
    if len(query) < 2:
        return {"products": []}
    import sqlite3
    db_path = Path(settings.database_path).resolve()
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "SELECT product_id, name, image_url, sub_type_name FROM products "
            "WHERE name LIKE ? OR clean_name LIKE ? ORDER BY name LIMIT 60",
            (f"%{query}%", f"%{query}%"),
        ).fetchall()
    return {"products": [{"product_id": r[0], "name": r[1], "image_url": r[2] or "", "sub_type_name": r[3] or ""} for r in rows]}


@app.put("/label-review/labels/{filename}", tags=["Label review"])
async def update_label_review(filename: str, update: LabelReviewUpdate):
    safe_name = Path(filename).name
    if safe_name != filename:
        raise HTTPException(status_code=400, detail="Invalid image filename")
    if safe_name not in {row["filename"] for row in _read_label_rows()}:
        raise HTTPException(status_code=404, detail="Label not found")
    product = _product_for_review(update.product_id)
    if not product:
        raise HTTPException(status_code=404, detail="TCGCSV product ID not found")
    overrides = _read_label_overrides()
    overrides[safe_name] = {
        "product_id": str(product["product_id"]),
        "candidate_name": product["name"],
        "catalog_image_url": product["image_url"],
        "review_note": update.review_note.strip(),
    }
    destination = _label_overrides_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".tmp")
    temp.write_text(json.dumps(overrides, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, destination)
    return {"status": "saved", "label": {"filename": safe_name, **overrides[safe_name]}}


# =============================================================================
# Request/Response Models
# =============================================================================

class ProductVariant(BaseModel):
    """One priced finish (Normal, Reverse Holofoil, etc.) of a matched
    card -- these share the same card_id/image, since TCGCSV assigns one
    productId per card regardless of finish."""
    sub_type_name: str
    low_price: Optional[float] = None
    mid_price: Optional[float] = None
    high_price: Optional[float] = None
    market_price: Optional[float] = None
    direct_low_price: Optional[float] = None


class ScanResult(BaseModel):
    card_id: int
    similarity: float
    box: List[float]
    details: Optional[dict] = None
    # Every priced finish for this card (Normal, Reverse Holofoil, etc.) --
    # the match itself can't tell which one is physically in hand, since
    # they share one catalog image (see ProductVariant).
    variants: List[ProductVariant] = []
    # RANSAC inlier count when geometric verification was requested
    # (verify=true); doubles as a match-confidence signal.
    inliers: Optional[int] = None


class ErrorResponse(BaseModel):
    error: str
    error_id: str


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    database: str
    scanner: str


# =============================================================================
# Health Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """
    Liveness probe - always returns 200 if the service is running.
    """
    return {"status": "healthy"}


@app.get("/ready", response_model=ReadyResponse, tags=["Health"])
async def ready():
    """
    Readiness probe - checks if the service is ready to accept requests.
    Verifies database connection and scanner initialization.
    """
    db_status = "unknown"
    scanner_status = "unknown"

    try:
        db = app.state.db
        if db and db.is_initialized:
            # Test database connection
            async with db.conn.execute("SELECT 1") as cursor:
                await cursor.fetchone()
            db_status = "ready"
        else:
            db_status = "not_initialized"
    except Exception as e:
        logger.warning(f"Readiness check - database error: {e}")
        db_status = "error"

    try:
        scanner = app.state.scanner
        if scanner and scanner.matcher and scanner.matcher.database:
            scanner_status = "ready"
        else:
            scanner_status = "not_initialized"
    except Exception as e:
        logger.warning(f"Readiness check - scanner error: {e}")
        scanner_status = "error"

    if db_status == "ready" and scanner_status == "ready":
        return {"status": "ready", "database": db_status, "scanner": scanner_status}
    else:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "database": db_status,
                "scanner": scanner_status,
            },
        )


# =============================================================================
# Scan Endpoints
# =============================================================================

LIVE_RECOGNITION_MAX_FPS = 20


def _save_live_detection_debug(frame: bytes, results: list[dict]) -> None:
    """TEMPORARY diagnostic aid (config.debug_save_live_detections, off by
    default): saves the exact raw frame bytes a /live-recognize client
    sent whenever it produced a non-empty detection, alongside a JSON
    sidecar with the full match info (card_id/similarity/box/track_id) --
    so a misidentification seen live can be inspected afterwards using
    the *actual* frame the client sent (real resolution/compression/
    frontend, not a guessed reproduction). Not meant to stay enabled long
    term -- this is unauthenticated write volume on every hit."""
    try:
        debug_dir = Path(settings.debug_live_detections_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        stamp = f"{time.time_ns()}-{uuid.uuid4().hex[:8]}"
        (debug_dir / f"{stamp}.jpg").write_bytes(frame)
        (debug_dir / f"{stamp}.json").write_text(json.dumps(results, indent=2, default=str))
    except Exception:
        logger.exception("Failed to save live-detection debug capture")


async def _live_frame_results(
    image: np.ndarray,
    tracker,
    top_n: Optional[int] = None,
    margin_pct: Optional[float] = None,
    min_similarity: Optional[float] = None,
) -> list[dict]:
    """Recognize one camera frame and enrich its candidate IDs from the DB.

    `tracker` is one connection's isolated Scanner.new_tracker() instance
    (see scanner.py) -- each detected card gets a `track_id` that stays
    stable for the same physical card across frames on this connection.
    No server-side score smoothing: raw per-frame similarity only, so
    frontends can implement whatever temporal aggregation they want,
    keyed by track_id.

    `top_n`/`margin_pct`/`min_similarity` behave exactly like /scan's:
    top_n fixes the match count per detected card; omitting it switches to
    margin mode (every match within margin_pct points of that card's best
    score); min_similarity drops matches below that cosine similarity
    entirely (config.min_match_similarity default) -- see
    siglip_matcher.SigLIPCardSearch.search().
    """
    scanner = getattr(app.state, "scanner", None)
    db = getattr(app.state, "db", None)
    if not scanner or not db or not db.is_initialized:
        raise RuntimeError("service_not_ready")

    detected_cards = await asyncio.wait_for(
        asyncio.to_thread(scanner.scan, image, k=top_n, verify=False,
                           tracker=tracker, margin_pct=margin_pct, min_similarity=min_similarity),
        timeout=settings.yolo_timeout,
    )
    columns = db.return_columns()
    results = []
    for card_segment in detected_cards:
        for match in card_segment["matches"]:
            card_id = int(match["card_id"])
            product_data = await db.query_by_id(card_id)
            results.append({
                "card_id": card_id,
                "track_id": card_segment["track_id"],
                "box": card_segment["box"],
                "details": dict(zip(columns, product_data)) if product_data else None,
                "variants": await db.query_variants_by_id(card_id),
                "similarity": round(float(match["similarity"]), 4),
            })
    return results


@app.websocket("/live-recognize")
async def live_recognize(
    websocket: WebSocket,
    top_n: Optional[int] = None,
    margin_pct: Optional[float] = None,
    min_similarity: Optional[float] = None,
):
    """Recognize JPEG camera frames, enforcing 20 FPS per connection.

    A connection owns one isolated tracker (Scanner.new_tracker()), so
    `track_id`s stay stable for the same physical card across frames on
    that connection without leaking into any other connection's tracks.
    There is no server-side score smoothing -- see `_live_frame_results`.

    `top_n`/`margin_pct`/`min_similarity` are query params on the
    connection URL (e.g. `/live-recognize?top_n=5`,
    `/live-recognize?margin_pct=3`, `/live-recognize?min_similarity=0.5`),
    fixed for the lifetime of the connection just like its tracker -- same
    semantics as /scan's (omitting top_n switches to margin mode).
    """
    await websocket.accept()
    scanner = getattr(app.state, "scanner", None)
    tracker = scanner.new_tracker() if scanner else None
    last_accepted_at: Optional[float] = None
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return
            frame = message.get("bytes")
            if frame is None:
                await websocket.send_json({"type": "error", "error": "invalid_frame"})
                continue
            now = time.monotonic()
            if last_accepted_at is not None and now - last_accepted_at < 1 / LIVE_RECOGNITION_MAX_FPS:
                await websocket.send_json({
                    "type": "error", "error": "frame_rate_exceeded", "max_fps": LIVE_RECOGNITION_MAX_FPS,
                })
                continue
            last_accepted_at = now
            if len(frame) > settings.max_file_size:
                await websocket.send_json({"type": "error", "error": "frame_too_large"})
                continue
            image = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                await websocket.send_json({"type": "error", "error": "invalid_jpeg"})
                continue
            try:
                results = await _live_frame_results(image, tracker, top_n=top_n, margin_pct=margin_pct, min_similarity=min_similarity)
                if results and settings.debug_save_live_detections:
                    _save_live_detection_debug(frame, results)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "error", "error": "recognition_timed_out"})
                continue
            except RuntimeError as exc:
                if str(exc) == "service_not_ready":
                    await websocket.send_json({"type": "error", "error": "service_not_ready"})
                    continue
                logger.exception("Live recognition failed")
                await websocket.send_json({"type": "error", "error": "recognition_failed"})
                continue
            except Exception:
                logger.exception("Live recognition failed")
                await websocket.send_json({"type": "error", "error": "recognition_failed"})
                continue
            await websocket.send_json({"type": "result", "results": results})
    except WebSocketDisconnect:
        return


@app.post(
    "/scan",
    response_model=List[ScanResult],
    responses={401: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
    tags=["Scan"],
)
@limiter.limit(f"{settings.rate_limit_scan}/minute")
async def scan(
    request: Request,
    image: UploadFile = fastapi.File(...),
    top_n: Optional[int] = None,
    margin_pct: Optional[float] = None,
    min_similarity: Optional[float] = None,
    verify: bool = False,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """
    Scan an image from file upload. Returns all data for all cards detected.
    Args:
        image: The image file to scan
        top_n: The number of top matches to return per card. If omitted,
            returns every match within margin_pct percentage points of the
            best match's similarity instead of a fixed count (useful for
            reprints/near-duplicates that shouldn't be arbitrarily narrowed
            down to one) -- see config.match_margin_pct.
        margin_pct: Overrides the default margin (config.match_margin_pct)
            used when top_n is omitted. Ignored if top_n is given.
        min_similarity: Overrides the default minimum-similarity floor
            (config.min_match_similarity) -- any match below this cosine
            similarity is dropped entirely, so a detected region that
            doesn't resemble anything real in the gallery reports no match
            instead of a false-confident "closest available" one.
        verify: Geometrically verify matches (RANSAC re-rank; adds inlier counts)
    Returns:
        JSON object containing all data for all cards detected
    """
    try:
        # Validate file size
        contents = await validate_file_size(image)

        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise APIError("Invalid image file", status_code=400)

        archived_image = archive_scan_image(contents, image.filename)
        logger.info("Archived /scan upload at %s", archived_image)

        # Use scanner to detect and match cards with timeout
        scanner = getattr(app.state, 'scanner', None)
        db = getattr(app.state, 'db', None)

        if not scanner or not db or not db.is_initialized:
            raise HTTPException(status_code=503, detail="Service not ready")

        try:
            detected_cards = await asyncio.wait_for(
                asyncio.to_thread(scanner.scan, img, k=top_n, verify=verify, margin_pct=margin_pct, min_similarity=min_similarity),
                timeout=settings.yolo_timeout,
            )
        except asyncio.TimeoutError:
            raise APIError(
                "Scan operation timed out",
                status_code=504,
                internal_message=f"YOLO inference exceeded {settings.yolo_timeout}s timeout",
            )

        results = []
        cols = db.return_columns()

        for card_segment in detected_cards:
            box = card_segment["box"]
            for match in card_segment["matches"]:
                product_id = match["card_id"]
                similarity = match["similarity"]

                # Query DB for product details
                product_data = await db.query_by_id(product_id)
                details = None
                if product_data:
                    details = dict(zip(cols, product_data))
                variants = await db.query_variants_by_id(product_id)

                results.append(
                    ScanResult(
                        card_id=product_id,
                        similarity=similarity,
                        box=box,
                        details=details,
                        variants=variants,
                        inliers=match.get("inliers"),
                    )
                )

        return results

    except APIError:
        raise
    except HTTPException:
        raise
    except Exception as e:
        raise APIError(
            "Scan failed",
            status_code=500,
            internal_message=str(e),
        )


@app.post(
    "/identify",
    response_model=List[ScanResult],
    responses={401: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
    tags=["Scan"],
)
@limiter.limit(f"{settings.rate_limit_identify}/minute")
async def identify(
    request: Request,
    image: UploadFile = fastapi.File(...),
    top_n: Optional[int] = None,
    margin_pct: Optional[float] = None,
    min_similarity: Optional[float] = None,
    verify: bool = False,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """
    Identify a pre-cropped card image. Skips YOLO detection.
    Args:
        image: The image file of the card
        top_n: The number of top matches to return. If omitted, returns
            every match within margin_pct percentage points of the best
            match's similarity instead of a fixed count -- see
            config.match_margin_pct.
        margin_pct: Overrides the default margin (config.match_margin_pct)
            used when top_n is omitted. Ignored if top_n is given.
        min_similarity: Overrides the default minimum-similarity floor
            (config.min_match_similarity) -- any match below this cosine
            similarity is dropped entirely. See /scan's min_similarity.
        verify: Geometrically verify matches (RANSAC re-rank; adds inlier counts)
    Returns:
        JSON object containing data for the identified card matches
    """
    try:
        # Validate file size
        contents = await validate_file_size(image)

        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise APIError("Invalid image file", status_code=400)

        scanner = getattr(app.state, 'scanner', None)
        db = getattr(app.state, 'db', None)

        if not scanner or not db or not db.is_initialized:
            raise HTTPException(status_code=503, detail="Service not ready")

        # Use identify_card (lighter than full scan) with timeout
        try:
            card_result = await asyncio.wait_for(
                asyncio.to_thread(scanner.identify_card, img, k=top_n, verify=verify, margin_pct=margin_pct, min_similarity=min_similarity),
                timeout=settings.yolo_timeout,
            )
        except asyncio.TimeoutError:
            raise APIError(
                "Identification timed out",
                status_code=504,
                internal_message=f"Identify operation exceeded {settings.yolo_timeout}s timeout",
            )

        results = []

        cols = db.return_columns()
        box = card_result["box"]

        for match in card_result["matches"]:
            product_id = match["card_id"]
            similarity = match["similarity"]

            # Query DB for product details
            product_data = await db.query_by_id(product_id)
            details = None
            if product_data:
                details = dict(zip(cols, product_data))
            variants = await db.query_variants_by_id(product_id)

            results.append(
                ScanResult(
                    card_id=product_id,
                    similarity=similarity,
                    box=box,
                    details=details,
                    variants=variants,
                    inliers=match.get("inliers"),
                )
            )

        return results

    except APIError:
        raise
    except HTTPException:
        raise
    except Exception as e:
        raise APIError(
            "Identification failed",
            status_code=500,
            internal_message=str(e),
        )


# =============================================================================
# Price Endpoints
# =============================================================================

@app.get("/price", tags=["Pricing"])
@limiter.limit(f"{settings.rate_limit_price}/minute")
async def price(
    request: Request,
    product_id: int,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """
    Get the price of a card.
    Args:
        product_id: The product ID of the card
    Returns:
        JSON object containing the price of the card
    """
    db = getattr(app.state, 'db', None)
    if not db or not db.is_initialized:
        raise HTTPException(status_code=503, detail="Service not ready")

    product_data = await db.query_by_id(product_id)
    if not product_data:
        raise HTTPException(status_code=404, detail="Product not found")

    cols = db.return_columns()
    product_dict = dict(zip(cols, product_data))

    price_cols = [
        "low_price",
        "mid_price",
        "high_price",
        "market_price",
        "direct_low_price",
    ]
    return {col: product_dict[col] for col in price_cols}


@app.post("/prices", tags=["Pricing"])
@limiter.limit(f"{settings.rate_limit_price}/minute")
async def prices(
    request: Request,
    product_ids: List[int],
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """
    Get prices for multiple cards in a single request.
    Args:
        product_ids: List of product IDs to get prices for
    Returns:
        JSON object mapping product IDs to their prices
    """
    db = getattr(app.state, 'db', None)
    if not db or not db.is_initialized:
        raise HTTPException(status_code=503, detail="Service not ready")

    result = await db.query_prices_batch(product_ids)
    return {"prices": result}


# =============================================================================
# Data Endpoints
# =============================================================================

@app.get("/categories", tags=["Data"])
@limiter.limit(f"{settings.rate_limit_price}/minute")
async def get_categories(
    request: Request,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """
    Get the categories in the database.
    Returns:
        JSON object containing the categories in the database
    """
    db = getattr(app.state, 'db', None)
    if not db or not db.is_initialized:
        raise HTTPException(status_code=503, detail="Service not ready")

    cats = await db.get_categories()
    return {"categories": [{"category_id": c[0], "category_name": c[1]} for c in cats]}


@app.get("/groups", tags=["Data"])
@limiter.limit(f"{settings.rate_limit_price}/minute")
async def get_groups(
    request: Request,
    category_id: int = 3,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """
    Get the groups for a category.
    Args:
        category_id: The category ID to get groups for (default: 3 for Pokemon)
    Returns:
        JSON object containing the groups in the category
    """
    db = getattr(app.state, 'db', None)
    if not db or not db.is_initialized:
        raise HTTPException(status_code=503, detail="Service not ready")

    groups = await db.get_groups(category_id)
    return {
        "groups": [
            {"group_id": g[0], "category_id": g[1], "group_name": g[2]} for g in groups
        ]
    }


@app.get("/columns", tags=["Data"])
@limiter.limit(f"{settings.rate_limit_price}/minute")
async def columns(
    request: Request,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """
    Get the columns in the database.
    Returns:
        JSON object containing the columns in the database
    """
    db = getattr(app.state, 'db', None)
    if not db or not db.is_initialized:
        raise HTTPException(status_code=503, detail="Service not ready")

    return {"columns": db.return_columns()}


@app.get("/ext-data", tags=["Data"])
@limiter.limit(f"{settings.rate_limit_price}/minute")
async def ext_data(
    request: Request,
    category_id: int = 3,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """
    Get the extended data column names for a category
    Args:
        category_id: The category ID to get the extended data column names for
    Returns:
        JSON object containing the extended data column names for the category
    """
    db = getattr(app.state, 'db', None)
    if not db or not db.is_initialized:
        raise HTTPException(status_code=503, detail="Service not ready")

    # Get a sample product from this category to see its ext_data keys
    async with db.conn.execute(
        "SELECT ext_data FROM products WHERE category_id = ? LIMIT 1",
        (category_id,),
    ) as cursor:
        row = await cursor.fetchone()
        if row and row[0]:
            try:
                ext_dict = pickle.loads(row[0])
                return {"ext_data_columns": list(ext_dict.keys())}
            except Exception as e:
                logger.warning(f"Failed to deserialize ext_data for category {category_id}: {e}")
                return {"ext_data_columns": []}

    return {"ext_data_columns": []}


# =============================================================================
# Admin Endpoints
# =============================================================================

@app.post("/update", tags=["Admin"])
@limiter.limit(f"{settings.rate_limit_update}/minute")
async def update(
    request: Request,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """
    Update the database.
    Returns:
        JSON object containing the status of the update
    """
    db = getattr(app.state, 'db', None)
    if not db or not db.is_initialized:
        raise HTTPException(status_code=503, detail="Service not ready")

    async def safe_update():
        """Wrapper to handle exceptions in background update task."""
        try:
            await db.update()
        except Exception as e:
            logger.error(f"Background database update failed: {e}")

    try:
        # We trigger update in background to not block the request
        task = asyncio.create_task(safe_update())
        background_tasks.append(task)
        return {"status": "Update started"}
    except Exception as e:
        raise APIError(
            "Failed to start update",
            status_code=500,
            internal_message=str(e),
        )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port)
