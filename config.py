"""
Centralized configuration for the card scanner service.
Uses pydantic-settings for environment variable support.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import List, Any
from datetime import time as dt_time


def parse_comma_list(value: Any) -> List[str]:
    """Parse comma-separated string into list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        if not value.strip():
            return []
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def parse_int_list(value: Any, default: List[int] | None = None) -> List[int]:
    """Parse comma-separated integers into list."""
    if value is None:
        return default or []
    if isinstance(value, list):
        return [int(x) for x in value]
    if isinstance(value, str):
        if not value.strip():
            return default or []
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return default or []


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    All settings can be overridden with CARD_SCANNER_ prefix.
    e.g., CARD_SCANNER_DATABASE_PATH, CARD_SCANNER_API_KEYS

    List fields (api_keys, cors_origins, default_categories) support
    comma-separated values: CARD_SCANNER_API_KEYS=key1,key2,key3
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CARD_SCANNER_",
        extra="ignore",
    )

    # Database settings
    database_path: str = "database.db"
    csv_path: str = "categories/"

    # Authentication - stored as string, parsed to list
    api_keys_str: str = Field(default="", validation_alias="CARD_SCANNER_API_KEYS")

    # CORS settings - stored as string, parsed to list
    cors_origins_str: str = Field(default="*", validation_alias="CARD_SCANNER_CORS_ORIGINS")

    # File upload limits
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    # Directory where validated images submitted to /scan are retained.
    scan_archive_dir: str = "scan-archive"
    # Review artifacts live outside the source tree so automated labeling can
    # append independently while the UI reads review overrides.
    label_review_dir: str = "/home/user/card-scanner-labeling"
    label_review_images_dir: str = "/home/user/experiments/card-scanner-corpus-audit-20260831/images"

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    # External service URLs
    tcg_csv_url: str = "https://tcgcsv.com/tcgplayer/"

    # Optional CSR pinned mode: verified local bundle, no Hub fallback/reloads.
    csr_manifest: str = ""

    # SigLIP2 LoRA global-embedding matcher (siglip_matcher.py). The adapter
    # + gallery embeddings live on the HF Hub (siglip_hf_repo_id); a local
    # siglip_vectors_path directory, if present, overrides the Hub for
    # offline dev -- see siglip_matcher.py's module docstring.
    siglip_vectors_path: str = "siglip_vectors"
    siglip_hf_repo_id: str = "jackttv/card-scanner-siglip-lora"

    # Margin-match mode: when a /scan or /identify caller omits top_n, the
    # matcher returns every gallery match within this many percentage
    # points of the single best match's similarity, instead of a fixed
    # count -- e.g. for reprints/near-duplicates that shouldn't be
    # arbitrarily narrowed down to one.
    match_margin_pct: float = 2.0
    match_margin_pool_size: int = 30

    # Below this cosine similarity, a match is dropped entirely rather
    # than returned -- a detected region that doesn't resemble anything
    # real in the gallery should report no match, not a false-confident
    # "closest available" one. Overridable per-request (see /scan's
    # min_similarity param); see siglip_matcher.SigLIPCardSearch.search()
    # for the incident this fixed.
    min_match_similarity: float = 0.3

    # Geometry-only sanity checks on the detected card quad, independent
    # of SigLIP similarity -- see geometry.py and Scanner.scan().
    # A detection more than this fraction off-frame is skipped entirely,
    # before it's even cropped/matched.
    max_offscreen_fraction: float = 0.2
    # Relative tolerance when comparing the quad's recovered 3D aspect
    # ratio against a matched candidate's real catalog image ratio (both
    # the ratio and its reciprocal count -- see
    # geometry.aspect_ratio_matches()). First-pass value: generous enough
    # to absorb real keypoint/lens-distortion noise on a genuine match,
    # tight enough to catch a genuinely wrong shape.
    aspect_ratio_tolerance: float = 0.15
    # Below this fraction of its own detection box (see
    # geometry.quad_box_area_fraction()), the pose model's 4 keypoints
    # are treated as a collapsed/failed prediction and discarded -- the
    # detection still gets matched, from the axis-aligned box crop
    # instead of a perspective-corrected one. A real card can't fill
    # much less than 0.486 of its bounding box at any in-plane rotation,
    # so this sits just under that. Recovered 13 of the 30 errors on the
    # 514-scan benchmark (94.16% -> 97.86% top-1 at the old imgsz=640).
    min_keypoint_quad_box_fraction: float = 0.45

    # Torch CUDA device for YOLO on NVIDIA hardware.
    yolo_device: str = "cuda"

    # Minimum detection confidence for the pose model. ultralytics'
    # own default (0.25) is quite permissive -- genuine cards in testing
    # scored 0.94-0.98, so this has a lot of headroom to cut down false
    # detections on non-card objects (a laptop screen, a water bottle)
    # without losing real ones.
    yolo_confidence_threshold: float = 0.6

    # Inference size for the pose model, below ultralytics' 640 default.
    # The keypoint head's accuracy depends on the card's *absolute* pixel
    # size in the network input, and it degrades badly once the card gets
    # large there -- the synthetic training corpus is mostly small cards
    # (median box height 0.376 of the frame, only 4% above 0.7), while a
    # card filling a fixed-camera fixture is ~0.77. Measured on the
    # 514-scan benchmark, the fraction of frames with a collapsed
    # keypoint quad (see min_keypoint_quad_box_fraction) by imgsz:
    # 416 -> 0%, 512 -> 7%, 640 -> 100% of the frames that fail at all,
    # 960 -> 37%, 1280 -> 61%; end-to-end top-1 went 94.16% at 640 to
    # 96.69% at 512/416. 640 sits right on a cliff for this model at
    # this card size. Lower is also cheaper. The real fix is retraining
    # the pose model with large-card examples, after which this can go
    # back to the default.
    yolo_imgsz: int = 512

    # TEMPORARY diagnostic aid, off by default -- see
    # api._save_live_detection_debug(). Saves every /live-recognize frame
    # that produced a non-empty detection (plus its match info) to disk,
    # so a misidentification seen live can be inspected using the actual
    # frame the client sent. Not meant to stay enabled long term.
    debug_save_live_detections: bool = False
    debug_live_detections_dir: str = "live-detections"

    # YOLO pose detector (scanner.py) -- corner-keypoint model, replacing
    # the earlier segmentation-mask + approxPolyDP approach. A local
    # yolo_model_path file, if present, overrides the HF Hub fetch --
    # same local-override pattern as siglip_hf_repo_id/siglip_vectors_path.
    yolo_model_path: str = "models/pose_best.pt"
    yolo_hf_repo_id: str = "jackttv/card-scanner-yolo-pose"
    yolo_hf_filename: str = "best.pt"

    # Timeouts
    http_timeout: float = 30.0
    yolo_timeout: float = 30.0

    # Default categories - stored as string, parsed to list
    default_categories_str: str = Field(
        default="3", validation_alias="CARD_SCANNER_DEFAULT_CATEGORIES"
    )

    # Scheduled update times (hour, minute)
    db_update_hour: int = 3
    db_update_minute: int = 0
    vectors_update_hour: int = 4
    vectors_update_minute: int = 0

    # Concurrency limits
    max_concurrent_downloads: int = 5

    # Retry settings
    retry_attempts: int = 3
    retry_min_wait: float = 1.0
    retry_max_wait: float = 10.0

    # Rate limiting (requests per minute)
    rate_limit_scan: int = 10
    rate_limit_identify: int = 30
    rate_limit_price: int = 60
    rate_limit_update: int = 1

    # Logging
    log_level: str = "INFO"
    log_json: bool = True

    @property
    def api_keys(self) -> List[str]:
        """Get API keys as a list."""
        return parse_comma_list(self.api_keys_str)

    @property
    def cors_origins(self) -> List[str]:
        """Get CORS origins as a list."""
        result = parse_comma_list(self.cors_origins_str)
        return result if result else ["*"]

    @property
    def default_categories(self) -> List[int]:
        """Get default categories as a list of integers."""
        return parse_int_list(self.default_categories_str, [3])

    @property
    def db_update_time(self) -> dt_time:
        """Return the database update time as a time object."""
        return dt_time(hour=self.db_update_hour, minute=self.db_update_minute)

    @property
    def vectors_update_time(self) -> dt_time:
        """Return the vectors update time as a time object."""
        return dt_time(hour=self.vectors_update_hour, minute=self.vectors_update_minute)

    @property
    def auth_enabled(self) -> bool:
        """Check if authentication is enabled."""
        return len(self.api_keys) > 0


# Global settings instance
settings = Settings()
