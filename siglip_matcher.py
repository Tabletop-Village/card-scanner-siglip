"""SigLIP2 LoRA global-embedding card search, replacing CudaSift RootSIFT+VLAD.

Cosine-similarity search over a precomputed gallery embedding index instead
of per-card local-feature (VLAD) aggregation -- see ~/siglip-scanner for the
fine-tuning project this model came from (98.8% top-1 / 100% top-3 on the
514-scan real-camera benchmark, vs the VLAD baseline's 94.2%/97.7%, at
~17ms/image encode vs VLAD's ~224ms).

Deliberately mirrors vlad_matcher.VLADCardSearch's public interface
(search/search_verified/compare_images/batch_search, the `database` dict,
`update_task` + start_scheduled_updates/scheduled_update) so scanner.py and
api.py need no changes beyond which matcher class gets instantiated.

The LoRA adapter and gallery index are hosted on the HuggingFace Hub
(https://huggingface.co/jackttv/card-scanner-siglip-lora) rather than a
synced git repo: `PeftModel.from_pretrained(repo_id)` fetches/caches the
adapter natively, and `embeddings.pt` (one consolidated {ids, embeds} file,
already-cached gallery embeddings from the fine-tuning project, fp16 -- no
need to regenerate them) is fetched with `hf_hub_download`. Both use HF's
own local cache (~/.cache/huggingface/hub by default), so only the first
startup needs network access. A local `siglip_vectors/` directory, if
present (e.g. for offline dev), takes priority over the Hub -- see
`_local_or_hub_path`.

search_verified() has no real geometric-verification equivalent to fall
back to (RANSAC re-ranking is a local-keypoint technique; a global
embedding has no keypoints) -- it returns inliers=0 same as
VLADCardSearch.search_verified() already does for this deployment's
CudaSift-tagged path, so this isn't a behavior regression.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from peft import PeftModel
from transformers import AutoModel, AutoProcessor

from config import settings
from logging_config import get_logger

logger = get_logger(__name__)

MODEL_ID = "google/siglip2-so400m-patch14-384"


class SigLIPCardSearch:
    """Load a SigLIP2 LoRA gallery embedding index and search it with cosine similarity."""

    def __init__(self, vectors_path=None, lora_path=None, hf_repo_id=None):
        self.repo_path = Path(vectors_path or settings.siglip_vectors_path)
        self.lora_path = Path(lora_path) if lora_path else self.repo_path / "lora_best"
        self.hf_repo_id = hf_repo_id or settings.siglip_hf_repo_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.database: dict[str, np.ndarray] = {}
        self._db_array = None
        self._db_ids = None
        self.update_task = None

        self._load_model()
        self.load_database()

    def _load_model(self):
        model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(self.device)
        if self.lora_path.exists():
            logger.info("Loading SigLIP2 base model + local LoRA adapter from %s", self.lora_path)
            model.vision_model = PeftModel.from_pretrained(model.vision_model, str(self.lora_path))
            model.vision_model = model.vision_model.merge_and_unload()
        elif self.hf_repo_id:
            logger.info("Loading SigLIP2 base model + LoRA adapter from HF Hub: %s", self.hf_repo_id)
            model.vision_model = PeftModel.from_pretrained(model.vision_model, self.hf_repo_id)
            model.vision_model = model.vision_model.merge_and_unload()
        else:
            logger.warning("No local or HF Hub LoRA adapter configured; using base SigLIP2 zero-shot")
        model.eval()
        self.model = model
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)

    def _rebuild_search_cache(self):
        if not self.database:
            self._db_array = self._db_ids = None
            return
        self._db_ids = np.array(list(self.database.keys()))
        values = np.vstack(list(self.database.values()))
        self._db_array = torch.as_tensor(values, dtype=torch.float16, device=self.device)
        logger.info("SigLIP search cache built: %s cards, shape %s on %s",
                    len(values), values.shape, self.device)

    def _resolve_embeddings_path(self):
        """Local siglip_vectors/embeddings.pt wins if present (offline dev);
        otherwise fetch/cache it from the HF Hub repo."""
        local_file = self.repo_path / "embeddings.pt"
        if local_file.exists():
            return local_file
        if not self.hf_repo_id:
            return None
        return Path(hf_hub_download(self.hf_repo_id, "embeddings.pt"))

    def load_database(self):
        self.database = {}
        embeddings_file = self._resolve_embeddings_path()
        if embeddings_file is None or not embeddings_file.exists():
            logger.warning("SigLIP vector index not found locally (%s) or on HF Hub (%s)",
                            self.repo_path / "embeddings.pt", self.hf_repo_id)
            self._rebuild_search_cache()
            return
        data = torch.load(embeddings_file, map_location="cpu")
        ids = data["ids"].tolist()
        embeds = data["embeds"].numpy().astype(np.float16)
        self.database = {str(pid): embeds[i] for i, pid in enumerate(ids)}
        self._rebuild_search_cache()

    def reload_database(self):
        self.load_database()

    def sync_and_reload(self, force=False):
        # No git repo to sync -- reload just re-resolves the embeddings path
        # (local override, or re-checks the HF Hub for a newer revision via
        # its own cache/ETag logic) and rebuilds the search cache.
        self.load_database()

    async def scheduled_update(self):
        while True:
            now = datetime.now()
            target = datetime.combine(now.date(), settings.vectors_update_time)
            if target <= now:
                target += timedelta(days=1)
            await asyncio.sleep((target - now).total_seconds())
            await asyncio.to_thread(self.sync_and_reload)

    def start_scheduled_updates(self):
        if self.update_task is None or self.update_task.done():
            self.update_task = asyncio.create_task(self.scheduled_update())

    def encode(self, image: np.ndarray) -> torch.Tensor:
        """image: BGR OpenCV array (as produced by Scanner.crop). Returns an
        L2-normalized embedding tensor on self.device."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        inputs = self.processor(images=[pil_img], return_tensors="pt").to(self.device, torch.bfloat16)
        with torch.inference_mode():
            feats = self.model.get_image_features(**inputs).pooler_output
            feats = F.normalize(feats.float(), dim=-1)
        return feats.squeeze(0)

    def search(self, query_image, top_k=5, margin_pct=None):
        """top_k=None switches to margin mode: instead of a fixed count,
        return every gallery match within `margin_pct` percentage points
        (defaults to settings.match_margin_pct) of the single best match's
        similarity -- e.g. for reprints/near-duplicate cards that a caller
        wants surfaced together rather than arbitrarily narrowed to one.
        Searches a settings.match_margin_pool_size-sized candidate pool
        first rather than the full gallery; real near-duplicates cluster
        within a couple of points of the top score, so this comfortably
        covers them without an expensive full sort."""
        if self._db_array is None:
            return []
        query = self.encode(query_image).to(torch.float16)
        scores = self._db_array @ query

        if top_k is None:
            margin = settings.match_margin_pct if margin_pct is None else margin_pct
            pool = min(settings.match_margin_pool_size, len(self._db_ids))
            if not pool:
                return []
            values, indices = torch.topk(scores, pool)
            threshold = float(values[0]) - margin / 100.0
            return [(str(self._db_ids[int(index)]), float(value))
                    for value, index in zip(values.cpu(), indices.cpu())
                    if float(value) >= threshold]

        count = min(max(int(top_k), 0), len(self._db_ids))
        if not count:
            return []
        values, indices = torch.topk(scores, count)
        return [(str(self._db_ids[int(index)]), float(value))
                for value, index in zip(values.cpu(), indices.cpu())]

    def search_verified(self, query_image, top_k=5, rerank_k=None, margin_pct=None):
        """API-compatible fallback -- a global embedding has no keypoints to
        run RANSAC geometric verification against, so this matches
        VLADCardSearch.search_verified()'s own current fallback behavior
        for the CudaSift-tagged deployment (inliers=0), not a regression."""
        return [(card_id, similarity, 0)
                for card_id, similarity in self.search(query_image, top_k, margin_pct=margin_pct)]

    def compare_images(self, image1, image2):
        a = self.encode(image1)
        b = self.encode(image2)
        return float((a * b).sum())

    def batch_search(self, images, top_k=5):
        return [self.search(image, top_k) for image in images]
