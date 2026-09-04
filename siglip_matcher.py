"""SigLIP2 LoRA global-embedding card search, replacing CudaSift RootSIFT+VLAD.

Cosine-similarity search over a precomputed gallery embedding index instead
of per-card local-feature (VLAD) aggregation -- see ~/siglip-scanner for the
fine-tuning project this model came from (98.8% top-1 / 100% top-3 on the
514-scan real-camera benchmark, vs the VLAD baseline's 94.2%/97.7%, at
~17ms/image encode vs VLAD's ~224ms).

Deliberately mirrors vlad_matcher.VLADCardSearch's public interface
(search/search_verified/compare_images/batch_search, the `database` dict,
`update_task` + start_scheduled_updates/scheduled_update) so scanner.py and
api.py need no changes beyond which matcher class gets instantiated -- the
"vectors repo" here is a small local directory (siglip_vectors/) holding a
single consolidated embeddings.pt (already-cached gallery embeddings from
the fine-tuning project, converted to fp16 -- no need to regenerate them)
plus the merged LoRA adapter, rather than VLAD's per-card sharded .pkl
files synced from an external git repo.

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
from PIL import Image
from peft import PeftModel
from transformers import AutoModel, AutoProcessor

from config import settings
from logging_config import get_logger

logger = get_logger(__name__)

MODEL_ID = "google/siglip2-so400m-patch14-384"


class SigLIPCardSearch:
    """Load a SigLIP2 LoRA gallery embedding index and search it with cosine similarity."""

    def __init__(self, vectors_path=None, lora_path=None):
        self.repo_path = Path(vectors_path or settings.siglip_vectors_path)
        self.lora_path = Path(lora_path) if lora_path else self.repo_path / "lora_best"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.database: dict[str, np.ndarray] = {}
        self._db_array = None
        self._db_ids = None
        self.update_task = None

        self._load_model()
        self.load_database()

    def _load_model(self):
        logger.info("Loading SigLIP2 base model + LoRA adapter from %s", self.lora_path)
        model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(self.device)
        if self.lora_path.exists():
            model.vision_model = PeftModel.from_pretrained(model.vision_model, str(self.lora_path))
            model.vision_model = model.vision_model.merge_and_unload()
        else:
            logger.warning("LoRA adapter not found at %s; using base SigLIP2 zero-shot", self.lora_path)
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

    def load_database(self):
        self.database = {}
        embeddings_file = self.repo_path / "embeddings.pt"
        if not embeddings_file.exists():
            logger.warning("SigLIP vector index not found at %s", embeddings_file)
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
        # No automatic repo sync -- this is a locally-built artifact, not an
        # externally cloned repo (see module docstring). Reload picks up
        # whatever's on disk if the artifact is ever refreshed in place.
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

    def search(self, query_image, top_k=5):
        if self._db_array is None:
            return []
        query = self.encode(query_image).to(torch.float16)
        scores = self._db_array @ query
        count = min(max(int(top_k), 0), len(self._db_ids))
        if not count:
            return []
        values, indices = torch.topk(scores, count)
        return [(str(self._db_ids[int(index)]), float(value))
                for value, index in zip(values.cpu(), indices.cpu())]

    def search_verified(self, query_image, top_k=5, rerank_k=None):
        """API-compatible fallback -- a global embedding has no keypoints to
        run RANSAC geometric verification against, so this matches
        VLADCardSearch.search_verified()'s own current fallback behavior
        for the CudaSift-tagged deployment (inliers=0), not a regression."""
        return [(card_id, similarity, 0) for card_id, similarity in self.search(query_image, top_k)]

    def compare_images(self, image1, image2):
        a = self.encode(image1)
        b = self.encode(image2)
        return float((a * b).sum())

    def batch_search(self, images, top_k=5):
        return [self.search(image, top_k) for image in images]
