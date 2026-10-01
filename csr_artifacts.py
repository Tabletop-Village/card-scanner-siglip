"""Verified, offline artifact snapshots for the CSR inference contract.

The header identifies this snapshot only after Scanner has loaded its files.
It is never copied from a caller's header or a configured expected digest.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
from tempfile import TemporaryDirectory

ARTIFACTS = ("detector", "preprocessing", "base_model", "lora", "gallery")
# Match internal/siglip/artifacts.go's struct field order and JSON escaping.
MANIFEST_FIELDS = (
    "schema_version", "scanner_revision", *ARTIFACTS,
    "gallery_base_sha256", "gallery_lora_sha256",
    "gallery_preprocessing_sha256", "input_contract",
)
PREPROCESSING = {
    "schema_version": 1,
    "input_contract": "csr-siglip-rgb-v1",
    "crop_width": 400,
    "crop_height": 558,
    "resize_interpolation": "opencv_INTER_LINEAR",
    "channel_conversion": "BGR_to_RGB",
    "embedding_normalization": "L2",
    "match_margin_pool_size": 30,
}


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate manifest field: {key}")
        result[key] = value
    return result


def read_manifest(path: Path) -> dict:
    with path.open("rb") as handle:
        data = handle.read(64 * 1024 + 1)
    if len(data) > 64 * 1024:
        raise ValueError("manifest too large")
    m = json.loads(data, object_pairs_hook=_json_object)
    if not isinstance(m, dict) or set(m) != set(MANIFEST_FIELDS):
        raise ValueError("unknown or missing manifest fields")
    if (type(m["schema_version"]) is not int or m["schema_version"] != 1
            or m["input_contract"] != "csr-siglip-rgb-v1"
            or not _hex(m["scanner_revision"], 40)):
        raise ValueError("unsupported scanner manifest or input contract")
    for name in ARTIFACTS:
        a = m[name]
        if not isinstance(a, dict) or set(a) != {"path", "sha256", "revision"}:
            raise ValueError("unknown or missing artifact fields")
        if not _hex(a["sha256"], 64) or not _hex(a["revision"], 40):
            raise ValueError("artifact requires SHA-256 and immutable source commit")
        _relative_path(a["path"])
    for field, name in (("gallery_base_sha256", "base_model"),
                        ("gallery_lora_sha256", "lora"),
                        ("gallery_preprocessing_sha256", "preprocessing")):
        if m[field] != m[name]["sha256"]:
            raise ValueError("gallery embedding artifacts are incompatible")
    return m


def _hex(value, length):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{%d}" % length, value) is not None


def _relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise ValueError("artifact requires a local relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) == ".":
        raise ValueError("artifact requires a local relative path")
    return path


def manifest_digest(m: dict) -> str:
    canonical = {}
    for field in MANIFEST_FIELDS:
        value = m[field]
        if field in ARTIFACTS:
            value = {key: value[key] for key in ("path", "sha256", "revision")}
        canonical[field] = value
    data = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))
    for char, escape in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"),
                         ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        data = data.replace(char, escape)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def verify_scanner_revision(expected: str):
    """CSR pinned mode runs from a clean committed scanner checkout."""
    root = Path(__file__).resolve().parent
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if revision != expected:
        raise ValueError("scanner source revision mismatch")
    # Verify all tracked source/config changes, including staged changes.
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=root, text=True).strip():
        raise ValueError("pinned scanner checkout has uncommitted changes")
    tracked = subprocess.check_output(["git", "ls-files", "--error-unmatch",
                                      "csr_artifacts.py"], cwd=root, text=True)
    if not tracked.strip():
        raise ValueError("artifact loader is not part of the pinned source commit")


def _copy_verified(source: Path, destination: Path, expected: str):
    digest = hashlib.sha256()
    with source.open("rb") as src, destination.open("xb") as dst:
        if not source.is_file():
            raise ValueError("artifact must be a regular file")
        while chunk := src.read(1024 * 1024):
            digest.update(chunk)
            dst.write(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"artifact checksum mismatch: {source.name}")


def _extract(archive: Path, destination: Path):
    """Model archives contain regular files/directories only, with no links."""
    destination.mkdir()
    seen = set()
    with tarfile.open(archive, "r:*") as tar:
        for entry in tar:
            name = _relative_path(entry.name)
            if name in seen or not (entry.isfile() or entry.isdir()):
                raise ValueError("unsupported or duplicate model archive entry")
            seen.add(name)
            target = destination / str(name)
            if entry.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(entry) as src, target.open("xb") as dst:
                    shutil.copyfileobj(src, dst)


class ArtifactBundle:
    """A private verified snapshot; model loaders never revisit mutable inputs."""

    def __init__(self, manifest_path: str):
        self._snapshot = TemporaryDirectory(prefix="csr-scanner-")
        try:
            path = Path(manifest_path).resolve(strict=True)
            manifest = read_manifest(path)
            verify_scanner_revision(manifest["scanner_revision"])
            root = path.parent
            snapshot = Path(self._snapshot.name)
            self.paths = {}
            for name in ARTIFACTS:
                artifact = manifest[name]
                source = (root / artifact["path"]).resolve(strict=True)
                if not source.is_relative_to(root) or not source.is_file():
                    raise ValueError("artifact escapes bundle or is not a regular file")
                copied = snapshot / (name + ".artifact")
                _copy_verified(source, copied, artifact["sha256"])
                if name in ("base_model", "lora", "preprocessing"):
                    extracted = snapshot / name
                    _extract(copied, extracted)
                    copied.unlink()
                    self.paths[name] = extracted
                else:
                    # YOLO determines task/file format from the .pt suffix.
                    renamed = snapshot / (name + ".pt")
                    copied.rename(renamed)
                    self.paths[name] = renamed
            preprocessing = json.loads(
                (self.paths["preprocessing"] / "csr-preprocessing.json").read_text())
            if preprocessing != PREPROCESSING:
                raise ValueError("unsupported pinned preprocessing contract")
            self.digest = manifest_digest(manifest)
        except BaseException:
            self.close()
            raise

    def close(self):
        self._snapshot.cleanup()
