"""Pinned deployment verification without downloading models or using a GPU."""
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

import csr_artifacts as artifacts

_verify_scanner_revision = artifacts.verify_scanner_revision


def archive(path, files):
    with tarfile.open(path, "w") as tar:
        for name, data in files.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            tar.addfile(entry, io.BytesIO(data))


def bundle_fixture(root, gallery=b"gallery"):
    root.mkdir(parents=True, exist_ok=True)
    archive(root / "base.tar", {"config.json": b"base config", "model.safetensors": b"weights"})
    archive(root / "lora.tar", {"adapter_config.json": b"lora config", "adapter_model.safetensors": b"lora"})
    archive(root / "processor.tar", {
        "preprocessor_config.json": b"processor config",
        "csr-preprocessing.json": json.dumps(artifacts.PREPROCESSING).encode(),
    })
    (root / "detector.pt").write_bytes(b"detector")
    (root / "embeddings.pt").write_bytes(gallery)
    m = {"schema_version": 1, "scanner_revision": "a" * 40,
         "input_contract": "csr-siglip-rgb-v1"}
    for name, path in zip(artifacts.ARTIFACTS, ("detector.pt", "processor.tar", "base.tar", "lora.tar", "embeddings.pt")):
        m[name] = {"path": path, "sha256": hashlib.sha256((root / path).read_bytes()).hexdigest(),
                   "revision": "b" * 40}
    m.update(gallery_base_sha256=m["base_model"]["sha256"],
             gallery_lora_sha256=m["lora"]["sha256"],
             gallery_preprocessing_sha256=m["preprocessing"]["sha256"])
    path = root / "manifest.json"
    # Deliberately reverse fields: manifest file formatting is not its identity.
    path.write_text(json.dumps(dict(reversed(list(m.items()))), indent=2))
    return path, m


@pytest.fixture(autouse=True)
def allow_fixture_revision(monkeypatch):
    monkeypatch.setattr(artifacts, "verify_scanner_revision", lambda revision: None)


def test_offline_snapshot_restart_refresh_and_rollback(tmp_path):
    old_path, m = bundle_fixture(tmp_path / "old")
    new_path, _ = bundle_fixture(tmp_path / "new", gallery=b"new gallery")
    old = artifacts.ArtifactBundle(str(old_path))
    restarted = artifacts.ArtifactBundle(str(old_path))
    new = artifacts.ArtifactBundle(str(new_path))
    rollback = artifacts.ArtifactBundle(str(old_path))
    try:
        assert old.digest == restarted.digest == rollback.digest == artifacts.manifest_digest(m)
        assert new.digest != old.digest
        assert old.paths["gallery"].read_bytes() == b"gallery"
        assert (old.paths["base_model"] / "model.safetensors").read_bytes() == b"weights"
        # The actual loaders read the private snapshot, not mutable source paths.
        (old_path.parent / "embeddings.pt").write_bytes(b"mutated after startup")
        assert old.paths["gallery"].read_bytes() == b"gallery"
        with pytest.raises(ValueError, match="checksum"):
            artifacts.ArtifactBundle(str(old_path))
    finally:
        for bundle in (old, restarted, new, rollback):
            snapshot = bundle.paths["gallery"].parent
            bundle.close()
            assert not snapshot.exists()


@pytest.mark.parametrize("change", [
    lambda m: m.update(schema_version=True),
    lambda m: m.update(scanner_revision="main"),
    lambda m: m.update(input_contract="gray"),
    lambda m: m.update(unexpected="field"),
    lambda m: m["base_model"].update(revision="main"),
    lambda m: m["gallery"].update(path="../outside.pt"),
    lambda m: m["gallery"].update(path="/outside.pt"),
    lambda m: m["gallery"].update(path="."),
    lambda m: m["gallery"].update(sha256="A" * 64),
    lambda m: m.update(gallery_lora_sha256="0" * 64),
])
def test_invalid_manifest_fails_closed(tmp_path, change):
    path, m = bundle_fixture(tmp_path)
    change(m)
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError):
        artifacts.ArtifactBundle(str(path))


def test_missing_file_and_symlink_escape_fail_closed(tmp_path):
    path, m = bundle_fixture(tmp_path / "bundle")
    gallery = path.parent / m["gallery"]["path"]
    gallery.unlink()
    with pytest.raises(FileNotFoundError):
        artifacts.ArtifactBundle(str(path))
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"gallery")
    gallery.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        artifacts.ArtifactBundle(str(path))


@pytest.mark.parametrize("name,kind", [("../escaped", "file"), ("/escaped", "file"),
                                       ("link", "symlink"), ("hardlink", "hardlink")])
def test_unsafe_archive_fails_closed(tmp_path, name, kind):
    path, m = bundle_fixture(tmp_path)
    base = tmp_path / m["base_model"]["path"]
    with tarfile.open(base, "w") as tar:
        entry = tarfile.TarInfo(name)
        if kind == "symlink":
            entry.type = tarfile.SYMTYPE
            entry.linkname = "../outside"
        elif kind == "hardlink":
            entry.type = tarfile.LNKTYPE
            entry.linkname = "../outside"
        tar.addfile(entry)
    m["base_model"]["sha256"] = hashlib.sha256(base.read_bytes()).hexdigest()
    m["gallery_base_sha256"] = m["base_model"]["sha256"]
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError):
        artifacts.ArtifactBundle(str(path))


def test_preprocessing_contract_is_checked(tmp_path):
    path, m = bundle_fixture(tmp_path)
    processor = tmp_path / m["preprocessing"]["path"]
    archive(processor, {"csr-preprocessing.json": b'{"channel_conversion":"grayscale"}'})
    m["preprocessing"]["sha256"] = hashlib.sha256(processor.read_bytes()).hexdigest()
    m["gallery_preprocessing_sha256"] = m["preprocessing"]["sha256"]
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="preprocessing"):
        artifacts.ArtifactBundle(str(path))


def test_duplicate_and_oversized_manifest_rejected(tmp_path):
    path, _ = bundle_fixture(tmp_path)
    path.write_text('{"schema_version":1,"schema_version":2}')
    with pytest.raises(ValueError, match="duplicate"):
        artifacts.ArtifactBundle(str(path))
    path.write_bytes(b" " * (64 * 1024 + 1))
    with pytest.raises(ValueError, match="large"):
        artifacts.ArtifactBundle(str(path))


def test_canonical_digest_matches_go_contract():
    manifest = artifacts.read_manifest(Path(__file__).parent / "fixtures/csr-manifest.json")
    # Shared golden with CSR's TestScannerManifestCanonicalDigest. It includes
    # Unicode and HTML-sensitive path characters that Go escapes in JSON.
    assert artifacts.manifest_digest(manifest) == "e76b01846a12db6df32f8ada48ec10d641fd2728cac43f98099d7bb422c47428"


@pytest.mark.parametrize("revision,dirty,tracked,ok", [
    ("a" * 40, "", "csr_artifacts.py\n", True),
    ("b" * 40, "", "csr_artifacts.py\n", False),
    ("a" * 40, " M scanner.py\n", "csr_artifacts.py\n", False),
    ("a" * 40, "M  csr_artifacts.py\n", "csr_artifacts.py\n", False),
    ("a" * 40, "", "", False),
])
def test_source_revision_must_match_clean_committed_loader(monkeypatch, revision, dirty, tracked, ok):
    outputs = iter((revision, dirty, tracked))
    monkeypatch.setattr(artifacts.subprocess, "check_output", lambda *args, **kwargs: next(outputs))
    if ok:
        _verify_scanner_revision("a" * 40)
    else:
        with pytest.raises(ValueError):
            _verify_scanner_revision("a" * 40)
