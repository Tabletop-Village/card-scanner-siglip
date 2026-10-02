"""Exercise real pinned startup and API headers with lightweight model loaders."""
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
from PIL import Image
import pytest
import torch
from fastapi.testclient import TestClient

import api
import csr_artifacts
import scanner as scanner_module
import siglip_matcher
from test_csr_artifacts import bundle_fixture


@pytest.fixture
def loaded_scanner(tmp_path, monkeypatch):
    path, _ = bundle_fixture(tmp_path)
    monkeypatch.setattr(csr_artifacts, "verify_scanner_revision", lambda revision: None)
    monkeypatch.setattr(scanner_module.settings, "csr_manifest", str(path))
    calls = {}
    model = SimpleNamespace(vision_model=object(), eval=lambda: None)
    model.to = lambda device: model

    def base(path, **kwargs):
        assert kwargs["local_files_only"] is True
        calls["base_model"] = path
        return model

    def lora(vision, path, **kwargs):
        assert vision is model.vision_model
        assert kwargs["local_files_only"] is True
        calls["lora"] = path
        return SimpleNamespace(merge_and_unload=lambda: vision)

    def processor(path, **kwargs):
        assert kwargs == {"local_files_only": True}
        calls["preprocessing"] = path
        return object()

    def detector(path):
        calls["detector"] = path
        return SimpleNamespace(to=lambda device: None)

    def gallery(path, **kwargs):
        calls["gallery"] = str(path)
        assert kwargs == {"map_location": "cpu", "weights_only": True}
        return {"ids": torch.tensor([123]), "embeds": torch.tensor([[1., 0.]])}

    monkeypatch.setattr(scanner_module, "YOLO", detector)
    monkeypatch.setattr(siglip_matcher.AutoModel, "from_pretrained", base)
    monkeypatch.setattr(siglip_matcher.AutoProcessor, "from_pretrained", processor)
    monkeypatch.setattr(siglip_matcher.PeftModel, "from_pretrained", lora)
    monkeypatch.setattr(siglip_matcher.torch, "load", gallery)
    # None of the pinned loaders may use the mutable HF/local override path.
    def network(*args, **kwargs):
        pytest.fail("pinned startup attempted a Hub download")
    monkeypatch.setattr(scanner_module, "hf_hub_download", network)
    monkeypatch.setattr(siglip_matcher, "hf_hub_download", network)
    loaded = scanner_module.Scanner()
    yield loaded, calls, path
    loaded.close()


def test_attestation_uses_actual_snapshot_and_pinned_gallery_never_reloads(loaded_scanner):
    loaded, calls, _ = loaded_scanner
    bundle = loaded._artifact_bundle
    assert set(calls) == set(csr_artifacts.ARTIFACTS)
    for name, path in calls.items():
        assert path == str(bundle.paths[name])
    assert loaded.artifact_digest == bundle.digest
    loaded.start_scheduled_updates()
    assert loaded.matcher.update_task is None
    for reload in (loaded.matcher.reload_database, loaded.matcher.sync_and_reload):
        with pytest.raises(RuntimeError, match="restart"):
            reload()
    assert loaded.artifact_digest == bundle.digest
    loaded.close()
    assert loaded.artifact_digest is None


class Cursor:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def fetchone(self):
        return (1,)


@pytest.fixture
def client(loaded_scanner, monkeypatch):
    loaded, _, _ = loaded_scanner
    db = SimpleNamespace(
        is_initialized=True,
        conn=SimpleNamespace(execute=lambda sql: Cursor()),
        return_columns=lambda: ["product_id"],
        query_by_id=AsyncMock(return_value=None),
        query_variants_by_id=AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(api.app.state, "scanner", loaded, raising=False)
    monkeypatch.setattr(api.app.state, "db", db, raising=False)
    monkeypatch.setattr(api.limiter, "enabled", False)
    monkeypatch.setattr(api.settings, "api_keys_str", "secret")
    # No lifespan: model startup above is exercised without a real SQLite sync.
    client = TestClient(api.app)
    try:
        yield client, loaded
    finally:
        client.close()


def color_png():
    output = io.BytesIO()
    Image.new("RGB", (3, 1), color="red").save(output, format="PNG")
    return output.getvalue()


@pytest.mark.parametrize("matches", [[], [("123", 0.97)]])
def test_ready_and_identify_attest_loaded_bundle_including_no_match(client, monkeypatch, matches):
    client, loaded = client
    def search(image, top_k, **kwargs):
        # The real identify_card resize retains color; OpenCV's BGR is explicit.
        assert image.shape == (558, 400, 3)
        assert np.array_equal(image[0, 0], [0, 0, 255])
        return matches
    monkeypatch.setattr(loaded.matcher, "search", search)
    ready = client.get("/ready", headers={"X-CSR-Artifact-SHA256": "forged"})
    assert ready.status_code == 200
    assert ready.headers["X-CSR-Artifact-SHA256"] == loaded.artifact_digest
    result = client.post("/identify?verify=false&margin_pct=5&min_similarity=0.5",
                         files={"image": ("card.png", color_png(), "image/png")},
                         headers={"X-API-Key": "secret", "X-CSR-Artifact-SHA256": "forged"})
    assert result.status_code == 200
    assert result.headers["X-CSR-Artifact-SHA256"] == ready.headers["X-CSR-Artifact-SHA256"]
    assert len(result.json()) == len(matches)


def test_unpinned_mode_does_not_claim_attestation(client, monkeypatch):
    client, loaded = client
    monkeypatch.setattr(loaded, "_artifact_bundle", None)
    monkeypatch.setattr(loaded.matcher, "search", lambda *args, **kwargs: [])
    for result in (client.get("/ready"), client.post(
            "/identify", files={"image": ("card.png", color_png(), "image/png")},
            headers={"X-API-Key": "secret"})):
        assert result.status_code == 200
        assert "X-CSR-Artifact-SHA256" not in result.headers


def test_failed_or_unauthorized_requests_do_not_attest(client, monkeypatch):
    client, loaded = client
    unauthorized = client.post("/identify", files={"image": ("card.png", color_png(), "image/png")})
    assert unauthorized.status_code == 401
    malformed = client.post("/identify", files={"image": ("card.png", b"bad", "image/png")},
                            headers={"X-API-Key": "secret"})
    assert malformed.status_code == 400
    monkeypatch.setattr(loaded.matcher, "database", {})
    not_ready = client.get("/ready")
    assert not_ready.status_code == 503
    for result in (unauthorized, malformed, not_ready):
        assert "X-CSR-Artifact-SHA256" not in result.headers


def test_pinned_artifacts_cannot_be_overridden(loaded_scanner):
    with pytest.raises(ValueError, match="overridden"):
        scanner_module.Scanner(model_path="different.pt")


@pytest.mark.parametrize("failure", ["model", "empty_gallery"])
def test_failed_pinned_load_cleans_snapshot_and_never_attests(loaded_scanner, monkeypatch, failure):
    observed = []
    real_bundle = scanner_module.ArtifactBundle

    def observe(path):
        bundle = real_bundle(path)
        observed.append(bundle.paths["gallery"].parent)
        return bundle

    monkeypatch.setattr(scanner_module, "ArtifactBundle", observe)
    if failure == "model":
        def fail(*args, **kwargs):
            raise ValueError("unloadable pinned model")
        monkeypatch.setattr(siglip_matcher.AutoModel, "from_pretrained", fail)
    else:
        monkeypatch.setattr(siglip_matcher.torch, "load", lambda *args, **kwargs: {
            "ids": torch.tensor([], dtype=torch.int64), "embeds": torch.empty((0, 2))})
    with pytest.raises(ValueError, match="pinned"):
        scanner_module.Scanner()
    assert len(observed) == 1
    assert not observed[0].exists()


@pytest.mark.parametrize("matches", [[], [{"card_id": "123", "similarity": .97}]])
def test_full_frame_scan_attests_and_requests_largest_card(client, monkeypatch, matches):
    client, loaded = client
    def scan(image, **kwargs):
        assert kwargs["largest_only"] is True
        assert kwargs["k"] is None
        assert kwargs["margin_pct"] == 5
        assert kwargs["min_similarity"] == .5
        assert np.array_equal(image[0, 0], [0, 0, 255])
        return [{"box": [0, 0, 3, 1], "matches": matches}]
    monkeypatch.setattr(loaded, "scan", scan)
    monkeypatch.setattr(api, "archive_scan_image", lambda *args: "local-test")
    response = client.post("/scan?largest_only=true&verify=false&margin_pct=5&min_similarity=0.5",
                           files={"image": ("frame.png", color_png(), "image/png")},
                           headers={"X-API-Key": "secret"})
    assert response.status_code == 200
    assert response.headers["X-CSR-Artifact-SHA256"] == loaded.artifact_digest
    assert len(response.json()) == len(matches)
