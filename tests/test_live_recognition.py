"""Contracts for live WebSocket card recognition."""
import cv2
import numpy as np
from fastapi.testclient import TestClient

from api import app


def _jpeg_bytes() -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


class _Scanner:
    def new_tracker(self):
        return object()  # opaque; just needs to round-trip through scan()

    def scan(self, image, k, verify=False, tracker=None):
        assert image.shape == (8, 8, 3)
        assert k == 3
        assert tracker is not None
        return [
            {"box": [0, 0, 8, 8], "track_id": 1, "matches": [{"card_id": 42, "similarity": 0.8}]},
            {"box": [4, 4, 8, 8], "track_id": 2, "matches": [{"card_id": 99, "similarity": 0.5}]},
        ]


class _Database:
    is_initialized = True

    def return_columns(self):
        return ["product_id", "clean_name"]

    async def query_by_id(self, product_id):
        return (product_id, "Test Card") if product_id == 42 else None

    async def query_variants_by_id(self, product_id):
        if product_id != 42:
            return []
        return [{"sub_type_name": "Normal", "low_price": 0.1, "mid_price": 0.2,
                 "high_price": 0.3, "market_price": 0.25, "direct_low_price": 0.15}]


def _client_with_live_dependencies():
    app.state.scanner = _Scanner()
    app.state.db = _Database()
    return TestClient(app)


def test_live_recognize_websocket_returns_per_card_track_ids_and_raw_similarity():
    client = _client_with_live_dependencies()
    try:
        with client.websocket_connect("/live-recognize") as websocket:
            websocket.send_bytes(_jpeg_bytes())
            response = websocket.receive_json()
    finally:
        client.close()

    assert response["type"] == "result"
    assert response["results"] == [
        {
            "card_id": 42,
            "track_id": 1,
            "box": [0, 0, 8, 8],
            "details": {"product_id": 42, "clean_name": "Test Card"},
            "variants": [{"sub_type_name": "Normal", "low_price": 0.1, "mid_price": 0.2,
                          "high_price": 0.3, "market_price": 0.25, "direct_low_price": 0.15}],
            "similarity": 0.8,
        },
        {
            "card_id": 99,
            "track_id": 2,
            "box": [4, 4, 8, 8],
            "details": None,
            "variants": [],
            "similarity": 0.5,
        },
    ]


def test_live_recognize_websocket_rejects_frames_above_twenty_per_second():
    client = _client_with_live_dependencies()
    try:
        with client.websocket_connect("/live-recognize") as websocket:
            websocket.send_bytes(_jpeg_bytes())
            assert websocket.receive_json()["type"] == "result"
            websocket.send_bytes(_jpeg_bytes())
            response = websocket.receive_json()
    finally:
        client.close()

    assert response == {"type": "error", "error": "frame_rate_exceeded", "max_fps": 20}


def test_live_recognize_websocket_rejects_non_jpeg_binary_data():
    client = _client_with_live_dependencies()
    try:
        with client.websocket_connect("/live-recognize") as websocket:
            websocket.send_bytes(b"not-a-jpeg")
            response = websocket.receive_json()
    finally:
        client.close()

    assert response == {"type": "error", "error": "invalid_jpeg"}


def test_live_recognize_websocket_uses_one_isolated_tracker_per_connection():
    """Each connection must get its own tracker (new_tracker() called once per
    connection, not shared/reused across connections or per-frame)."""
    calls = []

    class _CountingScanner(_Scanner):
        def new_tracker(self):
            tracker = object()
            calls.append(tracker)
            return tracker

    app.state.scanner = _CountingScanner()
    app.state.db = _Database()
    client = TestClient(app)
    try:
        with client.websocket_connect("/live-recognize") as ws1:
            ws1.send_bytes(_jpeg_bytes())
            ws1.receive_json()
        with client.websocket_connect("/live-recognize") as ws2:
            ws2.send_bytes(_jpeg_bytes())
            ws2.receive_json()
    finally:
        client.close()

    assert len(calls) == 2
    assert calls[0] is not calls[1]
