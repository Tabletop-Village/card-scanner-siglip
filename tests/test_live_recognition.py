"""Contracts for live WebSocket card recognition and its stability window."""
import io

import cv2
import numpy as np
from fastapi.testclient import TestClient

from api import app
from live_recognition import RollingMatchAggregator


def _jpeg_bytes() -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


class _Scanner:
    def scan(self, image, k, verify=False):
        assert image.shape == (8, 8, 3)
        assert k == 3
        return [{"box": [0, 0, 8, 8], "matches": [
            {"card_id": 42, "similarity": 0.8},
            {"card_id": 99, "similarity": 0.5},
        ]}]


class _Database:
    is_initialized = True

    def return_columns(self):
        return ["product_id", "clean_name"]

    async def query_by_id(self, product_id):
        return (product_id, "Test Card") if product_id == 42 else None


def _client_with_live_dependencies():
    app.state.scanner = _Scanner()
    app.state.db = _Database()
    return TestClient(app)


def test_rolling_match_aggregator_uses_linear_recency_weight_within_one_second():
    """Weight is similarity * (1 - age/window), with stale observations removed."""
    aggregator = RollingMatchAggregator(window_seconds=1.0)
    aggregator.add([{ "card_id": 42, "similarity": 0.60 }], observed_at=0.0)
    stable = aggregator.add([{ "card_id": 42, "similarity": 1.00 }], observed_at=0.5)

    # At t=.5: weights are .5 and 1.0: (.60*.5 + 1*1) / 1.5 = .866666...
    assert stable == {42: 0.8666666666666667}

    assert aggregator.add([], observed_at=1.51) == {}


def test_live_recognize_websocket_returns_enriched_raw_and_stable_percentages():
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
            "box": [0, 0, 8, 8],
            "details": {"product_id": 42, "clean_name": "Test Card"},
            "raw_percentage": 80.0,
            "stable_percentage": 80.0,
        },
        {
            "card_id": 99,
            "box": [0, 0, 8, 8],
            "details": None,
            "raw_percentage": 50.0,
            "stable_percentage": 50.0,
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


def test_live_recognize_websocket_allows_per_connection_stable_period_configuration():
    client = _client_with_live_dependencies()
    try:
        with client.websocket_connect("/live-recognize") as websocket:
            websocket.send_json({"type": "set_stable_period", "seconds": 2.5})
            response = websocket.receive_json()
    finally:
        client.close()

    assert response == {"type": "configured", "stable_period_seconds": 2.5}


def test_live_recognize_websocket_rejects_non_jpeg_binary_data():
    client = _client_with_live_dependencies()
    try:
        with client.websocket_connect("/live-recognize") as websocket:
            websocket.send_bytes(b"not-a-jpeg")
            response = websocket.receive_json()
    finally:
        client.close()

    assert response == {"type": "error", "error": "invalid_jpeg"}
