"""Static contract for browser-side live webcam recognition."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_webcam_ui_streams_jpegs_at_ten_fps_without_request_backlog():
    page = (ROOT / "static" / "index.html").read_text()

    assert 'new WebSocket(`${wsScheme}://${location.host}/live-recognize`)' in page
    assert "setInterval(sendLiveFrame, currentPollInterval())" in page
    assert 'if (liveSending || liveSocket.readyState !== WebSocket.OPEN)' in page
    assert "liveSending = true" in page
    assert "liveSending = false" in page
    assert "stable_percentage" in page
    assert "raw_percentage" in page
    assert "function scan()" in page  # Manual capture fallback remains available.
    assert 'id="pollInterval"' in page
    assert "pollInterval.addEventListener('input'" in page
    assert 'id="stablePeriod"' in page
    assert "stablePeriod.addEventListener('input'" in page
    assert "set_stable_period" in page
    assert "setInterval(sendLiveFrame, currentPollInterval())" in page


def test_live_results_update_in_place_without_entrance_animation():
    page = (ROOT / "static" / "index.html").read_text()

    # A 10 Hz stream must keep existing cards mounted; re-creating them runs the
    # entrance animation every response and makes the panel visually jumpy.
    assert "function createResultCard" in page
    assert "function updateResultCard" in page
    assert "resultsList.innerHTML = lastResults.map" not in page
    assert "animation: fadeUp" not in page
    assert "transition: width 0.5s ease" not in page
    # A transient miss must retain the current stable cards rather than showing
    # the flexing empty state above them (which pushes them to the bottom).
    assert "if (lastResults.length) return;" in page
