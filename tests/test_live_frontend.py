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
    assert "track_id" in page
    assert "function scan()" in page  # Manual capture fallback remains available.
    assert 'id="pollInterval"' in page
    assert "pollInterval.addEventListener('input'" in page
    assert 'id="stablePeriod"' not in page  # Server-side smoothing was dropped; no more slider.
    assert "set_stable_period" not in page
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


def test_live_results_are_keyed_by_track_id_not_card_id():
    page = (ROOT / "static" / "index.html").read_text()

    # Each physical card in view (track_id) gets its own row, even if two
    # distinct cards were (mis)identified as the same card_id -- rows must
    # not be deduped/keyed by card_id alone anymore.
    assert "function rowKey(r)" in page
    assert "r.track_id != null" in page
    assert "dataset.rowKey" in page
    assert "dataset.cardId" not in page
    # The old fixed 3-row cap was a per-card top-k limit, not a total-cards
    # limit -- it must not silently truncate the list of distinct tracks.
    assert ".slice(0, 3)" not in page
