from pathlib import Path

from api import archive_scan_image


def test_archive_scan_image_saves_uploaded_bytes_in_configured_directory(tmp_path, monkeypatch):
    monkeypatch.setattr("api.settings.scan_archive_dir", str(tmp_path))

    saved = archive_scan_image(b"uploaded-card-image", "front scan.JPG")

    assert saved.parent == tmp_path
    assert saved.suffix == ".jpg"
    assert saved.read_bytes() == b"uploaded-card-image"
    assert saved.stat().st_mode & 0o777 == 0o600
