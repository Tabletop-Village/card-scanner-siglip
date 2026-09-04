from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_upload_frontend_posts_selected_image_to_scan_endpoint():
    page = (ROOT / "static" / "upload.html").read_text()

    assert 'id="imageInput"' in page
    assert 'accept="image/*"' in page
    assert 'new FormData()' in page
    assert 'formData.append("image", file)' in page
    assert 'fetch("/scan"' in page
    assert 'id="results"' in page
    assert 'Drop an image' in page


def test_upload_frontend_renders_match_images_with_an_enlarge_dialog():
    page = (ROOT / "static" / "upload.html").read_text()

    assert 'details.image_url' in page
    assert 'class="card-art"' in page
    assert 'id="imageDialog"' in page
    assert 'imageDialog.showModal()' in page
