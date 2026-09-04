from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_label_review_frontend_has_visual_comparison_and_edit_controls():
    page = (ROOT / "static" / "label-review.html").read_text()

    assert 'id="scanImage"' in page
    assert 'id="catalogImage"' in page
    assert 'id="productSearch"' in page
    assert 'id="saveChoice"' in page
    assert 'fetch("/label-review/labels")' in page
    assert 'fetch("/label-review/products?q="' in page
    assert 'fetch(`/label-review/labels/${encodeURIComponent(current.filename)}`' in page


def test_label_review_frontend_explains_forced_low_choices():
    page = (ROOT / "static" / "label-review.html").read_text()

    assert 'forced_low' in page
    assert 'Original agent choice' in page
    assert 'Review status' in page
