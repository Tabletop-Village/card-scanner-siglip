"""Contracts for margin-based matching (top_k=None): when a caller leaves
top_n blank, search() should return every gallery match within a
percentage-point margin of the best match instead of a fixed count."""
import torch

from siglip_matcher import SigLIPCardSearch


def _fake_matcher(scores):
    """A SigLIPCardSearch with a rigged 1-D embedding space, so `scores`
    (already cosine similarities in [-1, 1]) come out of the dot product
    unchanged and no model/GPU is needed."""
    matcher = object.__new__(SigLIPCardSearch)
    matcher._db_ids = [str(i) for i in range(len(scores))]
    matcher._db_array = torch.tensor(scores, dtype=torch.float16).unsqueeze(1)
    matcher.encode = lambda image: torch.tensor([1.0], dtype=torch.float16)
    return matcher


def test_margin_mode_returns_every_match_within_percentage_points_of_best():
    matcher = _fake_matcher([0.95, 0.94, 0.90, 0.50])

    results = matcher.search(None, top_k=None, margin_pct=2.0)

    assert {card_id for card_id, _ in results} == {"0", "1"}


def test_margin_mode_falls_back_to_configured_default_margin():
    matcher = _fake_matcher([0.95, 0.94, 0.90, 0.50])

    results = matcher.search(None, top_k=None)  # margin_pct omitted too

    from config import settings
    expected = {i for i, s in enumerate([0.95, 0.94, 0.90, 0.50])
                if s >= 0.95 - settings.match_margin_pct / 100.0}
    assert {int(card_id) for card_id, _ in results} == expected


def test_explicit_top_k_is_unaffected_by_margin_mode():
    matcher = _fake_matcher([0.95, 0.94, 0.90, 0.50])

    results = matcher.search(None, top_k=2)

    assert [card_id for card_id, _ in results] == ["0", "1"]


def test_search_verified_forwards_margin_pct():
    matcher = _fake_matcher([0.95, 0.94, 0.90, 0.50])

    results = matcher.search_verified(None, top_k=None, margin_pct=2.0)

    assert {card_id for card_id, _, inliers in results} == {"0", "1"}
    assert all(inliers == 0 for _, _, inliers in results)
