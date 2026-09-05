"""Contract for the minimum-similarity floor (config.min_match_similarity):
a detected region that doesn't resemble anything real in the gallery
should report no match at all, not a false-confident "closest available"
one.

Real incident this guards against: a handful of gallery images turned out
to be a generic "Image Coming Soon" placeholder graphic (TCGplayer served
a 200 OK containing it instead of a 403 for a few photo-less products),
which acted as a false attractor for completely unrelated crops (a glue
gun box, a PSU fan, ...) at 50-65% similarity -- every one of them
returned a plausible-looking top-1 match with no way to tell it was
meaningless."""
import torch

from siglip_matcher import SigLIPCardSearch
from config import settings


def _fake_matcher(scores):
    matcher = object.__new__(SigLIPCardSearch)
    matcher._db_ids = [str(i) for i in range(len(scores))]
    matcher._db_array = torch.tensor(scores, dtype=torch.float16).unsqueeze(1)
    matcher.encode = lambda image: torch.tensor([1.0], dtype=torch.float16)
    return matcher


def test_top_k_mode_drops_matches_below_the_floor():
    below = settings.min_match_similarity - 0.05
    matcher = _fake_matcher([0.95, below])

    results = matcher.search(None, top_k=2)

    assert [card_id for card_id, _ in results] == ["0"]


def test_top_k_mode_returns_nothing_when_even_the_best_is_below_the_floor():
    matcher = _fake_matcher([settings.min_match_similarity - 0.05,
                              settings.min_match_similarity - 0.10])

    results = matcher.search(None, top_k=2)

    assert results == []


def test_margin_mode_does_not_pull_in_matches_below_the_floor():
    """A tight margin_pct around a below-floor top score must not smuggle
    a below-floor match through via the relative-margin path."""
    below = settings.min_match_similarity - 0.05
    matcher = _fake_matcher([below, below - 0.001])

    results = matcher.search(None, top_k=None, margin_pct=50.0)

    assert results == []


def test_genuine_high_confidence_match_is_unaffected():
    matcher = _fake_matcher([0.95, 0.94])

    results = matcher.search(None, top_k=2)

    assert [card_id for card_id, _ in results] == ["0", "1"]
