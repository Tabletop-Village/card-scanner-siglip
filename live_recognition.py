"""Per-connection rolling aggregation for live card recognition.

The stable score for a card is the recency-weighted mean of observations in the
last ``window_seconds``.  Each observation at age ``a`` has linear weight
``max(0, 1 - a / window_seconds)``; its similarity contributes
``similarity * weight``.  This favors the newest camera frames without making
one frame a permanent result.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class MatchObservation:
    observed_at: float
    card_id: int
    similarity: float


class RollingMatchAggregator:
    """Aggregate per-card similarities from a bounded time window."""

    def __init__(self, window_seconds: float = 1.0):
        self.window_seconds = window_seconds
        self._observations: deque[MatchObservation] = deque()

    def add(self, matches: Iterable[dict], observed_at: float) -> dict[int, float]:
        for match in matches:
            self._observations.append(
                MatchObservation(observed_at, int(match["card_id"]), float(match["similarity"]))
            )
        cutoff = observed_at - self.window_seconds
        while self._observations and self._observations[0].observed_at < cutoff:
            self._observations.popleft()

        totals: dict[int, tuple[float, float]] = {}
        for observation in self._observations:
            weight = max(0.0, 1.0 - ((observed_at - observation.observed_at) / self.window_seconds))
            if weight == 0:
                continue
            weighted_similarity, total_weight = totals.get(observation.card_id, (0.0, 0.0))
            totals[observation.card_id] = (
                weighted_similarity + observation.similarity * weight,
                total_weight + weight,
            )
        return {
            card_id: weighted_similarity / total_weight
            for card_id, (weighted_similarity, total_weight) in totals.items()
        }
