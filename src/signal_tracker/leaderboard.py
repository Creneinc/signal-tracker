"""
Leaderboard engine for ranking sources by prediction accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

from .models import Source, Claim, ClaimStatus, ScoreSnapshot
from .scoring import AccuracyScorer, AccuracyConfig


@dataclass
class LeaderboardEntry:
    rank: int
    source: Source
    score: ScoreSnapshot
    previous_score: Optional[ScoreSnapshot] = None

    @property
    def score_change(self) -> Optional[float]:
        if self.previous_score:
            return round(self.score.accuracy_score - self.previous_score.accuracy_score, 2)
        return None

    @property
    def is_rising(self) -> bool:
        return self.score_change is not None and self.score_change > 0

    @property
    def is_falling(self) -> bool:
        return self.score_change is not None and self.score_change < 0


@dataclass
class LeaderboardResult:
    entries: list[LeaderboardEntry]
    total_sources: int
    total_claims: int
    period: str
    generated_at: date = field(default_factory=lambda: date.today())

    @property
    def top_accurate(self) -> list[LeaderboardEntry]:
        return self.entries[:10]

    @property
    def worst_accurate(self) -> list[LeaderboardEntry]:
        qualified = [e for e in self.entries if e.score.total_claims >= 3]
        return list(reversed(qualified[-10:]))

    @property
    def biggest_risers(self) -> list[LeaderboardEntry]:
        with_change = [e for e in self.entries if e.score_change is not None and e.score_change > 0]
        return sorted(with_change, key=lambda e: e.score_change, reverse=True)[:10]

    @property
    def biggest_fallers(self) -> list[LeaderboardEntry]:
        with_change = [e for e in self.entries if e.score_change is not None and e.score_change < 0]
        return sorted(with_change, key=lambda e: e.score_change)[:10]

    @property
    def notable_wrongs(self) -> list[LeaderboardEntry]:
        notable = [e for e in self.entries if e.is_falling and e.score.accuracy_score > 40]
        return sorted(notable, key=lambda e: e.score_change)[:10]

    def to_dict(self) -> dict:
        return {
            "period": self.period, "generated_at": str(self.generated_at),
            "total_sources": self.total_sources, "total_claims": self.total_claims,
            "entries": [{
                "rank": e.rank, "source_name": e.source.name, "source_slug": e.source.slug,
                "source_type": e.source.source_type.value, "accuracy_score": e.score.accuracy_score,
                "total_claims": e.score.total_claims, "correct_claims": e.score.correct_claims,
                "wrong_claims": e.score.wrong_claims, "pending_claims": e.score.pending_claims,
                "score_change": e.score_change,
            } for e in self.entries],
        }


class Leaderboard:
    def __init__(self, scorer: Optional[AccuracyScorer] = None):
        self.scorer = scorer or AccuracyScorer()

    def build(self, sources: list[Source], claims: list[Claim], previous_claims: Optional[list[Claim]] = None,
              period: str = "all_time", min_claims: int = 1) -> LeaderboardResult:
        claims_by_source = {}
        for claim in claims:
            claims_by_source.setdefault(claim.source_id, []).append(claim)
        prev_by_source = {}
        if previous_claims:
            for claim in previous_claims:
                prev_by_source.setdefault(claim.source_id, []).append(claim)

        entries = []
        total_claims = 0
        for source in sources:
            source_claims = claims_by_source.get(source.id, [])
            if len(source_claims) < min_claims:
                continue
            total_claims += len(source_claims)
            score = self.scorer.score(source_claims, source_id=source.id, period=period)
            prev_score = None
            if source.id in prev_by_source:
                prev_score = self.scorer.score(prev_by_source[source.id], source_id=source.id, period=f"prev_{period}")
            entries.append(LeaderboardEntry(rank=0, source=source, score=score, previous_score=prev_score))

        entries.sort(key=lambda e: (e.score.accuracy_score, e.score.total_claims), reverse=True)
        for i, entry in enumerate(entries):
            entry.rank = i + 1

        return LeaderboardResult(entries=entries, total_sources=len(entries), total_claims=total_claims, period=period)

    def build_windowed(self, sources: list[Source], claims: list[Claim]) -> dict[str, LeaderboardResult]:
        results = {}
        today = date.today()
        for window_name, days in self.scorer.config.windows.items():
            if days is None:
                window_claims = claims
            else:
                cutoff = today - timedelta(days=days)
                window_claims = [c for c in claims if c.claim_date >= cutoff]
            results[window_name] = self.build(sources, window_claims, period=window_name)
        return results
