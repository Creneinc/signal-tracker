"""
Scoring engines for prediction accuracy and claim quality.
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

from .models import Claim, ClaimStatus, ScoreSnapshot


@dataclass
class AccuracyConfig:
    partial_weight: float = 0.5
    min_claims_for_score: int = 3
    windows: dict = field(default_factory=lambda: {
        "30d": 30, "90d": 90, "12mo": 365, "all_time": None,
    })
    recency_weighted: bool = False
    recency_half_life_days: int = 180


class AccuracyScorer:
    def __init__(self, config: Optional[AccuracyConfig] = None):
        self.config = config or AccuracyConfig()

    def score(self, claims: list[Claim], source_id: str = "", period: str = "all_time") -> ScoreSnapshot:
        resolved = [c for c in claims if c.is_resolved]
        pending = [c for c in claims if c.status == ClaimStatus.PENDING]
        correct = sum(1 for c in resolved if c.status == ClaimStatus.VERIFIED_CORRECT)
        wrong = sum(1 for c in resolved if c.status == ClaimStatus.VERIFIED_WRONG)
        partial = sum(1 for c in resolved if c.status == ClaimStatus.VERIFIED_PARTIAL)
        total_resolved = len(resolved)

        if total_resolved < self.config.min_claims_for_score:
            accuracy = 0.0
        else:
            weighted_correct = correct + (partial * self.config.partial_weight)
            accuracy = round((weighted_correct / total_resolved) * 100, 2)

        return ScoreSnapshot(
            source_id=source_id, accuracy_score=accuracy, total_claims=len(claims),
            correct_claims=correct, wrong_claims=wrong, pending_claims=len(pending),
            partial_claims=partial, period=period,
        )

    def score_windowed(self, claims: list[Claim], source_id: str = "") -> dict[str, ScoreSnapshot]:
        results = {}
        today = date.today()
        for window_name, days in self.config.windows.items():
            if days is None:
                window_claims = claims
            else:
                cutoff = today - timedelta(days=days)
                window_claims = [c for c in claims if c.claim_date >= cutoff]
            results[window_name] = self.score(window_claims, source_id=source_id, period=window_name)
        return results

    def score_with_recency(self, claims: list[Claim], source_id: str = "") -> ScoreSnapshot:
        resolved = [c for c in claims if c.is_resolved]
        if len(resolved) < self.config.min_claims_for_score:
            return ScoreSnapshot(source_id=source_id, accuracy_score=0.0, total_claims=len(claims),
                                 correct_claims=0, wrong_claims=0,
                                 pending_claims=sum(1 for c in claims if c.status == ClaimStatus.PENDING))

        today = date.today()
        half_life = self.config.recency_half_life_days
        weighted_correct = 0.0
        total_weight = 0.0
        correct_count = wrong_count = 0

        for claim in resolved:
            days_ago = (today - claim.claim_date).days
            weight = math.exp(-0.693 * days_ago / half_life)
            total_weight += weight
            if claim.status == ClaimStatus.VERIFIED_CORRECT:
                weighted_correct += weight
                correct_count += 1
            elif claim.status == ClaimStatus.VERIFIED_PARTIAL:
                weighted_correct += weight * self.config.partial_weight
            elif claim.status == ClaimStatus.VERIFIED_WRONG:
                wrong_count += 1

        accuracy = round((weighted_correct / total_weight) * 100, 2) if total_weight > 0 else 0.0
        return ScoreSnapshot(source_id=source_id, accuracy_score=accuracy, total_claims=len(claims),
                             correct_claims=correct_count, wrong_claims=wrong_count,
                             pending_claims=sum(1 for c in claims if c.status == ClaimStatus.PENDING),
                             period="recency_weighted")


@dataclass
class QualityConfig:
    weight_time_bound: float = 30.0
    weight_measurable: float = 30.0
    weight_falsifiable: float = 20.0
    weight_recency: float = 20.0
    max_age_years: float = 2.0
    min_quality_threshold: float = 50.0


class QualityScorer:
    TIME_PATTERNS = [
        r"\b(by|before|until|within)\s+(Q[1-4]\s*\d{4})",
        r"\b(by|before|until)\s+(end\s+of\s+\d{4})",
        r"\b(by|before|until)\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}",
        r"\b(within|in)\s+(\d+)\s+(months?|years?|weeks?|days?)",
        r"\b(by|before)\s+(\d{4})",
        r"\b(next\s+year|this\s+year|next\s+quarter|this\s+quarter)",
        r"\b\d{4}\b",
    ]
    MEASURE_PATTERNS = [
        r"\$[\d,.]+", r"\d+(\.\d+)?%",
        r"\d+(\.\d+)?\s*(billion|million|trillion|thousand)",
        r"(reach|hit|exceed|surpass)\s+\d+",
        r"\d+(\.\d+)?\s*(users|customers|subscribers|downloads)",
    ]
    FALSIFIABLE_PATTERNS = [
        r"\b(will|won't|will not)\b", r"\b(going to|is going to)\b",
        r"\b(predict|forecast|expect|anticipate)\b",
        r"\b(guarantee|certain|confident)\b", r"\b(never|always|impossible)\b",
    ]
    VAGUE_PATTERNS = [
        r"\b(might|could|may|possibly|perhaps|maybe)\b",
        r"\b(eventually|someday|soon|at some point)\b",
        r"\b(tend to|generally|usually|often)\b",
        r"\b(i think|i believe|in my opinion)\b",
    ]

    def __init__(self, config: Optional[QualityConfig] = None):
        self.config = config or QualityConfig()

    def score(self, claim: Claim) -> float:
        text = claim.text.lower()
        time_score = self._score_time_bound(claim, text)
        measure_score = self._score_measurable(text)
        falsifiable_score = self._score_falsifiable(text)
        recency_score = self._score_recency(claim)
        total = (
            time_score * (self.config.weight_time_bound / 100) +
            measure_score * (self.config.weight_measurable / 100) +
            falsifiable_score * (self.config.weight_falsifiable / 100) +
            recency_score * (self.config.weight_recency / 100)
        )
        return round(min(100.0, max(0.0, total)), 1)

    def is_high_quality(self, claim: Claim) -> bool:
        return self.score(claim) >= self.config.min_quality_threshold

    def score_batch(self, claims: list[Claim]) -> list[tuple[Claim, float]]:
        scored = [(c, self.score(c)) for c in claims]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _score_time_bound(self, claim: Claim, text: str) -> float:
        if claim.target_date is not None:
            return 100.0
        matches = sum(1 for p in self.TIME_PATTERNS if re.search(p, text, re.IGNORECASE))
        if matches >= 2: return 90.0
        elif matches == 1: return 60.0
        return 10.0

    def _score_measurable(self, text: str) -> float:
        matches = sum(1 for p in self.MEASURE_PATTERNS if re.search(p, text, re.IGNORECASE))
        if matches >= 2: return 100.0
        elif matches == 1: return 70.0
        return 15.0

    def _score_falsifiable(self, text: str) -> float:
        positive = sum(1 for p in self.FALSIFIABLE_PATTERNS if re.search(p, text, re.IGNORECASE))
        negative = sum(1 for p in self.VAGUE_PATTERNS if re.search(p, text, re.IGNORECASE))
        return max(0.0, min(100.0, positive * 35.0) - negative * 20.0)

    def _score_recency(self, claim: Claim) -> float:
        days_old = (date.today() - claim.claim_date).days
        max_days = self.config.max_age_years * 365
        if days_old <= 0: return 100.0
        elif days_old >= max_days: return 0.0
        return round(100.0 * (1 - days_old / max_days), 1)
