"""
SignalTracker — the main interface for tracking predictions.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional, Callable

from .models import Source, Claim, ClaimStatus, Verification, ScoreSnapshot, ClaimCategory, SourceType
from .scoring import AccuracyScorer, QualityScorer, AccuracyConfig, QualityConfig
from .extractors import ClaimExtractor
from .leaderboard import Leaderboard, LeaderboardResult


class SignalTracker:
    """
    Track predictions, score accuracy, build leaderboards.

    Example:
        >>> tracker = SignalTracker()
        >>> elon = tracker.add_source("Elon Musk", source_type="person", category="tech")
        >>> claim = tracker.add_claim(elon, "FSD by 2025", target_date=date(2025, 12, 31))
        >>> tracker.verify(claim, outcome="wrong", reasoning="Not achieved")
        >>> board = tracker.leaderboard()
    """

    def __init__(self, accuracy_config=None, quality_config=None, llm_fn=None):
        self._sources: dict[str, Source] = {}
        self._claims: dict[str, Claim] = {}
        self._verifications: dict[str, list[Verification]] = {}
        self._snapshots: list[ScoreSnapshot] = []
        self.accuracy_scorer = AccuracyScorer(accuracy_config)
        self.quality_scorer = QualityScorer(quality_config)
        self.extractor = ClaimExtractor(llm_fn=llm_fn)
        self._leaderboard = Leaderboard(scorer=self.accuracy_scorer)

    def add_source(self, name, source_type="person", category="other", **kwargs) -> Source:
        source = Source(name=name, source_type=SourceType(source_type), category=ClaimCategory(category), **kwargs)
        self._sources[source.id] = source
        return source

    def get_source(self, slug_or_id: str) -> Optional[Source]:
        if slug_or_id in self._sources:
            return self._sources[slug_or_id]
        for source in self._sources.values():
            if source.slug == slug_or_id:
                return source
        return None

    @property
    def sources(self) -> list[Source]:
        return list(self._sources.values())

    def add_claim(self, source, text, target_date=None, category=None, claim_date=None, **kwargs) -> Claim:
        source_id = source.id if isinstance(source, Source) else source
        claim = Claim(source_id=source_id, text=text, target_date=target_date,
                      category=ClaimCategory(category) if category else ClaimCategory.OTHER,
                      claim_date=claim_date or date.today(), **kwargs)
        claim.quality_score = self.quality_scorer.score(claim)
        self._claims[claim.id] = claim
        return claim

    def get_claims(self, source=None, status=None, category=None, min_quality=None) -> list[Claim]:
        claims = list(self._claims.values())
        if source:
            source_id = source.id if isinstance(source, Source) else source
            claims = [c for c in claims if c.source_id == source_id]
        if status:
            claims = [c for c in claims if c.status == ClaimStatus(status)]
        if category:
            claims = [c for c in claims if c.category == ClaimCategory(category)]
        if min_quality is not None:
            claims = [c for c in claims if c.quality_score >= min_quality]
        return claims

    @property
    def claims(self) -> list[Claim]:
        return list(self._claims.values())

    def extract_claims(self, text, source, use_llm=False, claim_date=None) -> list[Claim]:
        source_obj = source if isinstance(source, Source) else self.get_source(source)
        source_id = source_obj.id if source_obj else source
        source_name = source_obj.name if source_obj else ""
        if use_llm:
            result = self.extractor.extract_with_llm(text, source_id, source_name=source_name, claim_date=claim_date)
        else:
            result = self.extractor.extract(text, source_id, claim_date=claim_date)
        for claim in result.claims:
            claim.quality_score = self.quality_scorer.score(claim)
            self._claims[claim.id] = claim
        return result.claims

    def verify(self, claim, outcome, reasoning="", verifier="manual", confidence=1.0, evidence_url="") -> Verification:
        claim_id = claim.id if isinstance(claim, Claim) else claim
        claim_obj = self._claims.get(claim_id)
        if not claim_obj:
            raise ValueError(f"Claim {claim_id} not found")
        verification = Verification(claim_id=claim_id, outcome=ClaimStatus(outcome), verifier=verifier,
                                     confidence=confidence, reasoning=reasoning, evidence_url=evidence_url)
        claim_obj.status = verification.outcome
        claim_obj.updated_at = datetime.now(timezone.utc)
        self._verifications.setdefault(claim_id, []).append(verification)
        return verification

    def verify_with_consensus(self, claim, verifications: list[dict]) -> Verification:
        claim_id = claim.id if isinstance(claim, Claim) else claim
        for v in verifications:
            self._verifications.setdefault(claim_id, []).append(
                Verification(claim_id=claim_id, outcome=ClaimStatus(v["outcome"]),
                             verifier=v.get("verifier", "ai"), confidence=v.get("confidence", 1.0),
                             reasoning=v.get("reasoning", "")))
        outcome_weights = {"correct": 0.0, "wrong": 0.0, "partial": 0.0}
        for v in verifications:
            outcome = v["outcome"]
            if outcome in outcome_weights:
                outcome_weights[outcome] += v.get("confidence", 1.0)
        consensus = max(outcome_weights, key=outcome_weights.get)
        total_confidence = sum(outcome_weights.values())
        consensus_confidence = outcome_weights[consensus] / total_confidence if total_confidence > 0 else 0
        return self.verify(claim_id, outcome=consensus,
                           reasoning=f"Consensus: {consensus} ({consensus_confidence:.0%} agreement)",
                           verifier="consensus", confidence=consensus_confidence)

    def score(self, source) -> ScoreSnapshot:
        source_id = source.id if isinstance(source, Source) else source
        claims = self.get_claims(source=source_id)
        return self.accuracy_scorer.score(claims, source_id=source_id)

    def score_all(self) -> dict[str, ScoreSnapshot]:
        return {source.id: self.score(source) for source in self._sources.values()}

    def leaderboard(self, period="all_time", min_claims=1) -> LeaderboardResult:
        return self._leaderboard.build(sources=self.sources, claims=self.claims, period=period, min_claims=min_claims)

    def save(self, path) -> None:
        state = {
            "version": "1.0",
            "sources": {k: v.to_dict() for k, v in self._sources.items()},
            "claims": {k: v.to_dict() for k, v in self._claims.items()},
            "verifications": {k: [v.to_dict() for v in vlist] for k, vlist in self._verifications.items()},
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)

    @classmethod
    def load(cls, path, **kwargs) -> SignalTracker:
        with open(path) as f:
            state = json.load(f)
        tracker = cls(**kwargs)
        for source_data in state.get("sources", {}).values():
            source = Source.from_dict(source_data)
            tracker._sources[source.id] = source
        for claim_data in state.get("claims", {}).values():
            claim = Claim.from_dict(claim_data)
            tracker._claims[claim.id] = claim
        for claim_id, vlist in state.get("verifications", {}).items():
            tracker._verifications[claim_id] = [Verification.from_dict(v) for v in vlist]
        return tracker

    @property
    def stats(self) -> dict:
        claims = self.claims
        return {
            "total_sources": len(self._sources), "total_claims": len(claims),
            "pending": sum(1 for c in claims if c.status == ClaimStatus.PENDING),
            "correct": sum(1 for c in claims if c.status == ClaimStatus.VERIFIED_CORRECT),
            "wrong": sum(1 for c in claims if c.status == ClaimStatus.VERIFIED_WRONG),
            "partial": sum(1 for c in claims if c.status == ClaimStatus.VERIFIED_PARTIAL),
            "expired": sum(1 for c in claims if c.is_expired and c.status == ClaimStatus.PENDING),
        }

    def __repr__(self):
        return f"SignalTracker(sources={len(self._sources)}, claims={len(self._claims)})"
