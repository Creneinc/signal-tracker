#!/bin/bash
# Signal Tracker — Full project bootstrap
# Run: bash setup_signal_tracker.sh

cd ~/crene-signal-tracker

# ─── Directory structure ───
mkdir -p src/signal_tracker tests .github/workflows

# ─── src/signal_tracker/__init__.py ───
cat > src/signal_tracker/__init__.py << 'PYEOF'
"""
Signal Tracker — Open-source prediction tracking & accuracy scoring framework.

Track predictions from any source. Score accuracy over time.
Build leaderboards. Hold the world accountable.

Created by Crene (https://crene.com)
"""

__version__ = "0.1.0"
__author__ = "Crene, Inc."
__license__ = "MIT"

from .tracker import SignalTracker
from .models import Source, Claim, Verification, ScoreSnapshot
from .scoring import AccuracyScorer, QualityScorer
from .extractors import ClaimExtractor
from .leaderboard import Leaderboard

__all__ = [
    "SignalTracker",
    "Source",
    "Claim",
    "Verification",
    "ScoreSnapshot",
    "AccuracyScorer",
    "QualityScorer",
    "ClaimExtractor",
    "Leaderboard",
]
PYEOF

# ─── src/signal_tracker/models.py ───
cat > src/signal_tracker/models.py << 'PYEOF'
"""
Data models for prediction tracking.
All models are plain dataclasses — no ORM dependency.
"""

from __future__ import annotations

import uuid
import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timezone
from enum import Enum
from typing import Optional


class SourceType(str, Enum):
    PERSON = "person"
    MEDIA = "media"
    INSTITUTION = "institution"
    ANALYST = "analyst"
    MODEL = "model"


class ClaimStatus(str, Enum):
    PENDING = "pending"
    VERIFIED_CORRECT = "correct"
    VERIFIED_WRONG = "wrong"
    VERIFIED_PARTIAL = "partial"
    EXPIRED = "expired"
    RETRACTED = "retracted"


class ClaimCategory(str, Enum):
    TECH = "tech"
    FINANCE = "finance"
    POLITICS = "politics"
    CRYPTO = "crypto"
    SCIENCE = "science"
    SPORTS = "sports"
    GEOPOLITICS = "geopolitics"
    ECONOMICS = "economics"
    HEALTH = "health"
    ENERGY = "energy"
    OTHER = "other"


@dataclass
class Source:
    name: str
    source_type: SourceType = SourceType.PERSON
    category: ClaimCategory = ClaimCategory.OTHER
    slug: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        if not self.slug:
            self.slug = self.name.lower().replace(" ", "-").replace(".", "").replace("'", "")
        if isinstance(self.source_type, str):
            self.source_type = SourceType(self.source_type)
        if isinstance(self.category, str):
            self.category = ClaimCategory(self.category)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["source_type"] = self.source_type.value
        d["category"] = self.category.value
        d["created_at"] = self.created_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> Source:
        data = data.copy()
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class Claim:
    source_id: str
    text: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    category: ClaimCategory = ClaimCategory.OTHER
    tags: list[str] = field(default_factory=list)
    claim_date: date = field(default_factory=lambda: date.today())
    target_date: Optional[date] = None
    status: ClaimStatus = ClaimStatus.PENDING
    quality_score: float = 0.0
    source_url: str = ""
    context: str = ""
    content_hash: str = ""
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        if isinstance(self.status, str):
            self.status = ClaimStatus(self.status)
        if isinstance(self.category, str):
            self.category = ClaimCategory(self.category)
        if not self.content_hash:
            self.content_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        payload = json.dumps({
            "source_id": self.source_id,
            "text": self.text,
            "claim_date": str(self.claim_date),
            "target_date": str(self.target_date) if self.target_date else None,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def verify_integrity(self) -> bool:
        return self.content_hash == self._compute_hash()

    @property
    def is_expired(self) -> bool:
        if self.target_date is None:
            return False
        return date.today() > self.target_date

    @property
    def is_resolved(self) -> bool:
        return self.status in (
            ClaimStatus.VERIFIED_CORRECT,
            ClaimStatus.VERIFIED_WRONG,
            ClaimStatus.VERIFIED_PARTIAL,
            ClaimStatus.RETRACTED,
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        d["category"] = self.category.value
        d["claim_date"] = str(self.claim_date)
        d["target_date"] = str(self.target_date) if self.target_date else None
        d["created_at"] = self.created_at.isoformat()
        d["updated_at"] = self.updated_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> Claim:
        data = data.copy()
        for dt_field in ("claim_date", "target_date"):
            if data.get(dt_field) and isinstance(data[dt_field], str):
                try:
                    data[dt_field] = date.fromisoformat(data[dt_field])
                except (ValueError, TypeError):
                    data[dt_field] = None
        for dt_field in ("created_at", "updated_at"):
            if data.get(dt_field) and isinstance(data[dt_field], str):
                data[dt_field] = datetime.fromisoformat(data[dt_field])
        return cls(**data)


@dataclass
class Verification:
    claim_id: str
    outcome: ClaimStatus
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    verifier: str = "manual"
    confidence: float = 1.0
    reasoning: str = ""
    evidence_url: str = ""
    verified_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.outcome, str):
            self.outcome = ClaimStatus(self.outcome)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["outcome"] = self.outcome.value
        d["verified_at"] = self.verified_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> Verification:
        data = data.copy()
        if "verified_at" in data and isinstance(data["verified_at"], str):
            data["verified_at"] = datetime.fromisoformat(data["verified_at"])
        return cls(**data)


@dataclass
class ScoreSnapshot:
    source_id: str
    accuracy_score: float
    total_claims: int
    correct_claims: int
    wrong_claims: int
    pending_claims: int = 0
    partial_claims: int = 0
    period: str = "all_time"
    snapshot_date: date = field(default_factory=lambda: date.today())
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["snapshot_date"] = str(self.snapshot_date)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> ScoreSnapshot:
        data = data.copy()
        if "snapshot_date" in data and isinstance(data["snapshot_date"], str):
            data["snapshot_date"] = date.fromisoformat(data["snapshot_date"])
        return cls(**data)
PYEOF

# ─── src/signal_tracker/scoring.py ───
cat > src/signal_tracker/scoring.py << 'PYEOF'
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
PYEOF

# ─── src/signal_tracker/extractors.py ───
cat > src/signal_tracker/extractors.py << 'PYEOF'
"""
Claim extraction from text sources.
Supports rule-based and LLM-powered extraction.
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Callable

from .models import Claim, ClaimCategory


@dataclass
class ExtractionResult:
    claims: list[Claim]
    source_text: str
    extraction_method: str
    raw_extractions: list[dict] = field(default_factory=list)


def _months_from_now(months: int) -> date:
    today = date.today()
    month = today.month + months
    year = today.year + (month - 1) // 12
    month = (month - 1) % 12 + 1
    return date(year, month, min(today.day, 28))


def _years_from_now(years: int) -> date:
    today = date.today()
    return date(today.year + years, today.month, min(today.day, 28))


class ClaimExtractor:
    PREDICTION_SIGNALS = [
        r"(?:I\s+)?predict(?:s|ed)?\s+(?:that\s+)?",
        r"(?:I\s+)?forecast(?:s|ed)?\s+(?:that\s+)?",
        r"(?:I\s+)?expect(?:s|ed)?\s+(?:that\s+)?",
        r"(?:I\s+)?anticipate(?:s|ed)?\s+(?:that\s+)?",
        r"(?:I\s+)?believe(?:s|d)?\s+(?:that\s+)?",
        r"(?:I'm|I\s+am)\s+(?:very\s+)?confident\s+(?:that\s+)?",
        r"(?:will|is\s+going\s+to)\s+(?:definitely|certainly|absolutely)\s+",
        r"(?:it's|it\s+is)\s+(?:clear|obvious|inevitable)\s+(?:that\s+)?",
        r"(?:there's|there\s+is)\s+no\s+(?:doubt|question)\s+(?:that\s+)?",
        r"(?:I\s+)?guarantee(?:s|d)?\s+(?:that\s+)?",
        r"(?:my|our)\s+(?:view|thesis|bet|call)\s+is\s+(?:that\s+)?",
        r"(?:I'm|I\s+am)\s+(?:betting|wagering)\s+(?:that|on)\s+",
        r"(?:mark\s+my\s+words)",
    ]

    TIME_REFERENCES = {
        r"by\s+(?:the\s+)?end\s+of\s+(\d{4})": lambda m: date(int(m.group(1)), 12, 31),
        r"by\s+Q1\s+(\d{4})": lambda m: date(int(m.group(1)), 3, 31),
        r"by\s+Q2\s+(\d{4})": lambda m: date(int(m.group(1)), 6, 30),
        r"by\s+Q3\s+(\d{4})": lambda m: date(int(m.group(1)), 9, 30),
        r"by\s+Q4\s+(\d{4})": lambda m: date(int(m.group(1)), 12, 31),
        r"by\s+(\d{4})": lambda m: date(int(m.group(1)), 12, 31),
        r"within\s+(\d+)\s+months?": lambda m: _months_from_now(int(m.group(1))),
        r"within\s+(\d+)\s+years?": lambda m: _years_from_now(int(m.group(1))),
        r"next\s+year": lambda m: date(date.today().year + 1, 12, 31),
        r"this\s+year": lambda m: date(date.today().year, 12, 31),
    }

    CATEGORY_KEYWORDS = {
        ClaimCategory.TECH: ["ai", "artificial intelligence", "software", "hardware", "chip", "gpu", "semiconductor", "robot", "autonomous", "self-driving", "quantum"],
        ClaimCategory.FINANCE: ["stock", "market", "s&p", "dow", "nasdaq", "ipo", "earnings", "revenue", "profit", "valuation", "price target"],
        ClaimCategory.CRYPTO: ["bitcoin", "btc", "ethereum", "eth", "crypto", "blockchain", "defi", "nft", "token", "web3"],
        ClaimCategory.POLITICS: ["election", "vote", "president", "congress", "senate", "democrat", "republican", "policy", "regulation", "legislation"],
        ClaimCategory.ECONOMICS: ["gdp", "inflation", "interest rate", "fed", "recession", "unemployment", "trade", "tariff", "monetary policy"],
        ClaimCategory.ENERGY: ["oil", "gas", "solar", "wind", "nuclear", "renewable", "ev", "electric vehicle", "battery", "energy"],
        ClaimCategory.HEALTH: ["fda", "vaccine", "drug", "clinical trial", "treatment", "disease", "pandemic", "health"],
        ClaimCategory.GEOPOLITICS: ["war", "sanction", "nato", "china", "russia", "taiwan", "ukraine", "conflict", "treaty"],
    }

    def __init__(self, llm_fn: Optional[Callable[[str], str]] = None):
        self.llm_fn = llm_fn
        self._compiled_signals = [re.compile(p, re.IGNORECASE) for p in self.PREDICTION_SIGNALS]

    def extract(self, text: str, source_id: str, claim_date: Optional[date] = None, category: Optional[ClaimCategory] = None) -> ExtractionResult:
        claim_date = claim_date or date.today()
        sentences = self._split_sentences(text)
        claims = []
        raw = []
        for sentence in sentences:
            for pattern in self._compiled_signals:
                match = pattern.search(sentence)
                if match:
                    claim_text = sentence.strip()
                    target_date = self._extract_target_date(claim_text)
                    detected_category = category or self._detect_category(claim_text)
                    claim = Claim(source_id=source_id, text=claim_text, claim_date=claim_date,
                                  target_date=target_date, category=detected_category, context=text[:500])
                    raw.append({"sentence": claim_text, "pattern_matched": pattern.pattern,
                                "target_date": str(target_date) if target_date else None,
                                "category": detected_category.value})
                    claims.append(claim)
                    break
        return ExtractionResult(claims=claims, source_text=text, extraction_method="rules", raw_extractions=raw)

    def extract_with_llm(self, text: str, source_id: str, source_name: str = "", claim_date: Optional[date] = None, category: Optional[ClaimCategory] = None) -> ExtractionResult:
        if not self.llm_fn:
            raise ValueError("LLM function not provided. Initialize with llm_fn parameter, or use extract() for rule-based extraction.")
        claim_date = claim_date or date.today()
        prompt = self._build_extraction_prompt(text, source_name)
        response = self.llm_fn(prompt)
        try:
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            extracted = json.loads(json_match.group()) if json_match else json.loads(response)
        except json.JSONDecodeError:
            return self.extract(text, source_id, claim_date, category)
        claims = []
        for item in extracted:
            if not isinstance(item, dict) or "text" not in item:
                continue
            target_date = None
            if item.get("target_date"):
                try:
                    target_date = date.fromisoformat(item["target_date"])
                except (ValueError, TypeError):
                    target_date = self._extract_target_date(item["text"])
            detected_category = category
            if not detected_category and item.get("category"):
                try:
                    detected_category = ClaimCategory(item["category"])
                except ValueError:
                    detected_category = self._detect_category(item["text"])
            if not detected_category:
                detected_category = self._detect_category(item["text"])
            claim = Claim(source_id=source_id, text=item["text"], claim_date=claim_date,
                          target_date=target_date, category=detected_category, context=text[:500],
                          metadata={"llm_confidence": item.get("confidence", None)})
            claims.append(claim)
        return ExtractionResult(claims=claims, source_text=text, extraction_method="llm",
                                raw_extractions=extracted if isinstance(extracted, list) else [])

    def _build_extraction_prompt(self, text: str, source_name: str = "") -> str:
        source_context = f" by {source_name}" if source_name else ""
        return f"""Extract all verifiable predictions from the following text{source_context}.
Return a JSON array of objects with: "text", "target_date" (YYYY-MM-DD or null), "category" (tech/finance/politics/crypto/science/sports/geopolitics/economics/health/energy/other), "confidence" (0.0-1.0).
Only extract SPECIFIC, VERIFIABLE predictions. Return ONLY the JSON array.

Text:
\"\"\"{text[:3000]}\"\"\"

JSON:"""

    def _split_sentences(self, text: str) -> list[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 20]

    def _extract_target_date(self, text: str) -> Optional[date]:
        text_lower = text.lower()
        for pattern, date_fn in self.TIME_REFERENCES.items():
            match = re.search(pattern, text_lower)
            if match:
                try:
                    return date_fn(match)
                except (ValueError, OverflowError):
                    continue
        return None

    def _detect_category(self, text: str) -> ClaimCategory:
        text_lower = text.lower()
        scores = {}
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[category] = score
        return max(scores, key=scores.get) if scores else ClaimCategory.OTHER
PYEOF

# ─── src/signal_tracker/leaderboard.py ───
cat > src/signal_tracker/leaderboard.py << 'PYEOF'
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
PYEOF

# ─── src/signal_tracker/tracker.py ───
cat > src/signal_tracker/tracker.py << 'PYEOF'
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
PYEOF

# ─── src/signal_tracker/storage.py ───
cat > src/signal_tracker/storage.py << 'PYEOF'
"""
Storage backends for persisting tracker data.
Included: SQLiteBackend for larger datasets.
"""

from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from .models import Source, Claim, Verification, ScoreSnapshot, ClaimStatus


class StorageBackend(ABC):
    @abstractmethod
    def save_source(self, source: Source) -> None: ...
    @abstractmethod
    def get_source(self, source_id: str) -> Optional[Source]: ...
    @abstractmethod
    def list_sources(self) -> list[Source]: ...
    @abstractmethod
    def save_claim(self, claim: Claim) -> None: ...
    @abstractmethod
    def get_claim(self, claim_id: str) -> Optional[Claim]: ...
    @abstractmethod
    def list_claims(self, source_id: Optional[str] = None) -> list[Claim]: ...
    @abstractmethod
    def save_verification(self, verification: Verification) -> None: ...
    @abstractmethod
    def list_verifications(self, claim_id: str) -> list[Verification]: ...


class SQLiteBackend(StorageBackend):
    def __init__(self, db_path="signal_tracker.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sources (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, slug TEXT UNIQUE,
                    source_type TEXT, category TEXT, metadata TEXT DEFAULT '{}', created_at TEXT);
                CREATE TABLE IF NOT EXISTS claims (
                    id TEXT PRIMARY KEY, source_id TEXT REFERENCES sources(id), text TEXT NOT NULL,
                    category TEXT, tags TEXT DEFAULT '[]', claim_date TEXT, target_date TEXT,
                    status TEXT DEFAULT 'pending', quality_score REAL DEFAULT 0, source_url TEXT DEFAULT '',
                    context TEXT DEFAULT '', content_hash TEXT, metadata TEXT DEFAULT '{}',
                    created_at TEXT, updated_at TEXT);
                CREATE TABLE IF NOT EXISTS verifications (
                    id TEXT PRIMARY KEY, claim_id TEXT REFERENCES claims(id), outcome TEXT NOT NULL,
                    verifier TEXT DEFAULT 'manual', confidence REAL DEFAULT 1.0, reasoning TEXT DEFAULT '',
                    evidence_url TEXT DEFAULT '', verified_at TEXT, metadata TEXT DEFAULT '{}');
                CREATE INDEX IF NOT EXISTS idx_claims_source ON claims(source_id);
                CREATE INDEX IF NOT EXISTS idx_claims_status ON claims(status);
                CREATE INDEX IF NOT EXISTS idx_verifications_claim ON verifications(claim_id);
            """)

    def save_source(self, source):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT OR REPLACE INTO sources VALUES (?,?,?,?,?,?,?)",
                         (source.id, source.name, source.slug, source.source_type.value,
                          source.category.value, json.dumps(source.metadata), source.created_at.isoformat()))

    def get_source(self, source_id):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM sources WHERE id=? OR slug=?", (source_id, source_id)).fetchone()
            if row:
                return Source.from_dict(dict(row) | {"metadata": json.loads(row["metadata"])})
        return None

    def list_sources(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM sources ORDER BY name").fetchall()
            return [Source.from_dict(dict(r) | {"metadata": json.loads(r["metadata"])}) for r in rows]

    def save_claim(self, claim):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT OR REPLACE INTO claims VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (claim.id, claim.source_id, claim.text, claim.category.value,
                          json.dumps(claim.tags), str(claim.claim_date),
                          str(claim.target_date) if claim.target_date else None,
                          claim.status.value, claim.quality_score, claim.source_url,
                          claim.context, claim.content_hash, json.dumps(claim.metadata),
                          claim.created_at.isoformat(), claim.updated_at.isoformat()))

    def get_claim(self, claim_id):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM claims WHERE id=?", (claim_id,)).fetchone()
            if row:
                d = dict(row)
                d["tags"] = json.loads(d["tags"])
                d["metadata"] = json.loads(d["metadata"])
                return Claim.from_dict(d)
        return None

    def list_claims(self, source_id=None):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if source_id:
                rows = conn.execute("SELECT * FROM claims WHERE source_id=? ORDER BY claim_date DESC", (source_id,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM claims ORDER BY claim_date DESC").fetchall()
            claims = []
            for r in rows:
                d = dict(r)
                d["tags"] = json.loads(d["tags"])
                d["metadata"] = json.loads(d["metadata"])
                claims.append(Claim.from_dict(d))
            return claims

    def save_verification(self, verification):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT OR REPLACE INTO verifications VALUES (?,?,?,?,?,?,?,?,?)",
                         (verification.id, verification.claim_id, verification.outcome.value,
                          verification.verifier, verification.confidence, verification.reasoning,
                          verification.evidence_url, verification.verified_at.isoformat(),
                          json.dumps(verification.metadata)))

    def list_verifications(self, claim_id):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM verifications WHERE claim_id=?", (claim_id,)).fetchall()
            return [Verification.from_dict(dict(r) | {"metadata": json.loads(r["metadata"])}) for r in rows]
PYEOF

# ─── tests/__init__.py ───
touch tests/__init__.py

# ─── tests/test_signal_tracker.py ───
cat > tests/test_signal_tracker.py << 'PYEOF'
import json
import tempfile
from datetime import date, timedelta
from pathlib import Path
import pytest
from signal_tracker import SignalTracker, Source, Claim, Verification, ScoreSnapshot
from signal_tracker.models import SourceType, ClaimStatus, ClaimCategory
from signal_tracker.scoring import AccuracyScorer, QualityScorer, AccuracyConfig, QualityConfig
from signal_tracker.extractors import ClaimExtractor
from signal_tracker.leaderboard import Leaderboard
from signal_tracker.storage import SQLiteBackend


class TestSource:
    def test_create_source(self):
        source = Source(name="Elon Musk", source_type=SourceType.PERSON)
        assert source.name == "Elon Musk"
        assert source.slug == "elon-musk"
    def test_auto_slug(self):
        source = Source(name="Jim O'Brien")
        assert source.slug == "jim-obrien"
    def test_serialization(self):
        source = Source(name="Test", source_type=SourceType.MEDIA)
        restored = Source.from_dict(source.to_dict())
        assert restored.name == source.name
    def test_string_type_conversion(self):
        source = Source(name="Test", source_type="person")
        assert source.source_type == SourceType.PERSON


class TestClaim:
    def test_create_claim(self):
        claim = Claim(source_id="abc", text="BTC $100k by 2025", target_date=date(2025, 12, 31))
        assert claim.status == ClaimStatus.PENDING
        assert claim.content_hash
    def test_integrity_check(self):
        claim = Claim(source_id="abc", text="Test prediction")
        assert claim.verify_integrity() is True
        claim.text = "Tampered"
        assert claim.verify_integrity() is False
    def test_is_expired(self):
        assert Claim(source_id="a", text="t", target_date=date(2020, 1, 1)).is_expired is True
        assert Claim(source_id="a", text="t", target_date=date(2099, 12, 31)).is_expired is False
        assert Claim(source_id="a", text="t").is_expired is False
    def test_is_resolved(self):
        claim = Claim(source_id="a", text="t")
        assert claim.is_resolved is False
        claim.status = ClaimStatus.VERIFIED_CORRECT
        assert claim.is_resolved is True
    def test_serialization(self):
        claim = Claim(source_id="a", text="t", target_date=date(2025, 6, 30), tags=["finance"])
        restored = Claim.from_dict(claim.to_dict())
        assert restored.text == claim.text and restored.target_date == claim.target_date


class TestVerification:
    def test_create(self):
        v = Verification(claim_id="abc", outcome=ClaimStatus.VERIFIED_CORRECT)
        assert v.outcome == ClaimStatus.VERIFIED_CORRECT
    def test_serialization(self):
        v = Verification(claim_id="abc", outcome=ClaimStatus.VERIFIED_WRONG, reasoning="Nope", verifier="ai:claude")
        restored = Verification.from_dict(v.to_dict())
        assert restored.outcome == v.outcome


def _make_claims(correct=0, wrong=0, partial=0, pending=0):
    claims = []
    for _ in range(correct):
        c = Claim(source_id="test", text="t"); c.status = ClaimStatus.VERIFIED_CORRECT; claims.append(c)
    for _ in range(wrong):
        c = Claim(source_id="test", text="t"); c.status = ClaimStatus.VERIFIED_WRONG; claims.append(c)
    for _ in range(partial):
        c = Claim(source_id="test", text="t"); c.status = ClaimStatus.VERIFIED_PARTIAL; claims.append(c)
    for _ in range(pending):
        claims.append(Claim(source_id="test", text="t"))
    return claims

def _make_claims_for_source(source_id, correct=0, wrong=0):
    claims = []
    for _ in range(correct):
        c = Claim(source_id=source_id, text="t"); c.status = ClaimStatus.VERIFIED_CORRECT; claims.append(c)
    for _ in range(wrong):
        c = Claim(source_id=source_id, text="t"); c.status = ClaimStatus.VERIFIED_WRONG; claims.append(c)
    return claims


class TestAccuracyScorer:
    def setup_method(self):
        self.scorer = AccuracyScorer()
    def test_perfect(self):
        assert self.scorer.score(_make_claims(correct=10)).accuracy_score == 100.0
    def test_zero(self):
        assert self.scorer.score(_make_claims(wrong=10)).accuracy_score == 0.0
    def test_mixed(self):
        assert self.scorer.score(_make_claims(correct=7, wrong=3)).accuracy_score == 70.0
    def test_partial_weight(self):
        assert self.scorer.score(_make_claims(correct=5, partial=2, wrong=3)).accuracy_score == 60.0
    def test_min_claims(self):
        assert self.scorer.score(_make_claims(correct=1, wrong=1)).accuracy_score == 0.0
    def test_pending_ignored(self):
        score = self.scorer.score(_make_claims(correct=5, wrong=5, pending=100))
        assert score.accuracy_score == 50.0 and score.pending_claims == 100
    def test_windowed(self):
        claims = []
        for _ in range(5):
            c = Claim(source_id="t", text="t", claim_date=date.today() - timedelta(days=400))
            c.status = ClaimStatus.VERIFIED_WRONG; claims.append(c)
        for _ in range(5):
            c = Claim(source_id="t", text="t", claim_date=date.today() - timedelta(days=10))
            c.status = ClaimStatus.VERIFIED_CORRECT; claims.append(c)
        windows = self.scorer.score_windowed(claims, source_id="t")
        assert windows["30d"].accuracy_score == 100.0
        assert windows["all_time"].accuracy_score == 50.0
    def test_recency_weighted(self):
        scorer = AccuracyScorer(AccuracyConfig(recency_half_life_days=30))
        claims = []
        for _ in range(5):
            c = Claim(source_id="t", text="t", claim_date=date.today() - timedelta(days=365))
            c.status = ClaimStatus.VERIFIED_WRONG; claims.append(c)
        for _ in range(5):
            c = Claim(source_id="t", text="t", claim_date=date.today() - timedelta(days=5))
            c.status = ClaimStatus.VERIFIED_CORRECT; claims.append(c)
        assert scorer.score_with_recency(claims, source_id="t").accuracy_score > 80.0


class TestQualityScorer:
    def setup_method(self):
        self.scorer = QualityScorer()
    def test_high_quality(self):
        claim = Claim(source_id="t", text="Tesla will achieve full self-driving by end of 2025", target_date=date(2025, 12, 31))
        assert self.scorer.score(claim) > 60.0
    def test_low_quality(self):
        claim = Claim(source_id="t", text="Things might eventually get better someday", claim_date=date.today() - timedelta(days=800))
        assert self.scorer.score(claim) < 30.0
    def test_measurable(self):
        claim = Claim(source_id="t", text="Bitcoin will reach $150,000 by Q4 2025", target_date=date(2025, 12, 31))
        assert self.scorer.score(claim) > 70.0
    def test_is_high_quality(self):
        good = Claim(source_id="t", text="I predict Netflix will hit $800 by Q2 2025", target_date=date(2025, 6, 30))
        bad = Claim(source_id="t", text="I think things could maybe improve", claim_date=date.today() - timedelta(days=800))
        assert self.scorer.is_high_quality(good) and not self.scorer.is_high_quality(bad)
    def test_batch(self):
        claims = [
            Claim(source_id="t", text="I predict $500 by 2025", target_date=date(2025, 12, 31)),
            Claim(source_id="t", text="Maybe something might happen"),
            Claim(source_id="t", text="Revenue will exceed $10 billion by Q1 2025", target_date=date(2025, 3, 31)),
        ]
        scored = self.scorer.score_batch(claims)
        assert scored[0][1] >= scored[1][1] >= scored[2][1]


class TestClaimExtractor:
    def setup_method(self):
        self.extractor = ClaimExtractor()
    def test_extract_prediction(self):
        result = self.extractor.extract("I predict that Tesla will achieve full autonomy by 2026.", source_id="elon")
        assert len(result.claims) >= 1
    def test_extract_forecast(self):
        result = self.extractor.extract("Goldman Sachs forecasts that GDP will grow 3.2% in 2025.", source_id="gs")
        assert len(result.claims) >= 1
    def test_target_date(self):
        result = self.extractor.extract("I predict Bitcoin will reach $200k by end of 2025.", source_id="t")
        if result.claims:
            assert result.claims[0].target_date == date(2025, 12, 31)
    def test_category_detection(self):
        result = self.extractor.extract("I expect Bitcoin and Ethereum will rally by Q4 2025.", source_id="t")
        if result.claims:
            assert result.claims[0].category == ClaimCategory.CRYPTO
    def test_no_extraction_vague(self):
        result = self.extractor.extract("The weather is nice today. I had coffee.", source_id="t")
        assert len(result.claims) == 0
    def test_llm_requires_fn(self):
        with pytest.raises(ValueError):
            self.extractor.extract_with_llm("test", source_id="t")


class TestLeaderboard:
    def test_build(self):
        board = Leaderboard()
        sources = [Source(name="A", id="a"), Source(name="B", id="b"), Source(name="C", id="c")]
        claims = _make_claims_for_source("a", 8, 2) + _make_claims_for_source("b", 5, 5) + _make_claims_for_source("c", 3, 7)
        result = board.build(sources, claims)
        assert result.entries[0].source.name == "A" and result.entries[0].rank == 1
    def test_serialization(self):
        board = Leaderboard()
        result = board.build([Source(name="T", id="t")], _make_claims_for_source("t", 7, 3))
        d = result.to_dict()
        assert d["entries"][0]["accuracy_score"] == 70.0


class TestSignalTracker:
    def test_full_workflow(self):
        tracker = SignalTracker()
        elon = tracker.add_source("Elon Musk", source_type="person", category="tech")
        cramer = tracker.add_source("Jim Cramer", source_type="person", category="finance")
        c1 = tracker.add_claim(elon, "FSD by 2025", target_date=date(2025, 12, 31))
        c2 = tracker.add_claim(elon, "Mars by 2029", target_date=date(2029, 12, 31))
        c3 = tracker.add_claim(cramer, "Netflix $800", target_date=date(2025, 6, 30))
        tracker.verify(c1, outcome="wrong")
        tracker.verify(c3, outcome="correct")
        assert tracker.score(elon).wrong_claims == 1
        assert tracker.score(cramer).correct_claims == 1
        assert tracker.stats["total_claims"] == 3

    def test_consensus(self):
        tracker = SignalTracker()
        source = tracker.add_source("Test")
        claim = tracker.add_claim(source, "Test prediction")
        tracker.verify_with_consensus(claim, [
            {"outcome": "correct", "verifier": "ai:claude", "confidence": 0.9},
            {"outcome": "correct", "verifier": "ai:gpt-4", "confidence": 0.85},
            {"outcome": "wrong", "verifier": "ai:gemini", "confidence": 0.6},
        ])
        assert claim.status == ClaimStatus.VERIFIED_CORRECT

    def test_save_load(self):
        tracker = SignalTracker()
        source = tracker.add_source("Test Source")
        tracker.add_claim(source, "Test claim", target_date=date(2025, 12, 31))
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        tracker.save(path)
        loaded = SignalTracker.load(path)
        assert len(loaded.sources) == 1 and len(loaded.claims) == 1
        Path(path).unlink()

    def test_filtered_claims(self):
        tracker = SignalTracker()
        s = tracker.add_source("Test")
        c1 = tracker.add_claim(s, "Claim 1", category="tech")
        tracker.add_claim(s, "Claim 2", category="finance")
        tracker.verify(c1, outcome="correct")
        assert len(tracker.get_claims(category="tech")) == 1
        assert len(tracker.get_claims(status="correct")) == 1

    def test_extract_claims(self):
        tracker = SignalTracker()
        source = tracker.add_source("Elon Musk")
        claims = tracker.extract_claims("I predict that Tesla will dominate the EV market by 2026.", source=source)
        assert len(claims) >= 1 and len(tracker.claims) >= 1


class TestSQLiteBackend:
    def setup_method(self):
        self.db_path = tempfile.mktemp(suffix=".db")
        self.backend = SQLiteBackend(self.db_path)
    def teardown_method(self):
        Path(self.db_path).unlink(missing_ok=True)
    def test_source_crud(self):
        source = Source(name="Test Source", source_type=SourceType.PERSON)
        self.backend.save_source(source)
        assert self.backend.get_source(source.id) is not None
        assert self.backend.get_source("test-source") is not None
        assert len(self.backend.list_sources()) == 1
    def test_claim_crud(self):
        self.backend.save_source(Source(name="T", id="src1"))
        claim = Claim(source_id="src1", text="Test prediction", target_date=date(2025, 12, 31))
        self.backend.save_claim(claim)
        assert self.backend.get_claim(claim.id) is not None
        assert len(self.backend.list_claims()) == 1
    def test_verification_crud(self):
        v = Verification(claim_id="c1", outcome=ClaimStatus.VERIFIED_CORRECT, reasoning="It happened")
        self.backend.save_verification(v)
        loaded = self.backend.list_verifications("c1")
        assert len(loaded) == 1 and loaded[0].reasoning == "It happened"
PYEOF

# ─── pyproject.toml ───
cat > pyproject.toml << 'PYEOF'
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "signal-tracker"
version = "0.1.0"
description = "Open-source prediction tracking & accuracy scoring framework. Track predictions, score accuracy, build leaderboards."
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.10"
authors = [{name = "Crene, Inc.", email = "dev@crene.com"}]
keywords = ["predictions", "forecasting", "accuracy", "leaderboard", "track-record", "accountability", "signal", "intelligence"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

[project.urls]
Homepage = "https://crene.com"
Repository = "https://github.com/crene/signal-tracker"

[project.optional-dependencies]
llm = ["openai>=1.0", "anthropic>=0.20"]
all = ["openai>=1.0", "anthropic>=0.20"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
PYEOF

# ─── .gitignore ───
cat > .gitignore << 'EOF'
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
*.egg
.eggs/
*.db
.env
.venv
venv/
.pytest_cache/
.mypy_cache/
.DS_Store
EOF

# ─── MANIFEST.in ───
cat > MANIFEST.in << 'EOF'
include LICENSE
include README.md
include CONTRIBUTING.md
recursive-include src *.py
recursive-include tests *.py
EOF

echo ""
echo "✅ Signal Tracker project created at ~/crene-signal-tracker"
echo ""
echo "Files created:"
find . -type f | grep -v __pycache__ | grep -v .git/ | sort
echo ""
echo "Next steps:"
echo "  pip install -e ."
echo "  pytest tests/ -v"
echo "  git init && git add . && git commit -m 'Initial release: signal-tracker v0.1.0'"
