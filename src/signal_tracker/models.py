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
