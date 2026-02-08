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
