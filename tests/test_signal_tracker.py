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
