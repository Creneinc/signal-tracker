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
