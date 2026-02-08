#!/bin/bash
cd ~/crene-signal-tracker

# ─── README.md ───
cat > README.md << 'MDEOF'
# 📡 Signal Tracker

**Open-source prediction tracking & accuracy scoring framework.**

Track predictions from anyone — media, analysts, politicians, CEOs, AI models. Score accuracy over time. Build leaderboards. Hold the world accountable.

[![CI](https://github.com/Creneinc/signal-tracker/actions/workflows/ci.yml/badge.svg)](https://github.com/Creneinc/signal-tracker/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/signal-tracker)](https://pypi.org/project/signal-tracker/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## Why?

Everyone makes predictions. Almost nobody tracks them.

- Elon Musk said FSD would be ready "next year" — five years in a row
- Jim Cramer's stock picks are famously inverse-correlated with outcomes
- Media outlets make bold forecasts and quietly move on when they're wrong

**Signal Tracker** gives you the tools to track all of this systematically.

> Created by [Crene](https://crene.com) — the AI-powered prediction intelligence platform that tracks 420+ sources across tech, finance, politics, and geopolitics.

---

## Install

```bash
pip install signal-tracker
```

---

## Quick Start

```python
from signal_tracker import SignalTracker
from datetime import date

# Initialize
tracker = SignalTracker()

# Add sources to track
elon = tracker.add_source("Elon Musk", source_type="person", category="tech")
cramer = tracker.add_source("Jim Cramer", source_type="person", category="finance")

# Add predictions
claim1 = tracker.add_claim(
    source=elon,
    text="Tesla will achieve full self-driving by end of 2025",
    target_date=date(2025, 12, 31),
)

claim2 = tracker.add_claim(
    source=cramer,
    text="Netflix will hit $800 by Q2 2025",
    target_date=date(2025, 6, 30),
)

# Verify when outcomes are known
tracker.verify(claim1, outcome="wrong", reasoning="FSD not achieved by deadline")
tracker.verify(claim2, outcome="correct", reasoning="Netflix reached $820 in May 2025")

# Build leaderboard
board = tracker.leaderboard()
for entry in board.entries:
    print(f"{entry.rank}. {entry.source.name}: {entry.score.accuracy_score}%")

# Save state
tracker.save("my_tracker.json")
```

---

## Features

### 🎯 Prediction Tracking
Track specific, verifiable claims with time bounds, measurable targets, and clear success criteria.

```python
claim = tracker.add_claim(
    source=source,
    text="Bitcoin will reach $150k by end of 2025",
    target_date=date(2025, 12, 31),
    category="crypto",
)
```

### 📊 Accuracy Scoring
Multiple scoring modes — simple, time-windowed, and recency-weighted.

```python
score = tracker.score(source)
print(f"{source.name}: {score.accuracy_score}% ({score.correct_claims}/{score.total_claims})")
```

### 🏆 Leaderboards
Automatic ranking with risers, fallers, and notable results.

```python
board = tracker.leaderboard(min_claims=3)
board.top_accurate      # Best predictors
board.worst_accurate    # Worst predictors
board.biggest_risers    # Improving fast
board.biggest_fallers   # Getting worse
board.notable_wrongs    # High-profile misses
```

### 🔍 Claim Extraction
Extract predictions from text — rule-based (fast) or LLM-powered (accurate).

```python
# Rule-based (no API needed)
claims = tracker.extract_claims(
    text="Elon Musk said Tesla will achieve full self-driving by 2025.",
    source=elon,
)

# LLM-powered (bring your own LLM function)
def my_llm(prompt: str) -> str:
    return response.text

tracker = SignalTracker(llm_fn=my_llm)
claims = tracker.extract_claims(text, source=elon, use_llm=True)
```

### ✅ Multi-Model Verification
Consensus-based verification like Crene's 4-LLM system.

```python
tracker.verify_with_consensus(claim, [
    {"outcome": "correct", "verifier": "ai:claude", "confidence": 0.9},
    {"outcome": "correct", "verifier": "ai:gpt-4", "confidence": 0.85},
    {"outcome": "wrong", "verifier": "ai:gemini", "confidence": 0.6},
])
```

### 📈 Claim Quality Scoring
Automatically rate how verifiable a prediction is.

```python
from signal_tracker import QualityScorer
scorer = QualityScorer()
score = scorer.score(claim)  # 0-100
```

### 💾 Persistence
JSON file or SQLite for larger datasets.

```python
tracker.save("tracker.json")
tracker = SignalTracker.load("tracker.json")

from signal_tracker.storage import SQLiteBackend
backend = SQLiteBackend("tracker.db")
```

---

## Architecture

```
signal-tracker/
├── tracker.py       # SignalTracker — main interface
├── models.py        # Source, Claim, Verification, ScoreSnapshot
├── scoring.py       # AccuracyScorer, QualityScorer
├── extractors.py    # ClaimExtractor (rules + LLM)
├── leaderboard.py   # Leaderboard engine
└── storage.py       # SQLiteBackend
```

**Design principles:** Zero required dependencies (stdlib only), bring your own LLM, pluggable storage, plain dataclasses everywhere.

---

## Use Cases

| Use Case | Who It's For |
|----------|--------------|
| Track media accuracy | Journalists, researchers |
| Score analyst predictions | Finance professionals |
| Monitor political promises | Civic organizations |
| Track AI model forecasts | ML engineers |
| Build prediction markets | Developers |
| Personal prediction journal | Anyone |

---

## Roadmap

- [x] v0.1 — Core tracking, scoring, leaderboards, extraction
- [ ] v0.2 — REST API server (FastAPI)
- [ ] v0.3 — Auto-ingest from RSS, Twitter, YouTube transcripts
- [ ] v0.4 — Dashboard UI (React)
- [ ] v0.5 — Prediction market integrations (Polymarket, Kalshi)
- [ ] v0.6 — Blockchain anchoring for tamper-proof records

---

## About Crene

[Crene](https://crene.com) is an AI-powered prediction intelligence platform that tracks 420+ sources across tech, finance, politics, and geopolitics. We use a 4-LLM consensus system (Claude, GPT-4, Gemini, Grok) to verify claims and score credibility.

Signal Tracker is the open-source framework extracted from Crene's production system. The framework is free — the data is the moat.

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
git clone https://github.com/Creneinc/signal-tracker.git
cd signal-tracker
pip install -e ".[all]"
pytest
```

---

## License

MIT License. See [LICENSE](LICENSE).
MDEOF

# ─── LICENSE ───
cat > LICENSE << 'LICEOF'
MIT License

Copyright (c) 2025 Crene, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
LICEOF

# ─── CONTRIBUTING.md ───
cat > CONTRIBUTING.md << 'CONTEOF'
# Contributing to Signal Tracker

Thanks for your interest in contributing! Here's how to get started.

## Setup

```bash
git clone https://github.com/Creneinc/signal-tracker.git
cd signal-tracker
python -m venv venv
source venv/bin/activate
pip install -e ".[all]"
pip install pytest
```

## Running Tests

```bash
pytest tests/ -v
```

## What We're Looking For

- **New extractors**: RSS feeds, Twitter/X, YouTube transcripts, podcast transcripts
- **Storage backends**: PostgreSQL, MongoDB, Redis
- **Scoring algorithms**: ELO-style ratings, Brier scores, calibration metrics
- **Integrations**: Prediction market APIs (Polymarket, Kalshi, Metaculus)
- **Documentation**: Examples, tutorials, API reference
- **Bug fixes**: Always welcome

## Code Style

- Type hints on all public functions
- Docstrings on all public classes and methods
- No external dependencies in core (stdlib only)
- Optional deps go in `[project.optional-dependencies]`

## Pull Request Process

1. Fork the repo
2. Create a feature branch
3. Write tests for your changes
4. Ensure all tests pass
5. Submit a PR with a clear description

## Questions?

Open an issue or reach out at dev@crene.com.
CONTEOF

# ─── .github/workflows/ci.yml ───
cat > .github/workflows/ci.yml << 'CIEOF'
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[all]"
          pip install pytest

      - name: Run tests
        run: pytest tests/ -v
CIEOF

# ─── .github/workflows/publish.yml ───
cat > .github/workflows/publish.yml << 'PUBEOF'
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install build tools
        run: pip install build twine

      - name: Build package
        run: python -m build

      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
        run: twine upload dist/*
PUBEOF

echo "✅ All files created. Now run:"
echo "  git add ."
echo "  git commit -m 'Add README, LICENSE, CONTRIBUTING, CI workflows'"
echo "  git push"
