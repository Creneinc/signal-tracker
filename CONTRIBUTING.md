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
