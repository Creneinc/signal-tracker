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
