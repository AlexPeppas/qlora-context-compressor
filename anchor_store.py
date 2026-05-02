"""anchor_store.py — Append-only disk store for pinned facts extracted from turns.

Each turn is scanned with a lightweight regex extractor (+ spaCy if available)
to pull out named entities, decisions, constraints, key numbers, dates, and
currency amounts.  Extracted facts are written as JSON records to
``{base_dir}/anchors/turn_{n}.json``.

The store is intentionally append-only: existing records are never modified,
which means facts accumulate safely even across multiple sessions.

Usage::

    store = AnchorStore(base_dir=".")
    store.extract_and_save(turn_idx=3, text="User confirmed budget of $50,000.")
    hits = store.search("budget")
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Optional spaCy import (graceful fallback)
# ---------------------------------------------------------------------------

try:
    import spacy  # type: ignore[import]
    _NLP = spacy.load("en_core_web_sm")
    _SPACY_AVAILABLE = True
except Exception:
    _NLP = None
    _SPACY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Regex patterns for heuristic extraction
# ---------------------------------------------------------------------------

# Numeric values: integers, floats, percentages
_RE_NUMBERS = re.compile(r"\b\d[\d,]*(?:\.\d+)?(?:\s*%|\s*percent)?\b")

# Currency amounts: $1,000 / €50 / £200k etc.
_RE_CURRENCY = re.compile(
    r"(?:USD|EUR|GBP|JPY|\$|€|£|¥)\s*\d[\d,]*(?:\.\d+)?[KkMmBb]?\b"
    r"|\b\d[\d,]*(?:\.\d+)?[KkMmBb]?\s*(?:dollars?|euros?|pounds?|yen)\b",
    re.IGNORECASE,
)

# Dates (loose): "April 25, 2026", "2026-04-25", "Q3 2025", "next Monday"
_RE_DATES = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
    r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{1,2}(?:,\s*\d{4})?\b"
    r"|\b\d{4}-\d{2}-\d{2}\b"
    r"|\bQ[1-4]\s+\d{4}\b"
    r"|\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
    re.IGNORECASE,
)

# Decision keywords: phrases signalling a confirmed or rejected choice
_DECISION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(?:decided?|confirmed?|approved?|agreed?|accepted?)\b[^.!?]{0,60}",
        r"\b(?:rejected?|declined?|refused?|cancelled?|denied?)\b[^.!?]{0,60}",
        r"\b(?:chose|chosen|selected?|opted?\s+(?:for|in|out))\b[^.!?]{0,60}",
        r"\bwill\s+(?:use|proceed|go\s+with|implement)\b[^.!?]{0,60}",
    ]
]

# Constraint keywords: requirements and limitations
_CONSTRAINT_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(?:must|should|required?|mandatory|critical|essential)\b[^.!?]{0,80}",
        r"\b(?:must\s+not|should\s+not|cannot|prohibited|forbidden|not\s+allowed)\b[^.!?]{0,80}",
        r"\b(?:limit(?:ed)?\s+to|cap(?:ped)?\s+at|no\s+more\s+than|at\s+least)\b[^.!?]{0,80}",
    ]
]

# Capitalised multi-word proper-noun candidates (fallback NER without spaCy)
_RE_PROPER_NOUNS = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

def _extract_anchors(text: str) -> dict[str, list[str]]:
    """
    Extract structured facts from a piece of text using heuristics.

    Returns a dict with keys:
        - numbers:     numeric values and percentages
        - currencies:  currency amounts
        - dates:       date and time expressions
        - decisions:   sentences containing decision language
        - constraints: sentences containing constraint language
        - entities:    named entities (via spaCy if available, else proper nouns)
    """
    anchors: dict[str, list[str]] = {
        "numbers": [],
        "currencies": [],
        "dates": [],
        "decisions": [],
        "constraints": [],
        "entities": [],
    }

    # Numbers and currencies
    anchors["numbers"] = list(dict.fromkeys(_RE_NUMBERS.findall(text)))
    anchors["currencies"] = list(dict.fromkeys(_RE_CURRENCY.findall(text)))

    # Dates
    anchors["dates"] = list(dict.fromkeys(_RE_DATES.findall(text)))

    # Decisions
    for pat in _DECISION_PATTERNS:
        for m in pat.finditer(text):
            snippet = m.group(0).strip()
            if snippet and snippet not in anchors["decisions"]:
                anchors["decisions"].append(snippet)

    # Constraints
    for pat in _CONSTRAINT_PATTERNS:
        for m in pat.finditer(text):
            snippet = m.group(0).strip()
            if snippet and snippet not in anchors["constraints"]:
                anchors["constraints"].append(snippet)

    # Named entities
    if _SPACY_AVAILABLE and _NLP is not None:
        doc = _NLP(text[:5_000])  # cap to avoid huge inputs
        seen: set[str] = set()
        for ent in doc.ents:
            if ent.text not in seen:
                anchors["entities"].append(ent.text)
                seen.add(ent.text)
    else:
        # Fallback: capitalised multi-word phrases
        candidates = _RE_PROPER_NOUNS.findall(text)
        anchors["entities"] = list(dict.fromkeys(candidates))

    return anchors


# ---------------------------------------------------------------------------
# AnchorStore
# ---------------------------------------------------------------------------

class AnchorStore:
    """
    Append-only disk store for pinned facts extracted from conversation turns.

    Facts are written as JSON files to ``{base_dir}/anchors/turn_{n}.json``.
    The store never deletes or modifies existing records.

    Args:
        base_dir: Root directory under which the ``anchors/`` folder lives.
    """

    def __init__(self, base_dir: str = ".") -> None:
        self._anchors_dir = Path(base_dir) / "anchors"
        self._anchors_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_and_save(self, turn_idx: int, text: str) -> dict[str, Any]:
        """
        Extract facts from *text* and persist them to disk.

        The file ``anchors/turn_{turn_idx}.json`` is created (or overwritten
        if the turn was re-processed).

        Args:
            turn_idx: Index of the conversation turn this text belongs to.
            text:     Raw text of the turn (user or assistant content).

        Returns:
            The extracted anchor dict written to disk.
        """
        anchors = _extract_anchors(text)
        record: dict[str, Any] = {
            "turn_idx": turn_idx,
            "spacy_available": _SPACY_AVAILABLE,
            "anchors": anchors,
            # Keep a short excerpt for context during search
            "text_excerpt": text[:300],
        }
        path = self._anchors_dir / f"turn_{turn_idx}.json"
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        return record

    def search(self, query: str) -> list[dict[str, Any]]:
        """
        Search all anchor records for the given query string.

        Performs case-insensitive substring matching against every string value
        within each JSON record (including entity names, decision phrases, etc.).

        Args:
            query: The search string.

        Returns:
            A list of matching anchor records (dicts), sorted by turn_idx.
        """
        query_lower = query.lower()
        results: list[dict[str, Any]] = []

        for path in sorted(self._anchors_dir.glob("turn_*.json")):
            try:
                record: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

            if self._record_matches(record, query_lower):
                results.append(record)

        return results

    def load(self, turn_idx: int) -> dict[str, Any] | None:
        """
        Load the anchor record for a specific turn, or None if not found.

        Args:
            turn_idx: The turn index to load.

        Returns:
            The anchor dict, or None.
        """
        path = self._anchors_dir / f"turn_{turn_idx}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def all_turn_indices(self) -> list[int]:
        """Return a sorted list of all turn indices that have anchor records."""
        indices: list[int] = []
        for path in self._anchors_dir.glob("turn_*.json"):
            try:
                idx = int(path.stem.split("_")[1])
                indices.append(idx)
            except (IndexError, ValueError):
                pass
        return sorted(indices)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _record_matches(record: dict[str, Any], query: str) -> bool:
        """Recursively check if any string value in *record* contains *query*."""
        for value in record.values():
            if isinstance(value, str) and query in value.lower():
                return True
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and query in item.lower():
                        return True
            if isinstance(value, dict):
                if AnchorStore._record_matches(value, query):
                    return True
        return False
