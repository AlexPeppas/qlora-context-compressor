"""Deterministic sanity metrics (secondary quality-control layer).

These are NOT the headline metrics — the triad (faithfulness, downstream,
tier-appropriate) is. But they are fully deterministic (regex + counting),
cheap, and catch the specific failure modes the EOS bug produced. Reported
as a sanity sub-table so reviewers can see the pathologies are absent.

Every detector returns a `SanityResult` with a scalar value plus the
matched substrings/spans, so false positives are auditable (rubber-duck
recommendation from the Phase 0.5 review).

Detectors:
  * repetition_n8      fraction of duplicate 8-grams (continuous, 0-1)
  * bracketed_tag_format  count of [Capitalized: value] structural markers
  * meta_leakage       count of meta-conversational phrases (lower bound)
  * surface_natural_end  does output end with sentence-final punctuation

Note: `stopped_on_eos` is a REAL field recorded by the baselines (whether
the decoder emitted EOS), so it lives on the CompressionResult, not here.
surface_natural_end is only a text proxy kept for historical comparison
against the pre-EOS-fix bake-off, which lacked token-level stop reasons.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class SanityResult:
    name: str
    value: float  # scalar (bool as 0/1, count, or fraction)
    matches: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {"name": self.name, "value": self.value, "matches": list(self.matches)}


# ---------------------------------------------------------------------------
# Repetition collapse
# ---------------------------------------------------------------------------


def repetition_n8(text: str, n: int = 8) -> SanityResult:
    """Fraction of duplicate n-grams (default 8). A value near 0 is healthy;
    high values indicate phrase/sentence-level looping. Continuous — the
    paper reports the distribution; any binary 'collapse' threshold is
    calibrated separately against clean references (see rubric)."""
    tokens = text.split()
    if len(tokens) < n * 2:
        return SanityResult("repetition_n8", 0.0)
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return SanityResult("repetition_n8", 0.0)
    unique = len(set(grams))
    dup_fraction = 1.0 - unique / len(grams)
    # Surface a few example repeated grams for auditing
    seen: set[tuple[str, ...]] = set()
    repeated: list[str] = []
    for g in grams:
        if g in seen and " ".join(g) not in repeated:
            repeated.append(" ".join(g))
            if len(repeated) >= 5:
                break
        seen.add(g)
    return SanityResult("repetition_n8", dup_fraction, tuple(repeated))


# ---------------------------------------------------------------------------
# Hallucinated bracketed structural tags
# ---------------------------------------------------------------------------

# [Capitalized Phrase: value] — fabricated structural markers like
# [Status: ...], [Priority: ...]. See eosfix_results.md.
_BRACKET_TAG_RE = re.compile(r"\[[A-Z][A-Za-z ]+:\s*[^\]]+\]")


def bracketed_tag_format(text: str) -> SanityResult:
    """Count of bracketed structural-marker formats. Named for FORMAT, not
    hallucination — it detects the tag-like format that broken adapters
    emitted, not semantic hallucination (rubber-duck rename)."""
    matches = _BRACKET_TAG_RE.findall(text)
    return SanityResult(
        "bracketed_tag_format", float(len(matches)), tuple(matches[:10])
    )


# ---------------------------------------------------------------------------
# Meta-leakage (lower-bound detector)
# ---------------------------------------------------------------------------

# Principled taxonomy (rubber-duck recommendation), not ad-hoc phrase
# accretion. High-precision, low-recall -> rates are a LOWER BOUND.
_META_PATTERNS = [
    # conversation-state
    r"conversation is ongoing",
    r"conversation continues",
    r"ongoing conversation",
    r"the conversation (?:is|was|has)",
    # user-intent metacommentary
    r"user is asking",
    r"user asks",
    r"user wants",
    r"user requested",
    r"the user's request",
    # assistant-role metacommentary
    r"assistant is",
    r"the assistant (?:is|will|should|provided|diagnosed|explained)",
    r"assistant's response",
    r"as an ai",
    r"as the assistant",
]
_META_RE = re.compile("|".join(f"(?:{p})" for p in _META_PATTERNS), re.IGNORECASE)


def meta_leakage(text: str) -> SanityResult:
    """Count of meta-conversational phrases. A compression should compress
    CONTENT, not describe the conversation. Lower-bound (paraphrases evade)."""
    matches = _META_RE.findall(text)
    return SanityResult("meta_leakage", float(len(matches)), tuple(matches[:10]))


# ---------------------------------------------------------------------------
# Surface natural end (text proxy; prefer stopped_on_eos where available)
# ---------------------------------------------------------------------------

_SENTENCE_END_RE = re.compile(r"[.!?\"`)\]\}]\s*$")


def surface_natural_end(text: str) -> SanityResult:
    """Whether the output ends with sentence-final punctuation / closing
    delimiter. A weak proxy for healthy stopping; superseded by the real
    `stopped_on_eos` token-level field on CompressionResult for post-fix
    runs. Kept for comparability with the pre-fix bake-off."""
    stripped = text.rstrip()
    ok = bool(_SENTENCE_END_RE.search(stripped)) if stripped else False
    return SanityResult("surface_natural_end", 1.0 if ok else 0.0)


# ---------------------------------------------------------------------------
# Batch convenience
# ---------------------------------------------------------------------------


def all_sanity_metrics(text: str) -> dict[str, SanityResult]:
    """Run every detector on one compression, keyed by metric name."""
    return {
        r.name: r
        for r in (
            repetition_n8(text),
            bracketed_tag_format(text),
            meta_leakage(text),
            surface_natural_end(text),
        )
    }


__all__ = [
    "SanityResult",
    "all_sanity_metrics",
    "bracketed_tag_format",
    "meta_leakage",
    "repetition_n8",
    "surface_natural_end",
]
