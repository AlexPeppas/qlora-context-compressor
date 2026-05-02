"""compressor.py - Heuristic extractive compressor (stand-in for LongT5).

Uses sumy (LexRank or LSA) for extractive summarisation, with a graceful
fallback to a simple sentence-ranking heuristic if sumy is not installed or
its NLTK tokenizer data is unavailable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Optional sumy import
# ---------------------------------------------------------------------------

try:
    from sumy.parsers.plaintext import PlaintextParser  # type: ignore[import]
    from sumy.nlp.tokenizers import Tokenizer as SumyTokenizer  # type: ignore[import]
    from sumy.summarizers.lex_rank import LexRankSummarizer  # type: ignore[import]
    from sumy.summarizers.lsa import LsaSummarizer  # type: ignore[import]
    from sumy.nlp.stemmers import Stemmer  # type: ignore[import]
    from sumy.utils import get_stop_words  # type: ignore[import]
    _SUMY_AVAILABLE = True
except ImportError:
    _SUMY_AVAILABLE = False


# ---------------------------------------------------------------------------
# NLTK availability probe -- tested at import time so we know immediately
# whether punkt_tab data is present before any compression job runs.
# ---------------------------------------------------------------------------

def _probe_nltk() -> bool:
    """Return True only if nltk.sent_tokenize actually works in this environment."""
    try:
        import nltk  # type: ignore[import]
        nltk.sent_tokenize("Hello world. Test sentence.")
        return True
    except Exception:
        return False


_NLTK_AVAILABLE: bool = _probe_nltk()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LANGUAGE = "english"
MIN_SENTENCES_OUT = 1


# ---------------------------------------------------------------------------
# Fidelity helpers
# ---------------------------------------------------------------------------

_RE_KEY_TOKENS = re.compile(
    r"\b\d[\d,]*(?:\.\d+)?(?:\s*%|[KkMmBb])?\b"
    r"|(?:USD|EUR|GBP|\$|€|£)\s*\d[\d,]*(?:\.\d+)?"
    r"|\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b"
)


def _extract_key_tokens(text: str) -> set[str]:
    """Extract key tokens (numbers, currencies, proper nouns) from text."""
    return {m.group(0).strip() for m in _RE_KEY_TOKENS.finditer(text)}


def _fidelity_overlap(original: str, compressed: str) -> float:
    """
    Return the fraction of key tokens from original that appear in compressed.

    A score of 1.0 means all key tokens were preserved; 0.0 means none were.
    Returns 1.0 if the original has no key tokens (nothing to lose).
    """
    original_tokens = _extract_key_tokens(original)
    if not original_tokens:
        return 1.0
    preserved = sum(1 for tok in original_tokens if tok in compressed)
    return preserved / len(original_tokens)


# ---------------------------------------------------------------------------
# Sentence splitter -- regex-first, NLTK only if available
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences.

    Uses NLTK sent_tokenize if punkt_tab data is available, otherwise
    falls back to a punctuation-based regex splitter that requires no
    external data.
    """
    if _NLTK_AVAILABLE:
        try:
            import nltk  # type: ignore[import]
            return nltk.sent_tokenize(text)
        except Exception:
            pass  # data unavailable -- fall through to regex splitter
    # Regex fallback: no external data required
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Fallback sentence-ranking compressor (no external deps beyond regex)
# ---------------------------------------------------------------------------

def _score_sentence(sentence: str, word_freq: dict[str, int]) -> float:
    """Score a sentence by summing normalised word frequencies."""
    words = re.findall(r"\b\w+\b", sentence.lower())
    if not words:
        return 0.0
    return sum(word_freq.get(w, 0) for w in words) / len(words)


def _fallback_compress(text: str, target_sentences: int) -> str:
    """
    Extractive compression without external libraries.

    Ranks sentences by word frequency and returns the top target_sentences
    in their original order.
    """
    sentences = _split_sentences(text)
    if len(sentences) <= target_sentences:
        return text

    words = [w for w in re.findall(r"\b\w+\b", text.lower()) if len(w) > 3]
    freq: dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    scored = [(i, _score_sentence(s, freq), s) for i, s in enumerate(sentences)]
    top_indices = sorted(
        [idx for idx, _, _ in sorted(scored, key=lambda x: -x[1])[:target_sentences]]
    )
    return " ".join(sentences[i] for i in top_indices)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CompressionResult:
    """The output of a single compression pass."""

    original_text: str
    compressed_text: str
    target_ratio: float
    actual_ratio: float
    fidelity_overlap: float
    method: str  # "sumy_lexrank" | "sumy_lsa" | "fallback" | "passthrough"

    @property
    def original_chars(self) -> int:
        return len(self.original_text)

    @property
    def compressed_chars(self) -> int:
        return len(self.compressed_text)

    def summary_line(self) -> str:
        return (
            f"method={self.method}  "
            f"ratio={self.actual_ratio:.1f}x (target {self.target_ratio:.1f}x)  "
            f"fidelity={self.fidelity_overlap:.0%}  "
            f"chars={self.original_chars}>{self.compressed_chars}"
        )


# ---------------------------------------------------------------------------
# HeuristicCompressor
# ---------------------------------------------------------------------------

class HeuristicCompressor:
    """
    Heuristic extractive compressor -- stand-in for the future LongT5 model.

    Compression is performed by selecting the most important sentences from
    the input text using LexRank (preferred) or LSA (if LexRank fails), with
    a pure-Python fallback based on word frequency scoring.

    Args:
        prefer_lsa: If True, use LSA summariser instead of LexRank.
    """

    def __init__(self, prefer_lsa: bool = False) -> None:
        self._prefer_lsa = prefer_lsa
        self._method: str = self._detect_method()

    def _detect_method(self) -> str:
        if _SUMY_AVAILABLE:
            return "sumy_lsa" if self._prefer_lsa else "sumy_lexrank"
        return "fallback"

    def compress(
        self,
        text: str,
        target_ratio: float,
        is_recent: bool = False,
        max_ratio: float | None = None,
    ) -> CompressionResult:
        """
        Compress text to approximately 1/target_ratio of its original length.

        Args:
            text:         The raw text to compress (e.g. a formatted segment).
            target_ratio: Desired compression ratio (e.g. 3.0 means 1/3 of original).
            is_recent:    Hint that this is a recent segment (limits aggressiveness).
            max_ratio:    Hard cap on compression ratio. Defaults to 3.0 for recent
                          segments and 10.0 for older ones.

        Returns:
            A CompressionResult with the compressed text and fidelity metrics.
        """
        from .context_store import RECENT_MAX_RATIO, OLD_MAX_RATIO

        if max_ratio is None:
            max_ratio = RECENT_MAX_RATIO if is_recent else OLD_MAX_RATIO

        # Clamp the ratio to the allowed maximum
        effective_ratio = min(target_ratio, max_ratio)

        sentences = _split_sentences(text)
        if len(sentences) <= 1:
            return CompressionResult(
                original_text=text,
                compressed_text=text,
                target_ratio=effective_ratio,
                actual_ratio=1.0,
                fidelity_overlap=1.0,
                method="passthrough",
            )

        target_sentences = max(
            MIN_SENTENCES_OUT,
            int(len(sentences) / effective_ratio),
        )

        compressed = self._run_compression(text, sentences, target_sentences)
        actual_ratio = len(text) / max(1, len(compressed))
        fidelity = _fidelity_overlap(text, compressed)

        return CompressionResult(
            original_text=text,
            compressed_text=compressed,
            target_ratio=effective_ratio,
            actual_ratio=actual_ratio,
            fidelity_overlap=fidelity,
            method=self._method,
        )

    def _run_compression(
        self,
        text: str,
        sentences: list[str],
        target_sentences: int,
    ) -> str:
        """Dispatch to the appropriate backend."""
        if _SUMY_AVAILABLE:
            result = self._sumy_compress(text, target_sentences)
            if result:
                return result
        self._method = "fallback"
        return _fallback_compress(text, target_sentences)

    def _sumy_compress(self, text: str, target_sentences: int) -> str | None:
        """Run sumy LexRank or LSA and return the compressed text, or None on error."""
        try:
            parser = PlaintextParser.from_string(text, SumyTokenizer(LANGUAGE))
            stemmer = Stemmer(LANGUAGE)
            stop_words = get_stop_words(LANGUAGE)

            if self._prefer_lsa:
                summariser = LsaSummarizer(stemmer)
                self._method = "sumy_lsa"
            else:
                summariser = LexRankSummarizer(stemmer)
                self._method = "sumy_lexrank"

            summariser.stop_words = stop_words
            summary_sentences = summariser(parser.document, target_sentences)
            result = " ".join(str(s) for s in summary_sentences)
            if result.strip():
                return result
            return None
        except Exception:
            return None
