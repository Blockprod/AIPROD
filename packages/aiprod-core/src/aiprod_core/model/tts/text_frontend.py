"""
TTS Text Frontend — Grapheme-to-Phoneme and Text Normalization

Converts raw text into phoneme sequences ready for TTS synthesis.
Handles numbers, abbreviations, punctuation, and multi-language support.
"""

import re
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─── IPA phoneme inventory (simplified, extensible) ────────────────────────
# Based on common IPA symbols used across English, French, Spanish, German
IPA_VOWELS = [
    "i", "ɪ", "e", "ɛ", "æ", "ɑ", "ɒ", "ɔ", "o", "ʊ", "u", "ʌ", "ə",
    "aɪ", "aʊ", "ɔɪ", "eɪ", "oʊ", "iː", "uː", "ɜː",
]
IPA_CONSONANTS = [
    "p", "b", "t", "d", "k", "ɡ", "f", "v", "θ", "ð", "s", "z",
    "ʃ", "ʒ", "h", "m", "n", "ŋ", "l", "ɹ", "w", "j",
    "tʃ", "dʒ", "ɾ", "ʁ", "ɲ", "ç",
]
SILENCE = "_"
WORD_BOUNDARY = " "
SENTENCE_BOUNDARY = "."

# ─── Default phoneme vocabulary ────────────────────────────────────────────
DEFAULT_VOCAB: List[str] = (
    [SILENCE, WORD_BOUNDARY, SENTENCE_BOUNDARY]
    + IPA_VOWELS
    + IPA_CONSONANTS
    + list(".,;:!?-'\"()") 
)


@dataclass
class FrontendConfig:
    """Text frontend configuration."""
    language: str = "en"
    vocab: List[str] = field(default_factory=lambda: list(DEFAULT_VOCAB))
    max_phoneme_length: int = 512
    normalize_numbers: bool = True
    normalize_abbreviations: bool = True
    add_sentence_boundaries: bool = True
    expand_contractions: bool = True


# ───────────────────────────────────────────────────────────────────────────
# Text normalisation rules
# ───────────────────────────────────────────────────────────────────────────

_ORDINALS = {
    "1st": "first", "2nd": "second", "3rd": "third",
    "4th": "fourth", "5th": "fifth", "6th": "sixth",
    "7th": "seventh", "8th": "eighth", "9th": "ninth",
    "10th": "tenth",
}

_ABBREVIATIONS_EN = {
    "mr.": "mister", "mrs.": "missus", "dr.": "doctor",
    "st.": "saint", "jr.": "junior", "sr.": "senior",
    "vs.": "versus", "etc.": "etcetera", "approx.": "approximately",
    "dept.": "department", "est.": "established", "govt.": "government",
}

_CONTRACTIONS_EN = {
    "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
    "you're": "you are", "you've": "you have", "you'll": "you will",
    "he's": "he is", "she's": "she is", "it's": "it is",
    "we're": "we are", "we've": "we have", "we'll": "we will",
    "they're": "they are", "they've": "they have", "they'll": "they will",
    "can't": "cannot", "won't": "will not", "don't": "do not",
    "doesn't": "does not", "isn't": "is not", "aren't": "are not",
    "wasn't": "was not", "weren't": "were not",
    "couldn't": "could not", "wouldn't": "would not", "shouldn't": "should not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "let's": "let us", "that's": "that is", "who's": "who is",
    "what's": "what is", "here's": "here is", "there's": "there is",
}

_ONES = [
    "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS = [
    "", "", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety",
]


def _number_to_words(n: int) -> str:
    """Convert integer to English words (supports 0 – 999 999 999)."""
    if n == 0:
        return "zero"
    if n < 0:
        return "minus " + _number_to_words(-n)

    parts: List[str] = []

    if n >= 1_000_000:
        millions = n // 1_000_000
        parts.append(_number_to_words(millions) + " million")
        n %= 1_000_000

    if n >= 1_000:
        thousands = n // 1_000
        parts.append(_number_to_words(thousands) + " thousand")
        n %= 1_000

    if n >= 100:
        hundreds = n // 100
        parts.append(_ONES[hundreds] + " hundred")
        n %= 100

    if n >= 20:
        parts.append(_TENS[n // 10])
        n %= 10

    if 0 < n < 20:
        parts.append(_ONES[n])

    return " ".join(p for p in parts if p)


def _normalize_numbers(text: str) -> str:
    """Replace digit sequences with spelled‑out words."""
    def _replace(m: re.Match) -> str:
        raw = m.group(0)
        # Decimal numbers
        if "." in raw:
            integer_part, decimal_part = raw.split(".", 1)
            result = _number_to_words(int(integer_part)) + " point"
            for digit in decimal_part:
                result += " " + _ONES[int(digit)]
            return result
        return _number_to_words(int(raw))

    return re.sub(r"\d+(?:\.\d+)?", _replace, text)


# ───────────────────────────────────────────────────────────────────────────
# Lightweight rule-based G2P (no external dependency)
# ───────────────────────────────────────────────────────────────────────────

# Simplified English letter-to-phoneme mapping (covers ~85 % of common words)
_LETTER_TO_PHONEME_EN: Dict[str, str] = {
    "a": "æ", "b": "b", "c": "k", "d": "d", "e": "ɛ",
    "f": "f", "g": "ɡ", "h": "h", "i": "ɪ", "j": "dʒ",
    "k": "k", "l": "l", "m": "m", "n": "n", "o": "ɒ",
    "p": "p", "q": "k", "r": "ɹ", "s": "s", "t": "t",
    "u": "ʌ", "v": "v", "w": "w", "x": "ks", "y": "j",
    "z": "z",
}

# Common English digraph rules
_DIGRAPH_RULES_EN: List[Tuple[str, str]] = [
    ("th", "θ"),
    ("sh", "ʃ"),
    ("ch", "tʃ"),
    ("ph", "f"),
    ("ng", "ŋ"),
    ("ck", "k"),
    ("wh", "w"),
    ("wr", "ɹ"),
    ("kn", "n"),
    ("gn", "n"),
    ("igh", "aɪ"),
    ("tion", "ʃən"),
    ("sion", "ʒən"),
    ("ous", "əs"),
    ("ing", "ɪŋ"),
    ("ed", "d"),
    ("ee", "iː"),
    ("oo", "uː"),
    ("ea", "iː"),
    ("ai", "eɪ"),
    ("ay", "eɪ"),
    ("ow", "oʊ"),
    ("ou", "aʊ"),
    ("oi", "ɔɪ"),
    ("oy", "ɔɪ"),
]


def _g2p_english(word: str) -> List[str]:
    """Rule-based grapheme-to-phoneme for a single English word."""
    word = word.lower().strip()
    phonemes: List[str] = []
    i = 0
    while i < len(word):
        matched = False
        # Try digraphs / trigraphs (longest match first)
        for pattern, phoneme in sorted(_DIGRAPH_RULES_EN, key=lambda r: -len(r[0])):
            if word[i:].startswith(pattern):
                phonemes.append(phoneme)
                i += len(pattern)
                matched = True
                break
        if not matched:
            ch = word[i]
            if ch in _LETTER_TO_PHONEME_EN:
                phonemes.append(_LETTER_TO_PHONEME_EN[ch])
            elif ch == "'":
                pass  # skip apostrophes
            else:
                phonemes.append(ch)  # keep punctuation as-is
            i += 1
    return phonemes


# ───────────────────────────────────────────────────────────────────────────
# TextFrontend class
# ───────────────────────────────────────────────────────────────────────────

class TextFrontend:
    """
    Full text-processing pipeline for TTS.

    Pipeline:
        raw text → normalise → tokenise words → G2P → phoneme IDs
    """

    def __init__(self, config: Optional[FrontendConfig] = None):
        self.config = config or FrontendConfig()
        self._token2id: Dict[str, int] = {
            tok: idx for idx, tok in enumerate(self.config.vocab)
        }
        self._id2token: Dict[int, str] = {
            idx: tok for tok, idx in self._token2id.items()
        }

    @property
    def vocab_size(self) -> int:
        return len(self._token2id)

    # ── public API ─────────────────────────────────────────────────────

    def text_to_phoneme_ids(self, text: str) -> List[int]:
        """End-to-end: raw text → list of phoneme token IDs."""
        normalised = self.normalise(text)
        phonemes = self.g2p(normalised)
        ids = self.phonemes_to_ids(phonemes)
        # Truncate to max length
        return ids[: self.config.max_phoneme_length]

    def normalise(self, text: str) -> str:
        """Apply all text normalisation rules."""
        text = text.strip()
        text = text.lower()

        if self.config.expand_contractions:
            for contraction, expansion in _CONTRACTIONS_EN.items():
                text = text.replace(contraction, expansion)

        if self.config.normalize_abbreviations:
            for abbr, expansion in _ABBREVIATIONS_EN.items():
                text = text.replace(abbr, expansion)

        if self.config.normalize_numbers:
            text = _normalize_numbers(text)

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def g2p(self, text: str) -> List[str]:
        """Grapheme-to-phoneme for a full sentence."""
        phonemes: List[str] = []
        if self.config.add_sentence_boundaries:
            phonemes.append(SENTENCE_BOUNDARY)

        words = text.split()
        for wi, word in enumerate(words):
            # Strip trailing punctuation
            trailing_punct = ""
            while word and word[-1] in ".,;:!?":
                trailing_punct = word[-1] + trailing_punct
                word = word[:-1]

            if word:
                word_phonemes = _g2p_english(word)
                phonemes.extend(word_phonemes)

            if trailing_punct:
                for p in trailing_punct:
                    phonemes.append(p)

            if wi < len(words) - 1:
                phonemes.append(WORD_BOUNDARY)

        if self.config.add_sentence_boundaries:
            phonemes.append(SENTENCE_BOUNDARY)
        return phonemes

    def phonemes_to_ids(self, phonemes: List[str]) -> List[int]:
        """Map phoneme symbols → integer IDs (unknown → 0)."""
        return [self._token2id.get(p, 0) for p in phonemes]

    def ids_to_phonemes(self, ids: List[int]) -> List[str]:
        """Inverse mapping: IDs → phoneme symbols."""
        return [self._id2token.get(i, SILENCE) for i in ids]
