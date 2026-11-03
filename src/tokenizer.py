import re

import Stemmer
from bm25s.stopwords import STOPWORDS_EN

WORD_RE = re.compile(r"[A-Za-z0-9_]+")
CAMEL_BOUNDARY_RE = re.compile(
    r"(?<=[a-z]{2})(?=[A-Z])|(?<=[A-Za-z0-9])(?=[A-Z][a-z]{2,})"
)
QUESTION_WORDS = {
    "can", "do", "does", "how", "what", "when", "where", "which", "who", "why",
}
STOPWORDS = frozenset(STOPWORDS_EN) | QUESTION_WORDS
STEMMER = Stemmer.Stemmer("english")


def split_identifier(word: str) -> list[str]:
    parts: list[str] = []
    for piece in word.split("_"):
        parts.extend(part for part in CAMEL_BOUNDARY_RE.split(piece) if part)
    return parts


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for word in WORD_RE.findall(text):
        parts = split_identifier(word)
        candidates = [word.strip("_")] + parts if len(parts) > 1 else parts
        for candidate in candidates:
            token = candidate.lower()
            if len(token) > 1 and token not in STOPWORDS:
                tokens.append(token)
    return list(STEMMER.stemWords(tokens))
