import re

WORD_RE = re.compile(r"[A-Za-z0-9_]+")
CAMEL_BOUNDARY_RE = re.compile(
    r"(?<=[a-z]{2})(?=[A-Z])|(?<=[A-Za-z0-9])(?=[A-Z][a-z]{2,})"
)
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "how", "in", "is", "it", "of", "on", "or", "that",
    "the", "this", "to", "was", "what", "when", "where", "which", "who",
    "why", "with",
})


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
    return tokens
