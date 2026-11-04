import bm25s
import Stemmer
from bm25s.stopwords import STOPWORDS_EN

QUESTION_WORDS = [
    "can", "do", "does", "how", "what", "when", "where", "which", "who", "why",
]
STOPWORDS = list(STOPWORDS_EN) + QUESTION_WORDS
STEMMER = Stemmer.Stemmer("english")


def tokenize(texts: list[str]) -> list[list[str]]:
    tokens = bm25s.tokenize(
        texts,
        stopwords=STOPWORDS,
        stemmer=STEMMER,
        return_ids=False,
        show_progress=False,
    )
    return [list(document) for document in tokens]
