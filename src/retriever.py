"""BM25 retrieval: tokenize text, build the index, search it."""

import re
from pathlib import Path

import bm25s
import Stemmer
from bm25s.stopwords import STOPWORDS_EN

from .models import (
    Chunk,
    ChunkIndex,
    MinimalSource,
    RagError,
    load_model,
    save_model,
)

CHUNKS_FILE = "chunks.json"
QUESTION_WORDS = [
    "can", "do", "does", "how", "what", "when", "where", "which", "who", "why",
]
STOPWORDS = list(STOPWORDS_EN) + QUESTION_WORDS
STEMMER = Stemmer.Stemmer("english")

IDENTIFIER = re.compile(
    r"\b(?=\w*(?:[A-Za-z0-9]_[A-Za-z0-9]|[a-z][A-Z]|[A-Z][A-Z][a-z]))\w+\b"
)
IDENTIFIER_PART = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def add_identifier_parts(text: str) -> str:
    """Write the parts of each identifier after it.

    "max_num_tokens" becomes "max_num_tokens max num tokens", so a query can
    match the full name or its words.

    Args:
        text: Text to expand.

    Returns:
        The text with identifier parts added.
    """
    def expand(match: re.Match[str]) -> str:
        word = match.group(0)
        return word + " " + " ".join(IDENTIFIER_PART.findall(word))

    return IDENTIFIER.sub(expand, text)


def tokenize(texts: list[str]) -> list[list[str]]:
    """Split identifiers, lowercase, drop stopwords and stem each text.

    Used for both chunks and queries, so they produce the same tokens.

    Args:
        texts: Texts to tokenize.

    Returns:
        One token list per text.
    """
    tokens = bm25s.tokenize(
        [add_identifier_parts(text) for text in texts],
        stopwords=STOPWORDS,
        stemmer=STEMMER,
        return_ids=False,
        show_progress=False,
    )
    return [list(document) for document in tokens]


def build_index(chunks: list[Chunk], max_chunk_size: int,
                output_dir: str) -> None:
    """Build the BM25 index over chunks and save it in output_dir.

    Chunk i in chunks.json is BM25 document i.

    Args:
        chunks: Chunks to index.
        max_chunk_size: Chunk size used, saved with the chunks.
        output_dir: Folder to write the index to.

    Raises:
        RagError: If the index cannot be saved.
    """
    texts = [chunk.file_path + "\n" + chunk.text for chunk in chunks]
    print(f"Building BM25 index over {len(chunks)} chunks...")
    model = bm25s.BM25(k1=1.2, b=0.75)
    model.index(tokenize(texts), show_progress=False)
    try:
        model.save(output_dir, show_progress=False)
    except OSError as error:
        raise RagError(
            f"Cannot write index to {output_dir}: {error.strerror}"
        ) from None
    save_model(ChunkIndex(max_chunk_size=max_chunk_size, chunks=chunks),
               Path(output_dir) / CHUNKS_FILE)


class Retriever:
    """Search engine over a saved index."""
    def __init__(self, index_dir: str) -> None:
        """Load the index and check its two parts match.

        Args:
            index_dir: Folder holding the index.

        Raises:
            RagError: If the index is missing, damaged or inconsistent.
        """
        chunks_path = Path(index_dir) / CHUNKS_FILE
        if not chunks_path.is_file():
            raise RagError(
                f"No index found in {index_dir}. Run the index command first."
            )
        self.chunks = load_model(chunks_path, ChunkIndex).chunks
        try:
            self.model = bm25s.BM25.load(index_dir, show_progress=False)
        except (OSError, ValueError, KeyError):
            raise RagError(
                f"Index in {index_dir} is unreadable or damaged. "
                "Run the index command again."
            ) from None
        if len(self.chunks) != int(self.model.scores["num_docs"]):
            raise RagError(
                f"Index files in {index_dir} do not match. "
                "Run the index command again."
            )

    def search(self, query: str, k: int) -> list[MinimalSource]:
        """Return the top-k sources for a query, best first.

        Only chunks with a score above 0 are returned, so a query with no
        known word gets no result.

        Args:
            query: Question to search for.
            k: Number of sources to return.

        Returns:
            Sources, best first.
        """
        tokens = tokenize([query])[0]
        if not tokens:
            return []
        ids, scores = self.model.retrieve(
            [tokens], k=min(k, len(self.chunks)), show_progress=False
        )
        sources: list[MinimalSource] = []
        for chunk_id, score in zip(ids[0], scores[0]):
            if score > 0:
                chunk = self.chunks[int(chunk_id)]
                sources.append(MinimalSource(
                    file_path=chunk.file_path,
                    first_character_index=chunk.first_character_index,
                    last_character_index=chunk.last_character_index,
                ))
        return sources
