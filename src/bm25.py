"""BM25 lexical index built on bm25s."""

from pathlib import Path

import bm25s

from .io_utils import RagError

PARAMS_FILE = "params.index.json"


class BM25:
    """BM25 index over tokenized chunks."""
    def __init__(self, model: bm25s.BM25) -> None:
        """Wrap a bm25s model.

        Args:
            model: bm25s model to wrap.
        """
        self.model = model

    @classmethod
    def build(cls, documents: list[list[str]], k1: float = 1.5,
              b: float = 0.75) -> "BM25":
        """Build an index from tokenized documents.

        Args:
            documents: Token list of each document.
            k1: Term frequency saturation.
            b: Length normalization strength.

        Returns:
            The built index.

        Raises:
            RagError: If documents is empty.
        """
        if not documents:
            raise RagError("Cannot build a BM25 index without documents")
        model = bm25s.BM25(k1=k1, b=b)
        model.index(documents, show_progress=False)
        return cls(model)

    def save(self, directory: str | Path) -> None:
        """Save the index to a directory.

        Args:
            directory: Folder to write the index files to.

        Raises:
            RagError: If the files cannot be written.
        """
        try:
            self.model.save(str(directory), show_progress=False)
        except OSError as error:
            raise RagError(
                f"Cannot write BM25 index to {directory}: {error.strerror}"
            ) from None

    @classmethod
    def load(cls, directory: str | Path) -> "BM25":
        """Load an index saved with save().

        Args:
            directory: Folder holding the index files.

        Returns:
            The loaded index.

        Raises:
            RagError: If the index is missing or damaged.
        """
        folder = Path(directory)
        if not (folder / PARAMS_FILE).is_file():
            raise RagError(
                f"No index found in {folder}. Run the index command first."
            )
        try:
            model = bm25s.BM25.load(str(folder), show_progress=False)
        except (OSError, ValueError, KeyError):
            raise RagError(
                f"BM25 index in {folder} is unreadable or damaged. "
                "Run the index command again."
            ) from None
        return cls(model)

    @property
    def size(self) -> int:
        """Number of indexed documents."""
        return int(self.model.scores["num_docs"])

    def top_k(self, tokens: list[str], k: int) -> list[tuple[int, float]]:
        """Return up to k (document id, score) pairs with score > 0.

        Args:
            tokens: Query tokens.
            k: Maximum number of results.

        Returns:
            Pairs sorted by descending score.
        """
        if not tokens or k <= 0:
            return []
        ids, scores = self.model.retrieve(
            [tokens], k=min(k, self.size), show_progress=False
        )
        return [
            (int(doc_id), float(score))
            for doc_id, score in zip(ids[0], scores[0])
            if score > 0
        ]
