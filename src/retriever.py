"""BM25 search over the saved index."""

from pathlib import Path

from tqdm import tqdm

from .bm25 import BM25
from .indexer import CHUNKS_FILE
from .io_utils import RagError, load_model
from .models import ChunkIndex, MinimalSource
from .tokenizer import tokenize


class Retriever:
    """Search engine over a saved index."""
    def __init__(self, index_dir: str | Path) -> None:
        """Load the index and check its parts match.

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
        self.bm25 = BM25.load(index_dir)
        if len(self.chunks) != self.bm25.size:
            raise RagError(
                f"Index files in {index_dir} do not match. "
                "Run the index command again."
            )

    def search(self, query: str, k: int) -> list[MinimalSource]:
        """Return the top-k sources for one query.

        Args:
            query: Question to search for.
            k: Number of sources to return.

        Returns:
            Sources, best first.
        """
        return self.search_many([query], k, show_progress=False)[0]

    def search_many(self, queries: list[str], k: int,
                    show_progress: bool = True) -> list[list[MinimalSource]]:
        """Return the top-k sources for each query.

        Args:
            queries: Questions to search for.
            k: Number of sources per question.
            show_progress: Show a progress bar.

        Returns:
            One source list per question.
        """
        if not queries:
            return []
        token_lists = tokenize(queries)
        results: list[list[MinimalSource]] = []
        for tokens in tqdm(token_lists, desc="Searching",
                           disable=not show_progress):
            ranking = self.bm25.top_k(tokens, k)
            results.append([self.to_source(i) for i, _ in ranking])
        return results

    def to_source(self, chunk_id: int) -> MinimalSource:
        """Convert a chunk id to a MinimalSource.

        Args:
            chunk_id: Row of the chunk in the index.

        Returns:
            The chunk's span.
        """
        chunk = self.chunks[chunk_id]
        return MinimalSource(
            file_path=chunk.file_path,
            first_character_index=chunk.first_character_index,
            last_character_index=chunk.last_character_index,
        )
