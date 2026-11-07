"""Hybrid BM25 and embedding search."""

from pathlib import Path

from tqdm import tqdm

from .bm25 import BM25
from .embeddings import Embedder, dense_top_k, load_embeddings
from .indexer import CHUNKS_FILE
from .io_utils import RagError, load_model
from .models import ChunkIndex, MinimalSource
from .tokenizer import tokenize

CANDIDATES = 100
DENSE_WEIGHT = 0.25
RRF_CONSTANT = 60


def reciprocal_rank_fusion(rankings: list[list[int]],
                           weights: list[float]) -> list[int]:
    """Merge ranked id lists with weighted RRF.

    Args:
        rankings: Ranked chunk id lists.
        weights: Weight of each ranking.

    Returns:
        Chunk ids sorted by fused score.
    """
    scores: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = (
                scores.get(chunk_id, 0.0) + weight / (RRF_CONSTANT + rank)
            )
    return sorted(scores, key=scores.__getitem__, reverse=True)


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
        self.vectors = load_embeddings(index_dir)
        if not len(self.chunks) == self.bm25.size == len(self.vectors):
            raise RagError(
                f"Index files in {index_dir} do not match. "
                "Run the index command again."
            )
        self.embedder = Embedder()

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
        query_vectors = self.embedder.encode(queries)
        candidates = max(CANDIDATES, k)
        results: list[list[MinimalSource]] = []
        for query, tokens, query_vector in tqdm(
            zip(queries, token_lists, query_vectors),
            total=len(queries), desc="Searching", disable=not show_progress,
        ):
            bm25_ranking = [i for i, _ in self.bm25.top_k(tokens, candidates)]
            if not query.strip() or not bm25_ranking:
                results.append([])
                continue
            dense_ranking = [
                i for i, _ in dense_top_k(self.vectors, query_vector,
                                          candidates)
            ]
            fused = reciprocal_rank_fusion([bm25_ranking, dense_ranking],
                                           [1.0, DENSE_WEIGHT])
            results.append([self.to_source(i) for i in fused[:k]])
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
