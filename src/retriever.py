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
    scores: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = (
                scores.get(chunk_id, 0.0) + weight / (RRF_CONSTANT + rank)
            )
    return sorted(scores, key=scores.__getitem__, reverse=True)


class Retriever:
    def __init__(self, index_dir: str | Path) -> None:
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
        return self.search_many([query], k, show_progress=False)[0]

    def search_many(self, queries: list[str], k: int,
                    show_progress: bool = True) -> list[list[MinimalSource]]:
        token_lists = tokenize(queries)
        query_vectors = self.embedder.encode(queries)
        results: list[list[MinimalSource]] = []
        for query, tokens, query_vector in tqdm(
            zip(queries, token_lists, query_vectors),
            total=len(queries), desc="Searching", disable=not show_progress,
        ):
            if not query.strip():
                results.append([])
                continue
            bm25_ranking = [i for i, _ in self.bm25.top_k(tokens, CANDIDATES)]
            dense_ranking = [
                i for i, _ in dense_top_k(self.vectors, query_vector,
                                          CANDIDATES)
            ]
            fused = reciprocal_rank_fusion([bm25_ranking, dense_ranking],
                                           [1.0, DENSE_WEIGHT])
            results.append([self.to_source(i) for i in fused[:k]])
        return results

    def to_source(self, chunk_id: int) -> MinimalSource:
        chunk = self.chunks[chunk_id]
        return MinimalSource(
            file_path=chunk.file_path,
            first_character_index=chunk.first_character_index,
            last_character_index=chunk.last_character_index,
        )
