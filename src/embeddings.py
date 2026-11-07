from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .io_utils import RagError

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDINGS_FILE = "embeddings.npy"


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            try:
                self.model = SentenceTransformer(
                    model_name, device=device, local_files_only=True
                )
            except OSError:
                self.model = SentenceTransformer(model_name, device=device)
        except (OSError, ValueError) as error:
            raise RagError(
                f"Cannot load embedding model {model_name}: {error}"
            ) from None

    def encode(self, texts: list[str],
               show_progress: bool = False) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=128,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)


def save_embeddings(vectors: np.ndarray, directory: str | Path) -> None:
    folder = Path(directory)
    try:
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / EMBEDDINGS_FILE, vectors, allow_pickle=False)
    except OSError as error:
        raise RagError(
            f"Cannot write embeddings to {folder}: {error.strerror}"
        ) from None


def load_embeddings(directory: str | Path) -> np.ndarray:
    path = Path(directory) / EMBEDDINGS_FILE
    if not path.is_file():
        raise RagError(
            f"No embeddings found in {directory}. Run the index command first."
        )
    try:
        return np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    except (OSError, ValueError):
        raise RagError(
            f"Embeddings in {directory} are unreadable or damaged. "
            "Run the index command again."
        ) from None


def dense_top_k(vectors: np.ndarray, query_vector: np.ndarray,
                k: int) -> list[tuple[int, float]]:
    if k <= 0 or len(vectors) == 0:
        return []
    scores = vectors @ query_vector
    top = np.argsort(-scores)[:k]
    return [(int(index), float(scores[index])) for index in top]
