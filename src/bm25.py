import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse

from .io_utils import RagError

WEIGHTS_FILE = "bm25_weights.npz"
VOCABULARY_FILE = "bm25_vocabulary.json"


class BM25:
    def __init__(self, vocabulary: dict[str, int],
                 weights: sparse.csc_matrix) -> None:
        self.vocabulary = vocabulary
        self.weights = weights

    @classmethod
    def build(cls, documents: list[list[str]], k1: float = 1.5,
              b: float = 0.75) -> "BM25":
        vocabulary: dict[str, int] = {}
        rows: list[int] = []
        columns: list[int] = []
        counts: list[int] = []
        lengths: list[int] = []
        for document_id, tokens in enumerate(documents):
            for term, count in Counter(tokens).items():
                term_id = vocabulary.setdefault(term, len(vocabulary))
                rows.append(document_id)
                columns.append(term_id)
                counts.append(count)
            lengths.append(len(tokens))

        n_documents = len(documents)
        n_terms = len(vocabulary)
        row_array = np.array(rows, dtype=np.int64)
        column_array = np.array(columns, dtype=np.int64)
        tf = np.array(counts, dtype=np.float32)
        length_array = np.array(lengths, dtype=np.float32)
        average_length = max(float(length_array.mean()), 1.0) \
            if n_documents else 1.0

        df = np.bincount(column_array, minlength=n_terms)
        idf = np.log(1.0 + (n_documents - df + 0.5) / (df + 0.5))
        length_norm = 1.0 - b + b * length_array[row_array] / average_length
        values = idf[column_array] * tf * (k1 + 1.0) / (tf + k1 * length_norm)

        weights = sparse.csc_matrix(
            (values.astype(np.float32), (row_array, column_array)),
            shape=(n_documents, n_terms),
        )
        return cls(vocabulary, weights)

    def save(self, directory: str | Path) -> None:
        folder = Path(directory)
        terms = sorted(self.vocabulary, key=self.vocabulary.__getitem__)
        try:
            folder.mkdir(parents=True, exist_ok=True)
            sparse.save_npz(folder / WEIGHTS_FILE, self.weights)
            with open(folder / VOCABULARY_FILE, "w",
                      encoding="utf-8") as file:
                json.dump(terms, file)
        except OSError as error:
            raise RagError(
                f"Cannot write BM25 index to {folder}: {error.strerror}"
            ) from None

    @classmethod
    def load(cls, directory: str | Path) -> "BM25":
        folder = Path(directory)
        try:
            weights = sparse.load_npz(folder / WEIGHTS_FILE).tocsc()
            with open(folder / VOCABULARY_FILE, encoding="utf-8") as file:
                terms = json.load(file)
        except FileNotFoundError:
            raise RagError(
                f"No index found in {folder}. Run the index command first."
            ) from None
        except (OSError, ValueError, zipfile.BadZipFile):
            raise RagError(
                f"BM25 index in {folder} is unreadable or damaged. "
                "Run the index command again."
            ) from None
        if len(terms) != weights.shape[1]:
            raise RagError(
                f"BM25 index in {folder} is corrupted. "
                "Run the index command again."
            )
        return cls({term: i for i, term in enumerate(terms)}, weights)

    def scores(self, tokens: list[str]) -> np.ndarray:
        term_ids = [self.vocabulary[t] for t in tokens if t in self.vocabulary]
        if not term_ids:
            return np.zeros(self.weights.shape[0], dtype=np.float32)
        return np.asarray(self.weights[:, term_ids].sum(axis=1)).ravel()
