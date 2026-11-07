"""Build and save the search index."""

from pathlib import Path

from tqdm import tqdm

from .bm25 import BM25
from .chunking import chunk_file
from .corpus import find_files, read_source_file
from .embeddings import Embedder, save_embeddings
from .io_utils import RagError, save_model
from .models import Chunk, ChunkIndex
from .tokenizer import tokenize

CHUNKS_FILE = "chunks.json"


def indexed_text(chunk: Chunk) -> str:
    """Return the text indexed for a chunk.

    Args:
        chunk: Chunk to index.

    Returns:
        File path and chunk text.
    """
    return chunk.file_path + "\n" + chunk.text


def build_index(raw_dir: str | Path, output_dir: str | Path,
                max_chunk_size: int) -> int:
    """Index the corpus into output_dir, return chunk count.

    Args:
        raw_dir: Corpus folder.
        output_dir: Folder to write the index to.
        max_chunk_size: Maximum chunk length in chars.

    Returns:
        Number of indexed chunks.

    Raises:
        RagError: If the corpus is missing or empty, or the index cannot
            be saved.
    """
    chunks: list[Chunk] = []
    for path in tqdm(find_files(raw_dir), desc="Chunking files"):
        source = read_source_file(path)
        if source is not None:
            chunks.extend(chunk_file(source, max_chunk_size))
    if not chunks:
        raise RagError(f"No .py, .md or .txt content found in {raw_dir}")

    texts = [indexed_text(chunk) for chunk in chunks]
    print(f"Building BM25 index over {len(chunks)} chunks...")
    BM25.build(tokenize(texts)).save(output_dir)

    print("Computing embeddings...")
    vectors = Embedder().encode(texts, show_progress=True)
    save_embeddings(vectors, output_dir)

    save_model(ChunkIndex(max_chunk_size=max_chunk_size, chunks=chunks),
               Path(output_dir) / CHUNKS_FILE)
    return len(chunks)
