"""Find the corpus files and split them into chunks."""

import os
from pathlib import Path

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from tqdm import tqdm

from .models import Chunk, RagError

LANGUAGES = {
    ".py": Language.PYTHON,
    ".md": Language.MARKDOWN,
    ".txt": Language.MARKDOWN,
}


def find_files(raw_dir: str) -> list[Path]:
    """Return the sorted .py, .md and .txt files in raw_dir.

    Args:
        raw_dir: Corpus folder.

    Returns:
        Paths of the files.

    Raises:
        RagError: If raw_dir is not a directory.
    """
    root = Path(raw_dir)
    if not root.is_dir():
        raise RagError(f"Corpus directory not found: {raw_dir}")
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix in LANGUAGES
    )


def read_file(path: str | Path) -> str:
    """Read a file exactly as stored, or '' if unreadable.

    newline="" keeps \\r\\n line endings, so offsets match the file on disk.

    Args:
        path: File to read.

    Returns:
        The file text.
    """
    try:
        with open(path, encoding="utf-8", errors="replace",
                  newline="") as file:
            return file.read()
    except OSError:
        return ""


def split_text(text: str, file_path: str,
               splitter: RecursiveCharacterTextSplitter,
               max_chunk_size: int) -> list[Chunk]:
    """Split one file's text into chunks of at most max_chunk_size chars.

    Args:
        text: File text.
        file_path: Path stored in each chunk.
        splitter: Splitter for this file type.
        max_chunk_size: Maximum chunk length in chars.

    Returns:
        The chunks, in file order.
    """
    chunks: list[Chunk] = []
    for document in splitter.create_documents([text]):
        start = document.metadata["start_index"]
        end = start + len(document.page_content)
        if text[start:end] != document.page_content:
            continue
        for piece_start in range(start, end, max_chunk_size):
            piece_end = min(piece_start + max_chunk_size, end)
            chunks.append(Chunk(
                file_path=file_path,
                first_character_index=piece_start,
                last_character_index=piece_end,
                text=text[piece_start:piece_end],
            ))
    return chunks


def chunk_corpus(raw_dir: str, max_chunk_size: int) -> list[Chunk]:
    """Split every corpus file into chunks.

    Args:
        raw_dir: Corpus folder.
        max_chunk_size: Maximum chunk length in chars.

    Returns:
        The chunks of all files.

    Raises:
        RagError: If raw_dir is missing or holds no content.
    """
    splitters = {
        suffix: RecursiveCharacterTextSplitter.from_language(
            language,
            chunk_size=max_chunk_size,
            chunk_overlap=0,
            add_start_index=True,
        )
        for suffix, language in LANGUAGES.items()
    }
    chunks: list[Chunk] = []
    for path in tqdm(find_files(raw_dir), desc="Chunking files"):
        file_path = Path(os.path.relpath(path)).as_posix()
        chunks.extend(split_text(read_file(path), file_path,
                                 splitters[path.suffix], max_chunk_size))
    if not chunks:
        raise RagError(f"No .py, .md or .txt content found in {raw_dir}")
    return chunks
