"""Split source files into chunks with langchain text splitters."""

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from .io_utils import RagError
from .models import Chunk, SourceFile

SPLITTER_LANGUAGES = {
    "python": Language.PYTHON,
    "text": Language.MARKDOWN,
}


def chunk_file(source: SourceFile, max_chunk_size: int) -> list[Chunk]:
    """Split a file into chunks of at most max_chunk_size chars.

    Args:
        source: File to split.
        max_chunk_size: Maximum chunk length in chars.

    Returns:
        The chunks, in file order.

    Raises:
        RagError: If max_chunk_size is not positive.
    """
    if max_chunk_size <= 0:
        raise RagError("max_chunk_size must be a positive integer")
    splitter = RecursiveCharacterTextSplitter.from_language(
        SPLITTER_LANGUAGES[source.kind],
        chunk_size=max_chunk_size,
        chunk_overlap=0,
        add_start_index=True,
    )
    chunks: list[Chunk] = []
    for document in splitter.create_documents([source.text]):
        start = document.metadata["start_index"]
        end = start + len(document.page_content)
        if source.text[start:end] != document.page_content:
            continue
        chunks.append(Chunk(
            file_path=source.file_path,
            first_character_index=start,
            last_character_index=end,
            text=document.page_content,
            kind=source.kind,
        ))
    return chunks
