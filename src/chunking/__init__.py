from ..models import Chunk, SourceFile
from .python_chunker import chunk_python
from .text_chunker import chunk_text


def chunk_file(source: SourceFile, max_chunk_size: int) -> list[Chunk]:
    if source.kind == "python":
        return chunk_python(source, max_chunk_size)
    return chunk_text(source, max_chunk_size)
