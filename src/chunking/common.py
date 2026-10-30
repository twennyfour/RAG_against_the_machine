from dataclasses import dataclass

from ..io_utils import RagError
from ..models import Chunk, SourceFile

MIN_CHUNK_SIZE = 200


@dataclass
class Block:
    start: int
    end: int
    starts_section: bool
    context: str


def split_by_lines(text: str, start: int, end: int,
                   max_size: int) -> list[tuple[int, int]]:
    if end - start <= max_size:
        return [(start, end)]
    pieces: list[tuple[int, int]] = []
    piece_start = start
    position = start
    for line in text[start:end].splitlines(keepends=True):
        line_end = position + len(line)
        if line_end - piece_start > max_size and position > piece_start:
            pieces.append((piece_start, position))
            piece_start = position
        while line_end - piece_start > max_size:
            pieces.append((piece_start, piece_start + max_size))
            piece_start += max_size
        position = line_end
    if piece_start < end:
        pieces.append((piece_start, end))
    return pieces


def pack_blocks(source: SourceFile, blocks: list[Block],
                max_chunk_size: int) -> list[Chunk]:
    if max_chunk_size <= 0:
        raise RagError("max_chunk_size must be a positive integer")
    text = source.text
    spans: list[Block] = []
    for block in blocks:
        pieces = split_by_lines(text, block.start, block.end, max_chunk_size)
        for index, (start, end) in enumerate(pieces):
            starts_section = block.starts_section and index == 0
            current = spans[-1] if spans else None
            fits = (
                current is not None
                and end - current.start <= max_chunk_size
            )
            current_is_tiny_section = (
                current is not None
                and current.starts_section
                and current.end - current.start < MIN_CHUNK_SIZE
            )
            if current and fits and (
                not starts_section or current_is_tiny_section
            ):
                current.end = end
            else:
                spans.append(Block(start, end, starts_section, block.context))

    chunks: list[Chunk] = []
    for span in spans:
        content = text[span.start:span.end]
        if not content.strip():
            continue
        assert 0 < span.end - span.start <= max_chunk_size
        chunks.append(Chunk(
            file_path=source.file_path,
            first_character_index=span.start,
            last_character_index=span.end,
            text=content,
            kind=source.kind,
            context=span.context,
        ))
    return chunks
