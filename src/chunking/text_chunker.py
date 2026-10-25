import re
from dataclasses import dataclass

from ..io_utils import RagError
from ..models import Chunk, SourceFile
from .common import split_by_lines

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
MIN_CHUNK_SIZE = 200


@dataclass
class Block:
    start: int
    end: int
    starts_section: bool
    context: str


def split_blocks(text: str) -> list[Block]:
    blocks: list[Block] = []
    headings: list[tuple[int, str]] = []
    in_fence = False
    previous_blank = False
    offset = 0
    for line in text.splitlines(keepends=True):
        heading = None if in_fence else HEADING_RE.match(line)
        if heading:
            level = len(heading.group(1))
            while headings and headings[-1][0] >= level:
                headings.pop()
            headings.append((level, heading.group(2).strip()))
        is_blank = not line.strip()
        starts_paragraph = previous_blank and not is_blank and not in_fence
        if not blocks or heading or starts_paragraph:
            context = " > ".join(title for _, title in headings)
            blocks.append(Block(offset, offset, bool(heading), context))
        if FENCE_RE.match(line):
            in_fence = not in_fence
        offset += len(line)
        blocks[-1].end = offset
        previous_blank = is_blank
    return blocks


def chunk_text(source: SourceFile, max_chunk_size: int) -> list[Chunk]:
    if max_chunk_size <= 0:
        raise RagError("max_chunk_size must be a positive integer")
    text = source.text
    spans: list[Block] = []
    for block in split_blocks(text):
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
