import re

from ..models import Chunk, SourceFile
from .common import Block, pack_blocks

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")


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
    return pack_blocks(source, split_blocks(source.text), max_chunk_size)
