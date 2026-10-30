import ast
import re

from ..models import Chunk, SourceFile
from .common import Block, pack_blocks

DEFINITION_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def build_line_starts(text: str) -> list[int]:
    starts = [0] + [match.end() for match in re.finditer("\n", text)]
    if starts[-1] != len(text):
        starts.append(len(text))
    return starts


def split_python_blocks(text: str) -> list[Block]:
    line_starts = build_line_starts(text)
    blocks: list[Block] = []
    previous_last_line = 0
    for node in ast.parse(text).body:
        last_line = node.end_lineno or node.lineno
        start = line_starts[previous_last_line]
        end = line_starts[last_line]
        if isinstance(node, DEFINITION_TYPES):
            blocks.append(Block(start, end, True, node.name))
        else:
            blocks.append(Block(start, end, False, ""))
        previous_last_line = last_line
    if line_starts[previous_last_line] < len(text):
        blocks.append(Block(line_starts[previous_last_line], len(text),
                            False, ""))
    return blocks


def chunk_python(source: SourceFile, max_chunk_size: int) -> list[Chunk]:
    blocks = split_python_blocks(source.text)
    return pack_blocks(source, blocks, max_chunk_size)
