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


def node_blocks(nodes: list[ast.stmt], line_starts: list[int],
                previous_last_line: int, scope: str,
                max_chunk_size: int) -> list[Block]:
    blocks: list[Block] = []
    for node in nodes:
        last_line = node.end_lineno or node.lineno
        start = line_starts[previous_last_line]
        end = line_starts[last_line]
        if isinstance(node, DEFINITION_TYPES):
            name = f"{scope}.{node.name}" if scope else node.name
            if isinstance(node, ast.ClassDef) and end - start > max_chunk_size:
                inner = node_blocks(node.body, line_starts, previous_last_line,
                                    name, max_chunk_size)
                inner[0].starts_section = True
                inner[0].context = inner[0].context or name
                blocks.extend(inner)
            else:
                blocks.append(Block(start, end, True, name))
        else:
            blocks.append(Block(start, end, False, scope))
        previous_last_line = last_line
    return blocks


def split_python_blocks(text: str, max_chunk_size: int) -> list[Block]:
    line_starts = build_line_starts(text)
    blocks = node_blocks(ast.parse(text).body, line_starts, 0, "",
                         max_chunk_size)
    last_end = blocks[-1].end if blocks else 0
    if last_end < len(text):
        blocks.append(Block(last_end, len(text), False, ""))
    return blocks


def chunk_python(source: SourceFile, max_chunk_size: int) -> list[Chunk]:
    blocks = split_python_blocks(source.text, max_chunk_size)
    return pack_blocks(source, blocks, max_chunk_size)
