import ast
import re


def build_line_starts(text: str) -> list[int]:
    starts = [0] + [match.end() for match in re.finditer("\n", text)]
    if starts[-1] != len(text):
        starts.append(len(text))
    return starts


def node_line_range(node: ast.stmt) -> tuple[int, int]:
    decorators = getattr(node, "decorator_list", [])
    first_line = min([node.lineno] + [d.lineno for d in decorators])
    last_line = node.end_lineno or node.lineno
    return first_line, last_line


def lines_to_offsets(line_starts: list[int], first_line: int,
                     last_line: int) -> tuple[int, int]:
    return line_starts[first_line - 1], line_starts[last_line]
