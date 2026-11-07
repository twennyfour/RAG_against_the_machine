import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from .io_utils import RagError
from .models import MinimalSource, SourceFile

EXTENSION_KINDS: dict[str, Literal["python", "text"]] = {
    ".py": "python",
    ".md": "text",
    ".txt": "text",
}


def find_files(raw_dir: str | Path) -> list[Path]:
    root = Path(raw_dir)
    if not root.is_dir():
        raise RagError(f"Corpus directory not found: {raw_dir}")
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix in EXTENSION_KINDS
    )


def read_source_file(path: Path) -> SourceFile | None:
    try:
        with open(path, encoding="utf-8", errors="replace",
                  newline="") as file:
            text = file.read()
    except OSError:
        return None
    if not text.strip():
        return None
    return SourceFile(
        file_path=Path(os.path.relpath(path)).as_posix(),
        text=text,
        kind=EXTENSION_KINDS[path.suffix],
    )


@lru_cache(maxsize=256)
def read_text(file_path: str) -> str:
    try:
        with open(file_path, encoding="utf-8", errors="replace",
                  newline="") as file:
            return file.read()
    except OSError:
        return ""


def read_span(source: MinimalSource) -> str:
    text = read_text(source.file_path)
    return text[source.first_character_index:source.last_character_index]
