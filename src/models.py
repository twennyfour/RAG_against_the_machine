"""Pydantic models, their JSON files, and the project error."""

import uuid
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, Field, ValidationError


class RagError(Exception):
    """Expected error shown to the user without a traceback."""
    pass


class MinimalSource(BaseModel):
    """Character span [first, last) of a corpus file."""
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Question without a reference answer."""
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Question with its reference answer and sources."""
    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Dataset of questions."""
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Retrieved sources for one question."""
    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Retrieved sources and answer for one question."""
    answer: str


class StudentSearchResults(BaseModel):
    """Output of search_dataset."""
    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Output of answer_dataset."""
    search_results: list[MinimalAnswer]
    k: int


class Chunk(MinimalSource):
    """Indexed piece of a corpus file."""
    text: str


class ChunkIndex(BaseModel):
    """Saved chunks and the chunk size used."""
    max_chunk_size: int
    chunks: list[Chunk]


ModelT = TypeVar("ModelT", bound=BaseModel)


def load_model(path: str | Path, model_cls: type[ModelT]) -> ModelT:
    """Load and validate a JSON file as model_cls.

    Args:
        path: JSON file to read.
        model_cls: Pydantic model to validate with.

    Returns:
        The validated model.

    Raises:
        RagError: If the file is unreadable or invalid.
    """
    try:
        with open(path, encoding="utf-8") as file:
            content = file.read()
    except OSError as error:
        raise RagError(f"Cannot read {path}: {error.strerror}") from None
    except UnicodeDecodeError:
        raise RagError(f"Cannot read {path}: not a UTF-8 text file") from None

    try:
        return model_cls.model_validate_json(content)
    except ValidationError as error:
        first = error.errors()[0]
        location = ".".join(str(part) for part in first["loc"])
        raise RagError(
            f"Invalid {model_cls.__name__} file {path}: "
            f"{error.error_count()} error(s), first at '{location}': "
            f"{first['msg']}"
        ) from None


def save_model(model: BaseModel, path: str | Path) -> None:
    """Save a model as JSON, creating parent directories.

    Args:
        model: Model to save.
        path: Destination JSON file.

    Raises:
        RagError: If the file cannot be written.
    """
    output_path = Path(path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(model.model_dump_json(indent=2))
    except OSError as error:
        raise RagError(f"Cannot write {path}: {error.strerror}") from None
