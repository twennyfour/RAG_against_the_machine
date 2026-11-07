"""Pydantic models exchanged between pipeline stages."""

import uuid
from typing import Literal

from pydantic import BaseModel, Field


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


class SourceFile(BaseModel):
    """Corpus file loaded in memory."""
    file_path: str
    text: str
    kind: Literal["python", "text"]


class Chunk(MinimalSource):
    """Indexed piece of a corpus file."""
    text: str
    kind: Literal["python", "text"]


class ChunkIndex(BaseModel):
    """Saved chunks and the chunk size used."""
    max_chunk_size: int
    chunks: list[Chunk]
