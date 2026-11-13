"""Recall@k evaluation with the subject's IoU rule."""

from .models import (
    AnsweredQuestion,
    MinimalSource,
    RagDataset,
    RagError,
    StudentSearchResults,
)

RECALL_KS = (1, 3, 5, 10)
IOU_THRESHOLD = 0.05


def iou(first: MinimalSource, second: MinimalSource) -> float:
    """Return the IoU of two sources, 0.0 if in different files.

    Args:
        first: First span.
        second: Second span.

    Returns:
        Overlap length divided by union length.
    """
    if first.file_path != second.file_path:
        return 0.0
    overlap = (
        min(first.last_character_index, second.last_character_index)
        - max(first.first_character_index, second.first_character_index)
    )
    if overlap <= 0:
        return 0.0
    union = (
        max(first.last_character_index, second.last_character_index)
        - min(first.first_character_index, second.first_character_index)
    )
    return overlap / union


def question_recall(truth: list[MinimalSource],
                    retrieved: list[MinimalSource]) -> float:
    """Return the share of truth sources found.

    Args:
        truth: Ground-truth sources.
        retrieved: Retrieved sources.

    Returns:
        Recall between 0.0 and 1.0.
    """
    found = sum(
        any(iou(source, result) >= IOU_THRESHOLD for result in retrieved)
        for source in truth
    )
    return found / len(truth)


def recall_at_k(results: StudentSearchResults,
                dataset: RagDataset) -> dict[int, float]:
    """Return the mean recall for each k in RECALL_KS.

    Args:
        results: Search results to score.
        dataset: Ground-truth dataset.

    Returns:
        Mean recall keyed by k.

    Raises:
        RagError: If no question has sources.
    """
    truth_by_id = {
        question.question_id: question.sources
        for question in dataset.rag_questions
        if isinstance(question, AnsweredQuestion) and question.sources
    }
    if not truth_by_id:
        raise RagError("The dataset has no questions with sources to check")
    retrieved_by_id = {
        result.question_id: result.retrieved_sources
        for result in results.search_results
    }
    recalls: dict[int, float] = {}
    for k in RECALL_KS:
        scores = [
            question_recall(truth, retrieved_by_id.get(question_id, [])[:k])
            for question_id, truth in truth_by_id.items()
        ]
        recalls[k] = sum(scores) / len(scores)
    return recalls
