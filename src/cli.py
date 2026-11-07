"""Command-line interface exposed through Python Fire."""

from pathlib import Path

from .io_utils import RagError, load_model, save_model
from .models import (
    MinimalAnswer,
    MinimalSearchResults,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)

MAX_SOURCE_LENGTH = 2000


def check_positive_int(name: str, value: object) -> int:
    """Return value if it is a positive int.

    Args:
        name: Option name used in the message.
        value: Value given on the command line.

    Returns:
        The value as an int.

    Raises:
        RagError: If value is not a positive int.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RagError(f"--{name} must be a positive integer, got {value!r}")
    return value


def check_query(query: object) -> str:
    """Return the query as stripped, non-empty text.

    Args:
        query: Query given on the command line.

    Returns:
        The stripped query.

    Raises:
        RagError: If the query is empty.
    """
    text = str(query).strip()
    if not text:
        raise RagError("The query must not be empty")
    return text


class Cli:
    """RAG commands: index, search, answer and evaluate."""
    def index(self, max_chunk_size: int = 2000, raw_dir: str = "data/raw",
              output_dir: str = "data/processed") -> None:
        """Chunk the corpus and build the search index.

        Args:
            max_chunk_size: Maximum chunk length, at most 2000.
            raw_dir: Folder with the corpus.
            output_dir: Folder to write the index to.

        Raises:
            RagError: If an argument is invalid or indexing fails.
        """
        from .indexer import build_index

        size = check_positive_int("max_chunk_size", max_chunk_size)
        if size > MAX_SOURCE_LENGTH:
            raise RagError(
                f"--max_chunk_size must be at most {MAX_SOURCE_LENGTH}: "
                "longer sources are rejected by the grader"
            )
        build_index(raw_dir, output_dir, size)
        print(f"Ingestion complete! Indices saved under {output_dir}/")

    def search(self, query: str, k: int = 10,
               index_dir: str = "data/processed") -> None:
        """Print the top-k sources for a query.

        Args:
            query: Question to search for.
            k: Number of sources to return.
            index_dir: Folder holding the index.

        Raises:
            RagError: If an argument is invalid or the index is missing.
        """
        from .retriever import Retriever

        text = check_query(query)
        count = check_positive_int("k", k)
        sources = Retriever(index_dir).search(text, count)
        if not sources:
            print("No results.")
        for rank, source in enumerate(sources, start=1):
            print(f"{rank:2}. {source.file_path} "
                  f"[{source.first_character_index}:"
                  f"{source.last_character_index}]")

    def search_dataset(self, dataset_path: str, k: int = 10,
                       save_directory: str = "data/output/search_results",
                       index_dir: str = "data/processed") -> None:
        """Search every dataset question and save results.

        Args:
            dataset_path: RagDataset JSON file.
            k: Number of sources per question.
            save_directory: Folder for the results.
            index_dir: Folder holding the index.

        Raises:
            RagError: If an argument, file or the index is invalid.
        """
        from .retriever import Retriever

        count = check_positive_int("k", k)
        dataset = load_model(dataset_path, RagDataset)
        questions = dataset.rag_questions
        all_sources = Retriever(index_dir).search_many(
            [question.question for question in questions], count
        )
        results = StudentSearchResults(
            search_results=[
                MinimalSearchResults(
                    question_id=question.question_id,
                    question=question.question,
                    retrieved_sources=sources,
                )
                for question, sources in zip(questions, all_sources)
            ],
            k=count,
        )
        output_path = Path(save_directory) / Path(dataset_path).name
        save_model(results, output_path)
        print(f"Saved student_search_results to {output_path}")

    def answer(self, query: str, k: int = 10,
               index_dir: str = "data/processed") -> None:
        """Answer a query from its retrieved sources.

        Args:
            query: Question to answer.
            k: Number of sources to retrieve.
            index_dir: Folder holding the index.

        Raises:
            RagError: If an argument is invalid or a model or the index
                is missing.
        """
        from .generator import Generator
        from .retriever import Retriever

        text = check_query(query)
        count = check_positive_int("k", k)
        sources = Retriever(index_dir).search(text, count)
        print(Generator().answer(text, sources))
        if sources:
            print("\nSources:")
        for source in sources:
            print(f"  {source.file_path} "
                  f"[{source.first_character_index}:"
                  f"{source.last_character_index}]")

    def answer_dataset(
        self, student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
    ) -> None:
        """Generate answers for a search results file.

        Args:
            student_search_results_path: StudentSearchResults JSON file.
            save_directory: Folder for the answers.

        Raises:
            RagError: If the file is invalid or the output cannot be written.
        """
        from tqdm import tqdm

        from .generator import Generator

        results = load_model(student_search_results_path,
                             StudentSearchResults)
        total = len(results.search_results)
        print(f"Loaded {total} questions")
        generator = Generator()
        answers: list[MinimalAnswer] = []
        for result in tqdm(results.search_results, desc="Answering"):
            answer = generator.answer(result.question,
                                      result.retrieved_sources)
            answers.append(MinimalAnswer(**result.model_dump(),
                                         answer=answer))
        print(f"Processed {len(answers)} of {total} questions")
        output = StudentSearchResultsAndAnswer(search_results=answers,
                                               k=results.k)
        output_path = (Path(save_directory)
                       / Path(student_search_results_path).name)
        save_model(output, output_path)
        print(f"Saved student_search_results_and_answer to {output_path}")

    def evaluate(self, student_search_results_path: str,
                 dataset_path: str) -> None:
        """Print recall@k of search results against a dataset.

        Args:
            student_search_results_path: StudentSearchResults JSON file.
            dataset_path: Ground-truth RagDataset JSON file.

        Raises:
            RagError: If a file is missing or invalid.
        """
        from .evaluation import recall_at_k

        results = load_model(student_search_results_path,
                             StudentSearchResults)
        dataset = load_model(dataset_path, RagDataset)
        recalls = recall_at_k(results, dataset)
        print(f"Questions evaluated: {len(dataset.rag_questions)}")
        for k, recall in recalls.items():
            print(f"Recall@{k}: {recall:.3f} ({recall:.1%})")
