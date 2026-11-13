"""Command-line interface: uv run python -m src <command>."""

import sys
from pathlib import Path

import fire
from tqdm import tqdm

from .chunking import chunk_corpus
from .evaluation import recall_at_k
from .models import (
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    RagError,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    load_model,
    save_model,
)
from .retriever import Retriever, build_index

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


def print_sources(sources: list[MinimalSource]) -> None:
    """Print numbered sources, one per line.

    Args:
        sources: Sources to print, best first.
    """
    for rank, source in enumerate(sources, start=1):
        print(f"{rank:2}. {source.file_path} "
              f"[{source.first_character_index}:"
              f"{source.last_character_index}]")


class Cli:
    """RAG commands: index, search, answer and evaluate.

    The answer commands import the generator inside the method: torch and
    transformers are slow to import, and the other commands don't need them.
    """
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
        size = check_positive_int("max_chunk_size", max_chunk_size)
        if size > MAX_SOURCE_LENGTH:
            raise RagError(
                f"--max_chunk_size must be at most {MAX_SOURCE_LENGTH}: "
                "longer sources are rejected by the grader"
            )
        build_index(chunk_corpus(raw_dir, size), size, output_dir)
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
        text = check_query(query)
        count = check_positive_int("k", k)
        sources = Retriever(index_dir).search(text, count)
        if not sources:
            print("No results.")
        print_sources(sources)

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
        count = check_positive_int("k", k)
        dataset = load_model(dataset_path, RagDataset)
        retriever = Retriever(index_dir)
        search_results: list[MinimalSearchResults] = []
        for question in tqdm(dataset.rag_questions, desc="Searching"):
            sources = retriever.search(question.question, count)
            search_results.append(MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=sources,
            ))
        results = StudentSearchResults(search_results=search_results,
                                       k=count)
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
        text = check_query(query)
        count = check_positive_int("k", k)
        sources = Retriever(index_dir).search(text, count)

        from .generator import Generator
        print(Generator().answer(text, sources))
        if sources:
            print("\nSources:")
        print_sources(sources)

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
        results = load_model(student_search_results_path,
                             StudentSearchResults)
        total = len(results.search_results)
        print(f"Loaded {total} questions")

        from .generator import Generator
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
        results = load_model(student_search_results_path,
                             StudentSearchResults)
        dataset = load_model(dataset_path, RagDataset)
        recalls = recall_at_k(results, dataset)
        print(f"Questions evaluated: {len(dataset.rag_questions)}")
        for k, recall in recalls.items():
            print(f"Recall@{k}: {recall:.3f} ({recall:.1%})")


def main() -> None:
    """Run the CLI.

    Expected errors (RagError) print their message. Any other exception is
    a bug or an environment problem (out of memory, broken library); it is
    printed in one line too, so the user never sees a traceback.
    """
    try:
        fire.Fire(Cli())
    except RagError as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f"Unexpected error: {type(error).__name__}: {error}",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
